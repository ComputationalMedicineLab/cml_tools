from collections import defaultdict
from copy import deepcopy

import numpy as np
from cml.stats.incremental import concat_obs, extend_obs, IncrStats

# Default standardizer arguments, exactly as specified under the legacy format
DEFAULT_MODE_PARAMS = {
    'Age': {'kind': 'gelman'},
    'Sex': {'kind': 'identity'},
    'Race': {'kind': 'identity'},
    'Condition': {'kind': 'gelman_with_fallbacks',
                  'noshift': True,
                  'log': True,
                  'postfill': True,
                  'fill': 1e-6,
                  'eps': 1e-6,
                  'agg_mode': True},
    'Medication': {'kind': 'gelman',
                   'noshift': True,
                   'postfill': True,
                   'agg_mode': True},
    'Measurement': {'kind': 'gelman_with_fallbacks'},
}


class Log10Scaler:
    fields = ('labels', 'shift', 'scale', 'log10', 'eps')
    __slots__ = fields

    def __init__(self, labels, shift, scale, log10, eps=1e-6):
        self.labels = labels
        self.shift = shift
        self.scale = scale
        self.log10 = log10
        self.eps = eps

    @property
    def astuple(self):
        return (self.labels, self.shift, self.scale, self.log10, self.eps)

    def label_index(self, labels):
        _, K, _ = np.intersect1d(self.labels, labels, assume_unique=True, return_indices=True)
        return K

    def apply(self, X, labels):
        """
        Optionally log transforms the data. Then subtracts the shift and
        divides by the scale: `y = (x - shift) / scale`.

        Equivalent to a matricized, inplace version of:
        >>> if log:
        ...     x = np.log10(x + eps)
        >>> x = (x - shift) / scale
        """
        # What if not every label has an entry in self.labels? Do nothing by
        # default? Make ones/zeros and overwrite with scale/shift?
        K = self.label_index(labels)
        if np.any(mask := self.log10[K]):
            np.add(X, self.eps, out=X, where=mask)
            np.log10(X, out=X, where=mask)
        np.subtract(X, self.shift[K], out=X)
        np.divide(X, self.scale[K], out=X)
        return X
    __call__ = apply

    def apply_inverse(self, X, labels):
        """
        Undoes the effects of apply. Equivalent to a matricized, inplace
        version of:
        >>> y = (scale * x) + shift
        >>> if log:
        ...     x = np.power(10, x) - eps
        """
        K = self.label_index(labels)
        np.multiply(X, self.scale[K], out=X)
        np.add(X, self.shift[K], out=X)
        if np.any(mask := self.log10[K]):
            np.power(10, X, out=X, where=mask)
            np.subtract(X, self.eps, out=X, where=mask)
        return X

    def format_impact(self, label, delta, anchor=1.0, spec='+.2f'):
        """
        Provides the meaning of `delta` in the original space of the channel
        specified by `label`, with format `spec` and reference to `anchor`.

        For most transforms, an additive change in the amount `delta`
        corresponds to a scaled but still additive amount in the original
        space. In this case, `delta` in the transformed space is simply scaled
        to the original space and given an additive label. If the transform
        includes a logarithm operation, then an additive `delta` in the
        original space corresponds to a multiplicative change in the original
        space. In this case, `delta` is appropriately scaled and given the
        appropriate multiplicative label.
        """
        x = self.apply_inverse(np.array([anchor, anchor+delta]), [label])
        if self.log10[self.label_index(label)]:
            spec = spec.lstrip('+')
            if x[0] < x[1]:
                prefix = 'x'
                impact = x[1] / x[0]
            else:
                prefix = '/'
                impact = x[0] / x[1]
        else:
            prefix = ''
            impact = x[1] - x[0]
        return f'{prefix}{impact:{spec}}'


def make_log10scaler(base_stat: IncrStats,
                     lg10_stat: IncrStats,
                     label_modes: dict[str, np.ndarray],
                     n_obs: int = None,
                     mode_params=DEFAULT_MODE_PARAMS):
    """
    Makes a Log10Scaler from incrementally harvested statistics and metadata
    about the labels in the statistics.
    """
    # This doesn't *need* to be True, but the below would need changing
    assert np.all(base_stat.labels == lg10_stat.labels)

    n_channel = len(base_stat.labels)
    shift_vec = np.empty(n_channel)
    scale_vec = np.empty(n_channel)
    lg10_mask = np.zeros(n_channel).astype(bool)

    # If we need to do any mode-level adjustments do them up front
    if any(
            p.get('postfill') or p.get('agg_mode')
            for p in mode_params.values()
    ):
        # Invert and aggregate the label to mode mapping
        mode_labels = defaultdict(list)
        for k, v in label_modes.items():
            mode_labels[v].append(k)
        # np.unique(v) is equivalent to np.array(sorted(set(v)))
        mode_labels = {k: np.unique(v) for k, v in mode_labels.items()}
        for mode, params in mode_params.items():
            # If there are no mapped labels then this mode doesn't exist, so
            # its parameters don't matter. Just skip ahead.
            if (labels := mode_labels.get(mode)) is None:
                continue
            if params.get('postfill'):
                if n_obs is None:
                    raise ValueError('n_obs required with "postfill"')
                fill = params.get('fill', 0.0)
                base_stat = extend_obs(base_stat, n_obs, fill, labels)
                # If this mode isn't set to use log, or log10(fill) isn't a
                # real, finite number, then don't bother crunching these
                # numbers for the log transformed stats
                if fill > 0 and params.get('log'):
                    lg10_fill = np.log10(fill)
                    lg10_stat = extend_obs(lg10_stat, n_obs, lg10_fill, labels)
            if params.get('agg_mode'):
                base_stat = concat_obs(base_stat, labels)
                # Again, don't waste time doing this unless the mode wants log
                if params.get('log'):
                    lg10_stat = concat_obs(lg10_stat, labels)
        # This isn't available if the `if` never triggered
        del mode_labels

    for i, label in enumerate(base_stat.labels):
        mode = label_modes[label]
        params = mode_params[mode]

        if (log := params.get('log')):
            shift = lg10_stat.mean[i]
            scale = np.sqrt(lg10_stat.variance[i])
        else:
            shift = base_stat.mean[i]
            scale = np.sqrt(base_stat.variance[i])

        if params.get('noshift'):
            shift = 0.0

        bmax = base_stat.maxval
        bmin = base_stat.minval
        rel_const = (bmax - bmin) < 1e-6 * (bmax + bmin)

        match params.get('kind'):
            case 'identity':
                shift = 0.0
                scale = 1.0
            case 'standard':
                pass
            case 'gelman':
                scale *= 2
            case 'gelman_with_fallbacks':
                # If the curve is nearly constant shift by its own minval; i.e.
                # standardize the curve to be almost surely zero.
                if np.isclose(0.0, base_stat.variance[i]) or rel_const[i]:
                    shift = base_stat.minval[i]
                    scale = 1.0
                    log = False
                # Ignore the log param if too many (> 1%) negatives.
                elif base_stat.negative[i] > (0.01 * base_stat.count[i]):
                    shift = base_stat.mean[i]
                    scale = 2 * np.sqrt(base_stat.variance[i])
                    log = False
                # Otherwise we are clear to use the log transform (if given);
                # in which case the log (or not) and corresponding scale/shift
                # were determined above, we only need to apply the Gelman part.
                else:
                    scale *= 2
            case _:
                raise RuntimeError(f'Unknown standardization method: {kind}')

        lg10_mask[i] = log
        shift_vec[i] = shift
        scale_vec[i] = scale
    return Log10Scaler(base_stat.labels, shift_vec, scale_vec, lg10_mask)
