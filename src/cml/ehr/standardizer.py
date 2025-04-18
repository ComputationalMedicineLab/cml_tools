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
    'Measurement': {'kind': 'gelman_with_fallbacks',
                    'eps': 1e-6},
}


class Log10Scaler:
    __slots__ = ('labels', 'shift', 'scale', 'log10', 'eps')

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
        # What if not every label has an entry in self.labels? Do nothing by
        # default? Make ones/zeros and overwrite with scale/shift?
        K = self.label_index(labels)
        if np.any(mask := self.log10[K]):
            np.add(X, self.eps, out=X, where=mask)
            np.log10(X, out=X, where=mask)
        np.multiply(X, self.scale[K], out=X)
        np.add(X, self.shift[K], out=X)
        return X

    def apply_inverse(self, X, labels):
        K = self.label_index(labels)
        np.subtract(X, self.shift[K], out=X)
        np.divide(X, self.scale[K], out=X)
        if np.any(mask := self.log10[K]):
            np.power(10, X, out=X, where=mask)
            np.subtract(X, self.eps, out=X, where=mask)
        return X

    def format_impact(self, label, delta, anchor=1, spec='+.2f'):
        x = self.apply_inverse(np.array([anchor, anchor+delta]), [label])
        impact = x[1] - x[0]
        prefix = ''
        if self.log10[self.label_index(label)]:
            spec = spec.lstrip('+')
            if impact > 1.0:
                prefix = 'x'
            else:
                prefix = '/'
                impact = 1.0 / impact
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
            labels = mode_labels[mode]
            if params.get('postfill'):
                if n_obs is None:
                    raise ValueError('n_obs required with "postfill"')
                fill = params.get('fill', 0.0)
                base_stat = extend_obs(base_stat, n_obs, fill, labels)
                lg10_stat = extend_obs(lg10_stat, n_obs, np.log10(fill), labels)
            if params.get('agg_mode'):
                base_stat = concat_obs(base_stat, labels)
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
