"""
Functions for visualizing the results of a model which produces mean/variance
estimates, trained with e.g. pytorch's `torch.nn.GaussianNLLLoss` loss
function.
"""
import matplotlib.pyplot as plt
import numpy as np
RNG = np.random.default_rng()


def get_nmax_sorted_indices(vals, nmax=10_000, rng=RNG):
    """Get a subset of `nmax` indices into `vals`, sorted by `vals`"""
    if nmax < (n := len(vals)):
        indices = rng.choice(n, size=nmax, replace=False, shuffle=False)
        indices = indices[np.argsort(vals[indices])]
    else:
        indices = np.argsort(vals)
    return indices


def get_subset(pred_mean, pred_vars, true_vals, nmax=10_000, mask=None,
               return_indices=True, rng=RNG):
    """
    `pred_mean`, `pred_vars`, and `true_vals` are 1-dim ndarrays with the
    predicted mean, predicted variance, and true target value for some set of
    instances. Up to `nmax` indices into these arrays are selected uniformly at
    random. `mask`, if given, will subset the predictions *before* any random
    selection, so that, e.g., selection can be limited to categories of
    instance by the calling function. The selected values are sorted by
    predicted variance ascending before return. Return values are copies.
    """
    if mask is not None:
        pred_mean = pred_mean[mask]
        pred_vars = pred_vars[mask]
        true_vals = true_vals[mask]
    indices = get_nmax_sorted_indices(pred_vars, nmax=nmax, rng=rng)
    batch_mean = np.copy(pred_mean[indices], order='C')
    batch_vars = np.copy(pred_vars[indices], order='C')
    batch_true = np.copy(true_vals[indices], order='C')
    if return_indices:
        return batch_mean, batch_vars, batch_true, indices
    else:
        return batch_mean, batch_vars, batch_true


def get_conf_interval(variances, scale=1.96):
    """Produce a confidence interval from variances"""
    # Writing `intervals = 1.96 * np.sqrt(pred_vars))` causes some bizarre
    # error (silent segfault?) in jupyterlab notebooks which kills the kernel.
    # No idea why. Best to encapsulate this process.
    return np.multiply(scale, np.sqrt(variances))


def get_err_interval(pred_mean, pred_vars, true_vals):
    """
    Produce error and confidence interval ndarrays from input ndarrays of
    predicted mean, predicted variance, and prediction target, associated by
    index.
    """
    errors = (pred_mean - true_vals)
    intervals = get_conf_interval(pred_vars)
    return errors, intervals


def plot_errors_and_intervals(errors, intervals, clip=None, xoffset=0, ax=None,
                              marker='+', C0='C0', C1='C1'):
    """
    Produce a visualization of a set of errors and associated confidence
    intervals. The errors are scattered and the confidence intervals are filled
    between.

    If `clip` is given, the ylimits are truncated to +/- clip; any intervals
    extending beyond the clipped region are plotted in `C1` rather than `C0`.

    If `xoffset` is given, the x-axis is shifted by `xoffset`. This allows the
    caller to place multiple error-interval plots side by side in a single
    Axis.
    """
    # errors is going to be (predicted_mean - real_value)
    # intervals as a function of the predictions is 1.96*sqrt(predicted_var)
    # see get_err_interval
    assert np.all(intervals > 0.0)
    if ax is None:
        ax = plt.gca()
    xs = np.arange(len(intervals)) + xoffset
    if clip:
        mask = intervals < clip
        intervals = np.clip(intervals, a_min=-clip, a_max=clip)
        ax.fill_between(xs[mask], intervals[mask], -intervals[mask],
                        alpha=0.5, color=C0)
        ax.fill_between(xs[~mask], intervals[~mask], -intervals[~mask],
                        alpha=0.5, color=C1)
        mask = np.abs(errors) < clip
        errors = np.clip(errors, a_min=-clip, a_max=clip)
        ax.scatter(xs[mask], errors[mask], marker=marker, color=C0)
        ax.scatter(xs[~mask], errors[~mask], marker=marker, color=C1)
    else:
        ax.fill_between(xs, intervals, -intervals, color=C0, alpha=0.5)
        ax.scatter(xs, errors, marker=marker, color=C0)
    ax.set_xlabel(r'$i^{th}$ Prediction')
    ax.set_ylabel('Error')
    ax.set_title('Prediction Errors')
    return ax


def plot_prediction_errors(pred_mean, pred_vars, true_vals,
                           nmax=10_000, mask=None, figsize=(9, 6),
                           clip=None, xoffset=0, rng=RNG):
    """
    Produce a plot directly from predictions. For argument details, see the
    function `get_subset` `get_err_interval` `plot_errors_and_intervals`.
    """
    batch_mean, batch_vars, batch_true = get_subset(pred_mean, pred_vars,
                                                    true_vals, nmax=nmax,
                                                    mask=mask, rng=rng,
                                                    return_indices=False)
    errs, ints = get_err_interval(batch_mean, batch_vars, batch_true)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    plot_errors_and_intervals(errs, ints, clip=clip, xoffset=xoffset, ax=ax)

    if nmax is None:
        nmax_s = 'All Data'
    elif nmax >= 1000:
        if nmax % 1000 > 0:
            nmax_s = f'Approx. {nmax // 1000}k samples'
        else:
            nmax_s = f'{nmax // 1000}k samples'
    else:
        nmax_s = f'{nmax} samples'
    ax.set_title(f'Prediction Errors ({nmax_s})')
    return fig, ax


def plot_prediction_by_epoch(target, means, variances, ax=None):
    """
    Plot the target value, predicted value, and confidence interval over some
    number of batches or epochs (assumed to be epochs).
    """
    if ax is None:
        ax = plt.gca()

    means = np.array(means)
    confs = get_conf_interval(variances)
    xs = np.arange(len(means))

    ax.scatter(xs, means, alpha=0.5, label='Predicted Value')
    ax.vlines(xs, means+confs, means-confs, alpha=0.25)
    ax.axhline(target, linestyle='--', alpha=0.5)
    ax.set_title(f'Prediction and Confidence Interval as a function of Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    return ax


def plot_sequence_predictions(deltas, values, pred_mean, pred_vars, ax=None,
                              plot_error=None, err_color='C1'):
    """
    Plot a sequence of target values against a Gaussian model's predicted mean
    and variance. If `plot_error`, the targets which are beyond the predicted
    mean's confidence interval are highlighted `err_color` (C1 by default). If
    `plot_error` is a string containing:

    1. "inline": the errors are plotted between target and prediction.
    2. "oob" or "out_of_bound": the errs are *only* plotted which lie beyond
       the prediction confidence interval.
    3. "offset": the errors are plotted as magnitudes from y = 0.
    """
    if ax is None:
        ax = plt.gca()

    err, conf = get_err_interval(pred_mean, pred_vars, values)

    # Basic Plot: 'x' for targets, 'o' for predictions, vline conf intervals
    ax.scatter(deltas, pred_mean, zorder=-2.5, label='Predictions')
    ax.vlines(deltas, pred_mean+conf, pred_mean-conf, alpha=0.25,
              label='Prediction Confidence')
    ax.scatter(deltas, values, color='black', marker='x',
               zorder=2.5, label='Target Value')

    if plot_error:
        errw = np.where(err > 0, pred_mean-conf, pred_mean+conf)
        mask = np.abs(err) > conf
        # Always color the target markers for targets which are outside the
        # confidence interval of the prediction.
        ax.scatter(deltas[mask], values[mask], alpha=0.9,
                   color=err_color, marker='x', zorder=2.5,
                   label='Prediction Out of Interval')
        # Plot the distance from prediction to target
        if plot_error == 'inline':
            ax.vlines(deltas, values, pred_mean,
                      color=err_color, alpha=0.75,
                      label='Prediction Error')
        # Plot the distance from prediction to nearest confidence interval edge
        # only for predictions which are beyond the confidence interval.
        if plot_error in ('oob', 'out_of_bound'):
            ax.vlines(deltas[mask], errw[mask], values[mask],
                      color=err_color, alpha=0.75,
                      label='Prediction Error (Beyond CI)')
        # Plot the error as offset from zero.
        if plot_error == 'offset':
            ax.axhline(0.0, linestyle='--', color='black', alpha=0.25)
            ax.vlines(deltas, 0.0, err, color=err_color, alpha=0.75,
                      label='Prediction Error (Absolute)')

    # Generic, default labels - in practice, the caller will set these as
    # needed (i.e., xlabel "Fractional Days"; ylabel "Potassium Value"; etc).
    ax.set_xlabel('Offset Value')
    ax.set_ylabel('Target Value')
    ax.set_title('Predictions over a Sequence of Targets')

    ax.legend()
    return ax


def plot_sequence_interpolations(deltas, values, points,
                                 interpolations,
                                 interpolation_variances,
                                 predictions=None,
                                 prediction_variances=None,
                                 ax=None, plot_guides=True):
    """Plot a set of interpolations against the target values"""
    if ax is None:
        ax = plt.gca()

    iconf = get_conf_interval(interpolation_variances)

    ax.plot(points, interpolations, label='Interpolations', zorder=-2.5)
    ax.fill_between(points, interpolations+iconf, interpolations-iconf,
                    alpha=0.25, label='Interpolation Confidence')

    # TODO: annotate the target guide lines with the corresponding values?
    if plot_guides:
        for x in deltas:
            ax.axvline(x, linestyle='--', alpha=0.25)

    ax.scatter(deltas, values, color='black', marker='x',
               zorder=2.5, label='Target Value')

    if predictions is not None:
        ax.scatter(deltas, predictions, zorder=-2.5, label='Predictions')
        if prediction_variances is not None:
            pconf = get_conf_interval(prediction_variances)
            ax.vlines(deltas, predictions+pconf, predictions-pconf,
                      alpha=0.75, color='black', linestyle='--',
                      label='Prediction Confidence')

    # See note above about generic labels - caller is responsible for specifics
    ax.set_xlabel('Offset Value')
    ax.set_ylabel('Target Value')
    ax.set_title('Interpolations over a Sequence of Targets')

    ax.legend()
    return ax
