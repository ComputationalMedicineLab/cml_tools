"""
Functions for visualizing the results of a model which produces mean/variance
estimates, trained with e.g. pytorch's `torch.nn.GaussianNLLLoss` loss
function.
"""
import matplotlib.pyplot as plt
import numpy as np
RNG = np.random.default_rng()


def get_subset(pred_mean, pred_vars, true_vals, nmax=10_000, mask=None, rng=RNG):
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
    n = len(pred_mean)
    if nmax < n:
        indices = RNG.choice(n, size=nmax, replace=False, shuffle=False)
        indices = indices[np.argsort(pred_vars[indices])]
    else:
        indices = np.argsort(pred_vars)
    batch_mean = np.copy(pred_mean[indices], order='C')
    batch_vars = np.copy(pred_vars[indices], order='C')
    batch_true = np.copy(true_vals[indices], order='C')
    return batch_mean, batch_vars, batch_true


def get_err_interval(pred_mean, pred_vars, true_vals):
    """
    Produce error and confidence interval ndarrays from input ndarrays of
    predicted mean, predicted variance, and prediction target, associated by
    index.
    """
    errors = (pred_mean - true_vals)
    intervals = 1.96 * np.sqrt(pred_vars)
    return errors, intervals


def plot_errors_and_intervals(errors, intervals, clip=None, xoffset=0, ax=None):
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
                        alpha=0.5, color='C0')
        ax.fill_between(xs[~mask], intervals[~mask], -intervals[~mask],
                        alpha=0.5, color='C1')
        mask = np.abs(errors) < clip
        errors = np.clip(errors, a_min=-clip, a_max=clip)
        ax.scatter(xs[mask], errors[mask], marker='+', color='C0')
        ax.scatter(xs[~mask], errors[~mask], marker='+', color='C1')
    else:
        ax.fill_between(xs, intervals, -intervals, alpha=0.5)
        ax.scatter(xs, errors, marker='+', color='C0')
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
                                                    mask=mask, rng=rng)
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
    return ax
