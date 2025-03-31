"""Functions for visualizing loss curves"""
import matplotlib.pyplot as plt
import numpy as np


def plot_epoch_losses(train_loss=None, test_loss=None, *, model_id='',
                      reshape=None, agg=np.mean, a_min=None, a_max=None,
                      ax=None, marker='o', alpha=0.25, mark_min=True):
    """
    Each loss array (`train_loss` and `test_loss`) is a 2-dim ndarray of
    loss values of dimensions `[epochs, values]`. `values` can be loss values
    for either batches or instances.
    """
    if train_loss is None and test_loss is None:
        raise ValueError('Provide at least one of train_loss or test_loss')

    # plt.figure(figsize=(12, 4)); or (9, 3), etc - is usually the right dim.
    if ax is None:
        ax = plt.gca()

    # Use dummy variables
    ys0 = train_loss if train_loss is not None else np.array([0])
    ys1 = test_loss if test_loss is not None else np.array([0])
    # This is so that if, for example, train_loss is a 1-dim vector of 10_000
    # steps (or something like that) we can use reshape=(100, -1) and an agg
    # function like np.mean to get the average per 10 steps over 100 steps.
    if reshape is not None:
        ys0 = ys0.reshape(reshape)
        ys1 = ys1.reshape(reshape)

    # Using a list comp here instead of an axis argument allows (1) the agg
    # func to be a func that doesn't take that argument, but more importantly
    # (2) we can use array_split or the like to pass in inhomegeous lists of
    # arrays and still get an aggregation per batch (say we have some random
    # number of steps, 3512901 or something, and want to visualize the mean per
    # 1000 steps: this is difficult to do with reshaping and axis args etc).
    if agg is not None:
        ys0 = np.array([agg(v) for v in ys0])
        ys1 = np.array([agg(v) for v in ys1])

    # Handle clipping: mark where the clip occurs before actually doing it.
    def scatter_from_mask(mask, y):
        xs = np.argwhere(mask).ravel() + 1
        ys = np.full_like(xs, y)
        ax.scatter(xs, ys, marker='x', color='black', zorder=3)

    if a_min is not None and train_loss is not None:
        scatter_from_mask(ys0 < a_min, a_min)

    if a_min is not None and test_loss is not None:
        scatter_from_mask(ys1 < a_min, a_min)

    if a_max is not None and train_loss is not None:
        scatter_from_mask(ys0 > a_max, a_max)

    if a_max is not None and test_loss is not None:
        scatter_from_mask(ys1 > a_max, a_max)

    if (a_min is not None) or (a_max is not None):
        ys0 = np.clip(ys0, a_min=a_min, a_max=a_max)
        ys1 = np.clip(ys1, a_min=a_min, a_max=a_max)

    # x-axis should now have a point per epoch. Plot the loss curves at last.
    def plot_curve(ys, name):
        xs = np.arange(1, len(ys)+1)
        ys_min = np.min(ys)
        ys_argmin = np.argmin(ys)+1
        label = f'{name} Loss (min {ys_min:.4f}; {ys_argmin})'
        ax.plot(xs, ys, marker=marker, alpha=alpha, label=label)
        if mark_min:
            ax.scatter(ys_argmin, ys_min, color='C1', marker='x')

    if train_loss is not None:
        plot_curve(ys0, f'{model_id} Train')

    if test_loss is not None:
        plot_curve(ys1, f'{model_id} Test')

    # Get the aggregation function name: if it exists, format it. Set title.
    if (name := getattr(agg, '__name__', '')):
        name = name.title() + ' '
    if train_loss is None:
        title = f'{name}Test Loss per Epoch'
    elif test_loss is None:
        title = f'{name}Train Loss per Epoch'
    else:
        title = f'{name}Train/Test Loss per Epoch'

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{name}Loss')
    ax.set_title(title)
    return ax


def plot_result_histograms(losses, predictions, variances=None, epoch=None,
                           figsize=(12, 4), n_bins=100, yscale='log'):
    """
    Produce a 2 or 3-panel overview of a model's outputs. If `variances` is
    given, 3 panels are produces; otherwise only the losses and predictions are
    histogrammed.

    Arguments
    ---------
    losses: (ndarray, ndarray)
    predictions: (ndarray, ndarray)
    variances: (ndarray, ndarray)
        Each ndarray is either 1 or 2 dimensions. If 1 dim, then they are the
        losses (or predictions, or variances) per instance of a single epoch.
        If 2-dim, then the dimensions are [epochs, instances]. The histograms
        are flattened over all epochs provided. The first ndarray of each tuple
        are values from the training cycles, and the second ndarray are value
        from the test cycles.
    """
    ncols = 2 if variances is None else 3
    fig, axes = plt.subplots(figsize=figsize, ncols=ncols, layout='constrained')

    def _hist(ax, X, label, n_bins=n_bins):
        # X is a (train, test) 2-tuple: put them both into the same bins
        X0, X1 = X[0].ravel(), X[1].ravel()
        lo = min(min(X0), min(X1))
        hi = max(max(X0), max(X1))
        bins = np.linspace(lo, hi, n_bins)
        ax.hist(X0, bins=bins, histtype='step', label=f'Train {label}')
        ax.hist(X1, bins=bins, histtype='step', label=f'Test {label}')

    _hist(axes[0], losses, 'Loss')
    _hist(axes[1], predictions, 'Prediction')
    if variances is not None:
        _hist(axes[2], variances, 'Prediction Variance')

    for ax in axes:
        ax.legend()
        ax.set_yscale(yscale)

    if len(losses[0].shape) < 2:
        n_train = len(losses[0])
        n_test = len(losses[1])
        epochs = 1
    else:
        n_train = len(losses[0][0])
        n_test = len(losses[1][0])
        epochs = len(losses[0])

    fig.suptitle(f'Overall Results (N={n_train}/{n_test}; '
                 f'Trained {epochs:,} Epoch(s))')
    return fig, ax
