"""Functions for visualizing loss curves"""
import matplotlib.pyplot as plt
import numpy as np


def plot_epoch_losses(train_loss=None, test_loss=None, *,
                      reshape=None, agg=np.mean, a_min=None, a_max=None,
                      ax=None, marker='o', alpha=0.25):
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

    # Agg must be some kind of ufunc (min/max/mean) that has an "axis" arg
    if agg is not None:
        ys0 = agg(ys0, axis=-1)
        ys1 = agg(ys1, axis=-1)

    # Handle clipping: mark where the clip occurs before actually doing it.
    def scatter_from_mask(mask, y):
        xs = np.argwhere(mask).ravel()
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
    if train_loss is not None:
        ax.plot(ys0, marker=marker, alpha=alpha, label='Train Loss')

    if test_loss is not None:
        ax.plot(ys1, marker=marker, alpha=alpha, label='Test Loss')

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
