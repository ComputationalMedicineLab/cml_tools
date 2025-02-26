"""Functions for visualizing loss curves"""
import matplotlib.pyplot as plt
import numpy as np


def plot_epoch_losses(train_losses, test_losses=None, ax=None, agg=np.mean):
    """
    Each loss array (`train_losses` and `test_losses`) is a 2-dim ndarray of
    loss values of dimensions `[epochs, values]`. `values` can be loss values
    for either batches or instances.
    """
    # plt.figure(figsize=(12, 4)); or (9, 3), etc - is usually the right dim.
    if ax is None:
        ax = plt.gca()

    # Get the aggregation function name: if it exists, format it
    if (name := getattr(agg, '__name__', '')):
        name = name.title() + ' '

    # x-axis has a point per epoch
    xs = np.arange(len(train_losses))
    ys = agg(train_losses, axis=1)
    ax.plot(xs, ys, marker='o', label='Train Loss')
    title = f'{name}Train Loss per Epoch'

    if test_losses is not None:
        ys = agg(test_losses, axis=1)
        ax.plot(xs, ys, marker='o', label='Test Loss')
        title = f'{name}Train/Test Loss per Epoch'

    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{name}Loss')
    ax.set_title(title)
    return ax
