"""
We commonly group EHR data into channels using entities such as ICD codes,
medication ingredients, or the `concept_id` of the OMOP model. Frequently we
want to know per channel how many total observations there are, and how many
unique persons have observations in that channel. This module has functions
useful for visualizing the number of mentions and persons per channel for large
groups of channels (all SNOMEDs meeting a certain condition, for example). This
is useful for determining thresholds to use during data selection, and
understanding how many of our data channels are very rare or very common.

Throughout, the functions operate on pandas.DataFrame objects which are assumed
to have at least the three columns: `concept_id`, `mentions`, `persons`; the
`concept_id` is a unique id for the channel, mentions and persons are counts.
"""
from itertools import product

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


_what_is_df = """\
`df` is a dataframe with columns ['concept_id', 'mentions', 'persons'],
giving for each `concept_id` (aka data channel for this `mode`) the number
of total observations and the number of unique persons.
"""


def plot_threshold_heatmap(df, mode, ax=None, thresh=None):
    f"""
    {_what_is_df}

    This function produces a heatmap of how many channels are above each
    threshold given by `thresh` (defaults 1, 10, 100, 500, 1000, 2000). The
    x-axis values are mentions, the y-axis values are persons.
    """
    if ax is None:
        ax = plt.gca()
    N = thresh or (1, 10, 100, 500, 1000, 2000)
    vals = np.zeros((len(N), len(N)))
    for i, j in product(range(len(N)), repeat=2):
        vals[i, j] = len(df[(df.persons >= N[i]) & (df.mentions >= N[j])])
    annot = [[f'{int(x):,}' for x in row] for row in vals]
    sns.heatmap(vals, annot=annot, fmt='',
                xticklabels=N, yticklabels=N,
                cbar=False, linewidths=0.5,
                vmin=0.0, vmax=len(df), ax=ax)
    ax.invert_yaxis()
    ax.set_title(f'{mode} Channels above Threshold')
    ax.set_xlabel('Mentions')
    ax.set_ylabel('Persons')
    return ax


def plot_threshold_scatter(df, mode, ax=None, thresh=None):
    f"""
    {_what_is_df}

    This function produces a scatterplot with a marker per channel, where the
    x-axis values are number mentions and the y-axis values are number persons.
    The thresholds given by `thresh` are annotated (defaults 1, 10, 100, 1000).
    """
    if ax is None:
        ax = plt.gca()
    N = thresh or (1, 10, 100, 1000)
    X = df.mentions.values
    Y = df.persons.values
    ax.scatter(X, Y, alpha=0.2, marker='.')
    ax.set_xlabel('Mentions')
    ax.set_ylabel('Persons')
    ax.set_title(f'Distr. of Mentions and Persons per {mode} Channel (n={len(X):,})')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.vlines(N, 0, N, color='black', linestyle='--')
    ax.hlines(N, 0, N, color='black', linestyle='--')
    for n in N:
        count = len(df[(df.mentions >= n) & (df.persons >=n)])
        ax.text(1, n, f'{count}', ha='left', va='bottom')
    return ax


def plot_mentions_sorted(df, mode, ax=None, thresh=None, usebox=False):
    f"""
    {_what_is_df}

    Plots the numbers of mentions and persons per data channel, sorted by
    mentions. The x-axis is the sorted code ordinal. The y-axis is the count
    (persons or mentions).

    If `usebox` is True, a white box is drawn around the text annotations (this
    can help with legibility when the text is drawn on top of other markers).
    """
    if ax is None:
        ax = plt.gca()
    N = thresh or (1, 10, 100, 1000)

    # Sort df so that the plot is sorted by mention: persons is matched by code
    df = df.sort_values('mentions')
    X = df.mentions.values
    Y = df.persons.values

    # draw threshold lines at each of N with counts of how many channels are
    # above the threshold - counts by mention are left in blue (C0), counts by
    # person to the right in orange (C1)
    for n in N:
        ax.axhline(n, linestyle='--', color='black', alpha=0.25)
        if usebox:
            ax.text(0, n, f'{sum(X>=n)}',
                    ha='left', va='bottom', color='black',
                    bbox=dict(edgecolor='C0', facecolor='white', alpha=0.8))
            ax.text(len(X), n, f'{sum(Y>=n)}',
                    ha='right', va='bottom', color='black',
                    bbox=dict(edgecolor='C1', facecolor='white', alpha=0.8))
        else:
            ax.text(0, n, f'{sum(X>=n)}',
                    ha='left', va='bottom', color='C0')
            ax.text(len(X), n, f'{sum(Y>=n)}',
                    ha='right', va='bottom', color='C1')

    # Plot the number persons as scatters, the number mentions as a curve
    # Plot in this order so that the mentions curve is drawn on top
    ax.scatter(np.arange(len(X)), Y, color='C1', alpha=0.2,
               label=f'Persons (max={max(Y):,})', marker='.')
    ax.plot(X, label=f'Mentions (max={max(X):,})', color='C0')

    ax.set_yscale('log')
    ax.set_title(f'Distr. of Mentions and Persons per {mode} '
                 f'Channel (n={len(X):,})')
    ax.legend()
    return ax


def plot_threshold_metrics(df, name='', fig=None):
    f"""
    {_what_is_df}

    Visualizes, in three panels, the distributions of observations and persons
    for all the channels in a given data mode (identified by `name`).
    """
    # Looks best if called with a figure at least 9x9 and 'constrained'
    #fig = plt.figure(figsize=(9, 9), layout='constrained')
    if fig is None:
        fig = plt.gcf()

    axdict = fig.subplot_mosaic([['sorted', 'sorted'], ['scatter', 'heatmap']])
    plot_mentions_sorted(df, name, ax=axdict['sorted'], usebox=True)
    plot_threshold_scatter(df, name, ax=axdict['scatter'])
    plot_threshold_heatmap(df, name, ax=axdict['heatmap'])

    for k in ('scatter', 'heatmap'):
        axdict[k].set_title('')
    axdict['heatmap'].tick_params(axis='y',
                                  left=False, labelleft=False,
                                  right=True, labelright=True)
    axdict['heatmap'].set_ylabel('')
    return fig, axdict
