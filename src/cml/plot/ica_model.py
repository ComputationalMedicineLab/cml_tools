"""Plot an ICA model's `A` and `S` matrices"""
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_ic_scaled(component, source, label_func, color_func, title='',
                   cs_cutoff=0.95, min_elem=10, max_elem=60, bar_label=True):
    """
    Barplot the scaled significant elements of an independent component. The
    elements of the component are sorted by the magnitude of their projections
    onto the component. As many elements as are needed to travel `cs_cutoff` of
    the magnitude of the component are plotted, subject to `min_elem` and
    `max_elem`.

    Arguments
    ---------
    component : ndarray
    source : ndarray
        The component and source arguments are 1-dimensional ndarrays. In the
        Independent Component model `X = AS`, `component` is a column of `A`
        and `source` is the corresponding row of `S`.
    label_func : callable
    color_func : callable
        Each function has the signature (component, ordering), where component
        is the `component` argument to this function and `ordering` is a set of
        indices for which to retrieve information. `label_func` should return a
        list of labels to use for the elements of component to be plotted;
        `color_func` should be return a list of color names to use for the
        indicator dots between label and bar.
    title : str
        A string to prepend to the default suptitle string.
    cs_cutoff : float from 0 to 1, default = 0.95
        Plot elements along the component until `cs_cutoff` is reached.
    min_elem : int, default = 10
    max_elem : int, default = 60
        The minimum and maximum number of elements to plot.
    """
    # squared components sorted descending
    V = np.square(component)
    ordering = np.argsort(V)[::-1]
    # normalized sqrt cumulative sum curve (i.e. distance along that
    # component's projection normalized by total magnitude of the signature)
    V_cs = np.sqrt(np.cumsum(V[ordering]) / np.sum(V))
    cutoff = np.argwhere(V_cs > cs_cutoff)[0][0] + 1

    if cutoff > max_elem:
        n_plot = max_elem
        bar_colors = 'cornflowerblue'

    elif cutoff < min_elem:
        n_plot = min_elem
        bar_colors = ['cornflowerblue' for _ in range(cutoff)]
        for _ in range(min_elem-cutoff):
            bar_colors.append('orange')
        bar_colors = bar_colors[::-1]

    else:
        n_plot = cutoff
        bar_colors = 'cornflowerblue'

    title = f'{title}{cutoff} To Reach {cs_cutoff:.3f}, ({n_plot} Plotted)'
    ordering = ordering[:n_plot][::-1]

    # Get the basic values for the barplot - use normalized len for bar len
    xs = np.sign(component[ordering]) * np.sqrt(V[ordering] / np.sum(V))
    ys = np.arange(n_plot)

    # User supplied functions give us labels & colors for plotting
    labels = label_func(component, ordering)
    colors = color_func(component, ordering)

    # XXX: keep this hard-coded stuff or move to using a gridspec?
    width = 13
    # 1 inch for title, 0.5 for margin on either side, and 1 per 4 channels
    height = 1.5 + (n_plot*0.25)
    # add_axes(rect); rect=[left, bottom, width, height], as fraction of
    # whole (0..1) 1 inch as fraction of whole is 1/height
    margin = 0.5 / height
    ax_ht = 1 - (3 * margin)

    fig = plt.figure(figsize=(width, height))
    bars = fig.add_axes(rect=(0.55, margin, 0.44, ax_ht))
    dots = fig.add_axes(rect=(0.525, margin, 0.025, ax_ht))

    bars.axvline(0, color='black', linestyle='solid')
    b_ = bars.barh(ys, xs, color=bar_colors)
    xmax = max(map(abs, bars.get_xlim()))
    if bar_label:
        xmax *= 1.22
        bar_labels = [f'{x:.3f}' for x in V_cs[:n_plot][::-1]]
        bars.bar_label(b_, labels=bar_labels, padding=3, fmt='%.2f')
    else:
        xmax *= 1.11

    # Fill dots
    dots.scatter(np.full(ys.shape, (-xmax*0.8)),
                 ys, color=colors,
                 marker='o', s=100, edgecolors='black', linewidths=0.5)

    ylim = [-1, n_plot]
    # Remove borders
    bars.spines['left'].set_visible(False)
    dots.spines['right'].set_visible(False)
    # Set correct x/y limits/ticks/labels
    bars.tick_params(axis="x", bottom=True, top=True,
                     labelbottom=True, labeltop=True)
    bars.set_yticks([])
    bars.set_yticklabels([])
    dots.set_yticks(ys)
    dots.set_yticklabels(labels)
    dots.set_xticks([])
    bars.set_ylim(ylim)
    dots.set_ylim(ylim)
    bars.set_xlim([-xmax, +xmax])

    # Plot the expressions inset
    # https://matplotlib.org/3.3.3/gallery/axes_grid1/inset_locator_demo.html
    inset = inset_axes(bars, width=1.5, height=1.0, loc=3, borderpad=1.5)
    inset.hist(x=(source[source > 0], source[source < 0]),
               bins=100, histtype='stepfilled', log=True,
               color=None, alpha=0.4, label=('pos', 'neg'))
    inset.set_title('Expressions', fontsize='small')
    inset.tick_params(direction='in', labelsize='x-small', pad=1.2)
    inset.set_ylim(bottom=0.8, top=None)
    xlim = max(map(abs, inset.get_xlim()))
    inset.set_xlim([-xlim, xlim])
    inset.patch.set_alpha(0.6)

    fig.suptitle(title, fontsize=16)
    return fig, bars


def plot_model_to_pdf(outfile, A, S, plot_fn, log_every=0):
    """Produce a PDF of component plots produced from `plot_fn`

    Arguments
    ---------
    outfile : str | Pathlike
        The name of the output file.
    A : 2-dim ndarray
    S : 2-dim ndarray
        The A and S matrices from the ICA model equation `X = AS`. The shape of
        `A` is `[n_features, n_components]`; the shape of `S` is
        `[n_components, n_observations]`.
    plot_fn : Callable
        A function with the signature fn(i, component, source). `component` and
        `source` are a column of `A` and a corresponding row of `S`. `i` is the
        ordinal of the component & source pair in A and S. The function should
        produce a visualization for the component and source and return the
        figure and axes objects.
    log_every : int, default = 0
        If greater than zero, log progress every `log_every` components.
    """
    if log_every > 0:
        log = logging.getLogger('plot_model')
        log.info(f'plotting {A.shape[1]} independent components to {outfile}')
    with PdfPages(outfile) as pdf:
        for i in range(A.shape[1]):
            fig, axes = plot_fn(i, A[:, i], S[i])
            pdf.savefig()
            plt.close()
            if (log_every > 0) and ((i+1) % log_every == 0):
                log.info(f'{i} Independent Components plotted')
    if log_every > 0:
        log.info('DONE')
