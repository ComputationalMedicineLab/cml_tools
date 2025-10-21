"""Produce bar plots for an ICA model"""
# This is an inplace version 2 of ./ica_model.py; if it works well enough we'll
# eventually deprecate and remove ./ica_model.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_scaled_xs(component,
                  cs_cutoff=0.95,
                  min_elem=10,
                  max_elem=30,
                  norm_xaxis=False,
                  labels='cumulative'):
    """Determine which elements of `component` to plot.

    Generate the scaled significant elements of an independent component. The
    elements of the component are sorted by the magnitude of their projections
    onto the component. As many elements as are needed to travel `cs_cutoff` of
    the magnitude of the component are plotted, subject to `min_elem` and
    `max_elem`.
    """
    # Get the cutoff location
    V = np.square(component)
    ordering = np.argsort(V)[::-1]
    cs = np.sqrt(np.cumsum(V[ordering]) / np.sum(V))
    cutoff = np.argwhere(cs > cs_cutoff)[0][0] + 1

    # Find the actual number of plotted elements (i.e., cutoff is clamped)
    if cutoff > max_elem:
        n_plot = max_elem
    elif cutoff < min_elem:
        n_plot = min_elem
    else:
        n_plot = cutoff

    # Now select the indices into the component to plot
    indices = ordering[:n_plot][::-1]

    # Optionally rescale the component values from the A matrix space (the
    # component is a vector from the ICA A matrix) so that they are normalized.
    if norm_xaxis:
        xs = np.sign(component[indices]) * np.sqrt(V[indices] / np.sum(V))
    else:
        xs = component[indices]

    # Optionally calculate
    if labels == 'cumulative':
        bar_labels = [f'{x:.3f}' for x in cs[:n_plot][::-1]]
    elif labels == 'component':
        bar_labels = [f'{x:.3f}' for x in component[indices]]
    else:
        bar_labels = [f'{x:.3f}' for x in xs]

    return xs, indices, cutoff, bar_labels


def get_bar_colors(cutoff, min_elem):
    if cutoff < min_elem:
        colors = ['orange' for _ in range(min_elem - cutoff)]
        colors.extend(['cornflowerblue' for _ in range(cutoff)])
    else:
        colors = 'cornflowerblue'
    return colors


def draw_barplot(ax, xs, labels,
                 bar_colors='cornflowerblue',
                 dot_colors=None,
                 bar_labels=None,
                 grid=True):
    """Draw a barplot into an Axes object `ax`"""
    ax.axvline(0, color='black', linestyle='solid')

    if grid:
        ax.grid(axis='x', linestyle='--', color='black', alpha=0.25)

    b_ = ax.barh(labels, xs, color=bar_colors)

    if bar_labels:
        ax.bar_label(b_, labels=bar_labels, padding=3, fmt='%.2f')

    n_labels = len(labels)
    # Sets the x limits so that the barplot is centered at zero and has enough
    # margin to include the dots and bar labels (if reasonably short). Also set
    # the y limits so that barplots with many rows don't have weirdly huge
    # margins, since the y margins are determined as a percentage of the total
    # Axes height by default.
    xlim = max(map(abs, ax.get_xlim())) * 1.15
    ax.set_xlim([-xlim, xlim])
    ax.set_ymargin(0)
    ax.set_ylim([-0.75, n_labels-0.25])

    # Draw the dots. If the dot colors are not provided they are omitted.
    # Otherwise, dots are drawn aligned with the labels at the left inner edge
    # of the barplot Axes. These can indicate, e.g., what type of data (the
    # data mode) of the label, when the labels are data variables; etc.
    if dot_colors:
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        ax.scatter(np.full(n_labels, 0.025),
                   np.arange(n_labels),
                   color=dot_colors,
                   transform=transform,
                   marker='o', s=100, edgecolors='black', linewidths=0.5)

    return ax


def draw_inset_histogram(ax, source, title='Expressions'):
    """Plot the source values pos/neg histogram inset into the Axes"""
    bbox_transform = blended_transform_factory(ax.transAxes, ax.transData)
    # Plot the expressions inset
    # https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html
    # width and height are specified in inches;
    inset = inset_axes(ax, width=1.5, height=1.0,
                       bbox_to_anchor=(0.075, 0.0),
                       bbox_transform=bbox_transform,
                       loc="lower left", borderpad=0)
    inset.hist(x=(source[source > 0], source[source < 0]),
               bins=100, histtype='stepfilled', log=True,
               color=None, alpha=0.4, label=('pos', 'neg'))
    inset.set_title(title, fontsize='small')
    inset.tick_params(direction='in', labelsize='x-small', pad=1.2)
    inset.set_ylim(bottom=0.8, top=None)
    xlim = max(map(abs, inset.get_xlim()))
    inset.set_xlim([-xlim, xlim])
    inset.patch.set_alpha(0.6)
    return inset


def plot_ic(component, source, label_func, color_func, title='',
            cs_cutoff=0.95, min_elem=10, max_elem=30, norm_xaxis=False):
    """
    Plot an independent component and associated source expression
    distribution. The elements of the component to plot are selected via
    `get_scaled_xs`; arguments `cs_cutoff`, `min_elem`, `max_elem`, and
    `norm_xaxis` are forwarded to `get_scaled_xs`.

    `label_func` and `color_func` are used to acquire, respectively, the text
    tick labels and indicator colors for each element to be plotted. Each
    function must have signature `fn(component, indices)`.
    """
    xs, indices, cutoff, bar_labels = get_scaled_xs(component,
                                                    cs_cutoff=cs_cutoff,
                                                    min_elem=min_elem,
                                                    max_elem=max_elem,
                                                    norm_xaxis=norm_xaxis)
    bar_colors = get_bar_colors(cutoff, min_elem)
    dot_colors = color_func(indices)
    labels = label_func(component, indices)
    height = 0.75 + (len(labels) * 0.25)
    fig = plt.figure(layout='constrained', figsize=(12, height))
    gs = fig.add_gridspec(ncols=1, nrows=1)
    ax = fig.add_subplot(gs[0])
    draw_barplot(ax, xs, labels, bar_colors, dot_colors, bar_labels)
    draw_inset_histogram(ax, source)

    subtitle = f'{cutoff} to {cs_cutoff:.2f} ({len(indices)} plotted)'
    fig.suptitle(f'{title} {subtitle}', fontsize=16)
    return fig, ax


def plot_ic_and_inverse(component, source, unmixing, label_func, color_func,
                        title='', cs_cutoff=0.95, min_elem=10, max_elem=30,
                        norm_xaxis=False, label_width=88):
    """Combine the component and unmixing barplots into a single figure."""
    xs0, indices0, cutoff0, bar_labels0 = get_scaled_xs(component,
                                                        cs_cutoff=cs_cutoff,
                                                        min_elem=min_elem,
                                                        max_elem=max_elem,
                                                        norm_xaxis=norm_xaxis)

    xs1, indices1, cutoff1, bar_labels1 = get_scaled_xs(unmixing,
                                                        cs_cutoff=cs_cutoff,
                                                        min_elem=min_elem,
                                                        max_elem=max_elem,
                                                        norm_xaxis=norm_xaxis)

    bar_colors0 = get_bar_colors(cutoff0, min_elem)
    bar_colors1 = get_bar_colors(cutoff1, min_elem)

    dot_colors0 = color_func(indices0)
    dot_colors1 = color_func(indices1)

    labels0 = label_func(component, indices0)
    labels1 = label_func(1.0 / unmixing, indices1)

    # The height is adjusted to give each its own height - the number of
    # significant elements may differ between matching vectors of A and A
    # inverse. The height itself is specified in inches.
    height_ratios = [len(labels0), len(labels1)]
    height = 0.75 + (0.25 * sum(height_ratios))

    fig = plt.figure(layout='constrained', figsize=(12, height))
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=height_ratios)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    draw_barplot(ax0, xs0, labels0, bar_colors0, dot_colors0, bar_labels0)
    draw_barplot(ax1, xs1, labels1, bar_colors1, dot_colors1, bar_labels1)

    # Each barplot has the same inset histogram of source expressions
    draw_inset_histogram(ax0, source)
    draw_inset_histogram(ax1, source)

    ax0.set_title(rf'$\mathbf{{A}}$ Values ({cutoff0} to {cs_cutoff})')
    ax1.set_title(rf'$\mathbf{{A}}^{{-1}}$ Values ({cutoff1} to {cs_cutoff})')
    fig.suptitle(title, fontsize=16)

    return fig, (ax0, ax1)


def plot_n_ics(components, sources, label_func, color_func,
               title='', cs_cutoff=0.95, min_elem=10, max_elem=30,
               norm_xaxis=False, label_width=88):
    """Plot as many components as are given into a single figure."""
    xs = []
    indices = []
    cutoffs = []
    bar_labels = []
    _lists = [xs, indices, cutoffs, bar_labels]
    for component in components:
        results = get_scaled_xs(component, cs_cutoff=cs_cutoff,
                                min_elem=min_elem, max_elem=max_elem,
                                norm_xaxis=norm_xaxis)
        for _lst, _res in zip(_lists, results):
            _lst.append(_res)

    bar_colors = [get_bar_colors(c, min_elem) for c in cutoffs]
    dot_colors = [color_func(idx) for idx in indices]
    labels = [label_func(c, idx) for c, idx in zip(components, indices)]

    height_ratios = [len(labelset) for labelset in labels]
    height = 0.75 + (0.25 * sum(height_ratios))

    nrows = len(components)
    fig = plt.figure(layout='constrained', figsize=(12, height))
    gs = fig.add_gridspec(ncols=1, nrows=nrows, height_ratios=height_ratios)

    axes = [fig.add_subplot(gs[i]) for i in range(nrows)]
    for args in zip(axes, xs, labels, bar_colors, dot_colors, bar_labels):
        draw_barplot(*args)
    for ax, src in zip(axes, sources):
        draw_inset_histogram(ax, src)
    for ax, cutoff in zip(axes, cutoffs):
        ax.set_title(rf'$\mathbf{{A}}$ Values ({cutoff} to {cs_cutoff})')

    fig.suptitle(title, fontsize=16)
    return fig, tuple(axes)
