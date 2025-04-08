"""Longitudinal curve imputation"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from fast_intensity import infer_intensity


EPS = np.finfo(float).eps

# Need to use dtype in minutes (or hours) to get one year as 365.25 days
ONE_YEAR = np.timedelta64(((24*365)+6)*60, 'm')
ONE_DAY = np.timedelta64(24 * 60, 'm')


try:
    from bottleneck import move_mean
except ImportError:
    # strictly a fallback function: much slower than bottleneck's implementation
    def move_mean(x, window=365, **ignore):
        # Prefixing with nan and using nanmean mimics the behavior of pandas in
        # handling the beginning of the data before the window size is reached:
        #
        # >>> out = pd.DataFrame(x).rolling(window, min_periods=1).mean()
        # >>> out[0] == sum(x[0:1]) / 1.0
        # >>> out[1] == sum(x[0:2]) / 2.0
        # >>> # etc.
        #
        # And so on up until the window size is reached.
        x = np.concatenate((np.full(window-1, np.nan), x))
        w = np.lib.stride_tricks.sliding_window_view(x, window_shape=window)
        return np.nanmean(w, axis=1)


def _validate_args(grid, dates=None, values=None, **others):
    """Common input validation for daily resolution time curves"""
    assert len(grid) > 1
    assert grid.dtype == np.dtype('<M8[D]')
    assert (grid[1] - grid[0]) == np.timedelta64(1, 'D')

    if dates is not None:
        if values is not None:
            assert len(dates) == len(values)
        assert (grid[0] <= dates[0]) and (dates[-1] <= grid[-1])

    if (window := others.get('window')) is not None:
        assert isinstance(window, int) and window > 0


def fractional_time(grid, start=None, unit=ONE_YEAR, validate=False):
    if validate: _validate_args(grid)
    if start is None: start = grid[0]
    return (grid - start) / unit


def constant_curve(grid, constant=1.0, validate=False):
    if validate: _validate_args(grid)
    return np.full(len(grid), constant)


def pchip_regression(grid, dates, values, window=None, validate=False):
    if validate: _validate_args(grid, dates, values, window=window)
    # Need to prepare xi, yi, and x for PchipInterpolator; where yi = f(xi)
    # (both xi and yi are observed) for some f and the x values are the
    # locations to be interpolated. In our case, the xi correspond to
    # observation dates and the yi to observation values (e.g., the date and
    # value of a blood lab or other measurement), and x corresponds to the grid
    # over which we are computing the longitudinal curves. If there's only one
    # observation, the curve is constant at that value
    # Equivalent: return constant_curve(grid, constant=values[0])
    if len(dates) == 1:
        return np.full(len(grid), values[0])
    # Otherwise shift the grid and data observations to be fractional days from
    # the start of the grid, and if the grid extends beyond the observations
    # extend the observations to the edges of the grid.
    gridpt = (grid - grid[0])
    events = (dates - grid[0])
    # Can't use extrapolate arg of PchipInterpolator; it works differently
    if gridpt[0] != events[0]:
        events = np.concatenate((gridpt[:1], events))
        values = np.concatenate((values[:1], values))
    if gridpt[-1] != events[-1]:
        events = np.concatenate((events, gridpt[-1:]))
        values = np.concatenate((values, values[-1:]))
    curve = PchipInterpolator(events, values)(gridpt)
    if isinstance(window, int):
        if len(curve) < window:
            curve = np.cumsum(curve) / np.arange(1, len(curve)+1)
        else:
            curve = move_mean(curve, window=window, min_count=1)
    return curve


def event_intensity(grid, dates, min_count=10, iterations=100, window=None,
                    validate=False):
    if validate: _validate_args(grid, dates, window=window)
    # If the minimum number of elements required per bin is greater than half
    # the total number of elements, every bin contains every element. Hence the
    # curve is a constant. If the curve is a constant, the rolling mean of the
    # curve over any window is just that constant, so `window` is irrelevant.
    if (min_count * 2) > len(dates):
        return np.full(len(grid), len(dates)/len(grid))
    # The events are shifted by the start of the grid; the boundaries are
    # shifted to the start of the grid. The grid points are assumed to be
    # uniformly spaced and ascending. If not uniform, then:
    # >>> bounds = (grid - grid[0])
    # >>> bounds = np.concatenate((bounds, bounds[-1]+1)).astype(float)
    # >>> gdelta = (grid[1] - grid[0]).astype(float)
    # >>> bounds = bounds - (0.5 * gdelta)
    # Could make this a keyword argument? Or just hand-craft solutions per
    # normal resolution: do we *ever* have non-uniform grids in practice?
    events = (dates - grid[0]).astype(float)
    bounds = np.arange(len(grid) + 1) - 0.5
    curve = infer_intensity(events=events,
                            grid=bounds,
                            iterations=iterations,
                            min_count=max(min_count, 1))
    if isinstance(window, int):
        if len(curve) < window:
            curve = np.cumsum(curve) / np.arange(1, len(curve)+1)
        else:
            curve = move_mean(curve, window=window, min_count=1)
    return curve


def binary_signal(grid, dates, all_dates, window=None, round_up=True,
                  validate=False):
    if validate:
        _validate_args(grid, dates, window=window)
        _validate_args(grid, all_dates)
        assert np.all(np.isin(dates, all_dates))
        assert np.all(np.isin(all_dates, grid))

    gpts = (grid - grid[0]).astype(float)
    xpts = (all_dates - grid[0]).astype(float)
    ypts = np.isin(all_dates, dates).astype(float)
    # Default numpy behavior is to round exactly middle values (e.g. 0.5) to
    # the nearest even value (e.g., 0); but the behavior of pandas reindex is
    # to round 0.5 to 1.0 - so, round_up mimics pandas. Cf:
    # https://numpy.org/doc/stable/reference/generated/numpy.rint.html
    if round_up:
        curve = np.rint(np.interp(gpts, xpts, ypts) + EPS)
    else:
        curve = np.rint(np.interp(gpts, xpts, ypts))
    # Window here is not rolling mean but a kind of "fuzzing" of the event
    # dates by setting all the window after each event to the positive signal.
    if isinstance(window, int) and window > 0:
        _, index, _ = np.intersect1d(grid, dates,
                                     assume_unique=True,
                                     return_indices=True)
        # This is actually most efficient. The elements of `index` are
        # themselves already set; we want to set everything from i+1 to
        # i+1+window. Trying to resolve/merge these index slices is far slower,
        # btw, than just setting them all to 1.0 - the work being deduplicated
        # by merging the intervals is way cheaper than the work to merge the
        # intervals.
        index += 1
        for i in index:
            curve[i:i+window] = 1.0
    return curve
