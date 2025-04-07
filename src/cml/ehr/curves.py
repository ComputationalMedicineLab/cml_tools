"""Functions for constructing longitudinal curves specifically over EHR data"""
from functools import partial

import numpy as np
import pandas as pd

from cml.ehr.dtypes import *
from cml.time_curves import *


def build_ehr_curves(data, meta, *, start=None, until=None, window=365,
                     legacy_meds=True, **opts):
    """
    A function for applying our most standard curve construction configuration
    in an optimized manner (approx. 1/10 the time of the cml_data_tools.curves
    suite of tools).

    Curves are interpolated at daily resolution from the earliest the latest
    date given by the patient EHR, optionally extended or truncated by `start`
    and `until`. If `until` is given, it is included in the generated date
    range.

    Age, Sex, and Race are omitted (they are simple functions to calculate at
    sampling time). Conditions and Measurements are estimated by
    `event_intensity` and `pchip_regression` respectively, with a rolling mean
    window of 365 on both. Medications are estimated by `binary_signal` with a
    "fuzz window" of 365. Please see `cml.time_curves` for details of these
    functions.

    Returns a dictionary with keys `("person_id", "curves", "grid",
    "concepts")`. The curves are a double-precision ndarray of dimension
    `[len(concepts), len(grid)]` (note that this is the tranpose of the
    old-style curves dataframes). The grid is an ndarray of dtype('<M8[D]')
    (equiv. to `datetime.date`) from the first to the final date in the patient
    EHR. The concepts is an ndarray in sorted order of the unique integer
    concept IDs corresponding to the columns of the curves.
    """
    # TODO: think about how best to handle non-daily resolution or alternate
    # curve functions. Probably the best combination of legibility and
    # performance is to make a function per use case, but a carefully used
    # class pattern could be useful. The two goals are: (1) don't impact
    # performance, and (2) don't impact legibility (clarity of process flow).
    intensity_opts = {
        'min_count': opts.get('intensity_min_count', 10),
        'iterations': opts.get('intensity_iterations', 100),
        'window': opts.get('intensity_window', window),
        'validate': opts.get('validate', False),
    }
    pchip_opts = {
        'window': opts.get('pchip_window', window),
        'validate': opts.get('validate', False),
    }
    binary_opts = {
        'window': opts.get('binary_window', window),
        'validate': opts.get('validate', False),
    }
    fi_min_events = 2 * intensity_opts['min_count']

    modes = make_concept_map(meta)
    grid = patient_date_range(data, start=start, until=until)
    pid = int(data.person_id[0])

    # If either start or until is not None then the limits of the grid may not
    # coincide with the limits of the patient data; we may need to shrink the
    # patient data to the grid. There is no issue if they extend beyond the
    # patient data, the curves should just extrapolate to the grid limits.
    if start is not None or until is not None:
        data = data[(grid[0] <= data.date) & (data.date <= grid[-1])]

    # The medication curves need all the patient dates where there's data; or,
    # they need all the _Medication_ mode dates. The former behavior is
    # probably more correct, but the latter is the legacy behavior.
    if legacy_meds:
        is_med = np.array([modes[c] == 'Medication' for c in data.concept_id])
        all_dates = np.sort(np.copy(data.date[is_med], order='C'))
    else:
        all_dates = np.sort(np.copy(data.date, order='C'))

    groups, concepts, _ = split_by_concept_id(data)
    curves = np.empty((len(concepts), len(grid)), dtype=float)
    for i, (g, c) in enumerate(zip(groups, concepts)):
        # Fast-path computation of Condition and Measurement curves which can
        # be determined to be constant prior to calling the curve function
        match modes.get(c):
            case 'Condition' if len(g.date) < fi_min_events:
                x = np.float64(len(g.date) / len(grid))
            case 'Condition':
                x = event_intensity(grid, g.date, **intensity_opts)
            case 'Measurement' if len(g.date) == 1:
                x = g.value[0]
            case 'Measurement':
                x = pchip_regression(grid, g.date, g.value, **pchip_opts)
            case 'Medication':
                x = binary_signal(grid, g.date, all_dates, **binary_opts)
            case _:
                continue
        curves[i] = x
    return dict(person_id=pid, curves=curves, grid=grid, concepts=concepts)


def legacy_curve_gen(data, start=None, until=None, window=365, validate=False,
                     legacy_meds=True):
    """Function for creating curves from legacy DataFrame format for EHR.

    The curves are created using the old-style format: Age, Sex, and Race are
    included in the output, and the channels are the old-style (mode, channel)
    2-tuples of strings. The return keys are `("curves", "grid", "channels")`,
    as opposed to `("curves", "grid", "concepts")` from `build_ehr_curves`.
    Note also that the curves returned by this function are the transpose of
    those returned by `build_ehr_curves`.
    """
    # Emulate building curves from the old-style storage. Equivalent to the
    # older style curve spec, which we used on IPN and many other projects:
    #   CURVE_SPEC = {
    #       'Age': AgeCurveBuilder(),
    #       'Sex': ConstantCurveBuilder(),
    #       'Race': ConstantCurveBuilder(),
    #       'Condition': RollingIntensity(window=365),
    #       'Measurement': RollingRegression(window=365),
    #       'Medication': FuzzedBinaryCurveBuilder(fuzz_length=365),
    #   }
    grid = patient_date_range(data, start=start, until=until, from_df=True)
    m = len(np.unique(data['channel']))
    n = len(grid)
    curves = np.empty((m, n), dtype=float)

    # We want the channels sorted and unique
    all_channels = data[['mode', 'channel']].to_numpy()
    all_channels = sorted(set(map(tuple, all_channels)))

    if legacy_meds:
        all_dates = data[data['mode'] == 'Medication']['date'].to_numpy()
    else:
        all_dates = data['date'].to_numpy()
    all_dates = np.unique(all_dates[np.isfinite(all_dates)])
    all_dates = np.sort(all_dates.astype(np.dtype('<M8[D]')))

    # Patient birthdate is needed for the Age curve. Fail fast if not present.
    birthdate = np.datetime64(data[data['mode'] == 'Age']['value'].values[0])
    data = data.sort_values('date')

    opts = dict(window=window, validate=validate)
    eval_labs = partial(pchip_regression, **opts)
    eval_meds = partial(binary_signal, **opts)
    eval_codes = partial(event_intensity, **opts)

    for i, (mode, channel) in enumerate(all_channels):
        subdf = data[(data['mode'] == mode) & (data['channel'] == channel)]
        dates = subdf['date'].to_numpy().astype(np.dtype('<M8[D]'))
        match mode:
            case 'Age':
                curves[i] = (grid - birthdate) / ONE_YEAR
            case 'Sex'|'Race':
                curves[i] = np.ones(len(grid))
            case 'Condition':
                curves[i] = eval_codes(grid, dates)
            case 'Measurement':
                vals = subdf['value'].to_numpy().astype(float)
                curves[i] = eval_labs(grid, dates, vals)
            case 'Medication':
                curves[i] = eval_meds(grid, dates, all_dates)
    return dict(curves=curves, grid=grid, channels=all_channels)
