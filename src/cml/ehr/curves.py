"""Functions for constructing longitudinal curves specifically over EHR data"""
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd

from cml.ehr.dtypes import ConceptMeta
from cml.record import Record
from cml.time_curves import binary_signal, event_intensity, pchip_regression

# One year in minutes: used for constructing Age curves
ONE_YEAR = np.timedelta64(((24*365)+6)*60, 'm')


class CurveSet(Record):
    fields = ('person_id', 'concepts', 'curves', 'grid')
    __slots__ = fields

    def __init__(self, person_id, concepts, curves, grid):
        self.person_id = person_id
        self.concepts = concepts
        self.curves = curves
        self.grid = grid

    @property
    def astuple(self):
        return (self.person_id, self.concepts, self.curves, self.grid)


class CompressedCurveSet(Record):
    fields = ('person_id', 'const_concepts', 'const_curves', 'dense_concepts',
              'dense_curves', 'grid', 'full_nbytes')
    __slots__ = fields

    def __init__(self, person_id, const_concepts, const_curves, dense_concepts,
                 dense_curves, grid, full_nbytes):
        self.person_id = person_id
        self.const_concepts = const_concepts
        self.const_curves = const_curves
        self.dense_concepts = dense_concepts
        self.dense_curves = dense_curves
        self.grid = grid
        self.full_nbytes = full_nbytes

    @property
    def astuple(self):
        return (self.person_id, self.const_concepts, self.const_curves,
                self.dense_concepts, self.dense_curves, self.grid,
                self.full_nbytes)


def compress_curves(curveset: CurveSet):
    if isinstance(curveset, CompressedCurveSet):
        return curveset
    elif not isinstance(curveset, CurveSet):
        curveset = CurveSet(*curveset)
    # If every value in a curve is nearly the same as the first, it is const
    is_const = np.isclose(curveset.curves[:, 0, None], curveset.curves)
    is_const = np.all(is_const, axis=1)
    return CompressedCurveSet(curveset.person_id,
                              curveset.concepts[is_const],
                              curveset.curves[is_const, 0],
                              curveset.concepts[~is_const],
                              curveset.curves[~is_const],
                              curveset.grid,
                              curveset.curves.nbytes)


def split_by_concept_id(data, assume_sorted=True):
    """Returns views on data grouped by concept_id"""
    if not assume_sorted:
        data = data[np.argsort(data['concept_id'])]
    ids, positions = np.unique(data['concept_id'], return_index=True)
    groups = np.split(data, positions[1:])
    assert sum(len(x) for x in groups) == len(data)
    return groups, ids, positions


def construct_date_range(data, start=None, until=None, from_df=False):
    """Construct a date range at daily resolution for the EHR in `data`"""
    # Old-style storage for patient EHR was as dataframes with five columns:
    # ['patient_id', 'date', 'mode', 'channel', 'value']. *All* patient EHR was
    # stored here, including Age, Sex, Race, for which the date column was NaT.
    # If the data is not a dataframe it is assumed to be an EHR.dtype recarray.
    dts = data['date'].to_numpy() if from_df else data.date
    if start is None: start = np.nanmin(dts)
    if until is None: until = np.nanmax(dts)
    start = start.astype(np.dtype('<M8[D]'))
    until = until.astype(np.dtype('<M8[D]')) + 1
    return np.arange(start, until)


def grid_range_args(grid):
    """Produce start, stop, and step arguments from grid for np.arange"""
    assert len(grid) > 1 and np.all(np.sort(grid) == grid)
    return grid[0], grid[-1] + 1, (grid[1] - grid[0])


def build_ehr_curves(data, meta, *, start=None, until=None, window=365,
                     med_dates=True, **opts):
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

    Returns a Curveset.
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

    modes = ConceptMeta.make_mode_mapping(meta)
    grid = construct_date_range(data, start=start, until=until)

    # If either start or until is not None then the limits of the grid may not
    # coincide with the limits of the patient data; we may need to shrink the
    # patient data to the grid. There is no issue if they extend beyond the
    # patient data, the curves should just extrapolate to the grid limits.
    if start is not None or until is not None:
        data = data[(grid[0] <= data.date) & (data.date <= grid[-1])]

    # If `med_dates` is True then the Medication curves do the initial
    # interpolation using *only* dates in the patient record where there is at
    # least one med, of any kind. The idea is that if there's data for a
    # patient on a given date that doesn't include meds, then meds were not
    # being tracked during that visit (or equivalent), and therefore that date
    # does not give information about the presence or absence of any given med.
    if med_dates:
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
    return CurveSet(person_id=int(data.person_id[0]), concepts=concepts,
                    curves=curves, grid=grid_range_args(grid))


def legacy_curve_gen(data, start=None, until=None, window=365, validate=False,
                     med_dates=True):
    """Function for creating curves from legacy DataFrame format for EHR.

    Emulate building curves from the old-style storage. Equivalent to this
    older style curve specification, using tools from version 1 of the tools
    library (`cml_data_tools`):

    >>> from cml_data_tools.curves import *
    >>> CURVE_SPEC = {
    >>>     'Age': AgeCurveBuilder(),
    >>>     'Sex': ConstantCurveBuilder(),
    >>>     'Race': ConstantCurveBuilder(),
    >>>     'Condition': RollingIntensity(window=365),
    >>>     'Measurement': RollingRegression(window=365),
    >>>     'Medication': FuzzedBinaryCurveBuilder(fuzz_length=365),
    >>> }

    Returns a CurveSet object, with some differences in the usage of the
    CurveSet fields: the curves include Age, Sex, and Race; the concepts are
    specified as 2-tuples of ("mode", "channel") strings. The rows are channels
    and the columns are observations, which is in accordance with the current
    usage, but is the transpose of the version 1 dataframe based format.
    """
    grid = construct_date_range(data, start=start, until=until, from_df=True)
    m = len(np.unique(data['channel']))
    n = len(grid)
    curves = np.empty((m, n), dtype=float)

    # We want the channels sorted and unique
    all_channels = data[['mode', 'channel']].to_numpy()
    all_channels = sorted(set(map(tuple, all_channels)))

    if med_dates:
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

    return CurveSet(person_id=data.ptid[0], concepts=all_channels,
                    curves=curves, grid=grid_range_args(grid))
