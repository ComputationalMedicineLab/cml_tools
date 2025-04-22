"""Functions for constructing longitudinal curves specifically over EHR data"""
from functools import partial
from numbers import Number

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

    @property
    def concepts(self):
        """Concatenate const and dense concepts"""
        return np.concatenate((self.const_concepts, self.dense_concepts))


class CurvePointSet(Record):
    fields = ('person_id', 'dates', 'concepts', 'values')
    __slots__ = fields

    def __init__(self, person_id, dates, concepts, values):
        assert values.shape[0] == len(concepts)
        assert values.shape[1] == len(dates)
        self.person_id = person_id
        self.dates = dates
        self.concepts = concepts
        self.values = values

    @property
    def astuple(self):
        return (self.person_id, self.dates, self.concepts, self.values)


def compress_curves(curveset: CurveSet):
    """Construct a CompressedCurveSet from a CurveSet compatible object"""
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


def select_cross_section(curveset, dates):
    """Get the values of each curve in a curveset at the given dates"""
    # I think I've covered the cases... zero-dim ndarrays are annoying
    if not isinstance(dates, np.ndarray):
        if np.isscalar(dates):
            dates = np.array([dates])
        else:
            dates = np.array(dates)
    if dates.ndim == 0:
        dates = dates.reshape(1)
    dates = np.sort(dates)

    if not isinstance(curveset, (CurveSet, CompressedCurveSet)):
        curveset = CurveSet(*curveset)

    dmin, dmax, _ = curveset.grid
    if not (np.all(dmin <= dates) and np.all(dates < dmax)):
        grid = curveset.grid
        raise ValueError(f'Invalid {dates=} for {curveset=} ({grid=})')
    index = (dates - dmin).astype(int)

    # Values must be shape [features, dates] at this point
    if isinstance(curveset, CurveSet):
        values = curveset.curves[:, index]
    else:
        const = np.tile(curveset.const_curves, (len(index), 1))
        dense = curveset.dense_curves[:, index]
        values = np.vstack((const.T, dense))

    ordering = np.argsort(curveset.concepts)
    concepts = np.copy(curveset.concepts[ordering])
    values = np.copy(values[ordering])
    return CurvePointSet(curveset.person_id, dates, concepts, values)


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

    Returns a CurveSet.
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


def calculate_age_stats(space, cohort):
    """
    A cohort is a set of PersonMeta records, each of which defines a birthdate.
    Since a SampleSpace calculates the min and max dates per person in an
    implied cohort, the most convenient way to get the mean and stdev Age over
    the cohort curves is as a function of the SampleSpace and the cohort. See
    `cml.ehr.samplespace` and `cml.ehr.dtypes` for SampleSpace and PersonMeta
    objects.

    Returns a 2-tuple of floats, `(mean, stdev)`, for Age over the dates
    spanned by the given cohort, measured in fractional years.
    """
    D = {p.person_id: np.datetime64(p.birthdate) for p in cohort}
    D = np.array([D[i] for i in space.ids])
    D = space.dates - D[:, None]

    # Calculate in two passes. On sizeable datasets (which we expect) directly
    # instantiating the Age curves and passing them to np.mean / np.std is
    # slightly faster (it takes about 80-90% the time) but often requires
    # upwards of 100 Gb. So first we get the mean then var.
    n, m, s = 0, 0, 0
    for row in D:
        dr = np.arange(*row).astype(float) / 365.25
        m += np.sum(dr)
        n += len(dr)
    m /= n
    for row in D:
        dr = np.square((np.arange(*row).astype(float) / 365.25) - m)
        s += np.sum(dr)
    s = np.sqrt(s/n)
    return m, s


def build_age_curve(index, cohort, age_mean=None, age_sdev=None, unit=365.25):
    """
    Index is a DateSampleIndex or equivalent data structure which specifies the
    (person_id, date) pairs which index a given dataset. Cohort is an iterable
    of PersonMeta objects which provide demographic information about the
    persons in the cohort, including birthdate. Alternately, cohort is a
    dictionary mapping person_ids to np.datetime64 birthdates. age_mean and
    age_sdev are the mean and standard deviation to use in scaling the age
    curve.

    Returns the age curve, an ndarray of len(index.data) and dtype float,
    corresponding to the sample points in `index`. The curve is centered and
    scaled to unit stdev.

    Precomputing the person-birthdate mapping can save time when building age
    curves for many index sets (cohort in example below has approx 2.5 million
    persons, and index contains 600_000 sample points from approx 97 thousand
    unique persons):

    In [5]: %time age_map = {p.person_id: np.datetime64(p.birthdate) for p in cohort}
    CPU times: user 3.52 s, sys: 52.8 ms, total: 3.57 s
    Wall time: 3.59 s

    In [6]: %time age_curve = build_age_curve(index, cohort, age_mean, age_sdev)
    CPU times: user 6.33 s, sys: 71.8 ms, total: 6.4 s
    Wall time: 6.43 s

    In [7]: %time age_curve = build_age_curve(index, age_map, age_mean, age_sdev)
    CPU times: user 2.74 s, sys: 19.9 ms, total: 2.76 s
    Wall time: 2.77 s
    """
    if not isinstance(cohort, dict):
        D = {p.person_id: np.datetime64(p.birthdate) for p in cohort}
    else:
        D = cohort
    D = np.array([date-D[person] for person, date in index.data], dtype=float)
    D /= unit
    # Don't do useless work: x-0 and x/1 are identity ops
    if age_mean is not None and age_mean != 0.0: D -= age_mean
    if age_sdev is not None and age_sdev != 1.0: D /= age_sdev
    return D


def build_demographic_curves(index, demos, concepts):
    """One-hot encodes the demographics of a given index.

    "index" is a DateSampleIndex or equivalent data structure specifying pairs of
    (person_id, date/time) for some data set. "demos" is a mapping from
    demographic concept ids (or channel labels) to person ids. "concepts" is
    the universe of demographic concepts over which to operate, it is therefore
    the labelling of the rows of the output ndarray. Any element of "concepts"
    not also a key of "demos" (or a key to an empty list) will be identically
    zero. The output ndarray is of shape "[len(concepts), len(index)]".
    """
    dmap = {k: np.isin(index.data['person_id'], v) for k, v in demos.items()}
    curves = np.zeros((len(concepts), len(index.data)))
    for i, c in enumerate(concepts):
        if c in dmap:
            curves[i, dmap[c]] = 1.0
    return concepts, curves
