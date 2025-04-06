"""EHR related datatypes.

The `core_ehr_dtype` is a numpy record dtype object expressing the association
of a person_id with an OMOP concept_id on a given date, potentially with a
numeric real value (e.g. the value of a lab measurement). This data type is
used for storage of the basic EHR pulled from a data source, prior to
longitudinal curve construction.
"""
import datetime
import operator as op
import numpy as np

core_ehr_dtype = np.dtype([('person_id', np.int64),
                           ('concept_id', np.int64),
                           ('date', np.dtype('<M8[D]')),
                           ('value', np.double)])
# There *will* be variations: most likely person_id and concept_id which need
# to be strings. The variations can all live here in this file with whatever
# functions are needed for IO and interopability.

# These are the default field names produced by SELECT_CORE_CONCEPT_META (cf.
# cml/ehr/omop_templates.py). An OMOP channel metadata table is used to produce
# lookup tables for curve interpolation functions, fill values, etc.
omop_channel_meta_header = (
    'concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
    'concept_class_id', 'concept_code', 'data_mode', 'fill_value'
)
omop_channel_meta_types = (int, str, str, str, str, str, str, float)

# These are the default field names produced by SELECT_CORE_COHORT_DEMOGRAPHICS
# (cf. cml/ehr/omop_templates.py). Each member of the cohort is associated with
# a birthdate and information relating to sex and race. Typically the only
# inclusion requirement based on these three categories (Age, Sex, Race) is
# that all persons must have a valid birthdate, usually no earlier than 1920 -
# we do not generally accept EHR from the 1700s or 1800s. The concept ids
# should be keys in the OMOP concept table; the person ids should be keys in
# the OMOP person table (although there exist variations using alternate
# 'ID'ing strategies for custom de-identification purposes).
cohort_demographics_header = (
    'person_id', 'birth_date',
    'gender_concept_id', 'gender_source_value', 'gender_concept_name',
    'race_concept_id', 'race_source_value', 'race_concept_name',
)
cohort_demographics_types = (int, datetime.date, int, str, str, int, str, str)
# These types tuples can be used in automatic conversion from e.g. CSV files.


def core_ehr_from_arrow(table):
    """A constructor for a recarray of core_ehr_dtype from a pyarrow.Table"""
    # We need this b/c using arrow is the only efficient way to get data from
    # databricks-sql-connector, even though arrow itself is... cumbersome.
    return np.rec.fromarrays([
        table.column('person_id').to_numpy(),
        table.column('concept_id').to_numpy(),
        table.column('date').to_numpy(),
        table.column('value').fill_null(np.nan).to_numpy(),
    ], dtype=core_ehr_dtype)


def core_ehr_from_file(filename):
    """Load a recarray of core_ehr_dtype from a .npy file"""
    return np.rec.array(np.load(filename), dtype=core_ehr_dtype, copy=False)


def split_by_person_id(data):
    """Returns views on data grouped by person_id. data is assumed sorted"""
    ids, positions = np.unique(data.person_id, return_index=True)
    groups = np.split(data, positions[1:])
    assert sum(len(x) for x in groups) == len(data)
    return groups


def make_concept_map(data, keys=None, header=None):
    """
    `data` is a tuple of tuples with OMOP concept metadata records. The first
    element of `data` must contain string names for the fields of each tuple.
    The return value is a dict mapping OMOP concept id's to values specified
    by `keys`. If `header` is not None, it is used instead of `data[0]` and
    `data[0]` is assumed not to be the field names for the table.
    """
    if keys is None:
        keys = ('data_mode', )
    elif isinstance(keys, str):
        keys = (keys, )
    if header is None:
        header, *data = data
    concept_idx = header.index('concept_id')
    extract_row = op.itemgetter(*(header.index(k) for k in keys))
    return {row[concept_idx]: extract_row(row) for row in data}


def map_channels_to_concept_ids(channels, meta, keys=None, header=None):
    """Produce a mapping from channels to concept ids.

    By default, `channels` is assumed to be a list or ndarray of (mode, code)
    2-tuples, where `mode` is the data mode (corresponding to the `data_mode`
    field in `omop_channel_meta_header`) and `code` is the `concept_code` of
    the concept table. Concept codes are unique to a given vocabulary but may
    not be globally unique in the OMOP concept table; therefore, using them as
    primary keys for EHR data channels requires them to be annotated with the
    vocabulary or some other indication of source, such as `data_mode`.
    Furthermore, the concept code and data mode are typically strings,
    sometimes not very short strings, compared to the integer values of the
    concept ids. Therefore it is usually *much* more efficient to store channel
    information using concept ids, which pack into a single integer value both
    the mode and channel.
    """
    if keys is None: keys = ('data_mode', 'concept_code')
    mapping = make_concept_map(meta, keys=keys, header=header)
    channels = set(channels)
    return {v: k for k, v in mapping.items() if v in channels}


# XXX: how to handle different date resolutions? The base case for
# large-scale ICA has always been daily resolution, but we want sometimes
# to do hourly or minutely resolution for, e.g., ICA over inpatient
# admissions. Should probably keep it simple and have a simple function per use
# case; the project-specific calling code should know which use case it wants,
# so no need to complicate the implementation by over-generalization.
def patient_date_range(data, start=None, until=None, from_df=False):
    """Construct a date range at daily resolution for the EHR in `data`"""
    # Old-style storage for patient EHR was as dataframes with five columns:
    # ['patient_id', 'date', 'mode', 'channel', 'value']. *All* patient EHR was
    # stored here, including Age, Sex, Race, for which the date column was NaT.
    # If the data is not an old-stype dataframe it assumed to be a
    # core_ehr_dtype recarray.
    dts = data['date'].to_numpy() if from_df else data.date
    if start is None: start = np.nanmin(dts)
    if until is None: until = np.nanmax(dts)
    start = start.astype(np.dtype('<M8[D]'))
    until = until.astype(np.dtype('<M8[D]'))+1
    return np.arange(start, until)
