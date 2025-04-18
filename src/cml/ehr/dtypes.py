"""EHR related datatypes"""
import datetime
from operator import itemgetter

import numpy as np
from cml.record import Record


class ConceptMeta(Record):
    # These are the default field names produced by SELECT_CORE_CONCEPT_META
    # (cf. cml/ehr/omop_templates.py). An OMOP channel metadata table is used
    # to produce lookup tables for curve functions, fill values, etc.
    fields = (
        'concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
        'concept_class_id', 'concept_code', 'data_mode', 'fill_value'
    )
    dtypes = (int, str, str, str, str, str, str, float)
    __slots__ = fields

    def __init__(self, concept_id, concept_name, domain_id, vocabulary_id,
                 concept_class_id, concept_code, data_mode, fill_value):
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.domain_id = domain_id
        self.vocabulary_id = vocabulary_id
        self.concept_class_id = concept_class_id
        self.concept_code = concept_code
        self.data_mode = data_mode
        self.fill_value = fill_value

    @property
    def astuple(self):
        return (self.concept_id, self.concept_name, self.domain_id,
                self.vocabulary_id, self.concept_class_id, self.concept_code,
                self.data_mode, self.fill_value)

    @property
    def asdict(self):
        k, *v = self.astuple
        return {k: v}

    @classmethod
    def make_mode_mapping(cls, meta):
        """Produce a dictionary from concept_id to data_mode"""
        # Refer to cml.record.Record for from_iter
        if not isinstance(meta[0], cls):
            meta = cls.from_iter(meta)
        return {r.concept_id: r.data_mode for r in meta}

    @classmethod
    def inverse_channel_mapping(cls, channels, meta):
        """Produce a dictionary from (data_mode, concept_code) to concept_id"""
        channels = set(channels)
        if not isinstance(meta[0], cls):
            meta = cls.from_iter(meta)
        forward = {r.concept_id: (r.data_mode, r.concept_code) for r in meta}
        return {v: k for k, v in forward.items() if v in channels}


class PersonMeta(Record):
    # These are the default field names produced by our OMOP SQL script
    # SELECT_CORE_COHORT_DEMOGRAPHICS (cf. cml/ehr/omop_templates.py). Each
    # member of the cohort is associated with a birthdate and information
    # relating to sex and race. Typically the only inclusion requirement based
    # on these three categories (Age, Sex, Race) is that all persons must have
    # a valid birthdate, usually no earlier than 1920 - we do not generally
    # accept EHR from the 1700s or 1800s. The concept ids should be keys in the
    # OMOP concept table; the person ids should be keys in the OMOP person
    # table (although there exist variations using alternate 'ID'ing strategies
    # for custom de-identification purposes).
    fields = (
        'person_id', 'birthdate',
        'gender_concept_id', 'gender_source_value', 'gender_concept_name',
        'race_concept_id', 'race_source_value', 'race_concept_name',
    )
    dtypes = (int, datetime.date, int, str, str, int, str, str)
    __slots__ = fields

    def __init__(self, person_id, birthdate, gender_concept_id,
                 gender_source_value, gender_concept_name, race_concept_id,
                 race_source_value, race_concept_name):
        self.person_id = person_id
        self.birthdate = birthdate
        self.gender_concept_id = gender_concept_id
        self.gender_source_value = gender_source_value
        self.gender_concept_name = gender_concept_name
        self.race_concept_id = race_concept_id
        self.race_source_value = race_source_value
        self.race_concept_name = race_concept_name

    @property
    def astuple(self):
        return (self.person_id, self.birthdate, self.gender_concept_id,
                self.gender_source_value, self.gender_concept_name,
                self.race_concept_id, self.race_source_value,
                self.race_concept_name)

    @classmethod
    def from_frame(cls, df):
        """Attempt to extract a Person record from the v1 dataframe format"""
        person_id = df.ptid.values[0]
        # Try to convert to an integer, if the conversion is not lossy
        if isinstance(person_id, str) and str(int(person_id)) == person_id:
            person_id = int(person_id)

        birthdate = np.datetime64(df[df['mode'] == 'Age']['value'].values[0])

        # Consult the OMOP Gender domain documentation for the values below
        gender_source_value = df[df['mode'] == 'Sex']['channel'].values[0]
        assert isinstance(gender_source_value, str)
        match gender_source_value.upper():
            case 'F'|'FEMALE':
                gender_concept_id = 8532
                gender_concept_name = 'FEMALE'
            case 'M'|'MALE':
                gender_concept_id = 8507
                gender_concept_name = 'MALE'
            case _:
                gender_concept_id = 0
                gender_concept_name = 'No matching concept'

        # TODO: figure out how to back-map the various mappings we've used for
        # race. For now, just take the computed race_source_value and stub the
        # other two fields.
        race_source_value = df[df['mode'] == 'Race']['channel'].values[0]
        assert isinstance(race_source_value, str)

        return cls(person_id, birthdate, gender_concept_id,
                   gender_source_value, gender_concept_name, None,
                   race_source_value, None)


# The main dtype:
# There *will* be variations: most likely person_id and concept_id which need
# to be strings. The variations can all live here in this file with whatever
# functions are needed for IO and interopability.
class EHR:
    dtype = np.dtype([('person_id', np.int64),
                      ('concept_id', np.int64),
                      ('date', np.dtype('<M8[D]')),
                      ('value', np.double)])

    __slots__ = ('data',)
    def __init__(self, data):
        self.data = data

    # We need this b/c using arrow is the only efficient way to get data from
    # databricks-sql-connector, even though arrow itself is... cumbersome.
    @classmethod
    def from_arrow(cls, table):
        """Construct a recarray of EHR.dtype from a pyarrow.Table"""
        return cls(np.rec.fromarrays([
            table.column('person_id').to_numpy(),
            table.column('concept_id').to_numpy(),
            table.column('date').to_numpy(),
            table.column('value').fill_null(np.nan).to_numpy(),
        ], dtype=cls.dtype))

    @classmethod
    def from_npy(cls, filename):
        """Load a recarray of EHR.dtype from a .npy file"""
        return cls(np.rec.array(np.load(filename), dtype=cls.dtype, copy=False))

    def to_npy(self, filename):
        """Persist self.data to a .npy file"""
        np.save(filename, self.data)

    @classmethod
    def from_records(cls, records, sort=True):
        """Construct a recarray of EHR.dtype from tuples"""
        if sort: records = sorted(records)
        return cls(np.rec.fromrecords(records, dtype=cls.dtype))

    # XXX: split into a from_sharedmem classmethod and load_into_mem inst meth?
    @classmethod
    def wrap_sharedmem(cls, mem, shape, source=None):
        """
        Wraps a multiprocessing SharedMemory buffer with the appropriate dtype.
        This happens in two steps: first one must wrap the memory buffer with
        an np.ndarray using the buffer keyword argument, then one is able to
        wrap the ndarray with an np.rec.array. This function should make no
        copies of the underlying memory buffer.

        If `source` is not None then the contents of source are copied into the
        shared memory buffer (i.e., this is how the shared data is loaded into
        the shared memory space in the first place).
        """
        data = np.ndarray(shape, dtype=cls.dtype, buffer=mem.buf)
        if source is not None:
            data[:] = source
        return cls(np.rec.array(data, dtype=cls.dtype, copy=False))

    @classmethod
    def from_frame(cls, df, meta, hash_person_id=True, strict=False):
        """
        The old-style storage for patient EHR was in a dataframe with five
        columns: patient id, date, mode, channel, value. Dates and values were
        stored as Timestamps and floats; the other three were stored as
        strings. The numpy dtype is clearly more efficient.

        However, atemporal data modes (such as sex or race demographics) or
        data modes without dedicated OMOP concept ids (such as age) are
        captured by the Person tuple rather than a core_ehr_dtype recarray.

        Returns the ehr recarray, a Person tuple, and the mapping used from
        legacy ('mode', 'channel') pairs of strings to OMOP concept ids.
        """
        person = Person.from_frame(df)
        # The Person class constructor should already have tried type conversion
        if not isinstance(person_id := person.person_id, int):
            if hash_person_id:
                # XXX: do I need to use a np func to make sure the dtype fits?
                person_id = hash(person_id)
            elif strict:
                raise ValueError(f'Cannot coerce {person_id=} to integer')
            else:
                person_id = 0.0

        # Exclude age, sex, race, then get the channel to concepts map, if any
        # are missing then based on `strict` we bail, otherwise ignore.
        data = df[~np.isin(df['mode'].str.lower(), ('age', 'sex', 'race'))]
        channels = set(map(tuple, data[['mode', 'channel']].to_numpy()))
        to_concepts = ConceptMeta.inverse_channel_mapping(channels, meta)

        if strict and len(channels) != len(to_concepts):
            missing = [x for x in channels if x not in to_concepts]
            raise ValueError(f'Unable to map {missing=} to OMOP concepts')

        records = []
        for t in data.itertuples():
            key = (t.mode, t.channel)
            if (concept := to_concepts.get(key)) is not None:
                records.append((person_id, concept, t.date, t.value))
        ehr = EHR.from_records(records, sort=True)
        return ehr, person, to_concepts
