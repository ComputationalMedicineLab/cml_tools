"""EHR related datatypes"""
import datetime
import warnings
from collections import defaultdict
from functools import cached_property
from multiprocessing.managers import SharedMemoryManager
from operator import attrgetter

import bottleneck as bn
import numpy as np
import numpy.lib.format as npf

from cml import wrapdtypes
from cml.record import Record, SOA


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

    @classmethod
    def make_mode_mapping(cls, meta):
        """Produce a dictionary from concept_id to data_mode"""
        return {r.concept_id: r.data_mode for r in meta}

    @classmethod
    def inverse_channel_mapping(cls, meta, channels=None):
        """Produce a dictionary from (data_mode, concept_code) to concept_id"""
        mapping = {(r.data_mode, r.concept_code): r.concept_id for r in meta}
        if channels is not None:
            channels = set(channels)
            mapping = {k: v for k, v in mapping.items() if k in channels}
        return mapping


class PersonMeta(Record):
    # See the SELECT_CORE_COHORT_DEMOGRAPHICS and SELECT_DEMOGRAPHIC_META
    # templates in ./ehr/omop_templates.py for the exact meanings of these
    # fields. Typically, all persons are required to have a birthdate in the
    # Person table >= 1920-01-01 (we do not generally accept EHR from the 1700
    # or 1800's), and we use VUMC Race / Gender rather than the OMOP "standard"
    # vocabularies, which are lacking means of differentiating missing and
    # unknown data from refusal to answer, etc.
    fields = (
        'person_id',
        'birthdate',
        'gender_concept_id',
        'gender_concept_name',
        'gender_source_concept_id',
        'gender_source_value',
        'race_concept_id',
        'race_concept_name',
        'race_source_concept_id',
        'race_source_value',
    )
    dtypes = (int, datetime.date, int, str, int, str, int, str, int, str)
    __slots__ = fields

    # Arguments should match fields exactly
    def __init__(self, person_id, birthdate, gender_concept_id,
                 gender_concept_name, gender_source_concept_id,
                 gender_source_value, race_concept_id, race_concept_name,
                 race_source_concept_id, race_source_value):
        self.person_id = person_id
        self.birthdate = birthdate
        self.gender_concept_id = gender_concept_id
        self.gender_concept_name = gender_concept_name
        self.gender_source_concept_id = gender_source_concept_id
        self.gender_source_value = gender_source_value
        self.race_concept_id = race_concept_id
        self.race_concept_name = race_concept_name
        self.race_source_concept_id = race_source_concept_id
        self.race_source_value = race_source_value

    # Attributes should match fields exactly (boring code but fast)
    @property
    def astuple(self):
        return (self.person_id, self.birthdate, self.gender_concept_id,
                self.gender_concept_name, self.gender_source_concept_id,
                self.gender_source_value, self.race_concept_id,
                self.race_concept_name, self.race_source_concept_id,
                self.race_source_value)

    # These are for convenience, and also to enable customization of which
    # attribute should be used as gender or race concept ids; the "source
    # concept ids" currently being used map to VUMC Race / Gender vocabs.
    @property
    def race(self):
        return self.race_source_concept_id

    @property
    def gender(self):
        return self.gender_source_concept_id

    # FIXME: needs to work with new format using VUMC Race and Gender vocabs
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


class Cohort(tuple):
    """
    A convenience container for a tuple of PersonMeta objects which computes
    (and caches) some aggregate information about the Cohort demographics and
    other information which does not depend upon a specific underlying EHR
    object. Cohort objects should (usually) be considered immutable singletons;
    the tools in cml.ehr.samplespace are more tuned for creating subsets of the
    cohort and sampling points from the EHR.

    As a tuple subclass, all sequencing methods will work, but it is then the
    programmer's responsibility to maintain the Cohort class wrapper:

        >>> type(cohort)
        cml.ehr.dtypes.Cohort
        >>> # Will lose the type information:
        >>> subset = cohort[:10]
        >>> type(subset)
        tuple
        >>> # In order to keep the container class:
        >>> subset = Cohort(cohort[:10])
        >>> type(subset)
        cml.ehr.dtypes.Cohort

    Also, be aware that each instance will have its own cached properties.
    """
    def to_pickle(self, filename, header=False):
        """Convenience driver for PersonMeta.to_pickle_seq"""
        PersonMeta.to_pickle_seq(self, filename, header=header)

    @classmethod
    def from_pickle(cls, filename, header=False):
        """Convenience driver for PersonMeta.from_pickle_seq"""
        return cls(PersonMeta.from_pickle_seq(filename, header=header))

    @cached_property
    def by_person_id(self):
        """A dict mapping person ids to PersonMeta objects"""
        return {p.person_id: p for p in self}

    @cached_property
    def birthdays(self):
        """A dict mapping person ids to np.datetime64 birthdates"""
        return {p.person_id: np.datetime64(p.birthdate) for p in self}

    @cached_property
    def demographics(self):
        """A dict mapping race and gender concepts to ndarrays of person ids"""
        demo = defaultdict(list)
        for p in sorted(self, key=attrgetter('person_id')):
            demo[p.gender].append(p.person_id)
            demo[p.race].append(p.person_id)
        return {k: np.array(demo[k]) for k in sorted(demo)}


class EHR(SOA):
    """An "Electronic Health Record" styled as a Structure-of-Arrays"""
    fields = ('person_id', 'concept_id', 'datetime', 'value')
    dtypes = wrapdtypes('i8', 'i8', 'M8[s]', 'f8')
    __slots__ = fields

    def __init__(self, person_id, concept_id, datetime, value):
        self.person_id = person_id
        self.concept_id = concept_id
        self.datetime = datetime
        self.value = value

    @property
    def astuple(self):
        """Return the object values as a tuple in field order"""
        return (self.person_id, self.concept_id, self.datetime, self.value)

    # FIXME: needs to work with new format using VUMC Race and Gender vocabs
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
        to_concepts = ConceptMeta.inverse_channel_mapping(meta, channels)

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


# TODO: there is significant overlap between this and the SampleSpace concept.
# Possibly there should be either just the one class or the SampleSpace should
# be split into two: an EHR Index and a mapping from person ids to intervals.
# Then a "Visits" table is the basis for a particular sample space. As it is,
# this can be used directly *as* a samplespace:
#
#   >>> visits = Visits.from_npz('./inpatient_admissions.npz')
#   >>> sample_index = sample_uniform(visits, 600_000)
#
# This works just exactly as expected. So when sampling within inpatient
# admissions is desired, a samplespace instance is used to find the EHR for a
# given person (because it contains the EHR indices) and the Visits instance is
# used to generate the sample index. This feels backwards because of the naming
# but works fine. Need to think more about how to organize these ideas.
class Visits(SOA):
    """
    Provides a table of (person_id, start_datetime, end_datetime) tuples; one
    per visit. Not gauranteed (or even likely) to be unique with respect to
    person_id.
    """
    fields = ('person_id', 'datetimes')
    dtypes = wrapdtypes('i8', 'M8[s]')
    __slots__ = fields

    def __init__(self, person_id, datetimes):
        self.person_id = person_id
        self.datetimes = datetimes

    @classmethod
    def from_arrow(cls, table):
        # custom subclass impl. to merge the datetime cols into an ndarray
        person_id = table.column('person_id').to_numpy()
        datetimes = np.empty((len(table), 2), dtype=cls.dtypes[1])
        datetimes[:, 0] = table.column('start_datetime').to_numpy()
        datetimes[:, 1] = table.column('end_datetime').to_numpy()
        assert not bn.anynan(person_id)
        assert not bn.anynan(datetimes)
        return cls(person_id, datetimes)

    @property
    def ntimepoints(self):
        """Patient-time points spanned by the Visit datetime ranges"""
        return np.sum(self.datetimes[:, 1] - self.datetimes[:, 0])
