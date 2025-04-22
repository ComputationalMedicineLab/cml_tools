"""Tools for sampling patient data spaces"""
import functools
import pickle

import numpy as np

import cml
from cml.ehr.dtypes import EHR
from cml.record import Record


class DateSampleIndex:
    dtype = np.dtype([('person_id', np.int64), ('date', np.dtype('<M8[D]'))])
    __slots__ = ('data',)

    def __init__(self, data, sort=False):
        # FYI sorting returns a *copy* of the input array, so if sort=True then
        # we are guaranteed (one would hope) to have a contiguous data array
        if sort:
            data = np.sort(data)
        self.data = data

    def __eq__(self, other):
        return np.all(self.data == other.data)

    def split_by_person(self, asdict=True):
        """
        Return the index split by person id as either a list of structured
        ndarrays of self.dtype or a mapping from person ids to date arrays.
        """
        ids, index = np.unique(self.data['person_id'], return_index=True)
        # The zero-th split is always empty for some reason
        splits = np.split(self.data, index)[1:]
        if asdict:
            return {d['person_id'][0]: d['date'] for d in splits}
        return splits

    @classmethod
    def from_arrays(cls, ids, dates, sort=False):
        """Construct a DateSampleIndex from corresponding iterables"""
        assert len(ids) == len(dates)
        data = np.empty(len(ids), dtype=cls.dtype)
        data['person_id'] = ids
        data['date'] = dates
        return cls(data, sort=sort)

    @classmethod
    def from_tuples(cls, tuples, sort=False):
        """Construct a DateSampleIndex from tuples of (person_id, date)"""
        return cls.from_arrays(*zip(*tuples), sort=sort)

    def to_pickle(self, filename, mode='wb'):
        """Write a DateSampleIndex to a pickle file"""
        with open(filename, mode) as file:
            pickle.dump(self.data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filename, sort=False):
        """Load a DateSampleIndex from a pickle file"""
        with open(filename, 'rb') as file:
            return cls(pickle.load(file), sort=sort)

    @property
    def npersons(self):
        """Number of unique persons in the sample index"""
        return len(np.unique(self.data['person_id']))

    @property
    def ndates(self):
        """Number of unique dates in the sample index"""
        return len(np.unique(self.data['date']))


class SampleSpace(Record):
    fields = ('ids', 'indices', 'dates', '_index_map')
    __slots__ = fields

    def __init__(self, ids, indices, dates):
        self.ids = np.array(ids)
        self.indices = np.array(indices)
        self.dates = np.array(dates)
        self._index_map = {p: ij for p, ij, _ in self}

    @property
    def astuple(self):
        return (self.ids, self.indices, self.dates)

    @property
    def index_map(self):
        """Maps person_ids to their (i, j) indices in the EHR.data recarray"""
        return self._index_map

    @classmethod
    def from_ehr(cls, ehr):
        """Produce a SampleSpace from a recarray of EHR data"""
        if isinstance(ehr, EHR): ehr = ehr.data
        # np.unique will cause a brief doubling of memory usage
        ids, indices = np.unique(ehr.person_id, return_index=True)
        indices = indices.tolist()
        indices.append(len(ehr))
        indices = tuple(zip(indices, indices[1:]))
        # The min/max of the date is where most time is spent. This is
        # effectively an efficient groupby-reduce using direct indexing. The
        # date ranges are [min_date, max_date), following the semantics of
        # start and stop in range, np.arange, etc.
        dates = tuple(((d := ehr.date[i:j]).min(), d.max()+1) for i, j in indices)
        return cls(ids, indices, dates)

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        yield from zip(self.ids, self.indices, self.dates)

    # TODO: do I use this functionality? Would I rather a SampleSpace be a
    # Mapping than a Sequence? (i.e.: treat index as a person_id)
    def __getitem__(self, index):
        return (self.ids[index], self.indices[index], self.dates[index])

    # __eq__ and __hash__ are to support caching the results of subseting,
    # sampling, and restrictions (as well helping with debugging, testing, etc)
    def __eq__(self, other):
        return all((np.all(self.ids == other.ids),
                    np.all(self.indices == other.indices),
                    np.all(self.dates == other.dates)))

    def __hash__(self):
        return hash(id(self))

    @property
    def ntimepoints(self):
        """Patient-time points spanned by the SampleSpace date ranges"""
        return np.sum(self.dates[:, 1] - self.dates[:, 0])

    @property
    def ndatapoints(self):
        """Patient-data points spanned by the SampleSpace patient set"""
        return np.sum(self.indices[:, 1] - self.indices[:, 0])

    def batch_indices(self, n=None):
        X = np.hstack((self.ids[:, None], self.indices))
        return tuple(cml.iter_batches(X, n=n))


def overlap(a_start, a_stop, b_start, b_stop):
    """Predicate: do the intervals A and B have any overlap?"""
    # This function is actually documentation. It usually needs to be inlined
    # or vectorized over np.datetime64 arrays for performance.
    assert a_start <= a_stop and b_start <= b_stop
    return a_start <= b_stop and b_start <= a_stop


@functools.cache
def restrict_by_dates(space: SampleSpace, start, stop):
    """
    Clamps the dates within a SampleSpace to the range [start, stop), following
    the semantics of the start and stop arguments to builtins.range. Persons in
    the original SampleSpace without any dates in that range are filterd out.
    Returns a new SampleSpace.
    """
    start = np.datetime64(start)
    stop = np.datetime64(stop)
    mask = (space.dates[:, 0] <= stop) & (start <= space.dates[:, 1])
    dates = np.copy(space.dates[mask])
    dates[:, 0] = np.maximum(dates[:, 0], start)
    dates[:, 1] = np.minimum(dates[:, 1], stop)
    return SampleSpace(space.ids[mask], space.indices[mask], dates)


def sample_uniform(space: SampleSpace, size: int, rng=np.random.default_rng()):
    """
    Produce (ids, dates) sampled uniformly without replacement from the whole
    sampling space, envisioned as a sorted unique set of (person_id, date)
    tuples.
    """
    # It is *critical* to make sure `n` is an int before calling rng.choice
    n = space.ntimepoints.astype(int)
    points = np.sort(rng.choice(n, size=size, replace=False, shuffle=False))
    #
    # >>> def sample(space, points):
    # ...     ids, dates = [], []
    # ...     for (person_id, _, (min_date, max_date)) in space:
    # ...         drange = np.arange(min_date, max_date)
    # ...         ids.append(np.full(len(drange), person_id))
    # ...         dates.append(drange)
    # ...     ids = np.concatenate(ids)
    # ...     dates = np.concatenate(dates)
    # ...     return ids[points], dates[points]
    #
    # The following is equivalent to the above but without concretely realizing
    # the whole sample space, and therefore much more efficient (in both space
    # and time). It is also harder to understand (at least for me) so I've put
    # the slow but explicit reference code above.
    index = np.cumsum((space.dates[:, 1] - space.dates[:, 0]).astype(int))
    index = np.insert(index, 0, 0)
    locs = np.searchsorted(index, points, side='right') - 1
    offsets = points - index[locs]
    # sort=True in the DateSampleIndex constructor will make sure that the
    # persons and corresponding dates are *both* sorted and contiguous
    persons = space.ids[locs]
    dates = space.dates[locs, 0] + offsets
    return DateSampleIndex.from_arrays(persons, dates, sort=True)
