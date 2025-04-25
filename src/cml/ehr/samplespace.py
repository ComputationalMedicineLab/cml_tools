"""Tools for sampling patient data spaces"""
import contextlib
import functools
import pickle

import numpy as np

from cml import wrapdtypes, iter_batches
from cml.ehr.dtypes import EHR
from cml.record import Record, SOA


class SampleIndex(SOA):
    fields = ('person_id', 'datetime')
    dtypes = wrapdtypes('i8', 'M8[s]')
    __slots__ = fields + ('_npersons', '_ndates', '_mapping')

    def __init__(self, person_id, datetime):
        # Make sure the inputs are always lexically sorted (person, datetime)
        order = np.lexsort((datetime, person_id))
        self.person_id = np.copy(person_id[order])
        self.datetime = np.copy(datetime[order])

        # Make a mapping from person_id -> dates
        # The zero-th split is always empty for some reason, need [1:]
        ids, locs = np.unique(self.person_id, return_index=True)
        splits = zip(ids, np.split(self.datetime, locs)[1:])
        self._mapping = dict(splits)

        # Cache these two properties on the instance; why calculate repeatedly?
        self._npersons = len(ids)
        self._ndates = len(np.unique(self.datetime))

    @property
    def mapping(self):
        return self._mapping

    @property
    def npersons(self):
        """Number of unique persons in the sample index"""
        return self._npersons

    @property
    def ndates(self):
        """Number of unique dates in the sample index"""
        return self._ndates


class SampleSpace(SOA):
    fields = ('person_id', 'indices', 'datetimes')
    dtypes = wrapdtypes('i8', 'i8', 'M8[s]')
    __slots__ = fields + ('_index_map', )

    def __init__(self, person_id, indices, datetimes):
        self.person_id = np.array(person_id)
        self.indices = np.array(indices)
        self.datetimes = np.array(datetimes)
        self._index_map = {p: i for i, p in enumerate(self.person_id)}

    @property
    def astuple(self):
        return (self.person_id, self.indices, self.datetimes)

    def set_dt_unit(self, unit):
        """Changes the datetime unit to `unit` on this instance"""
        new_dtype = np.dtype(f'M8[{unit}]')
        if new_dtype != self.datetimes.dtype:
            self.datetimes = self.datetimes.astype(new_dtype)
        return self

    @contextlib.contextmanager
    def dt_unit_ctx(self, unit):
        prior = self.datetime_format[0]
        try:
            yield self.set_dt_unit(unit)
        finally:
            self.set_dt_unit(prior)

    @classmethod
    def from_ehr(cls, ehr: EHR):
        """Produce a SampleSpace from a recarray of EHR data"""
        # np.unique will cause a brief doubling of memory usage
        ids, indices = np.unique(ehr.person_id, return_index=True)
        indices = indices.tolist()
        indices.append(len(ehr))
        indices = tuple(zip(indices, indices[1:]))
        # The min/max of the datetime is where most time is spent. This is
        # effectively an efficient groupby-reduce using direct indexing. The
        # datetime ranges are [min_date, max_date), following the semantics of
        # start and stop in range, np.arange, etc.
        dts = (ehr.datetime[i:j] for i, j in indices)
        dts = tuple((d.min(), d.max()+1) for d in dts)
        return cls(ids, indices, dts)

    def __hash__(self):
        return hash(id(self))

    @property
    def ndates(self):
        """Number unique (person, date) pairs in the SampleSpace"""
        D = self.datetimes.astype('M8[D]').T
        return np.sum(D[1] - D[0])

    @property
    def ntimepoints(self):
        """Patient-time points spanned by the SampleSpace date ranges"""
        return np.sum(self.datetimes[:, 1] - self.datetimes[:, 0])

    @property
    def datetime_format(self):
        """Return the datetime format code and step size of self.datetimes"""
        return np.datetime_data(self.datetimes.dtype)

    @property
    def ndatapoints(self):
        """Patient-data points spanned by the SampleSpace patient set"""
        return np.sum(self.indices[:, 1] - self.indices[:, 0])

    def batch_indices(self, n=None):
        X = np.hstack((self.person_id[:, None], self.indices))
        return tuple(iter_batches(X, n=n))


def overlap(a_start, a_stop, b_start, b_stop):
    """Predicate: do the intervals A and B have any overlap?"""
    # This function is actually documentation. It usually needs to be inlined
    # or vectorized over np.datetime64 arrays for performance.
    assert a_start <= a_stop and b_start <= b_stop
    return a_start <= b_stop and b_start <= a_stop


@functools.cache
def restrict_by_dates(space: SampleSpace, start, stop):
    """
    Clamps the datetimes within a SampleSpace to the range [start, stop),
    following the semantics of the start and stop arguments to builtins.range.
    Persons in the original SampleSpace without any datetimes in that range are
    filterd out. Returns a new SampleSpace.
    """
    start = np.datetime64(start)
    stop = np.datetime64(stop)
    mask = (space.datetimes[:, 0] <= stop) & (start <= space.datetimes[:, 1])
    datetimes = np.copy(space.datetimes[mask])
    datetimes[:, 0] = np.maximum(datetimes[:, 0], start)
    datetimes[:, 1] = np.minimum(datetimes[:, 1], stop)
    return SampleSpace(space.person_id[mask], space.indices[mask], datetimes)


def sample_uniform(space: SampleSpace, size: int, rng=np.random.default_rng()):
    """
    Produce (person_id, datetime) sampled uniformly without replacement from
    the whole sampling space, envisioned as a sorted unique set of (person_id,
    datetime) tuples.
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
    delta = space.datetimes[:, 1] - space.datetimes[:, 0]
    index = np.cumsum(delta.astype(int))
    index = np.insert(index, 0, 0)
    locs = np.searchsorted(index, points, side='right') - 1
    offsets = points - index[locs]
    sample_persons = space.person_id[locs]
    sample_times = space.datetimes[locs, 0] + offsets
    return SampleIndex(sample_persons, sample_times)
