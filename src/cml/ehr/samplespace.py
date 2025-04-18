"""Tools for sampling patient data spaces"""
import functools
import numpy as np

import cml
from cml.ehr.dtypes import EHR


class SampleSpace:
    __slots__ = ('ids', 'indices', 'dates')

    def __init__(self, ids, indices, dates):
        self.ids = np.array(ids)
        self.indices = np.array(indices)
        self.dates = np.array(dates)

    @property
    def astuple(self):
        return (self.ids, self.indices, self.dates)

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

    @classmethod
    def from_iter(cls, iterable):
        """Constructor from an iterable of (person, ij, dates) tuples"""
        # Inverse of self.__iter__; notice that *zip(*) is inverse of zip
        return cls(*zip(*iterable))

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        yield from zip(self.ids, self.indices, self.dates)

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
        return tuple(cml.iter_batches(X, n=n, consume=False))


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
    return space.ids[locs], (space.dates[locs, 0] + offsets)
