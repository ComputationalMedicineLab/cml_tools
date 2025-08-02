"""Tools for sampling patient data spaces"""
import collections
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
        # The np function lexsort sorts from the last argument to the first.
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

    # Maybe TODO: build a generalized "datetime unit" abstraction for use here
    # and in ehr.dtypes (and many ehr.curves)?
    def set_dt_unit(self, unit):
        """Changes the datetime unit to `unit` on this instance"""
        new_dtype = np.dtype(f'M8[{unit}]')
        if new_dtype != self.datetime.dtype:
            self.datetime = self.datetime.astype(new_dtype)
        return self

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
    def index_map(self):
        return self._index_map

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
        """Produce a SampleSpace from an EHR instance"""
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


def dedupe_index(index: SampleIndex):
    """Drop duplicate sample points from a SampleIndex. Returns new object."""
    # Round trip the values through a recarray so that np.unique will work
    ra = np.unique(np.rec.fromarrays(index.astuple))
    return SampleIndex(ra.field(0), ra.field(1))


# Simple wrapper for the output of fit_index_into_space, see below
FitResults = collections.namedtuple('FitResults',
    'index space n_missing_persons n_missing_datapoints n_out_of_range'
)
def fit_index_into_space(index, space, match_dt_units=True):
    """
    Fits a SampleIndex into a given SampleSpace at index resolution. Persons
    who are not present in the space are dropped from the index, sample points
    which are no longer valid are clipped to the SampleSpace date ranges when
    possible else dropped. The return value is a namedtuple containing a new
    index, the corresponding subset of the input space (perhaps adjusted to the
    index's time resolution), and the numbers of persons dropped, sample points
    dropped, and sample points adjusted to be within target time boundaries.
    """
    # The index may contain persons not in the sample space: remove them
    p_idx = np.array([space.index_map.get(p, np.nan) for p in index.person_id])
    nan_mask = np.isnan(p_idx)
    n_missing_persons = len(np.unique(index.person_id[nan_mask]))
    n_missing_datapoints = sum(nan_mask)

    subindex = index[~nan_mask]
    subspace = space[p_idx[~nan_mask].astype(int)]

    # The index may contain sample points outside the space. But first, we may
    # need to drop the resolution of the samplespace to match the resolution of
    # the index (e.g. if the index is specified at daily resolution and the
    # space at seconds or minutes). The assumption is that the curves are built
    # at the resolution of the sampling index. But the samplespace may lose
    # some persons at a lower resolution (if, e.g., a person has data only
    # within a single day, we may be able to build curves and sample at second
    # resolution but not at daily). So we have to check if we're dropping more
    # people.
    if match_dt_units:
        subspace.datetimes = subspace.datetimes.astype(subindex.datetime.dtype)
        keep_mask = subspace.datetimes[:, 1] > subspace.datetimes[:, 0]
        n_missing_persons += len(np.unique(subspace.person_id[~keep_mask]))
        n_missing_datapoints += sum(~keep_mask)
        subindex = subindex[keep_mask]
        subspace = subspace[keep_mask]

    # Now truncate the sample index dates to fit within the sample space
    lo_mask = subindex.datetime < subspace.datetimes[:, 0]
    hi_mask = subindex.datetime > subspace.datetimes[:, 1] - 1
    subindex.datetime[lo_mask] = subspace.datetimes[lo_mask, 0]
    subindex.datetime[hi_mask] = subspace.datetimes[hi_mask, 1] - 1
    n_out_of_range = sum(lo_mask) + sum(hi_mask)
    return FitResults(dedupe_index(subindex), subspace, n_missing_persons,
                      n_missing_datapoints, n_out_of_range)


def overlap(a_start, a_stop, b_start, b_stop):
    """Predicate: do the intervals A and B have any overlap?"""
    # This function is actually documentation. It usually needs to be inlined
    # or vectorized over np.datetime64 arrays for performance.
    assert a_start <= a_stop and b_start <= b_stop
    return a_start <= b_stop and b_start <= a_stop


@functools.cache
def restrict_to_interval(space: SampleSpace, start, stop):
    """
    Clamps the datetimes within a SampleSpace to the range [start, stop),
    following the semantics of the start and stop arguments to builtins.range.
    Persons in the original SampleSpace without any datetimes in that range are
    filtered out. Returns a new SampleSpace.
    """
    start = np.datetime64(start)
    stop = np.datetime64(stop)
    mask = (space.datetimes[:, 0] < stop) & (start < space.datetimes[:, 1])
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
