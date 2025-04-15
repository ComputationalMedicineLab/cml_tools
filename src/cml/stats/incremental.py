"""
Incremental descriptive statistics container and functions.
"""
from copy import deepcopy
from functools import reduce

import bottleneck as bn
import numpy as np


class IncrStats:
    __slots__ = ('labels', 'count', 'mean', 'variance', 'negative', 'minval', 'maxval')

    def __init__(self, labels, count, mean, variance, negative, minval, maxval):
        self.labels = labels
        self.count = count
        self.mean = mean
        self.variance = variance
        self.negative = negative
        self.minval = minval
        self.maxval = maxval

    @property
    def astuple(self):
        return tuple(getattr(self, name) for name in self.__slots__)

    @property
    def asdict(self):
        return dict((name, getattr(self, name)) for name in self.__slots__)

    @property
    def asarrays(self):
        return (self.labels, np.stack(self.astuple[1:]))

    @property
    def asframe(self):
        import pandas as pd
        return pd.DataFrame(self.asdict).set_index('labels')

    # These are mostly for testing and debugging
    def __eq__(self, other):
        L_labels, L_stats = self.asarrays
        R_labels, R_stats = other.asarrays
        return np.all(L_labels == R_labels) and np.allclose(L_stats, R_stats)

    @property
    def anynan(self):
        return not any(map(bn.anynan, self.astuple))


def collect(data, labels, byrow=True, nansafe=True, nansqueeze=True):
    """Collect descriptive statistics from a data matrix `data`"""
    axis = 1 if byrow else 0
    if nansqueeze:
        mask = np.logical_not(bn.allnan(data, axis=axis))
        data = data[mask]
        labels = labels[mask]
    if nansafe:
        count = np.sum(np.isfinite(data), axis=axis)
        mean = bn.nanmean(data, axis=axis)
        variance = bn.nanvar(data, axis=axis)
        negative = bn.nansum(data < 0, axis=axis)
        minval = bn.nanmin(data, axis=axis)
        maxval = bn.nanmax(data, axis=axis)
    else:
        count = data.shape[axis]
        mean = np.mean(data, axis=axis)
        variance = np.var(data, axis=axis)
        negative = np.sum(data < 0, axis=axis)
        minval = np.min(data, axis=axis)
        maxval = np.max(data, axis=axis)
    return IncrStats(labels, count, mean, variance, negative, minval, maxval)


def merge_unlabeled(A, B):
    # Should work on either vectors or matrices
    assert A.shape == B.shape
    C = np.empty(A.shape)
    C[0] = A[0] + B[0]
    af = A[0] / C[0]
    bf = B[0] / C[0]
    dx = B[1] - A[1]
    C[1] = A[1] + (bf * dx)
    C[2] = (A[2] * af) + (B[2] * bf) + (af * bf * dx * dx)
    C[3] = np.add(A[3], B[3])
    C[4] = np.minimum(A[4], B[4])
    C[5] = np.maximum(A[5], B[5])
    return C


def merge(A: IncrStats, B: IncrStats) -> IncrStats:
    """Merge two sets of incremental statistics"""
    A_labels, A_stats = A.asarrays
    B_labels, B_stats = B.asarrays
    # C_labels is the *sorted and unique* union of both sets of labels, so the
    # intersection indices `*_sort` are the argsort of `*_labels`. The indices
    # `*_idx` slot the labels into their place in the sorted union.
    C_labels = np.union1d(A_labels, B_labels)
    _, A_idx, A_sort = np.intersect1d(C_labels, A_labels, assume_unique=True, return_indices=True)
    _, B_idx, B_sort = np.intersect1d(C_labels, B_labels, assume_unique=True, return_indices=True)
    X_idx, AX_idx, BX_idx = np.intersect1d(A_idx, B_idx, assume_unique=True, return_indices=True)
    # Step 1: copy each input set of stats _in toto_ into their location in the
    # merged stats. The stats for common labels will be overwritten twice;
    # doing so is faster (and easier to understand) than calculating which
    # labels are unique to which inputs and avoiding the extraneous copies.
    C = np.empty((6, len(C_labels)))
    C[:, A_idx] = A_stats[:, A_sort]
    C[:, B_idx] = B_stats[:, B_sort]
    # Step 2: Merge the stats for the common set of labels. `A_sort[AX_idx]` is
    # index black magic but correct: the common labels are selected and sorted.
    A = A_stats[:, A_sort[AX_idx]]
    B = B_stats[:, B_sort[BX_idx]]
    C[:, X_idx] = merge_unlabeled(A, B)
    return IncrStats(C_labels, *C)


def extend_obs(S: IncrStats, count, fill=0.0, labels=None):
    """
    Augments the statistics for a set of labels as though the existing data had
    been extended to `count` total observations, filling with `fill`.
    """
    _, idx, _ = np.intersect1d(S.labels, labels, assume_unique=True, return_indices=True)
    update = np.full((6, len(labels)), fill)
    update[0] = count - S.count[idx]
    update[2] = 0.0
    return merge(S, IncrStats(labels, *update))


def concat_obs(S: IncrStats, labels=None):
    """
    Set the statistic estimates for a set of labels to their joint estimates.
    Effectively performs a merge reduction of the statistics for the given
    labels and then broadcasts the results to all the labels.
    """
    # Make sure we don't modify the input IncrStats (esp. labels!). If no
    # labels are given then we're reducing over the whole label set of S. If
    # none of the labels given match the labels in S then there's nothing to do
    # so we return the _copy_ of S. Otherwise we do the reduction.
    S = deepcopy(S)
    if labels is None:
        idx = np.arange(len(S.labels))
    else:
        _, idx, _ = np.intersect1d(S.labels, labels, assume_unique=True, return_indices=True)
    if not np.any(idx):
        return S
    _, stats = S.asarrays
    stats[:, idx] = reduce(merge_unlabeled, stats.T[idx]).reshape(-1, 1)
    return IncrStats(S.labels, *stats)
