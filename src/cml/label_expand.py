"""Expands and merges a set of heterogeneously labeled ndarrays"""
import numpy as np


# TODO: rewrite this dostring
def expand(data, labels=None, fills=None, out=None):
    """
    `data` is a sequence of 4-tuples of (id, observation labels, feature
    labels, values). `values` is an ndarray of shape [num features, num
    observations].

    If `labels` is None then the labels are inferred from the union of feature
    labels across feature label arrays in `data`.

    If `fills` is None then the fill value is `np.nan`. If an array, then it
    must be of the same shape as `labels`. If a dict, then it must map feature
    labels to floating point values.
    """
    if labels is None:
        labels = np.unique(np.concatenate(tuple(x[2] for x in data)))

    if fills is None:
        fill_vec = np.full(len(labels), np.nan)
    elif isinstance(fills, dict):
        fill_vec = np.array([fills[x] for x in labels])
    else:
        fill_vec = np.copy(fills.ravel())
    fill_vec = fill_vec.reshape(-1, 1)

    col_dim = sum(len(x[1]) for x in data)
    # NB: index is *untyped* in this function. In order to convert to, e.g., a
    # DateSampleIndex, do something like this:
    # >>> DateSampleIndex.from_arrays(index[0], index[1], sort=False)
    index = np.empty((2, col_dim))
    if out is None:
        X = np.tile(fill_vec, (1, col_dim))
    else:
        X = out
        X[:] = fill_vec

    base = 0
    for (id_, obs, feats, values) in data:
        offset = len(obs)
        index[0][base:base+offset] = id_
        index[1][base:base+offset] = obs
        # If there are no features then there are no values - the observations
        # are empty (i.e, rows of entirely fill values)
        if np.any(feats):
            _, idx, _ = np.intersect1d(labels, feats, assume_unique=True, return_indices=True)
            X[idx, base:base+offset] = values
        base += offset
    return index, labels, X
