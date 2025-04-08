"""
Harvest descriptive statistics from data arrays and merge statistics harvested
from disparate sources.
"""
# bottleneck includes some very fast nan-safe drop-in replacements and
# extensions for numpy ndarrays: here we use it for nanmean, nanvar, etc.
import bottleneck as bn
import numpy as np


def collect_stats(data, byrow=True, nansafe=False, squeeze_nan=False):
    """
    Calculate base descriptive statistics from a potentially spares ndarray
    dataset `data`. By default `data` is expected to have feature rows and
    observation columns; i.e., the statistics will be calculated over (axis=1).
    If `byrow` is False the statistics are taken over columns instead.
    """
    axis = 1 if byrow else 0
    if squeeze_nan:
        mask = np.logical_not(bn.allnan(data, axis=axis))
        data = data[mask]
    out = np.empty((6, len(data)))
    if nansafe:
        out[0] = np.sum(np.isfinite(data), axis=axis)
        out[1] = bn.nanmean(data, axis=axis)
        out[2] = bn.nanvar(data, axis=axis)
        out[3] = bn.nansum(data < 0, axis=axis)
        out[4] = bn.nanmin(data, axis=axis)
        out[5] = bn.nanmax(data, axis=axis)
    else:
        out[0] = data.shape[axis]
        out[1] = np.mean(data, axis=axis)
        out[2] = np.var(data, axis=axis)
        out[3] = np.sum(data < 0, axis=axis)
        out[4] = np.min(data, axis=axis)
        out[5] = np.max(data, axis=axis)
    if squeeze_nan:
        return out, mask
    return out


def merge_unlabeled_stats(A, B):
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


def merge_stats(A_stats, B_stats, A_labels, B_labels):
    # Get the sorted union of A and B channels then the locations of each
    # subset within that union; then get the common indices with respect to
    # their own data matrices and with respect to the output matrix.
    C_labels = np.union1d(A_labels, B_labels)
    _, A_idx, A_sort = np.intersect1d(C_labels, A_labels, assume_unique=True, return_indices=True)
    _, B_idx, B_sort = np.intersect1d(C_labels, B_labels, assume_unique=True, return_indices=True)
    X_idx, AX_idx, BX_idx = np.intersect1d(A_idx, B_idx, assume_unique=True, return_indices=True)
    # Just copy the data in: any shared columns will be overwritten by the
    # update routines immediately below; simply copying all items and
    # overwriting the common columns is faster and easier than computing the
    # complement indices of A with respect to B in C.
    C = np.empty((6, len(C_labels)))
    C[:, A_idx] = A_stats[:, A_sort]
    C[:, B_idx] = B_stats[:, B_sort]
    # Run the mean/variance update formulae
    A = A_stats[:, A_sort[AX_idx]]
    B = B_stats[:, B_sort[BX_idx]]
    C[0, X_idx] = A[0] + B[0]
    af = A[0] / C[0, X_idx]
    bf = B[0] / C[0, X_idx]
    dx = B[1] - A[1]
    C[1, X_idx] = A[1] + (bf * dx)
    C[2, X_idx] = (A[2] * af) + (B[2] * bf) + (af * bf * dx * dx)
    # Update the number negative, data min, and data max values respectively
    C[3, X_idx] = np.add(A[3], B[3])
    C[4, X_idx] = np.minimum(A[4], B[4])
    C[5, X_idx] = np.maximum(A[5], B[5])
    return C, C_labels
