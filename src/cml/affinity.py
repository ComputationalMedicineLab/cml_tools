"""
A fast(ish) implementation of cosine similarity based affinity metrics.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_rows(X):
    """Normalize the *rows* of X to be unit vectors"""
    norms = np.einsum("ij,ij->i", X, X)
    norms = np.sqrt(norms)
    X /= norms[:, None]

def cosine_similarity(A, B, inplace=True, norm_A=True, norm_B=True):
    """Compute `AB / (||A|| * ||B||)`"""
    if not inplace:
        A = np.copy(A, order='C')
        B = np.copy(B, order='C')
    if norm_A: normalize_rows(A)
    if norm_B: normalize_rows(B)
    return np.dot(A, B.T)

def affinity_matrix(A, B, **kwargs):
    """Affinity matrices are the squared cosine similarities"""
    return np.square(cosine_similarity(A, B, **kwargs))

def apply_lsa(C):
    """Get the LSA decomposition of an affinity matrix C"""
    rows, cols = linear_sum_assignment(C, maximize=True)
    assert np.all(rows == np.arange(len(rows)))
    return C, rows, cols

def apply_lsa_both(C):
    """Apply LSA to both C and C.T and return the index permutations to match.

    Assuming `C` is an affinity matrix for two matrices `A` and `B`, the
    column indices returned by lsa(C) will show us how to reorder `B`, keeping
    `A` constant. Since we defined the affinity matrix such that transpose(C)
    is the same as affinity_matrix(B, A), the columns of lsa(C.T) give the
    reordering of `A`, keeping `B` constant. Then `A_idx` is the mapping to
    reorder the `A` matrix and `B_idx` is the mapping to reorder the `B` matrix.
    """
    _, _, B_idx = apply_lsa(C)
    _, _, A_idx = apply_lsa(np.copy(C.T, order='C'))
    return C, A_idx, B_idx

def apply_max(C):
    """Get the max affinity per direction of an affinity matrix C."""
    A_idx = np.argmax(C, axis=0)
    B_idx = np.argmax(C, axis=1)
    return C, A_idx, B_idx

def apply_max_ordering(C):
    n, m = C.shape
    _, A_idx, B_idx = apply_max(C)
    assert len(A_idx) == n and len(B_idx) == m
    A_ord = np.argsort(C[A_idx, np.arange(m)])[::-1]
    B_ord = np.argsort(C[np.arange(n), B_idx])[::-1]
    return C, A_idx, A_ord, B_idx, B_ord

def affinity_lsa(A, B, **kwargs):
    """Compute the affinity matrix and the corresponding LSA indices"""
    return apply_lsa(affinity_matrix(A, B, **kwargs))

def affinity_max(A, B, **kwargs):
    """Compute the indices in each direction of max affinity"""
    return apply_max(affinity_matrix(A, B, **kwargs))
