"""A torch implementation of the fastica fixed-point algorithm"""
import logging
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from cml_tools.whiten import apply_whitening

# See ./testing.py for why this is here. tanh is (usually) critical to fastica
torch.tanh(torch.tensor(0))


def apply_g(gX, gdv, tmpv=None, ones=None, batch_size=1):
    n = len(gX[0])
    if batch_size < 1: batch_size = 1
    if batch_size > n: batch_size = n
    # The preferred pattern is for caller to initialize and manage these. If
    # called properly this function should allocate no memory.
    if tmpv is None:
        tmpv = torch.empty(batch_size, n, dtype=gX.dtype, device=gX.device)
    if ones is None:
        ones = torch.ones(n, dtype=gX.dtype, device=gX.device)
    # gX = tanh(WX). The multiplcation WX is performed by calling function.
    torch.tanh(gX, out=gX)
    # To compute gdv (gX derivative), which is `mean(1 - gX**2, axis=1)`, we
    # rearrange it to 1 - mean(gX**2, axis=1) and compute in batches of rows.
    # torch.addmv (a wrapper around cblas_dgemv) computes
    #       beta * input + alpha * (mat @ vec)
    # First we copy a batch of the rows of gX into tmpv, then square them
    # inplace. Then, if we set `input` and `vec` to all ones and `alpha` to the
    # negative reciprocal of the sample len, then torch.addmv computes the mean
    # we want for the batched rows. This, it turns out, is very fast -
    # especially at larger sizes of gX. The required shapes are:
    #       - gX.shape == (n_components, n_samples)
    #       - gdv.shape == (n_components,)
    #       - tmpv.shape == (batch_size, n_samples)
    #       - ones.shape == (n_samples,)
    alpha = -1.0 / len(gX[0])
    n_batches, rem = divmod(len(gX), batch_size)
    for i in range(n_batches):
        index = slice(i*batch_size, (i+1)*batch_size)
        tmpv.copy_(gX[index]).square_()
        torch.addmv(ones[index], tmpv, ones, alpha=alpha, out=gdv[index])
    if rem > 0:
        start = n_batches * batch_size
        tmpv[:rem].copy_(gX[start:]).square_()
        torch.addmv(ones[:rem], tmpv[:rem], ones, alpha=alpha, out=gdv[start:])


def update_W(W1, W, gdv, tmp_W):
    # Function does not allocate any memory!
    # W1 -= W*gdv[:,np.newaxis]  # row operation
    tmp_W.copy_(W)
    tmp_W.mul_(gdv[:, None])
    W1.sub_(tmp_W)


def decorr_W(W, D, F, tmp_W, eps=None):
    # Function does not allocate any memory!
    if eps is None:
        eps = torch.finfo(W.dtype).eps
    # Get the eigenvalue decomposition of W @ W.T
    # D, F := (FDF^T) == WW^T
    torch.matmul(W, W.T, out=tmp_W)
    torch.linalg.eigh(tmp_W, out=(D, F))
    # Clamp D at eps for numeric stability of InvSqrt
    D.clamp_(min=eps)
    D.rsqrt_()

    # Three matrix multiplications to get W = FD^{-1/2}F^{T}W_{0}
    # This pattern of calls will (a) use only one working space matrix
    # additional to the three components F, D, W; and (b) ensure each
    # BLAS call has safe output matrices (BLAS is not in-place safe)

    # Start by copying W into W-shaped working space
    tmp_W.copy_(W)

    # W := F^TW_0 ; starting from the right
    torch.matmul(F.T, tmp_W, out=W)

    # W_0 := DW
    # NB: is this copy still necessary? In Cython, the above matmul is a
    # cblas_dgemm which apparently partially overwrites tmp_W; still an issue?
    tmp_W.copy_(W).mul_(D.reshape(-1, 1))

    # W := FW_0 ; Voila, W is now decorrelated in-place (no extra memory alloc)
    torch.matmul(F, tmp_W, out=W)


def max_change(W, W1):
    # Alternate formulation: lim = max(abs(abs(einsum("ij,ij->i", W1, W)) - 1))
    n = W.shape[0]
    # This bit of view magic uses batch matrix multiply at the BLAS level to
    # get the dot product per pairs of rows: x is of shape [n, 1, 1] and when
    # squeezed is the diagonal of (W @ W1.T); this is far the most efficient
    # way to get row by row dot products.
    x = torch.bmm(W.view(n, 1, n), W1.view(n, n, 1)).squeeze()
    return max(abs(abs(x) - 1.0)).item()


def fastica(X, W, *, max_iter=200, tol=1e-4, checkpoint_iter=0,
            checkpoint_dir='./checkpoints', checkpoint_iter_format='03d',
            start_iter=0, c=1, log_timing_format='.4f', apply_g=apply_g):
    """Run the fixed-point fastica algorithm.
    """
    log = logging.getLogger('fastica')
    n, m = X.shape

    if not (n == W.shape[0] and n == W.shape[1]):
        raise ValueError('W must be square and of dim X.shape[0]')
    if not (X.dtype == W.dtype and X.device == W.device):
        raise ValueError('W and X must be the same dtype and device')
    factory_kws = {'dtype': X.dtype, 'device': X.device}
    eps = torch.finfo(X.dtype).eps

    if start_iter > 0:
        log.info(f'FastICA: resume training from iteration: {start_iter}')

    if c < 1: c = 1
    if c > n: c = n

    # Equivalent to `mkdir -p $checkpoint_dir` if we're checkpointing
    if checkpoint_iter > 0:
        log.info('Checkpoints every %d iterations in %s',
                 checkpoint_iter, checkpoint_dir)
        Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

    # W1 holds the estimated ICA matrix and is the return value
    W1 = torch.empty(n, n, **factory_kws)
    # Internal buffers which should be released on function end
    gX = torch.empty(n, m, **factory_kws)
    gdv = torch.empty(n, **factory_kws)
    tmp_m = torch.empty(c, m, **factory_kws)
    tmp_W = torch.empty(n, n, **factory_kws)
    tmp_F = torch.empty(n, n, **factory_kws)
    tmp_D = torch.empty(n, **factory_kws)
    ones = torch.ones(m, **factory_kws)

    # dummy call to warm up dsecnd (first call to CPU clocks is pricy)
    perf_counter()

    log.info('Initial decorrelation of W')
    decorr_W(W, tmp_D, tmp_F, tmp_W, eps)

    log.info('Starting main loop of FastICA algorithm')
    log.info('it|gX|apply_g|W1|update_W|decorr_W|max_change|total|lim')
    # Used by logging calls in the main loop body
    ft = lambda t: format(t, log_timing_format)
    gt = lambda i: format(i, checkpoint_iter_format)

    t_start = perf_counter()
    for it in range(start_iter, start_iter+max_iter):
        # gX = np.dot(W, X)
        t0 = perf_counter()
        torch.matmul(W, X, out=gX)

        # gX = np.tanh(gX); gdv = np.mean(1 - gX**2, axis=1)
        t1 = perf_counter()
        apply_g(gX, gdv, tmp_m, ones, batch_size=c)

        # W1 = np.dot(gX, X.T) # ie. np.dot(np.tanh(np.dot(W, X)), X.T)
        t2 = perf_counter()
        torch.addmm(W1, gX, X.T, out=W1, alpha=1/m, beta=0)

        # W1 -= W*gdv[:,np.newaxis]  # row operation
        t3 = perf_counter()
        update_W(W1, W, gdv, tmp_W)

        # symmetric decorrelation of W1; i.e.  W1 = (WW^T)^{-1}W
        t4 = perf_counter()
        decorr_W(W1, tmp_D, tmp_F, tmp_W, eps=eps)

        # lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        t5 = perf_counter()
        lim = max_change(W, W1)

        t6 = perf_counter()
        log.info('%s|%s|%s|%s|%s|%s|%s|%s|%.8f', str(it),
                 ft(t1-t0), ft(t2-t1), ft(t3-t2), ft(t4-t3),
                 ft(t5-t4), ft(t6-t5), ft(t6-t0), lim)

        # Swaps the names; does not alter or copy any data
        W, W1 = W1, W
        if (checkpoint_iter > 0) and (it % checkpoint_iter == 0):
            torch.save(W, f'{checkpoint_dir}/W.{gt(it)}.pt')
        if lim < tol:
            break

    log.info(f'FastICA: {(it-start_iter)+1} iterations'
             f' run in {perf_counter()-t_start:.4f} seconds')
    return W, it+1


def recover_S_from_WX1(W, X1, S=None, scale_by_nsamples=False):
    """Returns S = WX1 = WK(X-m); i.e. X1 is X centered and whitened"""
    # Equivalent: S = np.dot(W, X)
    # If scale_by_m, S /= np.sqrt(X.shape[1])
    assert W.shape[0] == W.shape[1] and W.shape[1] == X1.shape[0]
    scale = 1.0/math.sqrt(X1.shape[0]) if scale_by_nsamples else 1.0
    if S is None:
        S = torch.empty_like(X1)
    return torch.addmm(S, W, X1, beta=0, alpha=scale, out=S)


def recover_A_from_WK(W, K, A=None):
    """Returns A = (WK)^{-1}; i.e. A is the pseudo-inverse of WK"""
    # Equivalent: A = scipy.linalg.pinv(np.dot(W, K))
    assert W.shape[0] == W.shape[1] and W.shape[1] == K.shape[0]
    if A is None:
        A = torch.empty_like(K)
    torch.matmul(W, K, out=A)
    torch.linalg.pinv(A, out=A)
    return A


def apply_model(W, K, Y, X_mean, scale_factors=None, out=None):
    """
    Applies the ICA model defined by S = WK(X-m) to arbitrary data Y; i.e.,
    produces the projection of Y into the given model. If `scale_factors` is not
    None, then the output will be scaled row-by-row by the given scale factors.
    """
    assert W.shape[0] == W.shape[1] and W.shape[1] == K.shape[0]
    assert K.shape[1] == Y.shape[0] and Y.shape[0] == X_mean.shape[0]
    if scale_factors is not None:
        assert SY.shape[0] == scale_factors.shape[0]
    Y1 = apply_whitening(Y, K, X_mean, out=out)
    SY = recover_S_from_WX1(W, Y1, scale_by_m=False)
    if scale_factors is not None:
        SY.div_(scale_factors)
    return SY


def scale_to_unit_variance(S, AT, alpha=1.0, sign_flip=False, inplace=True):
    """Compute and apply the scaling factors.

    The scaling factors are the per-component (per row) standard deviations of
    S scaled by alpha. If sign_flip is True, then the sign of each factor is
    flipped if the largest absolute value in the row is negative, so that the
    largest value per row is always positive. If inplace is True, S and A are
    scaled in place. S, A, and the scaling factors are all returned.
    """
    assert S.shape[0] == AT.shape[0], 'N comp. should be both S and AT 1st dim'
    factors = torch.std(S, axis=1, keepdim=True).mul_(alpha)
    if sign_flip:
        S_min, S_max = torch.aminmax(S, axis=1, keepdim=True)
        # If the abs value of S_min is greater than S_max then it must be < 0
        factors[abs(S_min) > S_max] *= -1
    if inplace:
        S.div_(factors)
        AT.mul_(factors)
    else:
        S = S / factors
        AT = AT * factors
    return S, AT, factors
