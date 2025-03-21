import torch
import torch.nn as nn
import torch.nn.functional as F


def cov_mean(X, block_size=None, eps=None):
    """
    Memory efficient and fast computation of cov(X) and E[X]. Equivalent usage:

    >>> X = torch.rand(121, 10_000, dtype=torch.float64)
    >>> C, m = low_mem_cov_mean(X)
    >>> assert torch.allclose(C, torch.cov(X, correction=0))
    >>> assert torch.allclose(m, torch.mean(X, axis=1, keepdim=True))

    The typical usage is for very rectangular matrices: the rows are features
    and the columns samples, with many more samples than there are features.
    """
    if block_size is None:
        block_size, n_samples = X.shape
    else:
        n_samples = X.shape[1]

    if eps is None:
        eps = torch.finfo(X.dtype).eps

    m = torch.mean(X, axis=1, keepdim=True)
    if torch.all(m <= eps):
        C = torch.zeros(len(X), len(X), dtype=X.dtype)
    else:
        C = torch.matmul(m, m.T).mul_(-n_samples)

    n_blocks, rem = divmod(n_samples, block_size)
    n_blocks += int(rem > 0)
    for i in range(n_blocks):
        Xb = X[:, i*block_size:(i+1)*block_size]
        torch.addmm(C, Xb, Xb.T, out=C)
    C.div_(n_samples)

    return C, m


def apply_whitening(Y, K, X_mean, inplace=True, out=None):
    """
    Computes the transformation `K(Y - X_mean)`, for `K` and `X_mean` learned
    from (a potentially different) matrix `X`.

    Arguments
    ---------
    Y : Tensor of shape (n, m)
    K : Tensor of shape (c, n)
    X_mean: Tensor of shape (n, 1)
    out : Tensor of shape (c, m)
        For number `n` of features, number `m` of samples (observations), and
        number `c` of components down to which to project `Y`.
    """
    assert X_mean.shape == (len(Y), 1)
    if inplace:
        Y.sub_(X_mean)
    else:
        Y = (Y - X_mean)
    return torch.matmul(K, Y, out=out)


def learn_whitening(X, n_component=None, component_thresh=None, apply=True,
                    inplace=True, out=None, eps=None):
    """Batch learns a whitening matrix for X.

    A zero-mean random vector `z` is said to be "white" if its elements `z[i]`
    are uncorrelated and have unit variances. A whitening transformation for a
    random vector `x` is a linear operator `K` such that `z = Vx` and `z` is
    white. This function fits a linear transformation `K` to a matrix of
    observations of `x`. The elements of `x` (the features) are the row
    dimension; the observations of `x` (the samples) are the column dimension;
    it is assumed therefore that `X.shape[0]` will usually be much larger than
    `X.shape[1]`.

    Let `E` be the eigenvector matrix and `D` the diagonal matrix of
    eigenvalues of the covariance matrix of `X`. Then `K = D^{-1/2}E^T` is a
    whitening matrix for `X`.

    We furthermore sort `K` by the magnitude of the singular vals `D^{1/2}`. If
    `n_component` is given, we select the top `n_component` rows of `K`; if
    `n_component is None` and `component_thresh` is given, we select as many
    rows as have corresponding singular vals greater than or equal to
    `component_thresh`.
    """
    if n_component is None and component_thresh is None:
        raise ValueError('must provide one of n_component or component_thresh')
    elif component_thresh is None and n_component < 1:
        raise ValueError('n_component must be nonzero positive or -1')

    if not eps or eps < 0:
        eps = torch.finfo(X.dtype).eps

    # Get whitening matrix from the eigh decomposition
    Cx, mx = cov_mean(X)
    svals, ecols = torch.linalg.eigh(Cx)
    # The eigenvalues and corresponding eigenvectors returned by `eigh` are in
    # *ascending* order. We get the singular values from the eigenvalues, then
    # divide the vectors by the corresponding singular values, and multiply by
    # the sign of the leading element (so that the largest eigenval's vector's
    # leading element is always positive). The scaled eigen columns are placed
    # in the *rows* of K.
    svals.clamp_(min=eps)
    svals.sqrt_()
    ecols.mul_(ecols[0].sgn())
    ecols.div_(svals)
    # Reverse and transpose the eigencolumns, then choose the top `n_component`
    # to acquire `K`, our whitening matrix. If `n_component` is not specified
    # as a positive integer but `component_thresh` is a positive number, we
    # choose as many as have singular vals (= sqrt(eigenvals)) above
    # `component_thresh`.
    if n_component is None:
        if (n_component := torch.sum(svals >= component_thresh)) < 1:
            raise RuntimeError(f'Invalid {n_component=}; {component_thresh=}')
    indices = torch.argsort(svals, descending=True)
    indices = indices[:n_component]
    K = (ecols[:, indices]).t_()
    if apply:
        X1 = apply_whitening(X, K, mx, inplace=inplace, out=out)
        return X1, K, mx
    return K, mx
