import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WGaussianNLLLoss(nn.Module):
    """
    A weighted decomposition of GaussianNLLLoss, implementing the following
    function (i.e., the Gaussian Negative Log Likelihood, with alpha and beta
    weights on the log variance and the error factors):

    >>> 0.5 * ((alpha * ln(var)) + (beta * (prediction-target)**2 / var))

    Arguments
    ---------
    alpha, beta : float
        Weights to apply to the parts of the Gaussian Negative Log Likelihood

    The rest of the arguments have the same meaning as in GaussianNLLLoss.
    """
    def __init__(self, alpha=1.0, beta=1.0, full=False, eps=1e-6,
                 reduction='mean', device=None, dtype=None):
        super().__init__()
        alpha = torch.tensor(alpha, device=device, dtype=dtype)
        beta = torch.tensor(beta, device=device, dtype=dtype)
        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)
        self._constant = 0.5 * math.log(2*math.pi)
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def extra_repr(self):
        return f'alpha={self.alpha}, beta={self.beta}'

    def forward(self, mean, target, var):
        # Follows (mostly) the pytorch implementation of F.gaussian_nll_loss,
        # but without the intermediate broadcasting: we accept a var tensor the
        # same shape as the mean/target or a float only.
        if isinstance(var, float):
            var = torch.full_like(mean, var)
        if torch.any(var < 0):
            raise ValueError('var has negative entry/entries')

        var = torch.clone(var)
        with torch.no_grad():
            var.clamp_(min=self.eps)

        ### Compute the following but (mostly) inplace:
        # err = torch.square(mean - target)
        # loss = 0.5 * (self.alpha*torch.log(var) + self.beta*(err/var))
        loss = (mean - target).square_().div_(var).mul_(self.beta)
        loss.add_(var.log_().mul_(self.alpha)).mul_(0.5)

        if self.full:
            loss += self._constant

        match self.reduction:
            case 'mean': return loss.mean()
            case 'sum': return loss.sum()
            case _: return loss
