"""A Pytorch Module for calculating model mean and variance inputs online"""
# see also: ../online_norm.py, which is designed for aggregating across pandas
# dataframes with assumed heterogeneous data channels. The basic update
# algorithm is the same though.

# TODO: fill_nan currently accepts None or a float; but we could use strings
# like "p01" (or something) to draw from the learned Gaussian and use that as
# the fill instead.

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineStandardScaler(nn.Module):
    """Shifts input instances by online estimations of mean and variance"""
    def __init__(self, num_features, eps=1e-5, fill_nan=None, frozen=False,
                 device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.fill_nan = fill_nan
        self.frozen = frozen
        factory_kwargs = {'device': device, 'dtype': dtype}
        running_mean = torch.zeros(num_features, **factory_kwargs)
        running_var = torch.ones(num_features, **factory_kwargs)
        # If we are nan-safe we may need counts per individual channel
        if self.fill_nan is None:
            running_num = torch.tensor(0, **factory_kwargs)
        else:
            running_num = torch.zeros(num_features, **factory_kwargs)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.register_buffer('running_num', running_num.to(torch.long))

    def extra_repr(self):
        return (
            f'num_features={self.num_features}, eps={self.eps}, '
            f'fill_nan={self.fill_nan}, frozen={self.frozen}'
        )

    def reset_running_stats(self):
        self.running_mean.zero_(0)
        self.running_var.copy_(1)
        self.running_num.zero_(0)

    def update_running_stats(self, batch):
        with torch.no_grad():
            # get the previous running count, mean, var
            pn = self.running_num.clone()
            pv = self.running_var.clone()
            pm = self.running_mean.clone()
            # There is no torch.nanvar yet; we have to calculate ourselves:
            # Var(X) = (E[x])^2 - (E[X])^2
            if self.fill_nan is None:
                cn = len(batch)
                cv, cm = torch.var_mean(batch, axis=0, correction=0)
            else:
                cn = torch.sum(torch.isfinite(batch), axis=0)
                cm = torch.nanmean(batch, axis=0)
                cv = torch.nanmean(batch**2, axis=0) - (cm**2)
            nan_mask = torch.isnan(cm)
            # Merge the two sets of stats using the update from Knuth
            N = pn + cn
            cf = cn / N
            pf = pn / N
            dx = cm - pm
            M = pm + (cf * dx)
            V = (pv * pf) + (cv * cf) + (pf * cf * dx * dx)
            # If any features were all-nan in this batch, the nans will still
            # have propagated through nanmean: ignore those updates. N does not
            # need updating (cn will already be 0 if the batch is all nan).
            M[nan_mask] = self.running_mean[nan_mask]
            V[nan_mask] = self.running_var[nan_mask]
            # Clip the variances to avoid division by very small numbers
            V.clamp_(min=self.eps)
            # Copy the results back into the module buffers
            self.running_num.copy_(N)
            self.running_var.copy_(V)
            self.running_mean.copy_(M)

    def apply_stats(self, batch):
        out = (batch - self.running_mean) / torch.sqrt(self.running_var)
        if self.fill_nan is None:
            return out
        else:
            return torch.nan_to_num(out, nan=self.fill_nan)

    def forward(self, batch):
        # Use an additional flag to determine if should update the running
        # stats. Then one is controlled by train/eval and one explicitly the
        # user. So this layer can be pretrained, or trained only on the first
        # epoch, to learn the mean/variance; then set `scaler.frozen = True`
        # and the module will stop learning mean/variance independently of
        # whether the model it is included in is training or evaluating.
        if self.training and (not self.frozen):
            self.update_running_stats(batch)
        return self.apply_stats(batch)
