"""Functions and Modules for Time Encodings"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sin_cos_encoding(dt, decay=None, encoding_dim=64, decay_rate=10_000,
                     interleave=True):
    """
    Produce the sine and cosine positional encoding detailed in "Attention is
    All You Need" (https://arxiv.org/abs/1706.03762), adapted for continuous
    positions.

    Arguments
    ---------
    dt : Tensor
        A tensor of real numbers, typically representing times or time offsets.
    decay : None
        If not given, calculated from `encoding_dim` and `decay_rate`
    encoding_dim : int, default = 64
        The output dimension of the encoding.
    decay_rate : float, default = 10_000
        How aggressively to expand the sinusoid function wavelengths.
    interleave : bool, default=True
        If True, each even index of the output encoding is a sine function and
        each odd index a cosine function; otherwise the first half are sine
        functions and the second half cosine functions.
    """
    shape = dt.shape
    dt = dt.flatten()
    dt = dt.reshape(len(dt), 1)
    if decay is None:
        decay = (torch.arange(0, encoding_dim, 2)
                 .to(dt.dtype).to(dt.device)
                 .mul_(-math.log(decay_rate) / encoding_dim)
                 .exp_())
    enc = torch.empty(len(dt), encoding_dim, dtype=dt.dtype, device=dt.device)
    if interleave:
        enc[:, 0::2] = torch.sin(dt * decay)
        enc[:, 1::2] = torch.cos(dt * decay)
    else:
        enc[:, :len(decay)] = torch.sin(dt * decay)
        enc[:, len(decay):] = torch.cos(dt * decay)
    return enc.reshape(*shape, encoding_dim)


class SinCosEncoder(nn.Module):
    def __init__(self, encoding_dim, decay_rate=10_000, interleave=True):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.decay_rate = decay_rate
        self.interleave = interleave
        # have to set the decay to the right float dtype explicitly
        decay = (torch.arange(0, encoding_dim, 2)
                 .to(torch.get_default_dtype())
                 .mul_(-math.log(decay_rate) / encoding_dim)
                 .exp_())
        self.register_buffer('decay', decay)

    def _sin_cos_encoding(self, x):
        return sin_cos_encoding(x, decay=self.decay,
                                encoding_dim=self.encoding_dim,
                                interleave=self.interleave)

    def forward(self, dt, y=None):
        # If targets `y` are given, add the delta from each dt to each target.
        # Conceptually, `y` here are cues or prediction times associated with
        # the targets.
        x = self._sin_cos_encoding(dt)
        if y is not None:
            x = x + self._sin_cos_encoding(torch.abs(y - dt))
        return x
