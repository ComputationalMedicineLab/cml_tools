"""Pytorch implemented Neurel Net Modules and associated blocks"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipBlock(nn.Module):
    """Sums the inputs and outputs of a fully connected linear layer"""
    def __init__(self, width, act_class=None):
        super().__init__()
        self.layer = nn.Linear(width, width)
        self.norm = nn.BatchNorm1d(width)
        if act_class is None:
            self.activation = nn.ReLU()
        else:
            self.activation = act_class()

    def forward(self, x):
        return x + self.activation(self.norm(self.layer(x)))


class LinearResidualBlock(nn.Module):
    """A Residual Block built on Linear layers instead of Convolutions"""
    def __init__(self, width, hidden=None):
        super().__init__()
        hidden = hidden or width
        self.block = nn.Sequential(
            nn.Linear(width, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class SplitGaussianModel(nn.Module):
    """Train two parallel networks to predict mean and variance"""
    def __init__(self, predict_mean, predict_vars):
        super().__init__()
        self.predict_mean = predict_mean
        self.predict_vars = predict_vars
        self.var_activation = nn.Softplus()

    def forward(self, x):
        m = self.predict_mean(x)
        v = self.predict_vars(x)
        return m, self.var_activation(v)


class GaussianModel(nn.Module):
    """Train a single network to predict mean and variance"""
    def __init__(self, predict, ntarget=1):
        super().__init__()
        self.ntarget = ntarget
        self.predict = predict
        self.var_activation = nn.Softplus()

    def forward(self, x):
        pred = self.predict(x)
        m = pred[:, :self.ntarget]
        v = pred[:, self.ntarget:]
        return m, self.var_activation(v)
