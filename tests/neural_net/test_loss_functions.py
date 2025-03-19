import itertools
import unittest
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cml_tools.neural_net.loss_functions import WGaussianNLLLoss


class TestWGaussianNLLLoss(unittest.TestCase):
    def setUp(self):
        self.n_inst = 1024
        self.n_feat = 121
        self.target = torch.zeros(120) + 4
        self.inputs = torch.zeros(120) + 3.5 + torch.rand(120)
        self.variances = torch.rand(120)

    def test_extra_repr(self):
        # Only a few basic checks... most real values won't be exact
        fn = WGaussianNLLLoss()
        self.assertEqual(str(fn), 'WGaussianNLLLoss(alpha=1.0, beta=1.0)')

        fn = WGaussianNLLLoss(alpha=0.5, beta=2.0)
        self.assertEqual(str(fn), 'WGaussianNLLLoss(alpha=0.5, beta=2.0)')

    def test_defaults(self):
        # alpha=1.0, beta=1.0 should be equivalent to F.gaussian_nll_loss
        inputs = self.inputs
        target = self.target
        var = self.variances
        args = itertools.product(('mean', 'sum', 'none'),
                                 (1e-6, 1e-4, 1e-2, 1e-1),
                                 (False, True))
        for reduction, eps, full in args:
            with self.subTest(reduction=reduction, eps=eps, full=full):
                kws = {'reduction': reduction, 'eps': eps, 'full': full}
                loss_fn = WGaussianNLLLoss(**kws)
                losses = loss_fn(inputs, target, var)
                gnll = F.gaussian_nll_loss(inputs, target, var, **kws)
                self.assertTrue(torch.all(losses == gnll))

    def test_mse_equivalence(self):
        # alpha=0.0, beta=2.0, var=1 should be equivalent to MSE
        inputs = self.inputs
        target = self.target
        var = torch.ones_like(self.inputs)

        for reduction in ('mean', 'sum', 'none'):
            with self.subTest(reduction=reduction):
                fn = WGaussianNLLLoss(alpha=0, beta=2, reduction=reduction)
                losses = fn(inputs, target, var)
                mse = F.mse_loss(inputs, target, reduction=reduction)
                self.assertTrue(torch.all(losses == mse))


if __name__ == '__main__':
    unittest.main()
