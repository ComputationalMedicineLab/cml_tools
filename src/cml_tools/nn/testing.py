"""Utilities for testing our torch-based code"""
import unittest

import torch
from torch import allclose


class TorchTestBase(unittest.TestCase):
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#numerical-accuracy
    # XXX: The expected accuracy of single-precision floats is about 7 decimal
    # places. The default atol of torch.allclose is 1e-8; 1e-7 is usually
    # sufficient (torch.mean(X - torch.mean(X)) is usuallly < 1e-7). But some
    # tests that check if an online algo works on an instance-by-instance basis
    # accumulates more error, so that the threshold may need to be specified to
    # be as high as atol=1e-5.
    def _atol(self, X, atol):
        if atol is None:
            if X.dtype == torch.float32:
                atol = 1e-6
            else:
                atol = 1e-8
        return atol

    def assert_close(self, X, Y, msg='', atol=None, equal_nan=True):
        atol = self._atol(X, atol)
        self.assertTrue(allclose(X, Y, atol=atol, equal_nan=equal_nan), msg)

    def assert_zero(self, X, msg='', atol=None, equal_nan=True):
        ZERO = torch.tensor(0, dtype=X.dtype)
        self.assert_close(X, ZERO, msg=msg, atol=atol, equal_nan=equal_nan)

    def assert_ones(self, X, msg='', atol=None, equal_nan=True):
        ONES = torch.tensor(1, dtype=X.dtype)
        self.assert_close(X, ONES, msg=msg, atol=atol, equal_nan=equal_nan)

    def assert_equal(self, X, Y, msg=''):
        self.assertTrue(torch.all(X == Y), msg)
