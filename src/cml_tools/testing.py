"""Utilities for testing our torch-based code"""
import os
import unittest

import torch
from torch import allclose


# This appears to prevents a very bizarre "illegal instruction" error that
# seems to happen whenever the first call to `torch.tanh` in a given process is
# on a sufficiently large tensor. By putting this here I make sure that the
# first call to `torch.tanh` in any subclass of TorchTestBase happens when
# TorchTestBase is imported and not inside any subclass's test function.
# https://github.com/pytorch/pytorch/issues/149156#issue-2918418704
torch.tanh(torch.tensor(1))


class TorchTestBase(unittest.TestCase):
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#numerical-accuracy
    @classmethod
    def setUpClass(cls):
        # Use double precision by default. But it can be useful to toggle
        # single precision, since this is the torch (and GPU) default, and some
        # algorithm accuracy can suffer from the loss in precision; looking at
        # which tests start to fail under single precision can be useful.
        dtype = os.getenv('DTYPE', 'double')
        match dtype.lower():
            case 'double'|'float64'|'f64':
                torch.set_default_dtype(torch.float64)
            case 'float'|'float32'|'f32':
                torch.set_default_dtype(torch.float32)
    # The expected accuracy of single-precision floats is about 7 decimal
    # places. The default atol of torch.allclose is 1e-8; 1e-7 is usually
    # sufficient (torch.mean(X - torch.mean(X)) is usuallly < 1e-7). But some
    # tests that check if an online algo works on an instance-by-instance basis
    # accumulates more error, so that the threshold may need to be specified to
    # be as high as atol=1e-5 or 5e-6
    def _atol(self, X, atol):
        if atol is None:
            match X.dtype:
                case torch.float64: atol = 1e-8
                case torch.float32: atol = 1e-6
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
