"""Test the online norm algorithm applied as a Pytorch module"""
import unittest
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from cml_tools.neural_net.online_norm import OnlineStandardScaler


class TestOnlineStandardScaler(unittest.TestCase):
    def setUp(self):
        self.n_inst = 1024
        self.n_feat = 121
        # Choose n_batch so that it divides n_inst
        self.n_batch = 32
        # Used in every test: a scaler and some data to scale
        self.X = torch.rand(self.n_inst, self.n_feat)
        self.scaler = self.make_scaler(self.n_feat)

    def make_scaler(self, N, eps=1e-5, fill_nan=None):
        """Create a scaler and test that it is initialized correctly"""
        scaler = OnlineStandardScaler(N, eps=eps, fill_nan=fill_nan)
        scaler.train()
        self.assertTrue(torch.all(scaler.running_num == 0.0))
        self.assertTrue(torch.all(scaler.running_mean == 0.0))
        self.assertTrue(torch.all(scaler.running_var == 1.0))
        return scaler

    def assert_equality(self, s0, s1):
        """Check that two scalers are identical"""
        self.assertTrue(torch.all(s0.running_num == s1.running_num))
        self.assertTrue(torch.all(s0.running_mean == s1.running_mean))
        self.assertTrue(torch.all(s0.running_mean == s1.running_mean))

    def get_var_mean(self, X, nansafe=False):
        if nansafe:
            m = torch.nanmean(X, axis=0)
            v = torch.nanmean(X**2, axis=0) - m**2
        else:
            v, m = torch.var_mean(X, axis=0, correction=0)
        return v, m

    def assert_torch_close(self, X, Y, atol=1e-8):
        self.assertTrue(torch.allclose(X, Y, atol=atol, equal_nan=True))

    def assert_fitness(self, X, scaler, n, nansafe=False):
        """Check that scaler has been fit correctly to X over n steps"""
        v, m = self.get_var_mean(X, nansafe=nansafe)
        self.assertTrue(torch.all(scaler.running_num == n))
        self.assert_torch_close(v, scaler.running_var.clone())
        self.assert_torch_close(m, scaler.running_mean.clone())
        # If nansafe, then Y = scaler(X) mean/variance will be altered by the
        # fill values. So before testing that the results are correctly
        # shifted, set them back to nan
        scaler.eval()
        Y = scaler(X)
        Y[torch.isnan(X)] = torch.nan
        yv, ym = self.get_var_mean(Y, nansafe=nansafe)
        self.assert_torch_close(yv, torch.tensor(1.0))
        # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#numerical-accuracy
        # XXX: The expected accuracy of single-precision floats is about 7
        # decimal places. The default atol of torch.allclose is 1e-8; 1e-7 is
        # usually sufficient (torch.mean(X - torch.mean(X)) is usuallly <
        # 1e-7). But the test that checks if this works on an
        # instance-by-instance basis accumulates more error, so that the
        # threshold needs to be atol=1e-5.
        self.assert_torch_close(ym, torch.tensor(0.0), atol=1e-5)

    def test_batch_size_one(self):
        # XXX: see the comment above in assert_fitness
        for x in self.X:
            self.scaler(x.reshape(1, -1))
        self.assert_fitness(self.X, self.scaler, self.n_inst)

    def test_full_batch_size(self):
        for batch in self.X.reshape(self.n_batch, -1, self.n_feat):
            self.scaler(batch)
        self.assert_fitness(self.X, self.scaler, self.n_inst)

    def test_two_scalers_equal(self):
        """Two OnlineStandardScalers should learn the same from the same data"""
        scaler0 = self.make_scaler(self.n_feat)
        scaler1 = self.make_scaler(self.n_feat)
        for batch in self.X.reshape(self.n_batch, -1, self.n_feat):
            x0 = scaler0(batch)
            x1 = scaler1(batch)
            self.assertTrue(torch.all(x0 == x1))
        # They should still be correctly fitted as well as identical
        self.assert_fitness(self.X, scaler0, self.n_inst)
        self.assert_fitness(self.X, scaler1, self.n_inst)
        self.assert_equality(scaler0, scaler1)

    def test_scaler_stops_training_in_eval(self):
        for batch in self.X.reshape(self.n_batch, -1, self.n_feat):
            self.scaler(batch)
        self.assert_fitness(self.X, self.scaler, self.n_inst)
        frozen_scaler = deepcopy(self.scaler)
        frozen_scaler.eval()
        self.scaler.eval()
        Y = torch.rand(self.n_inst, self.n_feat)
        for batch in Y.reshape(self.n_batch, -1, self.n_feat):
            y_scaled = self.scaler(batch)
        self.assert_equality(self.scaler, frozen_scaler)

    def test_scaler_stops_training_if_frozen(self):
        """Test that a frozen scaler won't train even if torch sets #train()"""
        for batch in self.X.reshape(self.n_batch, -1, self.n_feat):
            self.scaler(batch)
        self.assert_fitness(self.X, self.scaler, self.n_inst)
        # Set the copy to be neither frozen nor training
        frozen_scaler = deepcopy(self.scaler)
        frozen_scaler.train(False)
        self.assertFalse(frozen_scaler.frozen)
        self.assertFalse(frozen_scaler.training)
        # Set the scaler to be both frozen and training
        self.scaler.frozen = True
        self.scaler.train(True)
        self.assertTrue(self.scaler.training)
        self.assertTrue(self.scaler.frozen)
        Y = torch.rand(self.n_inst, self.n_feat)
        for batch in Y.reshape(self.n_batch, -1, self.n_feat):
            self.scaler(batch)
        self.assert_equality(self.scaler, frozen_scaler)

    def test_scaler_with_nan_inputs(self):
        fill_nan = 0.0
        scaler = self.make_scaler(self.n_feat, fill_nan=fill_nan)
        # Set the top right triangle of the input to NaN
        X = self.X.clone()
        X[*torch.triu_indices(self.n_feat, self.n_feat)] = torch.nan
        # calculate the expected number of NaN per channel
        N = torch.full((self.n_feat,), self.n_inst)
        N = N - torch.arange(1, self.n_feat+1)
        for batch in X.reshape(self.n_batch, -1, self.n_feat):
            scaler(batch)
        self.assert_fitness(X, scaler, N, nansafe=True)
        # Check that Y are all finite and filled with fill_nan
        scaler.eval()
        Y = scaler(X)
        self.assertTrue(torch.all(torch.isfinite(Y)))
        self.assertTrue(torch.all(Y[torch.isnan(X)] == fill_nan))


if __name__ == '__main__':
    unittest.main()
