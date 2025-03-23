import math
import unittest

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
# This is an sklearn internal API used in one of the tests... NB: it may break
# if they change their implementation!
from scipy import signal
from sklearn.decomposition import _fastica as sk_fastica


from cml.fastica import (
    apply_g, apply_model, decorr_W, fastica, max_change, recover_A_from_WK,
    recover_S_from_WX1, scale_to_unit_variance, update_W
)
from cml.test.base import TorchTestBase
from cml.whiten import apply_whitening, cov_mean, learn_whitening


class TestSignalReconstruction(TorchTestBase):
    @staticmethod
    def gen_test_signals(seed=0):
        np.random.seed(seed)
        n_samples = 2000
        time = np.linspace(0, 8, n_samples)

        s1 = np.sin(2 * time)
        s2 = np.sign(np.sin(3 * time))
        s3 = signal.sawtooth(2 * np.pi * time)

        S = np.vstack((s1, s2, s3))
        S += 0.2 * np.random.normal(size=S.shape)

        S /= S.std(axis=1, keepdims=True)
        A = np.array([[1.0, 0.5, 1.5],
                      [1.0, 2.0, 1.0],
                      [1.0, 1.0, 2.0]])
        # With X properly defined as [features, samples], the ICA equation is
        # straightforwardly calculated: X = AS
        X = np.dot(A, S)
        return X, A, S

    def setUp(self):
        self.X, self.A, self.S = self.gen_test_signals()
        self.W = np.random.normal(size=self.A.shape)
        self.sk_ica = sk_fastica.FastICA(n_components=3,
                                         w_init=np.copy(self.W),
                                         whiten_solver='eigh',
                                         whiten='unit-variance',
                                         # defaults
                                         max_iter=200,
                                         tol=1e-04)

    def test_signal_reconstruction(self):
        # Automatically tests the reconstruction error, but this really needs
        # to be checked visually well. Should this be two or more tests? Maybe
        # check against sklearn and against the original separately.
        n = self.X.shape[0]
        m = self.X.shape[1]

        # Get the reconstruction parts from scikit-learn. Bear in mind that out
        # interface specifies X as [features, samples], so there are certain
        # transposes that must be made to work with scikit-learn's API
        sk_S = self.sk_ica.fit_transform(np.copy(self.X.T)).T
        sk_A = self.sk_ica.mixing_
        sk_X = np.dot(sk_A, sk_S) + self.sk_ica.mean_[:,np.newaxis]
        sk_check = np.allclose(self.X, sk_X)
        self.assertTrue(sk_check, 'Scikit-learn reconstruction should work')

        # Get the reconstruction parts from our code
        X0 = torch.from_numpy(np.copy(self.X))
        X1, K, mean = learn_whitening(X0, 3)
        W, _ = fastica(X1, torch.from_numpy(self.W))

        S = recover_S_from_WX1(W, X1)
        A = recover_A_from_WK(W, K)
        fi_X = (A @ S) + mean

        sk_vs_cy = np.allclose(fi_X, sk_X)
        self.assertTrue(sk_vs_cy, 'Reconstruction should match sklearn recon')

        fi_check = np.allclose(self.X, fi_X)
        self.assertTrue(fi_check, 'Reconstruction should match actual input')

    def test_recover_S_from_WX(self):
        X = torch.from_numpy(self.X)
        X1, K, mean = learn_whitening(torch.clone(X), 3)
        X1_direct = (K @ (X - mean))
        self.assert_close(X1, X1_direct, msg='X1 should be K(X-mean)')

        W, _ = fastica(X1, torch.from_numpy(self.W))
        W_orig = torch.clone(W)
        X1_orig = torch.clone(X1)
        fi_S = recover_S_from_WX1(W, X1)
        self.assert_equal(W, W_orig, msg='should not mutate W')
        self.assert_equal(X1, X1_orig, msg='should not mutate X1')

        S = ((W @ K) @ (X - mean))
        self.assertTrue(np.allclose(fi_S, S), 'S should be WK(X - mean)')

    def test_recover_A_from_WK(self):
        X = torch.from_numpy(self.X)
        X1, K, mean = learn_whitening(X, 3)
        W, _ = fastica(X1, torch.from_numpy(self.W))

        W_orig = torch.clone(W)
        K_orig = torch.clone(K)
        fi_A = recover_A_from_WK(W, K)
        self.assert_equal(W, W_orig, msg='should not mutate W')
        self.assert_equal(K, K_orig, msg='should not mutate K')

        A = (W @ K).pinverse()
        self.assert_close(fi_A, A, 'A should be pinv(WK)')

    def test_scale_to_unit_variance(self):
        X = torch.from_numpy(self.X)
        X1, K, mean = learn_whitening(X, 3)
        W, _ = fastica(X1, torch.from_numpy(self.W))

        # Get and scale S and A directly
        # Get unit-variance scaled S and A via numpy
        S = (W @ K) @ (X - mean)
        A = (W @ K).pinverse()
        factors = torch.std(S, axis=1, keepdims=True)
        S = S / factors
        A = A.T * factors

        # Get S by the fast-path from W and X1 directly
        fi_S = recover_S_from_WX1(W, X1)
        fi_A = recover_A_from_WK(W, K)
        S_orig = torch.clone(fi_S)
        A_orig = torch.clone(fi_A)
        fi_factors = torch.std(fi_S, axis=1, keepdims=True)
        scale_to_unit_variance(fi_S, fi_A.T)

        # Our results should match the direct forms and intermediate steps
        self.assert_close(fi_S, S_orig / fi_factors, msg='scale S inplace')
        self.assert_close(fi_A.T, A_orig.T * fi_factors, msg='scale A inplace')
        self.assert_close(fi_A.T, A, msg='should match direct A')
        self.assert_close(fi_S, S, msg='should match direct S')


class TestFastICASubFunctions(TorchTestBase):
    def test_apply_g(self):
        n = 100
        m = 1000
        c = 0
        i = 0
        ones = torch.ones(m)
        for c in (1, 10, 32, 47, 100):
            # We have to make W *small* - apply_g is equivalent to:
            #       gX = torch.tanh(torch.matmul(W, X))
            #       gdv = torch.mean(1 - gX**2, axis=1)
            # Now: the first iteration through with default random sample
            # values, torch.matmul(W, X) will all be > 1.0 - hence it is true
            # that torch.allclose(1.0, gX). In turn, torch.allclose(0, gdv) is
            # true.  `gdv` is *also* (almost always) torch.allclose(0, gdv),
            # because random noise from malloc is almost always close to zero.
            # Therefore, these tests will almost always pass without ever
            # checking that apply_g does *anything* to `gdv` unless we make
            # sure that the output of tanh(WX) isn't all 1's. We do this by
            # dividing W by the number of samples.
            W = torch.rand(n, n) / m
            X = torch.rand(n, m)
            ext_gX = torch.tanh(torch.matmul(W, X))
            ext_gdv = torch.mean(1 - ext_gX**2, axis=1)

            gX = torch.matmul(W, X)
            gdv = torch.zeros(n)

            tmpv = torch.empty(c, m)
            apply_g(gX, gdv, tmpv, ones, batch_size=c)
            with self.subTest(c=c):
                self.assert_close(ext_gX, gX, 'gX should match')
                self.assert_close(ext_gdv, gdv, 'mean(g`X) should match')

    def test_update_W(self):
        n = 100
        m = 1000
        W = torch.rand(n, n)
        X = torch.rand(n, m)
        gX = torch.tanh(torch.matmul(W, X))
        gdv = torch.mean(1 - gX**2, axis=1)
        W1 = torch.zeros_like(W)
        tmp_W = torch.empty_like(W)

        W_orig = torch.clone(W)
        X_orig = torch.clone(X)
        gX_orig = torch.clone(gX)
        gdv_orig = torch.clone(gdv)

        torch.matmul(gX, X.T, out=W1).div_(m)
        update_W(W1, W, gdv, tmp_W)

        self.assert_equal(W, W_orig, 'update_W ought not mutate W')
        self.assert_equal(X, X_orig, 'update_W ought not mutate X')
        self.assert_equal(gX, gX_orig, 'update_W ought not mutate gX')
        self.assert_equal(gdv, gdv_orig, 'update_W ought not mutate gdv')

        tmp_W = ((gX @ X.T) / m) - (gdv[:, None] * W)
        self.assert_equal(W1, tmp_W, 'W1 should match reference impl.')

    def test_max_change(self):
        n = 1000
        X = torch.rand(n, n)
        Y = torch.rand(n, n)

        with self.subTest("Max deviation should match ref implementation"):
            ref_max = max(abs(abs(torch.einsum("ij,ij->i", X, Y)) - 1))
            self.assertEqual(ref_max, max_change(X, Y))

        with self.subTest("max_change : f(X, Y) should equal f(Y, X)"):
            self.assertEqual(max_change(X, Y), max_change(Y, X))


class TestSymmetricDecorrelation(TorchTestBase):
    """
    Symmetric Decorrelation is a critical step in the FastICA algorithm with a
    few useful, testable properties: chiefly, a decorrelated matrix W should
    not change under futher decorrelation (there are no correlations to
    remove), and the eigenvalues should all be identically 1.0
    """
    def setUp(self):
        n = 50
        w_init = torch.rand(n, n)
        W = w_init.clone()
        F = torch.empty_like(W)
        D = torch.empty(n, dtype=W.dtype)
        tmp_W = torch.empty_like(W)
        decorr_W(W, D, F, tmp_W)
        self.n = n
        self.w_init = w_init
        self.W = W

    def test_shape(self):
        self.assertEqual(self.w_init.shape, self.W.shape)

    def test_sym_decorrelation_against_sk(self):
        fail_msg = 'fails to match the reference implementation'
        # N.B. - This test uses scikit-learn's non-public _sym_decorrelation
        # function as a reference implementation to test against. If sklearn
        # changes this may break.
        w_init_np = self.w_init.clone().numpy()
        sk_W = torch.from_numpy(sk_fastica._sym_decorrelation(w_init_np))
        self.assert_close(sk_W, self.W, msg=fail_msg)

    def test_sym_decorrelation_eigenvalues(self):
        # If W has been correctly symmetrically decorrelated then the
        # eigenvalues of WW^T must all be identically one. This property leads
        # also to idempotence of the operation (tested next).
        #   LaTeX illustration of this property:
        # $$
        # \begin{align}
        # \mathbf{W}_{next}
        #       &= (\mathbf{W}\mathbf{W}^T)^{-1/2}\mathbf{W} \\
        #       &= (\mathbf{F}\mathbf{D}^{-1/2}\mathbf{F}^T)\mathbf{W} \\
        #       &= (\mathbf{F}\mathbf{I}\mathbf{F}^T)\mathbf{W} \\
        #       &= (\mathbf{F}\mathbf{F}^T)\mathbf{W} \\
        #       &= \mathbf{I}\mathbf{W} \\
        #       &= \mathbf{W} \\
        # \end{align}
        # $$
        D, _ = torch.linalg.eigh(torch.matmul(self.W, self.W.T))
        self.assert_ones(D, 'eigenvalues of W should be identically 1')

    def test_idempotence(self):
        W1 = torch.clone(self.W)
        F = torch.empty_like(W1)
        D = torch.empty(len(W1), dtype=W1.dtype)
        tmp_W = torch.empty_like(W1)
        # Run the decorrelation
        decorr_W(W1, D, F, tmp_W)
        # Decorrelating a decorrelated matrix should produce no change
        self.assert_close(self.W, W1, 'f(W) == f(f(W)) should be True')


if __name__ == '__main__':
    unittest.main()
