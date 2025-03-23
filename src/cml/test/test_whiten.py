import unittest

import torch
import torch.nn
import torch.nn.functional as F

from cml.test.base import TorchTestBase
from cml.whiten import apply_whitening, cov_mean, learn_whitening


class TestWhitening(TorchTestBase):
    def setUp(self):
        # XXX Need to add more tests with differing ratios of feature to
        # component dim, different distributions, etc; what is here are just
        # the most basic checks possible.
        self.n_samples = 1024
        self.n_features = 121
        self.n_component = 50
        self.run_whiten(torch.rand(self.n_features, self.n_samples),
                        n_component=self.n_component)

    def run_whiten(self, X, **kwargs):
        self.X = X
        self.X_orig = torch.clone(X)
        kwargs.setdefault('n_component', self.n_component)
        self.X1, self.K, self.X_mean = learn_whitening(X, **kwargs)

    def test_cov_mean(self):
        # Check various shapes to see that cov_mean behaves correctly
        for shape in ((1, 1), (100, 10_000), (121, 1024), (100, 50)):
            with self.subTest(shape=shape):
                X = torch.rand(shape)
                Cx, mx = cov_mean(X)
                self.assert_close(Cx, torch.cov(X, correction=0))
                self.assert_close(mx, torch.mean(X, axis=1, keepdim=True))

    def test_shapes(self):
        # Shape Tests: very important to keep clear and consistent semantics
        self.assertEqual(self.X1.shape, (self.n_component, self.n_samples))
        self.assertEqual(self.K.shape, (self.n_component, self.n_features))
        self.assertEqual(self.X_mean.shape, (self.n_features, 1))

    def test_X1_is_whitened(self):
        # XXX: need a definitive answer as to whether this atol can be improved
        # in the 32-bit precision case (or maybe its an acceptable atol).
        match self.X1.dtype:
            case torch.float64: atol = 1e-8
            case torch.float32: atol = 5e-6
        # X1 is white if centered and cov(X) == I
        self.assert_zero(torch.mean(self.X1, axis=1), atol=atol)
        X1_cov = torch.cov(self.X1, correction=0)
        X1_eye = torch.eye(self.n_component)
        self.assert_close(X1_cov, X1_eye, atol=atol)

    def test_adaptive_n_components(self):
        X = torch.rand(self.n_features, self.n_samples)
        Cx, _ = cov_mean(X)
        s = torch.linalg.eigvalsh(Cx).clamp_(1e-8).sqrt_()
        q = torch.tensor([0.2, 0.5, 0.8])
        for q, thresh in zip(q, torch.quantile(s, q)):
            n = torch.sum(s >= thresh).item()
            with self.subTest(q=q, thresh=thresh, n=n):
                self.run_whiten(torch.clone(X), n_component=None,
                                component_thresh=thresh, eps=1e-8)
                self.assertIn(len(self.X1), (n-1, n, n+1))
                self.assertIn(len(self.K), (n-1, n, n+1))

    def test_recover_all_components(self):
        X = torch.rand(self.n_features, self.n_samples)
        # If component_thresh is zero we should recover all components
        # If n_component == len(X) we should recover all components
        for (thresh, n) in ((0.0, None), (None, len(X))):
            with self.subTest(thresh=0, n=self.n_features):
                self.run_whiten(torch.clone(X), n_component=n,
                                component_thresh=thresh, eps=1e-8)
                self.assertEqual(len(self.X1), self.n_features)
                self.assertEqual(len(self.K), self.n_features)

    def test_recover_no_components(self):
        X = torch.rand(self.n_features, self.n_samples)
        with self.assertRaises(ValueError):
            learn_whitening(X, n_component=None, component_thresh=None)
        with self.assertRaises(RuntimeError):
            learn_whitening(X, n_component=None, component_thresh=1e+6)
        with self.assertRaises(ValueError):
            learn_whitening(X, n_component=0, component_thresh=None)

    # This should work, in theory, but... doesn't. Boo.
    @unittest.skip('Fix me!')
    def test_projecting_X1_through_K_inv_recovers_X(self):
        X_recov = (self.K.pinverse() @ self.X1) + self.X_mean
        self.assertTrue(torch.allclose(self.X_orig, X_recov, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
