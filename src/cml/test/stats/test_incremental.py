import unittest
from functools import partial
from random import randint

import bottleneck as bn
import numpy as np
import numpy.random as npr
import numpy.testing as npt

from cml.stats.incremental import (
    IncrStats, collect, merge, extend_obs, concat_obs
)


class TestIncrementalStats(unittest.TestCase):
    def test_collect(self):
        X = npr.rand(10, 1000)
        labels = np.arange(10)
        nstats = collect(X, labels)
        xstats = collect(X, labels, byrow=True, nansafe=False, nansqueeze=False)
        tstats = collect(X.T, labels, byrow=False, nansafe=False, nansqueeze=False)
        # The nan kws should not have a measurable effect if there's no nans,
        # and running the stats over the transpose shouldn't either.
        for st in (nstats, xstats, tstats):
            self.assertIsInstance(st, IncrStats)
            self.assertEqual(st, xstats)
        # Check that the collect script actually produces the right stats.
        # You'd be amazed how many bugs these stupid tests catch.
        npt.assert_equal(xstats.count, 1000)
        npt.assert_equal(xstats.mean, np.mean(X, axis=1))
        npt.assert_equal(xstats.variance, np.var(X, axis=1))
        npt.assert_equal(xstats.negative, np.sum(X < 0, axis=1))
        npt.assert_equal(xstats.minval, np.min(X, axis=1))
        npt.assert_equal(xstats.maxval, np.max(X, axis=1))

    def test_collect_nansafe(self):
        # Make X such that 25% are nan but no row is *all* nan
        X = npr.rand(10, 1000)
        X[npr.rand(10, 1000) < 0.25] = np.nan
        X[:, 0] = 1.0
        labels = np.arange(10)
        xstats = collect(X, labels, nansafe=True)
        self.assertIsInstance(xstats, IncrStats)
        self.assertFalse(bn.anynan(xstats.asarrays[1]))
        npt.assert_equal(xstats.count, np.sum(np.isfinite(X), axis=1))
        npt.assert_equal(xstats.mean, bn.nanmean(X, axis=1))
        npt.assert_equal(xstats.variance, bn.nanvar(X, axis=1))
        npt.assert_equal(xstats.negative, bn.nansum(X < 0, axis=1))
        npt.assert_equal(xstats.minval, bn.nanmin(X, axis=1))
        npt.assert_equal(xstats.maxval, bn.nanmax(X, axis=1))

    def test_collect_squeezenan(self):
        # Make X such that 25% are nan and two rows are *all* nan
        # First make sure only the two rows we select are all nan
        X = npr.rand(10, 1000)
        X[npr.rand(10, 1000) < 0.25] = np.nan
        X[:, 0] = 1.0
        # Now make the labelset and choose two at random to smash
        labels = np.arange(10)
        allnan = np.zeros(len(labels)).astype(bool)
        allnan[npr.choice(10, size=2, replace=False)] = True
        X[allnan] = np.nan
        # S for "allnan rows squeezed out"
        S = np.copy(X[~allnan], order='C')
        # Check class, labels, and that no nans crept through (very important!)
        xstats = collect(X, labels, nansafe=True)
        self.assertIsInstance(xstats, IncrStats)
        npt.assert_equal(xstats.labels, labels[~allnan])
        self.assertFalse(bn.anynan(xstats.asarrays[1]))
        # Notice these tests are against S, not X!
        npt.assert_equal(xstats.count, np.sum(np.isfinite(S), axis=1))
        npt.assert_equal(xstats.mean, bn.nanmean(S, axis=1))
        npt.assert_equal(xstats.variance, bn.nanvar(S, axis=1))
        npt.assert_equal(xstats.negative, bn.nansum(S < 0, axis=1))
        npt.assert_equal(xstats.minval, bn.nanmin(S, axis=1))
        npt.assert_equal(xstats.maxval, bn.nanmax(S, axis=1))

    def test_merge(self):
        X = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        Y = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        f = partial(collect, labels=np.arange(10))
        self.assertEqual(merge(f(X), f(Y)), f(np.hstack((X, Y))))

    def test_merge_disjoint(self):
        X = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        Y = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        xlabels = np.arange(10)
        ylabels = np.arange(10, 20)
        zlabels = np.arange(20)
        # Try a few permutations to make sure the outputs are always sorted and
        # the merging algorithm doesn't choke when the inputs are unsorted.
        xperms = [np.arange(10), *(npr.permutation(10) for _ in range(4))]
        yperms = [np.arange(10), *(npr.permutation(10) for _ in range(4))]
        for Px, Py in zip(xperms, yperms):
            xst = collect(X[Px], xlabels[Px])
            yst = collect(Y[Py], ylabels[Py])
            zst = merge(xst, yst)
            # This might look a little weird but it is correct. Merging
            # IncrStats objects sorts the labels as a (nearly) free byproduct,
            # but the basic IncrStats objects returned by `collect` are in the
            # input order of the labels, not sorted. On the last two lines,
            # permuting the sorted output `zst` is equivalent to unpermuting
            # the inputs (just feels sort of backwards).
            npt.assert_equal(xst.labels, xlabels[Px])
            npt.assert_equal(yst.labels, ylabels[Py])
            npt.assert_equal(zst.labels, zlabels)
            npt.assert_equal(xst.asarrays[1], zst.asarrays[1][:, :10][:, Px])
            npt.assert_equal(yst.asarrays[1], zst.asarrays[1][:, 10:][:, Py])

    def test_merge_overlapping(self):
        X = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        Y = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        xlabels = np.arange(10)
        ylabels = np.arange(5, 15)
        zlabels = np.arange(15)
        # The labels tossed in here are not used; the second half of X and
        # first half of Y are shared
        zcommon = collect(np.hstack((X[5:], Y[:5])), np.arange(5))
        zst_com = zcommon.asarrays[1]
        xperms = [np.arange(10), *(npr.permutation(10) for _ in range(4))]
        yperms = [np.arange(10), *(npr.permutation(10) for _ in range(4))]
        for Px, Py in zip(xperms, yperms):
            xst = collect(X[Px], xlabels[Px])
            yst = collect(Y[Py], ylabels[Py])
            zst = merge(xst, yst)
            # Check all the labeling
            npt.assert_equal(xst.labels, xlabels[Px])
            npt.assert_equal(yst.labels, ylabels[Py])
            npt.assert_equal(zst.labels, zlabels)
            # Check the x-only labels
            xst_arr = xst.asarrays[1]
            yst_arr = yst.asarrays[1]
            zst_arr = zst.asarrays[1]
            # inverse permutations
            Px_inv = np.argsort(Px)
            Py_inv = np.argsort(Py)
            # Now check the x-only, y-only, and commonly labeled.
            npt.assert_allclose(xst_arr[:, Px_inv][:, :5], zst_arr[:, :5])
            npt.assert_allclose(yst_arr[:, Py_inv][:, 5:], zst_arr[:, 10:])
            npt.assert_allclose(zst_com, zst_arr[:, 5:10])

    def test_extend_obs(self):
        X = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        X[npr.rand(10, 1000) < 0.25] = np.nan
        X[:, 0] = 1.0
        xst = collect(X, np.arange(10))
        for fill in (0.0, -1, 1e-6, np.nanmean(X), 1/(20*365.25)):
            Y = np.copy(X)
            Y[np.isnan(Y)] = fill
            yst = collect(Y, np.arange(10))
            zst = extend_obs(xst, 1000, fill=fill)
            self.assertEqual(yst, zst)

    def test_concat_obs(self):
        X = (npr.rand(10, 1000) * randint(-10, 10)) + randint(-10, 10)
        xst = collect(X, np.arange(10))

        # Test first a full reduction of the stats (all labels match)
        zst = concat_obs(xst, labels=None)
        zst_arr = zst.asarrays[1]
        self.assertTrue(np.all(zst_arr[:, 0, None] == zst_arr))
        npt.assert_allclose(zst.mean[0], np.mean(X))
        npt.assert_allclose(zst.variance[0], np.var(X))
        npt.assert_equal(zst.negative[0], np.sum(X < 0))
        npt.assert_equal(zst.minval[0], np.min(X))
        npt.assert_equal(zst.maxval[0], np.max(X))

        # And a total non-reduction of the stats (no labels match)
        zst = concat_obs(xst, labels=[100, 101, 102])
        self.assertEqual(zst, xst)

        # Test a few random proper subsets of the labels
        for labelset in (
                npr.choice(10, size=randint(1, 9), replace=False)
                for _ in range(10)
        ):
            zst = concat_obs(xst, labels=labelset)
            arr = zst.asarrays[1][:, labelset]
            X_subset = X[labelset]
            # Check that non-selected labels are not changed
            nonsubset = np.array([i for i in xst.labels if i not in labelset])
            npt.assert_equal(zst.asarrays[1][:, nonsubset],
                             xst.asarrays[1][:, nonsubset])
            # Check that the selected labels are all identical
            self.assertTrue(np.all(arr[:, 0, None] == arr))
            # Check that the selected label stats are correct
            npt.assert_allclose(arr[1], np.mean(X_subset))
            npt.assert_allclose(arr[2], np.var(X_subset))
            npt.assert_equal(arr[3], np.sum(X_subset < 0))
            npt.assert_equal(arr[4], np.min(X_subset))
            npt.assert_equal(arr[5], np.max(X_subset))


if __name__ == '__main__':
    unittest.main()
