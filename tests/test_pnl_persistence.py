"""pnl_persistence.py: every statistic is calibrated against a process with a KNOWN answer.

The variance-ratio estimator shipped its first draft with the wrong Lo-MacKinlay scaling --
(1 - 1/q) instead of (1 - q/T) -- which inflates VR by 2.0x at q=2. On real data that read
VR(2) = 1.97 with z = +89.9, i.e. overwhelming evidence of trending P&L. It was caught only
because the shuffled control returned 1.966 on the same data with time order destroyed, where
the true value is 1.0 by construction.

So each measure here is fed a series whose answer is known in advance. A statistic that cannot
be shown to read a planted effect correctly cannot be trusted to report a null.
"""
import importlib.util

import numpy as np
import pytest

# NOTE: do NOT stub read_mt1_dataset into sys.modules here. An earlier version of this file did
# `sys.modules.setdefault('read_mt1_dataset', types.ModuleType(...))` to load the script under
# test, which installs an EMPTY module that every later test importing the real one then receives.
# test_mt1_dataset.py passed alone and failed in the full suite, and the poisoning only surfaced
# when a file sorting alphabetically earlier acquired the same stub. The real module is importable
# from the repo root, so the stub bought nothing.


def _load():
    spec = importlib.util.spec_from_file_location('pnl_persistence', 'pnl_persistence.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load()


def ar1(phi, n, rng, sd=1.0):
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class TestVarianceRatio:
    """VR = Var(q-sum) / (q * Var(1)). Exactly 1 under a random walk, whatever q."""

    @pytest.mark.parametrize('q', [2, 3, 5, 10, 20])
    def test_iid_gives_one(self, q):
        rng = np.random.default_rng(q)
        v = [P.variance_ratio(rng.normal(0, 739, 1227), q)[0] for _ in range(40)]
        assert abs(np.mean(v) - 1.0) < 0.06, f'q={q}: iid VR {np.mean(v):.3f}, must be ~1.0'

    @pytest.mark.parametrize('q', [2, 5, 10])
    def test_positive_autocorrelation_reads_above_one(self, q):
        rng = np.random.default_rng(7)
        v = [P.variance_ratio(ar1(+0.30, 1227, rng), q)[0] for _ in range(30)]
        assert np.mean(v) > 1.15, f'q={q}: trending series read VR {np.mean(v):.3f}'

    @pytest.mark.parametrize('q', [2, 5, 10])
    def test_negative_autocorrelation_reads_below_one(self, q):
        """phi = -0.13 is the P&L's own measured lag-1, attributed to the fill model."""
        rng = np.random.default_rng(8)
        v = [P.variance_ratio(ar1(-0.13, 1227, rng), q)[0] for _ in range(30)]
        assert np.mean(v) < 0.95, f'q={q}: mean-reverting series read VR {np.mean(v):.3f}'

    def test_stronger_dependence_reads_stronger(self):
        rng = np.random.default_rng(9)
        weak = np.mean([P.variance_ratio(ar1(0.15, 1227, rng), 10)[0] for _ in range(25)])
        strong = np.mean([P.variance_ratio(ar1(0.45, 1227, rng), 10)[0] for _ in range(25)])
        assert strong > weak > 1.0

    def test_scale_invariance(self):
        """VR is a ratio of variances, so multiplying the series must not move it."""
        rng = np.random.default_rng(10)
        x = ar1(0.25, 1227, rng)
        a, _ = P.variance_ratio(x, 5)
        b, _ = P.variance_ratio(x * 1000.0, 5)
        assert abs(a - b) < 1e-6


class TestSignRuns:
    def test_known_sequence(self):
        r = P.sign_runs(np.array([1., 1., 1., -1., -1., 1.]))
        assert list(r) == [3, 2, 1]

    def test_zeros_are_dropped_not_treated_as_a_sign(self):
        """Reset days are exactly $0. Counting them as a sign would split real runs."""
        assert list(P.sign_runs(np.array([1., 0., 1.]))) == [2]

    def test_alternating_gives_all_ones(self):
        assert P.sign_runs(np.array([1., -1., 1., -1.])).mean() == 1.0

    def test_iid_mean_run_is_about_two(self):
        """A fair coin gives mean run 2.0; this is the bar real data must beat to show momentum."""
        rng = np.random.default_rng(3)
        m = np.mean([P.sign_runs(rng.normal(0, 1, 4000)).mean() for _ in range(20)])
        assert abs(m - 2.0) < 0.1

    def test_momentum_lengthens_runs(self):
        rng = np.random.default_rng(4)
        m = np.mean([P.sign_runs(ar1(0.6, 4000, rng)).mean() for _ in range(10)])
        assert m > 2.4, f'a strongly trending series gave mean run {m:.2f}'


class TestRankLife:
    def test_a_persistent_ranking_is_recovered(self):
        """Industries with permanently different means must stay ranked far past the window."""
        rng = np.random.default_rng(5)
        T, N = 800, 12
        level = np.linspace(-300, 300, N)
        pnl = rng.normal(0, 200, (T, N)) + level
        out = P.rank_life(pnl, 20, [20, 40])
        assert out[20][0] > 0.7, f'persistent ranking lost: {out[20][0]:.2f}'
        assert out[40][1] > 0.8, 'top tercile should stay top when means truly differ'

    def test_iid_ranking_dies_once_windows_stop_overlapping(self):
        """The control that matters: overlapping trailing windows correlate MECHANICALLY, so
        apparent persistence at lag < k is not evidence of anything."""
        rng = np.random.default_rng(6)
        pnl = rng.normal(0, 700, (800, 12))
        out = P.rank_life(pnl, 20, [5, 20, 40])
        assert out[5][0] > 0.4, 'overlap should induce correlation at lag < k'
        assert abs(out[20][0]) < 0.12, f'non-overlapping iid rank corr {out[20][0]:.2f}, must be ~0'
        assert abs(out[40][0]) < 0.12


class TestSlopeLife:
    def test_a_regime_change_shortens_the_life(self):
        rng = np.random.default_rng(11)
        steady = rng.normal(50, 100, 600)
        flip = np.concatenate([rng.normal(50, 100, 300), rng.normal(-50, 100, 300)])
        assert np.median(P.slope_life(flip)) < np.median(P.slope_life(steady))

    def test_it_returns_finite_lives_on_pure_noise(self):
        """A finite life on iid data is expected -- the opening estimate is noisy. That is why
        the reported number is only meaningful next to the shuffled control."""
        rng = np.random.default_rng(12)
        lives = P.slope_life(rng.normal(0, 700, 600))
        assert lives.size > 100 and np.all(lives > 0)
