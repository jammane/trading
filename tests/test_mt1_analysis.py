"""mt1_analysis — the harness that decides A, B and C.

Its answers get acted on, so it has to be shown to have two properties that are easy to assume
and easy to get wrong:

  POWER   — a planted signal must be recovered. A harness that cannot find a signal it was handed
            reports nulls that mean nothing.
  NO LEAK — a shuffled target must give IC ~ 0. Earlier in this project an off-by-one in a forward
            window made every horizon look predictable, and nothing in the output showed it.

Plus the error bars. Treating T x 12 overlapping rows as independent produced a t of -7.20 on a
signal indistinguishable from zero. The block bootstrap exists to stop that, so it is tested
against a series with a KNOWN dependence length rather than assumed to work.
"""
import numpy as np
import pytest

import mt1_analysis as M
import read_mt1_dataset as D


def synth(T=900, N=12, F=74, seed=0, signal=0.0, sig_feat=0):
    """A dataset dict shaped like read_mt1_dataset.parse output.

    `signal` plants a known dependence of the NEXT day's book P&L on today's feature `sig_feat`,
    in units of the P&L's own sd.
    """
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(T, N, F)).astype(np.float32)
    mkt = rng.normal(0, 300, (T, N))
    trade = rng.normal(0, 80, (T, N))
    if signal:
        # today's feature moves TOMORROW's P&L: shift the driver forward by one day
        drive = np.zeros((T, N))
        drive[1:] = feat[:-1, :, sig_feat]
        mkt = mkt + signal * 300.0 * drive
    book_prev = np.full((T, N), 25000.0)
    book_now = book_prev + mkt + trade
    return {'pass': np.zeros(T, np.int64), 'day': np.arange(T, dtype=np.int64),
            'feat': feat,
            'book_prev': book_prev.astype(np.float32), 'book_now': book_now.astype(np.float32),
            'mkt_move': mkt.astype(np.float32), 'trade_delta': trade.astype(np.float32)}


class TestNoLookAhead:
    """A fit for test day t may only use rows whose forward window CLOSED before t."""

    @pytest.mark.parametrize('h', [1, 3, 5, 10])
    def test_fit_rows_exclude_the_unclosed_window(self, h, monkeypatch):
        seen = []
        real = M.ridge_fit

        def spy(X, y, lam):
            seen.append(len(y))
            return real(X, y, lam)

        monkeypatch.setattr(M, 'ridge_fit', spy)
        T = 600
        X = np.random.default_rng(0).normal(size=(T, 8))
        y = np.arange(T, dtype=float)
        pred, act, rows = M.walk_forward(X, y, h, min_train=200, refit_every=50)
        assert rows.size > 0
        # the largest fit must still stop h rows short of the last test day
        assert max(seen) <= rows.max() - h, \
            f'a fit used rows up to {max(seen)} for a test day {rows.max()} at h={h}'

    def test_a_pure_future_feature_is_not_predictable(self):
        """Feature = tomorrow's target. Honest walk-forward can still fit it (the RELATION is
        learnable from closed windows), so this pins the weaker but real property: no row is
        predicted before enough closed history exists."""
        T = 500
        y = np.random.default_rng(1).normal(size=T)
        X = np.zeros((T, 3))
        X[:-1, 0] = y[1:]
        pred, act, rows = M.walk_forward(X, y, 1, min_train=200, refit_every=20)
        assert rows.min() >= 200

    def test_min_train_is_respected(self):
        T, mt = 400, 300
        X = np.random.default_rng(2).normal(size=(T, 5))
        y = np.random.default_rng(3).normal(size=T)
        _, _, rows = M.walk_forward(X, y, 5, min_train=mt, refit_every=10)
        assert rows.min() >= mt


class TestPower:
    """If a planted signal is not recovered, every null this harness reports is uninformative."""

    def test_a_planted_signal_is_found(self):
        ds = synth(signal=0.35, seed=5)
        P, A = M.run_formulation(ds, 1, 'per-industry', min_train=250, refit_every=25)
        r = M.evaluate(P, A, 1, n_boot=200)
        assert r is not None
        assert r['ic'] > 0.05, f"planted signal not recovered: IC {r['ic']:.4f}"
        assert r['ic_t'] > 2, f"planted signal found but not significant: t {r['ic_t']:.2f}"

    def test_a_stronger_signal_gives_a_stronger_reading(self):
        weak = M.evaluate(*M.run_formulation(synth(signal=0.15, seed=6), 1, 'per-industry',
                                             min_train=250, refit_every=25), 1, n_boot=150)
        strong = M.evaluate(*M.run_formulation(synth(signal=0.50, seed=6), 1, 'per-industry',
                                               min_train=250, refit_every=25), 1, n_boot=150)
        assert strong['ic'] > weak['ic']

    def test_the_pooled_formulation_also_finds_it(self):
        """Pooling shares one model across 12 industries. If the plumbing misaligns industries the
        signal vanishes, and it would look like a real negative result about pooling."""
        ds = synth(signal=0.35, seed=7)
        r = M.evaluate(*M.run_formulation(ds, 1, 'pooled', min_train=250, refit_every=25),
                       1, n_boot=150)
        assert r['ic'] > 0.05, f"pooled lost the planted signal: IC {r['ic']:.4f}"


class TestNoLeak:
    def test_shuffled_targets_give_no_signal(self):
        ds = synth(signal=0.0, seed=8)
        r = M.shuffle_control(ds, 1, 'per-industry', min_train=250, refit_every=25)
        assert abs(r['ic']) < 0.05, f'shuffled control found signal: IC {r["ic"]:.4f} — leak'

    def test_shuffling_destroys_a_planted_signal(self):
        """The sharpest version: plant a signal, then shuffle. Recovering it anyway would mean the
        pipeline is reading the target through some path other than the features."""
        ds = synth(signal=0.50, seed=9)
        found = M.evaluate(*M.run_formulation(ds, 1, 'per-industry', min_train=250,
                                              refit_every=25), 1, n_boot=150)
        killed = M.shuffle_control(ds, 1, 'per-industry', min_train=250, refit_every=25)
        assert found['ic'] > 0.05
        assert abs(killed['ic']) < found['ic'] / 3

    def test_pure_noise_gives_no_signal_at_any_horizon(self):
        ds = synth(signal=0.0, seed=10)
        for h in (1, 5):
            r = M.evaluate(*M.run_formulation(ds, h, 'per-industry', min_train=250,
                                              refit_every=25), h, n_boot=150)
            assert abs(r['ic']) < 0.06, f'h={h}: IC {r["ic"]:.4f} on pure noise'


class TestBlockBootstrap:
    """The correction that stops overlapping windows manufacturing significance."""

    def test_it_recovers_the_iid_se_when_there_is_no_dependence(self):
        rng = np.random.default_rng(11)
        n = 2000
        x = rng.normal(size=n)

        def stat(idx):
            return float(x[idx].mean())

        se, _ = M.block_bootstrap(stat, n, block=1, n_boot=600)
        assert abs(se - 1 / np.sqrt(n)) < 0.25 / np.sqrt(n)

    def test_a_longer_block_gives_a_LARGER_se_on_dependent_data(self):
        """The whole point. On a positively autocorrelated series the independent SE is too small;
        blocks long enough to span the dependence recover the larger, honest one."""
        rng = np.random.default_rng(12)
        n = 3000
        e = rng.normal(size=n)
        x = np.convolve(e, np.ones(20) / 20, mode='same')     # dependence length ~20

        def stat(idx):
            return float(x[idx].mean())

        se_short, _ = M.block_bootstrap(stat, n, block=1, n_boot=400)
        se_long, _ = M.block_bootstrap(stat, n, block=60, n_boot=400)
        assert se_long > 2 * se_short, f'block SE {se_long:.5f} vs iid {se_short:.5f}'

    def test_the_evaluate_block_scales_with_the_horizon(self):
        ds = synth(T=700, seed=13)
        for h in (1, 5, 20):
            r = M.evaluate(*M.run_formulation(ds, h, 'per-industry', min_train=250,
                                              refit_every=50), h, n_boot=60)
            assert r['block'] >= 2 * h, 'block must span the overlap it is correcting for'

    def test_it_returns_a_ci_that_brackets_the_estimate(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=1500)

        def stat(idx):
            return float(x[idx].mean())

        se, (lo, hi) = M.block_bootstrap(stat, 1500, block=25, n_boot=400)
        assert lo < x.mean() < hi and se > 0


class TestTargetHandling:
    def test_standardizing_keeps_the_ranking_within_an_industry(self):
        ds = synth(T=600, seed=15)
        raw = M._target(ds, 5, 'book', standardize=False)
        std = M._target(ds, 5, 'book', standardize=True)
        i, t = 3, 400
        m = np.isfinite(raw[:, i]) & np.isfinite(std[:, i])
        assert M.pearson(raw[m, i], std[m, i]) > 0.95
        del t

    def test_components_can_be_targeted_separately(self):
        ds = synth(T=600, seed=16)
        b = M._target(ds, 3, 'book', standardize=False)
        m = M._target(ds, 3, 'mkt', standardize=False)
        tr = M._target(ds, 3, 'trade', standardize=False)
        ok = np.isfinite(b) & np.isfinite(m) & np.isfinite(tr)
        assert np.allclose(b[ok], m[ok] + tr[ok], atol=1e-1)

    def test_unclosed_windows_never_reach_the_fit(self):
        ds = synth(T=400, seed=17)
        y = M._target(ds, 10, 'book', standardize=False)
        assert np.all(np.isnan(y[-10:]))
        P, A = M.run_formulation(ds, 10, 'per-industry', min_train=200, refit_every=50)
        assert np.all(np.isnan(A[-10:])), 'an unclosed window was scored'
