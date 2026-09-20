"""living_bn.py: the state machine, the decay, the shrinkage, and the data-driven half-life.

This model is meant to run inside daily training and update itself, so a wrong transition or a
decay that silently does nothing would drift the numbers with nothing to notice it. Every part is
checked against a case whose answer is known in advance.
"""
import numpy as np
import pytest

import living_bn as L


class TestClassify:
    @pytest.mark.parametrize('prev,cur,want', [
        (100., 200., L.UP_MORE),
        (200., 100., L.UP_LESS),
        (100., -50., L.UP_DOWN),
        (-100., -200., L.DN_MORE),
        (-200., -100., L.DN_LESS),
        (-100., 50., L.DN_UP),
    ])
    def test_every_branch(self, prev, cur, want):
        assert L.classify(prev, cur) == want

    def test_flat_days_are_excluded(self):
        """A reset day writes book_prev == book_now, i.e. exactly $0. It is not a real flat day
        and must not be classified as either direction."""
        assert L.classify(0.0, 100.0) is None
        assert L.classify(100.0, 0.0) is None

    def test_non_finite_is_excluded(self):
        assert L.classify(np.nan, 1.0) is None
        assert L.classify(1.0, np.inf) is None

    def test_groups_cover_all_states(self):
        assert set(L.GROUP) == set(range(L.N_STATES))
        assert set(L.GROUP.values()) == {'accel', 'decel', 'flip'}


class TestUpdating:
    def test_a_deterministic_continuation_drives_p_to_one(self):
        m = L.LivingBN(n_ind=1, half_life=1e9)
        seq = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]      # always UP_MORE, always continues
        for t in range(2, len(seq)):
            m.observe(np.array([seq[t - 2]]), np.array([seq[t - 1]]), np.array([seq[t]]))
        assert m.p_continue(0, L.UP_MORE) > 0.8

    def test_a_deterministic_reversal_drives_p_to_zero(self):
        m = L.LivingBN(n_ind=1, half_life=1e9)
        seq = []
        for _ in range(40):
            seq += [10.0, 5.0, -5.0]            # up, LESS up, then reverses
        for t in range(2, len(seq)):
            m.observe(np.array([seq[t - 2]]), np.array([seq[t - 1]]), np.array([seq[t]]))
        assert m.p_continue(0, L.UP_LESS) < 0.3

    def test_decay_forgets(self):
        """The whole point of 'living'. A short half-life must let new evidence overturn old."""
        rng = np.random.default_rng(0)
        old = np.abs(rng.normal(0, 1, (200, 1))) + 1.0          # all positive: always continues
        new = np.tile([[5.0], [3.0], [-3.0]], (60, 1))          # decelerating, always reverses
        series = np.vstack([old, new])
        fast = L.LivingBN(n_ind=1, half_life=25)
        slow = L.LivingBN(n_ind=1, half_life=1e9)
        for m in (fast, slow):
            for t in range(2, len(series)):
                m.observe(series[t - 2], series[t - 1], series[t])
        assert fast.p_continue(0, L.UP_LESS) < slow.p_continue(0, L.UP_LESS), \
            'the short half-life should have forgotten the old regime faster'

    def test_weights_actually_decay(self):
        m = L.LivingBN(n_ind=1, half_life=10)
        s = np.tile([[3.0], [2.0], [1.0]], (200, 1))
        for t in range(2, len(s)):
            m.observe(s[t - 2], s[t - 1], s[t])
        assert m.w_n.sum() < 40, 'effective sample size must saturate, not grow without bound'


class TestShrinkage:
    def test_homogeneous_industries_shrink_to_pooled(self):
        """The measured situation: tau^2 = 0, so per-industry estimates must collapse to pooled.
        Estimating 12 separate rates on ~187 observations each would be fitting noise."""
        rng = np.random.default_rng(1)
        m = L.LivingBN(n_ind=12, half_life=1e9)
        s = rng.normal(0, 100, (900, 12))
        for t in range(2, len(s)):
            m.observe(s[t - 2], s[t - 1], s[t])
        ps = [m.p_continue(i, L.UP_LESS) for i in range(12)]
        assert max(ps) - min(ps) < 0.01, 'identical industries must get ~identical estimates'

    @pytest.mark.parametrize('seed', [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    def test_identical_industries_stay_effectively_pooled(self, seed):
        """Identical industries must come out with the same estimate, across seeds.

        The criterion is the SPREAD of the resulting estimates, not `tau2 == 0`. tau^2 is a moment
        estimator on 12 groups and is inherently noisy; demanding an exact zero from it would mean
        ratcheting the penalty until genuine heterogeneity could no longer be detected either --
        trading a harmless error for a harmful one. What matters is that leftover noise cannot
        move the answer: a residual tau^2 of 3e-5 gives each industry ~2% weight on its own data
        at n = 187, which is pooling.
        """
        rng = np.random.default_rng(seed)
        m = L.LivingBN(n_ind=12, half_life=1e9)
        s = rng.normal(0, 100, (900, 12))
        for t in range(2, len(s)):
            m.observe(s[t - 2], s[t - 1], s[t])
        for st in range(L.N_STATES):
            ps = [m.p_continue(i, st) for i in range(12)]
            assert max(ps) - min(ps) < 0.01, (
                f'state {st}: identical industries spread by {max(ps)-min(ps):.4f} '
                f'(tau^2 {m.tau2(st):.2e}) -- un-pooling on noise')

    def test_genuinely_different_industries_separate(self):
        """The shrinkage must RELAX when industries really do differ, or the model can never
        learn that they do."""
        rng = np.random.default_rng(2)
        T = 3000
        s = np.zeros((T, 4))
        for t in range(T):
            s[t, 0] = 10.0 if t % 2 == 0 else 5.0       # industry 0: rigid up_less, continues
            s[t, 1] = 10.0 if t % 3 else -8.0
            s[t, 2] = rng.normal(0, 100)
            s[t, 3] = rng.normal(0, 100)
        m = L.LivingBN(n_ind=4, half_life=1e9)
        for t in range(2, T):
            m.observe(s[t - 2], s[t - 1], s[t])
        assert m.tau2(L.UP_LESS) > 0, 'heterogeneous industries must produce tau^2 > 0'
        ps = [m.p_continue(i, L.UP_LESS) for i in range(4)]
        assert max(ps) - min(ps) > 0.05, 'estimates must separate once tau^2 > 0'


class TestStreakN:
    def test_n_follows_the_probability(self):
        m = L.LivingBN(n_ind=1)
        m.w_n[0, L.UP_MORE] = 1000
        m.w_same[0, L.UP_MORE] = 750                 # p = 0.75 -> N = 4
        assert 3.5 < m.streak_n(0, L.UP_MORE) < 4.5

    def test_a_coin_flip_gives_n_of_two(self):
        m = L.LivingBN(n_ind=1)
        m.w_n[0, L.UP_MORE] = 1000
        m.w_same[0, L.UP_MORE] = 500
        assert 1.8 < m.streak_n(0, L.UP_MORE) < 2.2


class TestHalfLifeSelection:
    def test_a_stationary_series_prefers_a_long_memory(self):
        rng = np.random.default_rng(3)
        pnl = rng.normal(0, 700, (900, 6))
        best, _ = L.fit_half_life(pnl, grid=(30, 250, 1e9), warmup=200)
        assert best >= 250, f'stationary data chose half-life {best}'

    def test_a_regime_switch_prefers_a_short_memory(self):
        rng = np.random.default_rng(4)
        a = np.abs(rng.normal(0, 1, (500, 6))) + 1.0        # always continues
        b = np.tile([[6.0], [4.0], [-4.0]], (200, 1)) * np.ones((1, 6))
        pnl = np.vstack([a, b])
        best, _ = L.fit_half_life(pnl, grid=(30, 250, 1e9), warmup=200)
        assert best <= 250, f'regime-switching data chose half-life {best}'


class TestPersistence:
    def test_round_trip_preserves_every_posterior(self, tmp_path):
        rng = np.random.default_rng(5)
        m = L.LivingBN(n_ind=12, half_life=180)
        s = rng.normal(0, 500, (400, 12))
        for t in range(2, len(s)):
            m.observe(s[t - 2], s[t - 1], s[t])
        p = tmp_path / 'bn.json'
        m.save(p)
        back = L.LivingBN.load(p)
        assert back.half_life == m.half_life and back.days_seen == m.days_seen
        for i in range(12):
            for st in range(L.N_STATES):
                assert abs(back.p_continue(i, st) - m.p_continue(i, st)) < 1e-12


class TestDailyUpkeep:
    """upkeep_living_bn: the daily re-evaluation hook production_v2 calls.

    The point of 'living' is that these numbers move with recent StockNN performance, so the
    state has to survive between runs. A silently-reset file would look identical from outside --
    the model would just permanently report its prior.
    """

    def test_state_accumulates_across_runs(self, tmp_path):
        import upkeep
        inds = [f'i{k}' for k in range(4)]
        rng = np.random.default_rng(0)
        first = None
        for _ in range(60):
            row = {ind: float(rng.normal(0, 100)) for ind in inds}
            rep = upkeep.upkeep_living_bn(str(tmp_path), row, inds)
            if first is None:
                first = sum(r['n_eff'] for r in rep)
        last = sum(r['n_eff'] for r in rep)
        assert last > first, 'effective sample size did not grow across runs'
        assert (tmp_path / upkeep.LIVING_BN_FILE).exists()

    def test_a_missing_industry_is_nan_not_zero(self, tmp_path):
        """0.0 is what a hard-floor reset writes, and classify() treats it as 'no state'. Passing
        0.0 for 'unknown' would silently merge the two."""
        import upkeep
        inds = ['a', 'b']
        for _ in range(5):
            upkeep.upkeep_living_bn(str(tmp_path), {'a': 10.0, 'b': None}, inds)
        import json as _j
        blob = _j.loads((tmp_path / upkeep.LIVING_BN_FILE).read_text())
        assert all(np.isnan(r[1]) for r in blob['history']), 'missing industry not stored as NaN'

    def test_industry_count_change_resets_rather_than_misaligns(self, tmp_path):
        """A symbol swap can change the industry list. Carrying counts across would silently
        attribute one industry's history to another."""
        import upkeep
        for _ in range(5):
            upkeep.upkeep_living_bn(str(tmp_path), {'a': 5.0, 'b': -5.0}, ['a', 'b'])
        rep = upkeep.upkeep_living_bn(str(tmp_path), {'a': 5.0, 'b': -5.0, 'c': 1.0},
                                      ['a', 'b', 'c'])
        assert sum(r['n_eff'] for r in rep) == 0.0, 'stale counts survived an industry change'

    def test_a_corrupt_state_file_does_not_break_the_run(self, tmp_path):
        import upkeep
        (tmp_path / upkeep.LIVING_BN_FILE).write_text('{not json')
        rep = upkeep.upkeep_living_bn(str(tmp_path), {'a': 1.0}, ['a'])
        assert isinstance(rep, list) and len(rep) == L.N_STATES
