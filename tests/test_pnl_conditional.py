"""pnl_conditional.py: the six-state classification and the permutation null.

The state definitions are load-bearing -- every conditional number in the report is a mean taken
over one of these buckets, so a mis-assigned boundary case silently moves the answer without
changing anything visible. The hypothesis under test assigns each state TWO competing outcomes
(continuation vs reversal), so the buckets must match the spec exactly.
"""
import importlib.util
import sys
import types

import numpy as np
import pytest


def _load():
    sys.modules.setdefault('read_mt1_dataset', types.ModuleType('read_mt1_dataset'))
    spec = importlib.util.spec_from_file_location('pnl_conditional', 'pnl_conditional.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _load()

UP_MORE, UP_LESS, UP_DOWN, DN_MORE, DN_LESS, DN_UP = 0, 1, 2, 3, 4, 5


def state_of(yesterday, today):
    """Classify one (yesterday, today) pair; a third value is needed only for alignment."""
    s, _ = C.method2_states(np.array([yesterday, today, 0.0]))
    return int(s[0])


class TestStateDefinitions:
    @pytest.mark.parametrize('y,t,want', [
        (100., 200., UP_MORE),    # profit, then MORE profit
        (200., 100., UP_LESS),    # profit, then LESS profit
        (100., -50., UP_DOWN),    # profit, then loss
        (-100., -200., DN_MORE),  # loss, then BIGGER loss
        (-200., -100., DN_LESS),  # loss, then SMALLER loss
        (-100., 50., DN_UP),      # loss, then profit
    ])
    def test_each_branch_of_the_spec(self, y, t, want):
        assert state_of(y, t) == want

    def test_every_pair_lands_in_exactly_one_state(self):
        """No gaps and no overlaps: an unclassified pair would be dropped from every bucket and
        quietly shrink the sample it belonged to."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 500, 4000)
        t = rng.normal(0, 500, 4000)
        for a, b in zip(y, t, strict=True):
            if a == 0 or b == 0:
                continue
            assert 0 <= state_of(a, b) <= 5

    def test_equal_magnitudes_are_assigned_consistently(self):
        """t == y is a boundary. It must land somewhere deterministic, not flip between runs."""
        assert state_of(100., 100.) == state_of(100., 100.)
        assert state_of(-100., -100.) == state_of(-100., -100.)
        assert state_of(100., 100.) in (UP_MORE, UP_LESS)
        assert state_of(-100., -100.) in (DN_MORE, DN_LESS)

    def test_sign_dominates_magnitude(self):
        """A tiny profit after a huge profit is still 'profit then profit', never a loss state."""
        assert state_of(10000., 0.01) == UP_LESS
        assert state_of(-10000., -0.01) == DN_LESS


class TestMethod1States:
    def test_split_is_on_the_sign_of_today(self):
        s, nxt = C.method1_states(np.array([5.0, -5.0, 3.0]))
        assert list(s) == [1, 0]
        assert list(nxt) == [-5.0, 3.0]


class TestPermutationNull:
    def test_a_statistic_with_no_time_structure_gives_z_near_zero(self):
        """iid data: the conditional mean must not differ from its own permutation null."""
        rng = np.random.default_rng(1)
        pnl = rng.normal(0, 700, (900, 12))

        def stat(p):
            st, nxt = C.statefn_panel(p, C.method2_states)
            m = st == UP_LESS
            return nxt[m].mean() - nxt.mean() if m.sum() > 30 else np.nan

        _, _, z, pv = C.perm_test(stat, pnl, 150, 0)
        assert abs(z) < 2.5, f'iid data gave z {z:+.2f}'
        assert pv > 0.05, f'iid data gave p {pv:.3f}'

    def test_a_planted_conditional_effect_is_detected(self):
        """Power check. If the test cannot find an effect it was handed, its nulls say nothing."""
        rng = np.random.default_rng(2)
        T, N = 900, 12
        pnl = rng.normal(0, 300, (T, N))
        for i in range(N):                       # after 'up then LESS up', force tomorrow down
            for t in range(2, T - 1):
                if pnl[t - 1, i] > 0 and 0 < pnl[t, i] < pnl[t - 1, i]:
                    pnl[t + 1, i] -= 400.0

        def stat(p):
            st, nxt = C.statefn_panel(p, C.method2_states)
            m = st == UP_LESS
            return nxt[m].mean() - nxt.mean() if m.sum() > 30 else np.nan

        o, _, z, pv = C.perm_test(stat, pnl, 150, 0)
        assert o < -50, f'planted effect not recovered: {o:+.1f}'
        assert z < -3 and pv < 0.05, f'planted effect not significant: z {z:+.2f} p {pv:.3f}'

    def test_the_null_preserves_each_industry_marginal(self):
        """The permutation must destroy time order ONLY. If it also changed the cross-industry
        composition, it would stop being the right null for a pooled statistic."""
        rng = np.random.default_rng(3)
        pnl = rng.normal(0, 700, (400, 12)) + np.arange(12) * 50.0
        sh = C.shuffled(pnl, 5)
        assert np.allclose(np.sort(pnl, axis=0), np.sort(sh, axis=0))
