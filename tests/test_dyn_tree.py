"""dyn_tree.py: the adaptive split must fire when structure is there, and not when it isn't.

Both halves are load-bearing. A tree that never splits would report "no deeper structure" on any
input, which is exactly the conclusion drawn from it -- so the power check (it DOES split on
planted structure) is what makes the null meaningful.
"""
import importlib.util

import numpy as np
import pytest


def _load():
    spec = importlib.util.spec_from_file_location('dyn_tree', 'dyn_tree.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DT = _load()


class TestSymbols:
    @pytest.mark.parametrize('prev,cur,want', [
        (10., 20., DT.ACCEL), (20., 10., DT.DECEL), (10., -5., DT.FLIP), (-5., 10., DT.FLIP),
        (-10., -20., DT.ACCEL), (-20., -10., DT.DECEL),
    ])
    def test_relation(self, prev, cur, want):
        assert DT.relation(prev, cur) == want

    def test_level_one_is_the_sign_of_today(self):
        col = np.array([1.0, 2.0, 3.0])
        assert DT.context(col, 2, 3)[0] == 0
        assert DT.context(np.array([1.0, -2.0, 3.0]), 2, 3)[0] == 1

    def test_a_flat_day_truncates_the_path(self):
        """A hard-floor reset writes exactly $0; it cannot be a context symbol."""
        col = np.array([5.0, 0.0, 3.0, 4.0])
        p = DT.context(col, 3, 4)
        assert p is not None and len(p) == 1        # stops at the zero
        assert DT.context(np.array([0.0, 4.0]), 1, 3) is None


class TestSplitCriterion:
    def _fill(self, node, n, k):
        node.n, node.k = float(n), float(k)

    def test_identical_children_do_not_justify_a_split(self):
        t = DT.DynTree(min_obs=40)
        for s in (0, 1):
            t.root.kids[s] = DT.Node()
            self._fill(t.root.kids[s], 500, 250)
        assert not t._split_justified(t.root)

    def test_clearly_different_children_do_justify_a_split(self):
        t = DT.DynTree(min_obs=40)
        t.root.kids[0] = DT.Node(); self._fill(t.root.kids[0], 500, 350)   # 70%
        t.root.kids[1] = DT.Node(); self._fill(t.root.kids[1], 500, 150)   # 30%
        assert t._split_justified(t.root)

    def test_a_thin_child_cannot_trigger_a_split(self):
        """Otherwise the tree splits on a handful of observations, which is the failure mode the
        fixed-depth sweep already demonstrated at k=4-5."""
        t = DT.DynTree(min_obs=40)
        t.root.kids[0] = DT.Node(); self._fill(t.root.kids[0], 500, 250)
        t.root.kids[1] = DT.Node(); self._fill(t.root.kids[1], 5, 5)
        assert not t._split_justified(t.root)


class TestAdaptiveDepth:
    def test_it_finds_planted_depth_two_structure(self):
        """POWER CHECK. Tomorrow depends on the accel/decel relation, not just today's sign."""
        rng = np.random.default_rng(0)
        T, N = 4000, 6
        x = np.zeros((T, N))
        for i in range(N):
            v = 1.0
            for t in range(T):
                x[t, i] = v
                if t >= 1:
                    dec = abs(x[t, i]) < abs(x[t - 1, i]) and (x[t, i] > 0) == (x[t - 1, i] > 0)
                    keep = rng.random() < (0.15 if dec else 0.85)
                else:
                    keep = rng.random() < 0.5
                mag = 1.0 + rng.random()
                v = mag * (1 if (x[t, i] > 0) == keep else -1)
        h, ll, dep, base, _td, _rn = DT.run(x, 1e9, 0, judge=1000, min_obs=40)
        assert dep.max() >= 2, 'tree never reached depth 2 on planted depth-2 structure'
        assert dep.mean() > 1.2, f'mean depth {dep.mean():.2f} -- barely using the structure'

    def test_it_stays_shallow_on_noise(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 700, (3000, 8))
        h, ll, dep, base, _td, _rn = DT.run(x, 1e9, 0, judge=800, min_obs=40)
        assert dep.mean() < 1.2, f'tree split on pure noise to mean depth {dep.mean():.2f}'


class TestKernels:
    def test_a_window_bounds_the_effective_sample(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 500, (1200, 4))
        t = DT.DynTree(max_depth=3, min_obs=40, half_life=1e9, window=100)
        for tt in range(1, len(x)):
            obs = []
            for i in range(4):
                p = DT.context(x[:, i], tt, 3)
                if p is None:
                    continue
                obs.append((p, float(np.sign(x[tt, i]) == np.sign(x[tt - 1, i]))))
            t.observe(obs)
        assert t.root.n <= 100 * 4 + 1, f'window did not bound the root count ({t.root.n})'
