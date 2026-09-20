"""depth_sweep.py: the ordered-set state function.

The claim that k=2 reproduces the existing six states is load-bearing -- it is what makes the
sweep a generalisation of the current model rather than a different model that happens to share a
number. If the mapping were not a bijection, every column of the sweep would be measuring
something unrelated to what is deployed.
"""
import importlib.util
import sys
import types

import numpy as np
import pytest


def _load():
    sys.modules.setdefault('read_mt1_dataset', types.ModuleType('read_mt1_dataset'))
    spec = importlib.util.spec_from_file_location('depth_sweep', 'depth_sweep.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load()


class TestStateSpace:
    @pytest.mark.parametrize('k,want', [(0, 1), (1, 2), (2, 6), (3, 24), (4, 120), (5, 720)])
    def test_size(self, k, want):
        assert S.n_states(k) == want

    def test_k2_is_a_bijection_onto_the_original_six(self):
        """Run inline by the script too, but pinned here so a refactor cannot quietly break it."""
        mapping = S.verify_k2()
        assert len(mapping) == 6
        assert len(set(mapping.values())) == 6

    @pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
    def test_indices_stay_in_range(self, k):
        rng = np.random.default_rng(k)
        for _ in range(2000):
            vals = rng.normal(0, 100, k)
            st = S.state_k(vals)
            assert st is None or 0 <= st < S.n_states(k)

    def test_flat_or_nonfinite_days_give_no_state(self):
        """A hard-floor reset writes exactly $0; it is not a real day and must not be encoded."""
        assert S.state_k([1.0, 0.0, 2.0]) is None
        assert S.state_k([1.0, np.nan]) is None

    def test_the_state_depends_on_ORDER_not_just_values(self):
        """The whole premise: the same two days in the opposite order are a different state."""
        assert S.state_k([100.0, 50.0]) != S.state_k([50.0, 100.0])

    def test_scale_invariance_of_the_ordering(self):
        """Doubling every day preserves both the profitability order and the sign count, so the
        state must not move. If it did, the model would be conditioning on volatility by accident.
        """
        for k in range(1, 6):
            rng = np.random.default_rng(100 + k)
            v = rng.normal(0, 100, k)
            if S.state_k(v) is not None:
                assert S.state_k(v) == S.state_k(v * 3.0)

    def test_sign_count_separates_otherwise_identical_orderings(self):
        """Same profitability ORDER, different number of positive days -> different state."""
        a = S.state_k([20.0, 10.0])      # both positive, older more profitable
        b = S.state_k([-10.0, -20.0])    # both negative, older more profitable
        assert a is not None and b is not None and a != b
