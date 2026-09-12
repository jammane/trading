"""`--flat-allocation`: even capital split with MT1/MT2 inference skipped.

Used while MT1/MT2 have no demonstrated out-of-sample skill, so paper results measure the
StockNN layer rather than an unvalidated allocator.

The flat path is written explicitly rather than reusing the "no models" fallback, and these
tests pin the two reasons why:

  * `tiers_to_alloc` returns $0.00 for every industry when no industry has a positive tier,
    so the fallback deploys an EMPTY book, not an even one (`test_without_flat_...`).
  * `tiers_to_alloc` re-ranks positives into terciles weighted 1.0/1.5/2.25, so even a uniform
    positive tier_map does not come back equal (`test_tiers_to_alloc_cannot_produce_flat`).

And the liquidation interaction: three consecutive tier-0 readings sell an industry's holdings,
so the flat path must CLEAR `zero_counts`, never let it accumulate.
"""
import pytest

import production_v2
from training_lib import tiers_to_alloc

INDS = {f'ind_{i}': [f'SYM{i}{j}' for j in range(12)] for i in range(12)}


class _ExplodingModel:
    """Stands in for MasterNN — records any call. The flat path must never reach it."""

    def __init__(self):
        self.called = False

    def __call__(self, *a, **kw):
        self.called = True
        raise AssertionError('master model must not be called under flat allocation')


def _flat(total_cash, industries=INDS, zero_counts=None, master=None):
    zc = {ind: 0 for ind in industries} if zero_counts is None else zero_counts
    alloc, tiers, mt1 = production_v2.run_master_allocation(
        master, industries, {}, {}, zc, total_cash, model_dir=None, flat=True)
    return alloc, tiers, mt1, zc


class TestFlatAllocation:
    def test_splits_evenly_across_all_industries(self):
        alloc, _, _, _ = _flat(120_000.0)
        assert len(alloc) == 12
        assert set(alloc.values()) == {10_000.0}

    def test_uneven_division_preserves_total(self):
        total = 100_000.0                      # 8333.33... per industry
        alloc, _, _, _ = _flat(total)
        assert sum(alloc.values()) == pytest.approx(total)
        assert len(set(alloc.values())) == 1, 'every industry must get the identical share'

    def test_every_industry_gets_a_positive_tier(self):
        # Anything reading tier 0 treats the industry as "allocate nothing / count toward
        # liquidation". Under flat, nothing may read as tier 0.
        _, tiers, _, _ = _flat(120_000.0)
        assert set(tiers) == set(INDS)
        assert all(t > 0 for t in tiers.values())

    def test_zero_counts_are_cleared(self):
        # The regression this guards: zero_counts at 3+ liquidates that industry's holdings.
        # Pre-seed above the threshold — flat must reset, not increment.
        seeded = {ind: 5 for ind in INDS}
        _, _, _, zc = _flat(120_000.0, zero_counts=seeded)
        assert set(zc.values()) == {0}

    def test_mt1_outputs_is_none(self):
        _, _, mt1, _ = _flat(120_000.0)
        assert mt1 is None

    def test_master_model_is_never_called(self):
        master = _ExplodingModel()
        alloc, _, _, _ = _flat(120_000.0, master=master)
        assert master.called is False
        assert set(alloc.values()) == {10_000.0}

    def test_zero_cash(self):
        alloc, _, _, _ = _flat(0.0)
        assert set(alloc.values()) == {0.0}

    def test_no_industries_does_not_divide_by_zero(self):
        alloc, tiers, _, _ = _flat(120_000.0, industries={})
        assert alloc == {}
        assert tiers == {}


class TestFlatIsNotTheFallback:
    def test_without_flat_and_without_models_allocates_nothing(self):
        # The contrast that motivates an explicit flat path: the documented "equal allocation"
        # fallback allocates $0.00 everywhere, which is an empty book and not an even one.
        zc = {ind: 0 for ind in INDS}
        alloc, tiers, mt1 = production_v2.run_master_allocation(
            None, INDS, {}, {}, zc, 120_000.0, model_dir=None, flat=False)
        assert set(alloc.values()) == {0.0}
        assert set(tiers.values()) == {0}
        assert mt1 is None
        assert set(zc.values()) == {1}, 'tier-0 readings must accumulate toward liquidation'

    def test_tiers_to_alloc_cannot_produce_flat(self):
        # Even handed a uniform positive tier_map, tiers_to_alloc re-ranks into terciles and
        # weights them 1.0/1.5/2.25 — so it can never be reused to express "even".
        uniform = {ind: 1 for ind in INDS}
        alloc = tiers_to_alloc(uniform, list(INDS), 120_000.0)
        assert len(set(round(v, 6) for v in alloc.values())) > 1
