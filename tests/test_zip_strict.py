"""Parallel-array (zip strict=) guards.

Background: v0.4.1.0 added `strict=` to all 24 bare `zip()` calls (ruff B905). Bare `zip` silently
truncates to the shortest operand, so a length mismatch produced a plausible WRONG NUMBER rather
than an error — the worst failure mode for a system whose outputs are scores computed from parallel
arrays. `strict=True` turns that into a `ValueError`.

These tests pin that behaviour down where it is reachable without a full simulation, plus the
length invariants the zips depend on. `training_lib.step_industry` and `selection_and_mutation`
also contain strict zips but need a whole day of fill simulation to reach, so they are covered
indirectly by the C++/Python parity work rather than here.
"""

import pytest
import torch

from training_lib import (
    compute_alloc_from_predicted,
    compute_weighted_avg_model,
    compute_weighted_avg_portfolio,
    save_slot_model,
)


class TestAllocFromPredicted:
    """training_lib.compute_alloc_from_predicted: dict(zip(industry_list, weights, strict=True)).

    `weights` is always sliced/padded to exactly 12, so strict=True asserts the caller passed a
    12-industry list. Previously a short list silently produced a partial allocation map.
    """

    @staticmethod
    def _predicted():
        return torch.zeros(1, 36)

    def test_accepts_twelve_industries(self):
        inds = [f"ind{i}" for i in range(12)]
        alloc, depth, trigger = compute_alloc_from_predicted(self._predicted(), inds)
        assert set(alloc) == set(inds)
        assert len(depth) == 12
        assert len(trigger) == 12

    @pytest.mark.parametrize("n", [1, 11, 13, 24])
    def test_rejects_wrong_industry_count(self, n):
        inds = [f"ind{i}" for i in range(n)]
        with pytest.raises(ValueError):
            compute_alloc_from_predicted(self._predicted(), inds)


class TestWeightedAvgPortfolio:
    """training_lib.compute_weighted_avg_portfolio: zip(portfolios, weights, strict=True) ×2."""

    @staticmethod
    def _pf(cash, qty):
        return {"cash": cash, "holdings": {"AAA": qty}, "label": "unused"}

    def test_equal_lengths_average_correctly(self):
        out = compute_weighted_avg_portfolio(
            [self._pf(100.0, 10.0), self._pf(300.0, 30.0)], [1.0, 1.0]
        )
        assert out["cash"] == pytest.approx(200.0)
        assert out["holdings"]["AAA"] == pytest.approx(20.0)

    def test_weights_are_applied(self):
        out = compute_weighted_avg_portfolio(
            [self._pf(0.0, 0.0), self._pf(100.0, 10.0)], [0.0, 1.0]
        )
        assert out["cash"] == pytest.approx(100.0)

    @pytest.mark.parametrize("n_pf,n_val", [(3, 2), (2, 3), (1, 5)])
    def test_rejects_length_mismatch(self, n_pf, n_val):
        pfs = [self._pf(100.0, 1.0) for _ in range(n_pf)]
        with pytest.raises(ValueError):
            compute_weighted_avg_portfolio(pfs, [1.0] * n_val)


class TestWeightedAvgModel:
    """training_lib.compute_weighted_avg_model: zip(slots, weights, strict=True).

    Uses MT1Net (3,501 params) rather than StockNN (928,825) to keep the fixture cheap.
    """

    @staticmethod
    def _seed(tmp_path, n):
        from models import MT1Net

        for slot in range(n):
            save_slot_model("zt", str(tmp_path), slot, MT1Net())
        return MT1Net

    def test_equal_lengths_produce_a_model(self, tmp_path):
        cls = self._seed(tmp_path, 3)
        out = compute_weighted_avg_model("zt", str(tmp_path), [0, 1, 2], [1.0, 1.0, 1.0], cls)
        assert isinstance(out, cls)
        assert all(torch.isfinite(p).all() for p in out.parameters())

    @pytest.mark.parametrize("n_slots,n_vals", [(3, 2), (2, 3)])
    def test_rejects_length_mismatch(self, tmp_path, n_slots, n_vals):
        cls = self._seed(tmp_path, max(n_slots, n_vals))
        with pytest.raises(ValueError):
            compute_weighted_avg_model(
                "zt", str(tmp_path), list(range(n_slots)), [1.0] * n_vals, cls
            )


class TestLengthInvariants:
    """Invariants the remaining strict=True zips rely on, asserted at their source."""

    def test_plot_training_industries_match_colors(self):
        """plot_training: dict(zip(INDUSTRIES, COLORS, strict=True)) — one colour per industry."""
        import plot_training

        assert len(plot_training.INDUSTRIES) == len(plot_training.COLORS)
        assert len(plot_training.IND_COLOR) == len(plot_training.INDUSTRIES)

    def test_read_mt_log_industry_names_match_n_ind(self):
        """read_mt_log unpacks fixed-width N_IND blocks and labels them from INDUSTRY_NAMES."""
        import read_mt_log

        assert len(read_mt_log.INDUSTRY_NAMES) == read_mt_log.N_IND

    def test_industry_name_lists_agree_across_modules(self):
        """plot_training, read_mt_log and the universe must describe the same 12 industries."""
        import plot_training
        import read_mt_log
        from universe import INDUSTRIES

        assert plot_training.INDUSTRIES == read_mt_log.INDUSTRY_NAMES
        assert set(plot_training.INDUSTRIES) == set(INDUSTRIES.keys())


class TestDiversityInjectionOffByOne:
    """training_lib diversity injection is the one strict=False site that is NOT a sliding window.

    ELITE_COUNT is odd, so `half = ELITE_COUNT // 2` splits the ranked elites unevenly:
    source_slots gets `half`, inject_slots gets `ELITE_COUNT - half` (one more). The zip truncates
    to the shorter, so the last inject slot is never replaced — while the log line above it claims
    the full count. This test pins the CURRENT behaviour so a deliberate fix shows up as a failing
    test rather than a silent change in training dynamics. See the comment at that zip.
    """

    def test_split_is_uneven_and_one_slot_is_skipped(self):
        from training_lib import ELITE_COUNT

        half = ELITE_COUNT // 2
        n_source = half
        n_inject = ELITE_COUNT - half
        n_paired = min(n_source, n_inject)

        assert ELITE_COUNT % 2 == 1, "even ELITE_COUNT would make the split exact"
        assert n_inject == n_source + 1
        assert n_inject - n_paired == 1, "exactly one inject slot is currently skipped"
