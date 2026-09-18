"""Tests for StockNN, MasterNN, and MT2NN model architecture. MT1Net: tests/test_mt1net.py."""

import pytest
import torch
import torch.nn.functional as F

import models
from models import MT2NN, MasterNN, StockNN, stock_close_pos, stock_close_vs_wap

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def stock_inputs():
    torch.manual_seed(0)
    history = torch.randn(1, 15, 60)
    today   = torch.randn(1, 232)
    return history, today


@pytest.fixture
def master_inputs():
    torch.manual_seed(0)
    today = torch.randn(1, 444)
    return (today,)


# ── StockNN ────────────────────────────────────────────────────────────────────

class TestStockNN:
    def test_output_shape(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today)
        assert out.shape == (1, 48)

    def test_output_reshapes_to_12x4(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today).view(12, 4)
        assert out.shape == (12, 4)

    def test_buy_qty_nonnegative(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today).view(12, 4)
        assert (out[:, 0] >= 0).all(), "buy_qty must be non-negative (ReLU)"

    def test_buy_price_frac_in_range(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today).view(12, 4)
        assert (out[:, 1] >= 0).all() and (out[:, 1] <= 1).all(), \
            "buy_price_frac must be in [0, 1] (sigmoid)"

    def test_sell_all_price_frac_in_range(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today).view(12, 4)
        assert (out[:, 2] >= 0).all() and (out[:, 2] <= 1).all(), \
            "sell_all_price_frac must be in [0, 1] (sigmoid)"

    def test_sell_qty_nonnegative(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today).view(12, 4)
        assert (out[:, 3] >= 0).all(), "sell_qty must be non-negative (ReLU)"

    def test_deterministic(self, stock_inputs):
        history, today = stock_inputs
        model = StockNN()
        model.eval()
        with torch.no_grad():
            out1 = model(history, today)
            out2 = model(history, today)
        assert torch.equal(out1, out2)

    def test_serialization_roundtrip(self, stock_inputs, tmp_path):
        history, today = stock_inputs
        model = StockNN()
        model.eval()
        with torch.no_grad():
            out_before = model(history, today)

        path = tmp_path / "stock_model.pt"
        torch.save(model.state_dict(), path)

        model2 = StockNN()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            out_after = model2(history, today)

        assert torch.allclose(out_before, out_after)

    def test_inject_layers_grow(self):
        model = StockNN()
        for i, layer in enumerate(model.fc_inject):
            assert layer.in_features  == 180 + 5 * i, \
                f"fc_inject[{i}] in_features: expected {180 + 5*i}, got {layer.in_features}"
            assert layer.out_features == 125 + 5 * i, \
                f"fc_inject[{i}] out_features: expected {125 + 5*i}, got {layer.out_features}"

    def test_inject_layer_count(self):
        assert len(StockNN().fc_inject) == 14

    def test_seed_layer_dims(self):
        model = StockNN()
        assert model.fc_seed.in_features  == 60
        assert model.fc_seed.out_features == 120

    def test_today_layer_dims(self):
        model = StockNN()
        assert model.fc_today.in_features  == 422  # 190 (final hidden) + 232 (today features)
        assert model.fc_today.out_features == 300

    def test_output_layer_dims(self):
        model = StockNN()
        assert model.fc_out.in_features  == 111
        assert model.fc_out.out_features == 48

    def test_batch_size_one(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today)
        assert out.shape[0] == 1

    def test_no_nan_in_output(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, stock_inputs):
        history, today = stock_inputs
        out = StockNN()(history, today)
        assert not torch.isinf(out).any()


# ── MasterNN ───────────────────────────────────────────────────────────────────

class TestMasterNN:
    def test_output_shape(self, master_inputs):
        (today,) = master_inputs
        out = MasterNN()(today)
        assert out.shape == (1, 48)

    def test_output_reshapes_to_12x4(self, master_inputs):
        (today,) = master_inputs
        out = MasterNN()(today)
        assert out.view(12, 4).shape == (12, 4)

    def test_per_industry_softmax_sums_to_one(self, master_inputs):
        (today,) = master_inputs
        out = MasterNN()(today)
        probs = F.softmax(out.view(12, 4), dim=1)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(12), atol=1e-5), \
            "Per-industry softmax rows must sum to 1"

    def test_tier_argmax_in_range(self, master_inputs):
        (today,) = master_inputs
        out  = MasterNN()(today)
        tiers = F.softmax(out.view(12, 4), dim=1).argmax(dim=1)
        assert ((tiers >= 0) & (tiers <= 3)).all(), "Tier argmax must be in {0,1,2,3}"

    def test_output_has_12x4_logits(self, master_inputs):
        (today,) = master_inputs
        out = MasterNN()(today)
        assert out.shape[-1] == 48  # 12 industries × 4 class logits

    def test_deterministic(self, master_inputs):
        (today,) = master_inputs
        model = MasterNN()
        model.eval()
        with torch.no_grad():
            out1 = model(today)
            out2 = model(today)
        assert torch.equal(out1, out2)

    def test_serialization_roundtrip(self, master_inputs, tmp_path):
        (today,) = master_inputs
        model = MasterNN()
        model.eval()
        with torch.no_grad():
            out_before = model(today)

        path = tmp_path / "master_model.pt"
        torch.save(model.state_dict(), path)

        model2 = MasterNN()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            out_after = model2(today)

        assert torch.allclose(out_before, out_after)

    def test_layer_dims(self):
        model = MasterNN()
        assert model.fc1.in_features    == 444
        assert model.fc1.out_features   == 444
        assert model.fc2.in_features    == 444
        assert model.fc2.out_features   == 444
        assert model.fc3.in_features    == 444
        assert model.fc3.out_features   == 312
        assert model.fc4.in_features    == 312
        assert model.fc4.out_features   == 180
        assert model.fc_out.in_features  == 180
        assert model.fc_out.out_features == 48

    def test_no_nan_in_output(self, master_inputs):
        (today,) = master_inputs
        out = MasterNN()(today)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, master_inputs):
        (today,) = master_inputs
        out = MasterNN()(today)
        assert not torch.isinf(out).any()


# ── MT2NN ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mt2_inputs():
    torch.manual_seed(0)
    # One MT1 prediction per industry, in dollars — was 48 (4 channels × 12) before the MT1
    # rebuild. Scaled like a real prediction so the test exercises the magnitudes MT2 will see.
    return torch.randn(1, 12) * 500.0


class TestMT2NN:
    def test_output_shape(self, mt2_inputs):
        out = MT2NN()(mt2_inputs)
        assert out.shape == (1, 48)

    def test_param_count(self):
        n = sum(p.numel() for p in MT2NN().parameters())
        assert n == 32844, f"MT2NN param count: expected 32844, got {n}"

    def test_output_reshapes_to_12x4(self, mt2_inputs):
        out = MT2NN()(mt2_inputs)
        assert out.view(12, 4).shape == (12, 4)

    def test_tier_argmax_in_range(self, mt2_inputs):
        out = MT2NN()(mt2_inputs)
        tiers = out.view(12, 4).argmax(dim=1)
        assert ((tiers >= 0) & (tiers <= 3)).all()

    def test_fc_branch_dims(self):
        m = MT2NN()
        assert m.fc1.in_features == 12 and m.fc1.out_features == 36
        assert m.fc2.in_features == 36 and m.fc2.out_features == 36

    def test_lstm_dims(self):
        m = MT2NN()
        assert m.lstm.input_size  == 1
        assert m.lstm.hidden_size == 36
        assert m.lstm.num_layers  == 2

    def test_lstm_walks_industries_one_at_a_time(self, mt2_inputs):
        """The LSTM must see 12 steps of 1 feature, not 1 step of 12.

        The reshape is x.view(batch, 12, 1); a view(batch, 1, 12) has the same element count and
        would run without error, collapsing the sequence into a single step and silently throwing
        away the recurrence the branch exists for.
        """
        m = MT2NN().eval()
        lstm_in = mt2_inputs.view(mt2_inputs.size(0), 12, 1)
        assert lstm_in.shape == (1, 12, 1)
        _, (h_n, _) = m.lstm(lstm_in)
        assert h_n.shape == (2, 1, 36)

    def test_every_industry_reaches_the_output(self, mt2_inputs):
        """Perturbing any one industry's prediction must move the logits.

        With a 12-wide input a fencepost in the FC or the reshape would drop an industry entirely,
        and the shape checks above would all still pass.
        """
        m = MT2NN().eval()
        with torch.no_grad():
            base = m(mt2_inputs)
            for i in range(12):
                x = mt2_inputs.clone()
                x[0, i] += 1000.0
                assert not torch.equal(m(x), base), f'industry {i} does not reach the output' 

    def test_taper_dims(self):
        m = MT2NN()
        assert m.taper1.in_features == 72  and m.taper1.out_features == 66
        assert m.taper2.in_features == 66  and m.taper2.out_features == 60
        assert m.taper3.in_features == 60  and m.taper3.out_features == 54
        assert m.fc_out.in_features == 54  and m.fc_out.out_features == 48

    def test_deterministic(self, mt2_inputs):
        model = MT2NN()
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(mt2_inputs), model(mt2_inputs))

    def test_serialization_roundtrip(self, mt2_inputs, tmp_path):
        model = MT2NN()
        model.eval()
        with torch.no_grad():
            out_before = model(mt2_inputs)
        path = tmp_path / "mt2.pt"
        torch.save(model.state_dict(), path)
        model2 = MT2NN()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            assert torch.allclose(out_before, model2(mt2_inputs))

    def test_no_nan(self, mt2_inputs):
        assert not torch.isnan(MT2NN()(mt2_inputs)).any()

    def test_no_inf(self, mt2_inputs):
        assert not torch.isinf(MT2NN()(mt2_inputs)).any()


class TestStockClosePos:
    """close_pos = (4C - 2O - H - L) / (7(H - L)) — StockNN today feature 15 (v0.6.0.0).

    Must stay identical to stock_close_pos() in training_v4.cpp.
    """

    def test_close_at_high_of_a_bar_that_opened_at_the_low(self):
        # O=L=0, H=C=1  ->  (4 - 0 - 1 - 0) / 7 = 3/7
        assert stock_close_pos(0.0, 1.0, 0.0, 1.0) == pytest.approx(3.0 / 7.0)

    def test_close_at_low_of_a_bar_that_opened_at_the_high(self):
        # O=H=1, C=L=0  ->  (0 - 2 - 1 - 0) / 7 = -3/7
        assert stock_close_pos(1.0, 1.0, 0.0, 0.0) == pytest.approx(-3.0 / 7.0)

    def test_unchanged_midrange_bar_is_zero(self):
        # O=C=0.5, H=1, L=0  ->  (2 - 1 - 1 - 0) / 7 = 0
        assert stock_close_pos(0.5, 1.0, 0.0, 0.5) == pytest.approx(0.0)

    def test_zero_range_bar_returns_zero_not_nan(self):
        assert stock_close_pos(10.0, 10.0, 10.0, 10.0) == 0.0

    def test_invalid_bar_returns_zero(self):
        assert stock_close_pos(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_scale_free(self):
        """Same bar shape at two price levels must give the same value — the whole point of
        dividing by (H-L) rather than feeding the raw dollar numerator."""
        cheap = stock_close_pos(10.0, 11.0, 9.0, 10.5)
        rich  = stock_close_pos(400.0, 440.0, 360.0, 420.0)
        assert cheap == pytest.approx(rich)

    def test_equals_two_intraday_plus_clv(self):
        """Identity: close_pos == (2*(C-O)/(H-L) + CLV) / 7 * 7 ... i.e. 2A + B over 7."""
        o, hi, lo, c = 12.0, 15.0, 11.0, 14.0
        a = (c - o) / (hi - lo)
        clv = ((c - lo) - (hi - c)) / (hi - lo)
        assert stock_close_pos(o, hi, lo, c) == pytest.approx((2 * a + clv) / 7.0)

    def test_bounded_for_real_bars(self):
        """L <= O,C <= H implies close_pos in [-3/7, 3/7]."""
        for o, hi, lo, c in [(1, 2, 0.5, 1.5), (5, 5.2, 4.1, 4.2), (100, 140, 99, 139)]:
            v = stock_close_pos(float(o), float(hi), float(lo), float(c))
            assert -3.0 / 7.0 - 1e-9 <= v <= 3.0 / 7.0 + 1e-9


class TestTodayLayout:
    """The `today` vector's section arithmetic (v0.6.0.0).

    These constants mirror training_v4.cpp; a change on one side that is not mirrored on the
    other silently misaligns every feature past the first symbol block, which stays in bounds
    and so produces plausible wrong numbers rather than a crash.
    """

    def test_width_matches_the_model_input(self):
        model = StockNN()
        assert model.fc_today.in_features == 190 + models.TODAY_WIDTH

    def test_sections_tile_the_vector_without_gap_or_overlap(self):
        per_sym = models.TODAY_N_SYMS * models.TODAY_PER_SYM
        assert per_sym == models.TODAY_AGG_OFF
        assert models.TODAY_AGG_OFF + models.TODAY_AGG_LEN == models.TODAY_STATE_OFF
        state_len = 1 + models.TODAY_N_SYMS
        assert models.TODAY_STATE_OFF + state_len == models.TODAY_WIDTH

    def test_close_pos_and_reserved_are_the_last_two_slots_of_each_block(self):
        assert models.TODAY_RESERVED == models.TODAY_PER_SYM - 1
        assert models.TODAY_CLOSE_POS == models.TODAY_PER_SYM - 2

    def test_reserved_indices_stay_inside_the_symbol_region(self):
        for j in range(models.TODAY_N_SYMS):
            assert j * models.TODAY_PER_SYM + models.TODAY_RESERVED < models.TODAY_AGG_OFF

    def test_expected_absolute_indices(self):
        """Pinned literals — if these move, every .bin on disk is invalidated."""
        assert models.TODAY_WIDTH == 232
        assert models.TODAY_AGG_OFF == 204
        assert models.TODAY_STATE_OFF == 219
        close_pos = [j * models.TODAY_PER_SYM + models.TODAY_CLOSE_POS
                     for j in range(models.TODAY_N_SYMS)]
        assert close_pos == [15, 32, 49, 66, 83, 100, 117, 134, 151, 168, 185, 202]


class TestStockCloseVsWap:
    """close_vs_wap = (C − A)/A with A = (2O+3C+H+L)/7 — today feature 16 (v0.6.1.0).

    Must stay identical to stock_close_vs_wap() in training_v4.cpp.
    """

    def test_matches_the_explicit_definition(self):
        o, hi, lo, c = 12.0, 15.0, 11.0, 14.0
        a = (2 * o + 3 * c + hi + lo) / 7.0
        assert stock_close_vs_wap(o, hi, lo, c) == pytest.approx((c - a) / a)

    def test_shares_the_numerator_with_close_pos(self):
        """Both features are (4C−2O−H−L) over a different denominator; the ratio between them
        is (H−L)/A, the range as a fraction of price. That relationship is the reason both slots
        are carried, so pin it."""
        o, hi, lo, c = 20.0, 23.0, 19.0, 22.0
        a = (2 * o + 3 * c + hi + lo) / 7.0
        assert (stock_close_pos(o, hi, lo, c) / stock_close_vs_wap(o, hi, lo, c)
                == pytest.approx(a / (hi - lo)))

    def test_flat_bar_is_zero(self):
        assert stock_close_vs_wap(10.0, 10.0, 10.0, 10.0) == pytest.approx(0.0)

    def test_strong_close_is_positive_weak_close_negative(self):
        assert stock_close_vs_wap(10.0, 11.0, 9.5, 11.0) > 0     # closes at the high
        assert stock_close_vs_wap(11.0, 11.0, 9.5, 9.5) < 0      # closes at the low

    def test_scale_free(self):
        cheap = stock_close_vs_wap(10.0, 11.0, 9.0, 10.5)
        rich = stock_close_vs_wap(400.0, 440.0, 360.0, 420.0)
        assert cheap == pytest.approx(rich)

    def test_degenerate_bar_returns_zero_not_nan(self):
        assert stock_close_vs_wap(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_slot_16_is_no_longer_reserved(self):
        assert models.TODAY_CLOSE_WAP == 16
        assert models.TODAY_RESERVED == models.TODAY_CLOSE_WAP   # alias kept
        assert models.TODAY_PER_SYM == 17                        # unchanged by filling the slot
        assert models.TODAY_WIDTH == 232                         # unchanged — that was the point
