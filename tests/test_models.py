"""Tests for StockNN and MasterNN model architecture."""

import io
import pytest
import torch

from models import MasterNN, StockNN


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def stock_inputs():
    torch.manual_seed(0)
    history = torch.randn(1, 15, 60)
    today   = torch.randn(1, 208)
    return history, today


@pytest.fixture
def master_inputs():
    torch.manual_seed(0)
    history = torch.randn(1, 15, 61)
    today   = torch.randn(1, 229)
    return history, today


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
        assert model.fc_today.in_features  == 398  # 190 (final hidden) + 208 (today features)
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
        history, today = master_inputs
        out = MasterNN()(history, today)
        assert out.shape == (1, 36)

    def test_allocation_sums_to_one(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        alloc_sum = out[0, :12].sum().item()
        assert abs(alloc_sum - 1.0) < 1e-5, \
            f"Allocation weights must sum to 1 (softmax); got {alloc_sum}"

    def test_allocation_all_positive(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        assert (out[0, :12] > 0).all(), "All allocation weights must be positive (softmax)"

    def test_liquidation_depth_in_range(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        depth = out[0, 12:24]
        assert (depth >= 0).all() and (depth <= 1).all(), \
            "Liquidation depth must be in [0, 1] (sigmoid)"

    def test_liquidation_trigger_in_range(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        trigger = out[0, 24:36]
        assert (trigger >= 0).all() and (trigger <= 1).all(), \
            "Liquidation trigger must be in [0, 1] (sigmoid)"

    def test_output_has_12_industries(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        assert out.shape[-1] == 36  # 12 alloc + 12 depth + 12 trigger

    def test_deterministic(self, master_inputs):
        history, today = master_inputs
        model = MasterNN()
        model.eval()
        with torch.no_grad():
            out1 = model(history, today)
            out2 = model(history, today)
        assert torch.equal(out1, out2)

    def test_serialization_roundtrip(self, master_inputs, tmp_path):
        history, today = master_inputs
        model = MasterNN()
        model.eval()
        with torch.no_grad():
            out_before = model(history, today)

        path = tmp_path / "master_model.pt"
        torch.save(model.state_dict(), path)

        model2 = MasterNN()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            out_after = model2(history, today)

        assert torch.allclose(out_before, out_after)

    def test_inject_layers_grow(self):
        model = MasterNN()
        for i, layer in enumerate(model.fc_inject):
            assert layer.in_features  == 181 + 5 * i, \
                f"fc_inject[{i}] in_features: expected {181 + 5*i}, got {layer.in_features}"
            assert layer.out_features == 125 + 5 * i, \
                f"fc_inject[{i}] out_features: expected {125 + 5*i}, got {layer.out_features}"

    def test_inject_layer_count(self):
        assert len(MasterNN().fc_inject) == 14

    def test_seed_layer_dims(self):
        model = MasterNN()
        assert model.fc_seed.in_features  == 61   # 60 OHLCV means + 1 flat_cos regime signal
        assert model.fc_seed.out_features == 120

    def test_today_layer_dims(self):
        model = MasterNN()
        assert model.fc_today.in_features  == 419  # 190 (final hidden) + 229 (today features)
        assert model.fc_today.out_features == 300

    def test_output_layer_dims(self):
        model = MasterNN()
        assert model.fc_out.in_features  == 102
        assert model.fc_out.out_features == 36

    def test_no_nan_in_output(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self, master_inputs):
        history, today = master_inputs
        out = MasterNN()(history, today)
        assert not torch.isinf(out).any()
