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

import torch.nn.functional as F

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
