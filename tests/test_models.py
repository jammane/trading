"""Tests for StockNN, MasterNN, MT1NN, and MT2NN model architecture."""

import io
import pytest
import torch

from models import MasterNN, MT1NN, MT1Head, MT1Tail, MT2NN, StockNN


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


# ── MT1NN ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mt1_inputs():
    torch.manual_seed(0)
    return torch.randn(1, 37)


class TestMT1NN:
    def test_output_shape(self, mt1_inputs):
        out = MT1NN()(mt1_inputs)
        assert out.shape == (1, 4)

    def test_param_count(self):
        # Composed = shared head (998) + 4 specialized tails (1187 each) = 5746
        assert sum(p.numel() for p in MT1Head().parameters()) == 998
        assert sum(p.numel() for p in MT1Tail().parameters()) == 1187
        n = sum(p.numel() for p in MT1NN().parameters())
        assert n == 5746, f"MT1NN param count: expected 5746, got {n}"

    def test_head_tail_shapes(self, mt1_inputs):
        h = MT1Head()(mt1_inputs)
        assert h.shape == (1, 28)                 # concat A20+B4+C4
        assert MT1Tail()(h).shape == (1, 1)       # single-output tail

    def test_composition(self, mt1_inputs):
        # Composed forward == head then the four tails concatenated.
        m = MT1NN()
        h = m.head(mt1_inputs)
        manual = torch.cat([t(h) for t in m.tails], dim=1)
        assert torch.allclose(m(mt1_inputs), manual)

    def test_head_tail_roundtrip(self):
        import numpy as np
        from prepare_models import state_dict_to_arr, HEAD_LAYER_DEFS, TAIL_LAYER_DEFS
        from convert_weights import arr_to_state_dict
        x = torch.randn(1, 37)
        head = MT1Head()
        arr = state_dict_to_arr(head.state_dict(), HEAD_LAYER_DEFS)
        assert arr.size == 998
        head2 = MT1Head(); head2.load_state_dict(arr_to_state_dict(arr, HEAD_LAYER_DEFS, MT1Head))
        assert torch.allclose(head(x), head2(x))
        tail = MT1Tail(); h = head(x)
        tarr = state_dict_to_arr(tail.state_dict(), TAIL_LAYER_DEFS)
        assert tarr.size == 1187
        tail2 = MT1Tail(); tail2.load_state_dict(arr_to_state_dict(tarr, TAIL_LAYER_DEFS, MT1Tail))
        assert torch.allclose(tail(h), tail2(h))

    def test_confidence_after_sigmoid(self, mt1_inputs):
        out = MT1NN()(mt1_inputs)
        conf = torch.sigmoid(out[:, 0])
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_calib_confidence_after_sigmoid(self, mt1_inputs):
        out = MT1NN()(mt1_inputs)
        conf4 = torch.sigmoid(out[:, 3])
        assert (conf4 >= 0).all() and (conf4 <= 1).all()

    def test_range_after_softplus(self, mt1_inputs):
        out = MT1NN()(mt1_inputs)
        rng = F.softplus(out[:, 2])
        assert (rng > 0).all()

    def test_deterministic(self, mt1_inputs):
        model = MT1NN()
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(mt1_inputs), model(mt1_inputs))

    def test_serialization_roundtrip(self, mt1_inputs, tmp_path):
        model = MT1NN()
        model.eval()
        with torch.no_grad():
            out_before = model(mt1_inputs)
        path = tmp_path / "mt1.pt"
        torch.save(model.state_dict(), path)
        model2 = MT1NN()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            assert torch.allclose(out_before, model2(mt1_inputs))

    def test_no_nan(self, mt1_inputs):
        assert not torch.isnan(MT1NN()(mt1_inputs)).any()

    def test_no_inf(self, mt1_inputs):
        assert not torch.isinf(MT1NN()(mt1_inputs)).any()


# ── MT2NN ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mt2_inputs():
    torch.manual_seed(0)
    return torch.randn(1, 48)


class TestMT2NN:
    def test_output_shape(self, mt2_inputs):
        out = MT2NN()(mt2_inputs)
        assert out.shape == (1, 48)

    def test_param_count(self):
        n = sum(p.numel() for p in MT2NN().parameters())
        assert n == 34572, f"MT2NN param count: expected 34572, got {n}"

    def test_output_reshapes_to_12x4(self, mt2_inputs):
        out = MT2NN()(mt2_inputs)
        assert out.view(12, 4).shape == (12, 4)

    def test_tier_argmax_in_range(self, mt2_inputs):
        out = MT2NN()(mt2_inputs)
        tiers = out.view(12, 4).argmax(dim=1)
        assert ((tiers >= 0) & (tiers <= 3)).all()

    def test_fc_branch_dims(self):
        m = MT2NN()
        assert m.fc1.in_features == 48 and m.fc1.out_features == 36
        assert m.fc2.in_features == 36 and m.fc2.out_features == 36

    def test_lstm_dims(self):
        m = MT2NN()
        assert m.lstm.input_size  == 4
        assert m.lstm.hidden_size == 36
        assert m.lstm.num_layers  == 2

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
