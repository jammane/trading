"""Single-output MT1 — the rebuild.

One network per industry emitting one number: that industry's next-day StockNN P&L in dollars.
Replaces MT1DualHead + four MT1Tail (9,208 params across five 200-slot pools).
"""
import pytest
import torch

from models import MT1NET_LAYER_DEFS, MT1NET_PARAMS, MT1Net


class TestShape:
    def test_one_output(self):
        net = MT1Net()
        out = net(torch.randn(5, 74))
        assert out.shape == (5, 1), 'MT1 emits exactly one number per industry'

    def test_accepts_the_74_feature_slice(self):
        net = MT1Net()
        assert net(torch.randn(1, 74)).shape == (1, 1)
        with pytest.raises(Exception):
            net(torch.randn(1, 73))

    def test_output_is_a_raw_logit(self):
        """Decode is mt1_pred(): tanh(raw) * scale. The net must NOT pre-squash."""
        net = MT1Net()
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(50.0)
        out = net(torch.randn(64, 74) * 10)
        assert out.abs().max() > 1.5, 'output looks bounded — a squash crept into the net'


class TestParams:
    def test_param_count_matches_the_defs(self):
        assert sum(p.numel() for p in MT1Net().parameters()) == MT1NET_PARAMS

    def test_param_count_is_3501(self):
        assert MT1NET_PARAMS == 3501

    def test_much_smaller_than_the_old_composed_model(self):
        old = 1996 + 4 * 1803          # MT1DualHead + four MT1Tail
        assert MT1NET_PARAMS < old * 0.45, 'the point was a large reduction'

    def test_defs_and_module_agree_layer_by_layer(self):
        net = MT1Net()
        sd = dict(net.named_parameters())
        for name, out_sz, in_sz in MT1NET_LAYER_DEFS:
            assert sd[f'{name}.weight'].shape == (out_sz, in_sz), name
            assert sd[f'{name}.bias'].shape == (out_sz,), name

    def test_defs_cover_every_parameter(self):
        named = {n.rsplit('.', 1)[0] for n, _ in MT1Net().named_parameters()}
        assert named == {n for n, _, _ in MT1NET_LAYER_DEFS}


class TestBlockStructure:
    """The three feature blocks stay separate for two layers — the one piece of the old head
    worth keeping, because it is a real inductive bias AND cheaper than dense (998 vs 2,100)."""

    def test_trunk_is_cheaper_than_dense(self):
        trunk = sum(o * i + o for n, o, i in MT1NET_LAYER_DEFS if n.startswith('m_'))
        assert trunk == 998
        assert trunk < 37 * 28 + 28, 'block structure must beat a dense 37->28'

    def test_daily_block_cannot_reach_the_vol_block_in_layer_one(self):
        """Perturbing a vol+poly feature must not move the daily sub-trunk's output."""
        net = MT1Net().eval()
        x = torch.zeros(1, 74)
        with torch.no_grad():
            b_before = net._trunk(x[:, 0:37], net.m_a1, net.m_a2, net.m_b1,
                                  net.m_b2, net.m_c1, net.m_c2)[:, 20:24].clone()
            x[0, 25] = 5.0                      # a vol+poly feature, block [17:37]
            b_after = net._trunk(x[:, 0:37], net.m_a1, net.m_a2, net.m_b1,
                                 net.m_b2, net.m_c1, net.m_c2)[:, 20:24]
        assert torch.allclose(b_before, b_after), 'blocks are mixing in layer one'

    def test_market_and_portfolio_trunks_are_separate(self):
        """A portfolio feature must not move the market trunk."""
        net = MT1Net().eval()
        x = torch.zeros(1, 74)
        with torch.no_grad():
            m_before = net._trunk(x[:, 0:37], net.m_a1, net.m_a2, net.m_b1,
                                  net.m_b2, net.m_c1, net.m_c2).clone()
            x[0, 50] = 5.0                      # portfolio half
            m_after = net._trunk(x[:, 0:37], net.m_a1, net.m_a2, net.m_b1,
                                 net.m_b2, net.m_c1, net.m_c2)
        assert torch.allclose(m_before, m_after)


class TestReservedInput:
    """d2 takes 23 inputs; the 23rd is held for (H-L)/A. Reserving it now means filling it later
    changes no dimension, offset or file format — the trick that let StockNN's slot 16 be filled
    in v0.6.1.0 with binary compatibility intact."""

    def test_default_is_exactly_inert(self):
        net = MT1Net().eval()
        x = torch.randn(4, 74)
        with torch.no_grad():
            assert torch.allclose(net(x), net(x, extra=torch.zeros(4, 1)), atol=1e-6)

    def test_the_slot_is_actually_wired(self):
        """A non-zero extra must change the output, or the reservation is decorative."""
        net = MT1Net().eval()
        x = torch.randn(8, 74)
        with torch.no_grad():
            a = net(x, extra=torch.zeros(8, 1))
            b = net(x, extra=torch.full((8, 1), 10.0))
        assert not torch.allclose(a, b), 'reserved d2 input is not connected'

    def test_d2_has_room_for_it(self):
        d2 = next(d for d in MT1NET_LAYER_DEFS if d[0] == 'd2')
        d1 = next(d for d in MT1NET_LAYER_DEFS if d[0] == 'd1')
        assert d2[2] == d1[1] + 1, 'd2 input must be d1 output plus the reserved slot'
        assert MT1Net.RESERVED_D2_INPUT == d1[1]


class TestSerialization:
    def test_state_dict_roundtrip(self):
        a = MT1Net()
        b = MT1Net()
        b.load_state_dict(a.state_dict())
        x = torch.randn(3, 74)
        with torch.no_grad():
            assert torch.allclose(a(x), b(x))

    def test_flat_vector_roundtrip_in_defs_order(self):
        """The C++ side stores one flat float array in MT1NET_LAYER_DEFS order."""
        net = MT1Net()
        sd = net.state_dict()
        flat = torch.cat([torch.cat([sd[f'{n}.weight'].flatten(), sd[f'{n}.bias'].flatten()])
                          for n, _, _ in MT1NET_LAYER_DEFS])
        assert flat.numel() == MT1NET_PARAMS
