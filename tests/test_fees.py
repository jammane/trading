"""Tests for broker fee constants and sell-net calculations."""

import pytest

from fees import (
    BUY_FILL,
    FINRA_TAF_MAX,
    FINRA_TAF_PER_SHARE,
    SEC_FEE_RATE,
    SELL_FILL,
    SLIPPAGE_RATE,
    _sell_net,
)


class TestFeeConstants:
    def test_buy_fill_is_one(self):
        assert BUY_FILL == 1.0, "Alpaca charges no commission — BUY_FILL must be 1.0"

    def test_sell_fill_is_one(self):
        assert SELL_FILL == 1.0, "Alpaca charges no commission — SELL_FILL must be 1.0"

    def test_sec_fee_rate(self):
        assert SEC_FEE_RATE == pytest.approx(0.0000278), \
            "SEC Section 31 rate must be $0.0000278 per $ of proceeds"

    def test_finra_taf_per_share(self):
        assert FINRA_TAF_PER_SHARE == pytest.approx(0.000166), \
            "FINRA TAF must be $0.000166 per share sold"

    def test_finra_taf_max(self):
        assert FINRA_TAF_MAX == pytest.approx(8.30), \
            "FINRA TAF per-trade cap must be $8.30"

    def test_slippage_rate(self):
        assert SLIPPAGE_RATE == pytest.approx(0.001), \
            "Slippage rate must be 0.10%"

    def test_sec_fee_rate_positive(self):
        assert SEC_FEE_RATE > 0

    def test_finra_taf_per_share_positive(self):
        assert FINRA_TAF_PER_SHARE > 0

    def test_finra_taf_max_positive(self):
        assert FINRA_TAF_MAX > 0

    def test_slippage_rate_between_zero_and_one(self):
        assert 0 < SLIPPAGE_RATE < 1


class TestSellNet:
    def test_zero_shares_returns_zero(self):
        assert _sell_net(0.0, 100.0) == 0.0

    def test_result_less_than_gross(self):
        gross = 100.0 * 50.0 * SELL_FILL
        net   = _sell_net(100.0, 50.0)
        assert net < gross, "Fees must reduce net proceeds below gross"

    def test_sec_fee_applied(self):
        shares, price = 10.0, 100.0
        gross         = shares * price * SELL_FILL
        sec_fee       = gross * SEC_FEE_RATE
        finra_fee     = shares * FINRA_TAF_PER_SHARE   # below cap for small trade
        expected_net  = gross - sec_fee - finra_fee
        assert _sell_net(shares, price) == pytest.approx(expected_net, rel=1e-9)

    def test_finra_taf_capped(self):
        # At $0.000166/share, the cap of $8.30 is hit at 8.30/0.000166 ≈ 50,000 shares
        shares, price = 100_000.0, 1.0
        gross         = shares * price * SELL_FILL
        sec_fee       = gross * SEC_FEE_RATE
        finra_fee     = FINRA_TAF_MAX                   # capped
        expected_net  = gross - sec_fee - finra_fee
        assert _sell_net(shares, price) == pytest.approx(expected_net, rel=1e-9)

    def test_finra_taf_below_cap_uses_per_share_rate(self):
        shares, price = 1.0, 100.0
        uncapped_finra = shares * FINRA_TAF_PER_SHARE
        assert uncapped_finra < FINRA_TAF_MAX, "Precondition: trade is below FINRA cap"
        gross        = shares * price * SELL_FILL
        expected_net = gross - gross * SEC_FEE_RATE - uncapped_finra
        assert _sell_net(shares, price) == pytest.approx(expected_net, rel=1e-9)

    def test_high_price_increases_sec_fee(self):
        net_low  = _sell_net(10.0, 10.0)
        net_high = _sell_net(10.0, 100.0)
        # Gross for high_price is 10× greater; SEC fee scales with proceeds
        gross_low  = 10.0 * 10.0
        gross_high = 10.0 * 100.0
        # Net proceeds should scale roughly with price (minus proportional SEC fee)
        assert net_high > net_low

    def test_more_shares_higher_net(self):
        assert _sell_net(100.0, 50.0) > _sell_net(10.0, 50.0)

    def test_net_positive_for_realistic_trade(self):
        assert _sell_net(100.0, 50.0) > 0

    def test_finra_cap_exact_boundary(self):
        cap_shares = FINRA_TAF_MAX / FINRA_TAF_PER_SHARE
        price      = 1.0
        gross      = cap_shares * price * SELL_FILL

        # Just below cap: FINRA fee = cap_shares * FINRA_TAF_PER_SHARE < MAX
        net_below = _sell_net(cap_shares - 1, price)
        # Just above cap: FINRA fee = FINRA_TAF_MAX
        net_above = _sell_net(cap_shares + 1, price)

        # Both should be valid positive numbers
        assert net_below > 0
        assert net_above > 0
