"""
fees.py — Broker fee constants and sell-net helper.

Alpaca charges $0 commission. Regulatory fees still apply on sells.
SEC Section 31 rate is set annually; FINRA TAF is fixed.
"""

BUY_FILL            = 1.0        # multiplier on buy  fill price (1.0 = no commission)
SELL_FILL           = 1.0        # multiplier on sell fill price (1.0 = no commission)
SEC_FEE_RATE        = 0.0000278  # SEC Section 31: $0.0000278 per $ of sale proceeds
FINRA_TAF_PER_SHARE = 0.000166   # FINRA TAF: $0.000166 per share sold
FINRA_TAF_MAX       = 8.30       # FINRA TAF per-trade cap
SLIPPAGE_RATE       = 0.001      # 0.10% adverse slippage on triggered STOPS (market orders)
# Calibrated to PAPER fills until prod runs, then to prod (training_v4.cpp has the measurements).
# Limits fill at the limit, never worse (0 of 162 paper buys filled above their limit). A buy whose
# limit is at or above the next open fills at the open plus this premium, capped at the limit:
# paper paid +0.60% over the open on average (119 fills, 20 sessions).
OPEN_BUY_PREMIUM    = 0.0070
# A sell that fills at the open (a market sell, a sell-all limit the open gaps through, a stop the
# open gaps below) fills this far below the open, never below a limit: paper's market sells at the
# open filled 0.91% below it and its gapped sell-all limits 0.42% (124 sells, 2026-10-14..11-18).
OPEN_SELL_DISCOUNT  = 0.0080
# The fill simulation's version (training_v4.cpp FILL_MODEL, which lists them). Bump it with any
# change to a fill price: a run's champions are re-scored under the new model before it trains on
# (rescore_champions.py).
FILL_MODEL          = 3


def _sell_net(shares: float, price: float) -> float:
    """Net cash received on an equity sell after SEC fee and FINRA TAF."""
    gross = shares * price * SELL_FILL
    fees  = gross * SEC_FEE_RATE + min(shares * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX)
    return gross - fees


def open_sell_price(nd_open: float, limit: float = 0.0) -> float:
    """Price a sell fills at the open: OPEN_SELL_DISCOUNT below it, never below `limit` (0 =
    market). Mirrors open_sell_price() in training_v4.cpp."""
    return max(nd_open * (1.0 - OPEN_SELL_DISCOUNT), limit)


def buy_fill_price(nd_open: float, nd_low: float, nd_high: float, limit: float) -> float:
    """Price a limit buy fills at against the next bar, or 0.0 if it does not fill.
    Mirrors buy_fill_price() in training_v4.cpp. Strict at the low: a limit exactly at the day's
    low does not fill intraday."""
    if nd_open <= limit:
        return min(nd_open * (1.0 + OPEN_BUY_PREMIUM), limit)
    if nd_low < limit < nd_high:
        return limit
    return 0.0
