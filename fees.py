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
SLIPPAGE_RATE       = 0.001      # 0.10% adverse price slippage on limit & stop fills


def _sell_net(shares: float, price: float) -> float:
    """Net cash received on an equity sell after SEC fee and FINRA TAF."""
    gross = shares * price * SELL_FILL
    fees  = gross * SEC_FEE_RATE + min(shares * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX)
    return gross - fees
