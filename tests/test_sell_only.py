"""Sell-only mode (the loser zone): StockNN keeps running, its sells go in, nothing is bought.

An industry in the loser zone drains at the model's own sell rate. It is not liquidated: a held
position the model does not sell stays held. Because production cancels a held symbol's open stop
before selling (Alpaca reserves the shares a stop covers), sell-only must re-place that stop for
whatever the day's sells leave held, or the remainder would sit unprotected.
"""
import pytest

pv = pytest.importorskip("production_v2")

SYMS = ["AAA", "BBB", "CCC", "DDD"]
DAY = {s: {"open": 50.0, "high": 52.0, "low": 48.0, "close": 50.0} for s in SYMS}
HOLD = {"AAA": 10.0, "BBB": 8.0, "CCC": 5.0}             # DDD not held
# (buy_qty, buy_price_frac, sell_all_price_frac, sell_qty)
SUG = {
    "AAA": (5.0, 0.5, 2.0, 4.0),    # buys 5, sells 4 of 10 (sell-all limit out of range)
    "BBB": (3.0, 0.5, 0.5, 0.0),    # buys 3, sells all 8 at the limit
    "CCC": (0.0, 0.5, 2.0, 0.0),    # no sell, no buy: must stay untouched
    "DDD": (4.0, 0.5, 2.0, 0.0),    # buy only
}


def plan(**kw):
    return pv.plan_industry_orders(SYMS, SUG, HOLD, DAY, 10_000.0, {}, **kw)


def of(orders, action):
    return {(o["symbol"], o["quantity"]) for o in orders if o["action"] == action}


def test_normal_mode_buys_and_sells():
    o = plan()
    assert of(o, "sell") == {("AAA", 4), ("BBB", 8)}
    assert {s for s, _ in of(o, "buy")} == {"AAA", "BBB", "DDD"}
    assert of(o, "stop_loss") == {(s, q) for s, q in of(o, "buy")}     # one stop per buy


def test_sell_only_places_exactly_the_same_sells_and_no_buys():
    o = plan(sell_only=True)
    assert of(o, "sell") == of(plan(), "sell")
    assert not of(o, "buy")


def test_sell_only_is_not_liquidation():
    o = plan(sell_only=True)
    assert not [x for x in o if x["symbol"] in ("CCC", "DDD")]       # held-but-unsold untouched
    assert sum(q for _, q in of(o, "sell")) < sum(HOLD.values())


def test_sell_only_re_protects_the_remainder_at_the_old_stop_price():
    o = plan(sell_only=True, stops={"AAA": 45.0, "BBB": 44.0})
    stops = [x for x in o if x["action"] == "stop_loss"]
    assert [(x["symbol"], x["quantity"], x["price"]) for x in stops] == [("AAA", 6, 45.0)]
    # BBB was sold in full, so nothing is left to protect; CCC's stop was never cancelled
    sells = [i for i, x in enumerate(o) if x["symbol"] == "AAA" and x["action"] == "sell"]
    assert sells[0] < o.index(stops[0])                              # sell submitted first


def test_sell_only_without_a_prior_stop_invents_none():
    assert not of(plan(sell_only=True), "stop_loss")


def test_new_symbol_hold_still_applies():
    o = pv.plan_industry_orders(SYMS, SUG, HOLD, DAY, 10_000.0, {"AAA": 3}, sell_only=True)
    assert not [x for x in o if x["symbol"] == "AAA"]


def test_buys_still_spend_down_the_budget():
    o = pv.plan_industry_orders(SYMS, SUG, HOLD, DAY, 260.0, {})
    assert sum(x["quantity"] * x["price"] for x in o if x["action"] == "buy") <= 260.0
