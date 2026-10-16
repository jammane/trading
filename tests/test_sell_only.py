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
    # AAA and BBB are sold today, so they are not also bought (Alpaca: wash trade)
    assert {s for s, _ in of(o, "buy")} == {"DDD"}
    assert all(x["stop"] for x in o if x["action"] == "buy")           # the stop rides the buy
    assert not of(o, "stop_loss")                                      # no prior stops to re-place


def test_every_price_is_a_whole_cent_and_rounds_in_the_orders_favour():
    day = {s: {"open": 50.0, "high": 52.003, "low": 48.0071, "close": 50.0049} for s in SYMS}
    o = pv.plan_industry_orders(SYMS, SUG, HOLD, day, 10_000.0, {}, stops={"CCC": 45.4449})
    for x in o:
        for k in ("price", "stop"):
            if k in x:
                assert round(x[k] * 100, 6) == int(round(x[k] * 100)), (x, k)
    buy = next(x for x in o if x["action"] == "buy")
    raw = 48.0071 + 0.5 * (52.003 - 48.0071)
    assert raw - 0.01 < buy["price"] <= raw                            # a buy never pays more
    assert pv.to_cents(45.4449, "near") == 45.44 and pv.to_cents(0.123456, "down") == 0.1234


def test_held_shares_are_re_protected_in_every_industry_unless_bought_today():
    o = plan(stops={"AAA": 45.0, "CCC": 40.0, "DDD": 41.0})
    st = {(x["symbol"], x["quantity"], x["price"]) for x in o if x["action"] == "stop_loss"}
    assert st == {("AAA", 6, 45.0), ("CCC", 5, 40.0)}                 # DDD is not held
    o = pv.plan_industry_orders(SYMS, {**SUG, "CCC": (2.0, 0.5, 2.0, 0.0)}, HOLD, DAY,
                                10_000.0, {}, stops={"CCC": 40.0})
    assert not of(o, "stop_loss") and ("CCC", 2) in of(o, "buy")      # open buy: no stop beside it


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
    # BBB was sold in full, so nothing is left to protect
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


# ── Suspended symbols: data stopped arriving, so that symbol alone stops trading ─────────────

def test_a_symbol_whose_data_stopped_is_suspended_and_the_rest_are_not():
    pv = pytest.importorskip("production_v2")
    last = {"A": "2026-10-13", "B": "2026-10-13", "C": "2026-10-09"}
    session, sus = pv.suspended_symbols(["A", "B", "C", "D"], last)
    assert session == "2026-10-13"
    assert sus == {"C": "2026-10-09", "D": None}          # D: no data at all


def test_a_suspended_symbol_gets_no_orders_while_its_industry_trades():
    """Suspension removes the symbol's bar, and plan_industry_orders places nothing on a symbol
    without one -- no sell, no buy, no stop -- while its holdings stay as they are."""
    pv = pytest.importorskip("production_v2")
    syms = ["A", "B"]
    bar = {"open": 50.0, "high": 52.0, "low": 48.0, "close": 50.0}
    day = {"A": bar}                                       # B suspended: its bar was removed
    sug = {"A": (5, 0.5, 2.0, 0), "B": (5, 0.5, 0.0, 9)}
    orders = pv.plan_industry_orders(syms, sug, {"B": 10.0}, day, 1000.0, {})
    assert orders and all(o["symbol"] == "A" for o in orders)


def test_sell_qty_is_a_market_order_at_the_open():
    # The trainer sells sell_qty at the next open whatever the price; a limit at the decision
    # close is an order the trainer never models (AR, 2026-10-15: held while the trainer sold).
    o = plan()
    aaa = [x for x in o if x["action"] == "sell" and x["symbol"] == "AAA"]
    assert aaa == [{"symbol": "AAA", "action": "sell", "quantity": 4, "at_open": True}]
    bbb = [x for x in o if x["action"] == "sell" and x["symbol"] == "BBB"]
    assert len(bbb) == 1 and not bbb[0].get("at_open") and bbb[0]["price"] == 50.0  # sell-all limit
