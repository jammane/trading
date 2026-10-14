"""fill_check: the trainer's simulation of a decision day set beside what the broker did.

The comparison is only useful if it attributes a difference to the right level: a different
intent (the two forward passes disagree) must be reported as such, a fill the simulator gave and
the market did not must be counted on its own side, and an untraded symbol must not appear.
"""
import pytest

import fill_check as fc

COLS = ("date,fill_date,industry,symbol,hold_before,cash_before,bqty,bfrac,sfrac,sqty,buy_lim,"
        "sell_all_lim,stop_before,low_first,fill_open,fill_high,fill_low,fill_close,sold_open_q,"
        "sold_open_px,sold_all_q,sold_all_px,stopped_q,stopped_px,bought_q,bought_px,stop_after,"
        "hold_after,cash_after")


def _sim(tmp_path, rows):
    p = tmp_path / "orders_sim.csv"
    lines = [COLS]
    for sym, d in rows.items():
        base = dict(hold_before=0, cash_before=500, bqty=5, bfrac=0.5, sfrac=2.0, sqty=0,
                    buy_lim=50.0, sell_all_lim=60.0, stop_before=0, low_first=1, fill_open=51,
                    fill_high=52, fill_low=49, fill_close=50, sold_open_q=0, sold_open_px=0,
                    sold_all_q=0, sold_all_px=0, stopped_q=0, stopped_px=0, bought_q=0,
                    bought_px=0, stop_after=0, hold_after=0, cash_after=500) | d
        lines.append(",".join(["2026-10-13", "2026-10-14", "ind", sym]
                              + [str(base[c]) for c in COLS.split(",")[4:]]))
    p.write_text("\n".join(lines) + "\n")
    return fc.load_sim(str(p))


def _buy(sym, limit, qty, filled, px=None, status="filled"):
    return dict(symbol=sym, side="buy", type="limit", qty=qty, limit=limit, stop=None,
                filled_qty=filled, filled_px=px, status=status, legs=[])


def test_matching_day_reports_all_levels_agreeing(tmp_path):
    sim = _sim(tmp_path, {"A": dict(bought_q=5, bought_px=50.05, hold_after=5),
                          "B": dict(bqty=0), "C": dict(bqty=0)})
    pending = dict(intents={"A": [5, 0.5, 2.0, 0], "B": [0, 0.5, 2.0, 0], "C": [0, 0.5, 2.0, 0]},
                   holdings={})
    rows, s = fc.compare(pending, sim, [_buy("A", 50.0, 5, 5, 50.0)], {"A": 5})
    assert [r["symbol"] for r in rows] == ["A"]          # untraded symbols are not listed
    assert s["intent_mismatch"] == 0 and s["intent_checked"] == 3
    assert s["buy_both"] == 1 and s["hold_mismatch"] == 0
    assert s["buy_px_diff"] == [pytest.approx(-0.05)]                   # the market filled 5c under the sim


def test_each_way_a_fill_can_disagree_is_counted_on_its_own_side(tmp_path):
    sim = _sim(tmp_path, {"A": dict(bought_q=5, bought_px=50.05, hold_after=5),
                          "B": dict(hold_after=0), "C": dict(hold_after=0)})
    orders = [_buy("A", 50.0, 5, 0, status="expired"), _buy("B", 50.0, 5, 5, 49.9),
              _buy("C", 50.0, 5, 0, status="expired")]
    rows, s = fc.compare(dict(holdings={}), sim, orders, {"B": 5})
    assert (s["buy_sim_only"], s["buy_real_only"], s["buy_neither"]) == (1, 1, 1)
    assert s["hold_mismatch"] == 2                      # A: sim holds 5, B: market holds 5
    by = {r["symbol"]: r for r in rows}
    assert by["A"]["fill"].startswith("SIM filled") and by["B"]["fill"].startswith("market")


def test_a_different_intent_is_flagged_as_the_first_cause(tmp_path):
    sim = _sim(tmp_path, {"A": dict(bqty=5.0, bfrac=0.5)})
    pending = dict(intents={"A": [5.0, 0.73, 2.0, 0]}, holdings={})
    rows, s = fc.compare(pending, sim, [_buy("A", 51.0, 5, 0, status="expired")], {})
    assert s["intent_mismatch"] == 1 and rows[0]["intent_match"] is False
    assert rows[0]["note"] == "buy limit differs"
    assert any("MISMATCH" in line for line in fc.summary_lines(s, "2026-10-13"))


def test_sells_and_stops_compare_the_quantities_that_left_the_book(tmp_path):
    sim = _sim(tmp_path, {"A": dict(hold_before=10, sqty=4, sold_open_q=4, sold_open_px=51,
                                    stopped_q=6, stopped_px=45, hold_after=0, bqty=0)})
    orders = [dict(symbol="A", side="sell", type="limit", qty=4, limit=50.5, stop=None,
                   filled_qty=4, filled_px=50.6, status="filled", legs=[]),
              dict(symbol="A", side="sell", type="stop", qty=6, limit=None, stop=45.0,
                   filled_qty=0, filled_px=None, status="new", legs=[])]
    rows, s = fc.compare(dict(holdings={"A": 10}), sim, orders, {"A": 6})
    r = rows[0]
    assert (r["sim_sold"], r["real_sold"], r["sim_stopped"], r["real_stopped"]) == (4, 4, 6, 0)
    assert r["hold_match"] is False and s["notes"]      # no intents recorded -> said so


def test_rows_append_under_one_header(tmp_path):
    sim = _sim(tmp_path, {"A": dict(bought_q=5, bought_px=50.05, hold_after=5)})
    rows, _ = fc.compare(dict(holdings={}), sim, [_buy("A", 50.0, 5, 5, 50.0)], {"A": 5})
    out = tmp_path / "fill_check.csv"
    fc.append_rows(str(out), rows)
    fc.append_rows(str(out), rows)
    lines = out.read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("date,fill_date,industry,symbol")
