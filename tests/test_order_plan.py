"""Training fills production's orders, and only those.

production_v2 decides its orders at the close, before anything fills, under limits the broker
imposes: a symbol sold today is not bought (Alpaca rejects the buy as a wash trade), buys are
sized against the cash the industry has (tomorrow's sale proceeds are not in it, and an order
that does not fill still used its share of the budget), at the limit, in whole shares, and every
price is rounded to an increment Alpaca accepts. The simulators used to re-buy what they had just
sold with the proceeds and size at the fill price -- on paper (2026-10-14) only 16 of 26 holdings
matched. Now training_lib.plan_orders is the one place these rules live: production queues its
plan, the Python simulators fill it, and training_v4.cpp's plan_orders() is its twin.
"""
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import training_lib as tl

SYMS = [f"S{i}" for i in range(12)]


def f32(x):
    return float(np.float32(x))


def random_book(rng):
    """One industry at the close: bars, holdings, cash and the model's (saturated or not) output."""
    day, hold, sug = {}, {}, {}
    for s in SYMS:
        if rng.random() < 0.08:
            continue                                        # no bar today
        low = f32(rng.uniform(20, 90))
        high = f32(low * rng.uniform(1.0, 1.08))
        close = f32(rng.uniform(low, high))
        day[s] = {"open": low, "high": high, "low": low, "close": close}
        hold[s] = float(rng.choice([0, 0, 0, 1, 2, 5, 10]))
        q = rng.choice([0.0, 0.0, rng.uniform(0, 20), 1e9, 2.4e15])
        sug[s] = (f32(q), f32(rng.choice([0.0, 1.0, rng.random()])),
                  f32(rng.choice([0.0, 1.0, rng.random(), 1.5])),
                  f32(rng.choice([0.0, 0.0, rng.uniform(0, 4), 1e12])))
    for s in SYMS:
        sug.setdefault(s, (0.0, 0.0, 0.0, 0.0))
    return day, hold, sug, f32(rng.choice([0.0, 75.0, 562.5, rng.uniform(0, 3000)]))


# ── The rules ───────────────────────────────────────────────────────────────────────────────────

DAY = {s: {"open": 50.0, "high": 52.0, "low": 48.0, "close": 50.0} for s in ("AAA", "BBB", "CCC")}


def test_a_held_symbol_is_sold_and_not_bought():
    p = tl.plan_orders(["AAA"], {"AAA": (1e9, 0.0, 1.0, 0.0)}, {"AAA": 3.0}, DAY, 10_000.0)
    assert p["sell_all"] == {"AAA": (3, 52.0)} and not p["buys"] and p["sold"] == {"AAA"}


def test_buys_are_sized_against_the_cash_at_the_close():
    # three buys wanted, $120 of cash: one share of the first two at $48, none of the third
    sug = {s: (1e9, 0.0, 1.0, 0.0) for s in ("AAA", "BBB", "CCC")}
    p = tl.plan_orders(["AAA", "BBB", "CCC"], sug, {}, DAY, 120.0)
    assert {s: q for s, (q, _, _) in p["buys"].items()} == {"AAA": 1, "BBB": 1}


def test_sale_proceeds_do_not_fund_buys():
    sug = {"AAA": (0.0, 0.0, 0.0, 0.0), "BBB": (1e9, 0.0, 1.0, 0.0)}
    p = tl.plan_orders(["AAA", "BBB"], sug, {"AAA": 100.0}, DAY, 0.0)
    assert "AAA" in p["sell_all"] and not p["buys"]


def test_prices_are_whole_cents_rounded_in_the_orders_favour():
    day = {"AAA": {"open": 50.0, "high": 52.003, "low": 48.0071, "close": 50.0049}}
    p = tl.plan_orders(["AAA"], {"AAA": (5.0, 0.5, 0.5, 0.0)}, {}, day, 10_000.0)
    _, limit, stop = p["buys"]["AAA"]
    raw = 48.0071 + 0.5 * (52.003 - 48.0071)
    assert raw - 0.01 < limit <= raw and round(limit * 100) == limit * 100
    assert stop == tl.to_cents(tl.buy_stop_price(limit, 50.0049), "near")


# ── The Python simulator fills the plan ─────────────────────────────────────────────────────────

def fill(sym, port, plan, nd, low_first=True):
    day = {sym: DAY[sym]}
    b, s = tl._fill_sym_open_low(sym, port, plan, day, {sym: nd}, low_first)
    if low_first:
        s += tl._fill_sell_all_intraday(sym, port, plan, day, {sym: nd})
    return b, s


def test_the_planned_quantity_is_bought_even_when_the_fill_is_cheaper():
    plan = tl.plan_orders(["AAA"], {"AAA": (2.0, 0.0, 1.0, 0.0)}, {}, DAY, 1000.0)  # 2 @ $48
    port = {"cash": 1000.0, "holdings": {}, "stop_prices": {}}
    fill("AAA", port, plan, {"open": 30.0, "high": 31.0, "low": 29.0, "close": 30.0})
    assert port["holdings"]["AAA"] == 2 and port["cash"] == pytest.approx(940.0)


def test_a_position_bought_today_is_not_sold_back_the_same_day():
    plan = tl.plan_orders(["AAA"], {"AAA": (2.0, 0.0, 0.0, 0.0)}, {}, DAY, 1000.0)
    port = {"cash": 1000.0, "holdings": {}, "stop_prices": {}}
    fill("AAA", port, plan, {"open": 47.0, "high": 60.0, "low": 46.0, "close": 55.0})
    assert port["holdings"]["AAA"] == 2                     # bought at the open, still held


def test_no_stop_beside_a_sell():
    # production cancels a held symbol's stop before selling it, and re-places none
    plan = tl.plan_orders(["AAA"], {"AAA": (0.0, 0.0, 1.0, 0.0)}, {"AAA": 3.0}, DAY, 0.0)
    port = {"cash": 0.0, "holdings": {"AAA": 3.0}, "stop_prices": {"AAA": 45.0}}
    fill("AAA", port, plan, {"open": 46.0, "high": 47.0, "low": 40.0, "close": 41.0})
    assert port["holdings"]["AAA"] == 3.0                   # the $52 limit never filled


def test_sell_qty_goes_at_the_open():
    plan = tl.plan_orders(["AAA"], {"AAA": (0.0, 0.0, 2.0, 2.0)}, {"AAA": 5.0}, DAY, 0.0)
    port = {"cash": 0.0, "holdings": {"AAA": 5.0}, "stop_prices": {}}
    assert plan["sell_open"] == {"AAA": 2} and not plan["sell_all"]   # $56 limit: out of range
    fill("AAA", port, plan, {"open": 49.0, "high": 50.0, "low": 45.0, "close": 46.0})
    assert port["holdings"]["AAA"] == 3.0
    assert port["cash"] == pytest.approx(tl._sell_net(2, 49.0))


def test_production_queues_the_plan():
    pv = pytest.importorskip("production_v2")
    rng = random.Random(7)
    for _ in range(200):
        day, hold, sug, cash = random_book(rng)
        plan = tl.plan_orders(SYMS, sug, hold, day, cash)
        orders = pv.plan_industry_orders(SYMS, sug, hold, day, cash, {})
        buys = {o["symbol"]: (o["quantity"], o["price"], o["stop"])
                for o in orders if o["action"] == "buy"}
        assert buys == plan["buys"]
        assert {o["symbol"] for o in orders if o["action"] == "sell"} == plan["sold"]


# ── The C++ twin ────────────────────────────────────────────────────────────────────────────────

def cpp_block(src):
    """The constants and functions plan_orders() needs, lifted from training_v4.cpp verbatim."""
    parts = [re.search(r"static constexpr int IND_SYMS = [^;]+;", src).group(0),
             re.search(r"static constexpr float MAX_SINGLE_STOCK_PCT = [^;]+;", src).group(0),
             re.search(r"static constexpr float STOP_LOSS_FRAC = [^;]+;", src).group(0),
             re.search(r"static inline float buy_stop_price\(.*?\n}\n", src, re.S).group(0)]
    a = src.index("static inline double to_cents(")
    b = src.index("\n}\n", src.index("static void plan_orders(")) + 3
    return "\n".join(parts) + "\n" + src[a:b]


def test_cpp_twin_plans_the_same_orders(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("g++ not available")
    prog = tmp_path / "plan.cpp"
    prog.write_text(
        "#include <algorithm>\n#include <cmath>\n#include <cstdio>\n"
        + cpp_block(Path("training_v4.cpp").read_text()) + r"""
int main() {
    float cash;
    while (scanf("%f", &cash) == 1) {
        float out48[48], low[IND_SYMS], high[IND_SYMS], close[IND_SYMS], hold[IND_SYMS];
        unsigned char valid[IND_SYMS];
        for (int j = 0; j < IND_SYMS; j++) {
            int v;
            scanf("%d %f %f %f %f %f %f %f %f", &v, &low[j], &high[j], &close[j], &hold[j],
                  &out48[4*j], &out48[4*j+1], &out48[4*j+2], &out48[4*j+3]);
            valid[j] = (unsigned char)v;
        }
        OrderPlan p;
        plan_orders(out48, low, high, close, valid, hold, cash, p);
        for (int j = 0; j < IND_SYMS; j++)
            printf("%.0f %.4f %.0f %.4f %.4f %d\n", p.sell_open[j], p.sell_lim[j], p.buy_q[j],
                   p.buy_lim[j], p.buy_stop[j], p.sold[j] ? 1 : 0);
    }
}
""")
    exe = tmp_path / "plan"
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe), str(prog)], check=True)

    rng = random.Random(20261015)
    books = [random_book(rng) for _ in range(300)]
    lines = []
    for day, hold, sug, cash in books:
        lines.append(repr(cash))
        for s in SYMS:
            d = day.get(s, {"low": 0.0, "high": 0.0, "close": 0.0})
            lines.append(" ".join(repr(x) for x in (
                int(s in day), d["low"], d["high"], d["close"], hold.get(s, 0.0), *sug[s])))
    res = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout.split("\n")

    n_buys = 0
    for k, (day, hold, sug, cash) in enumerate(books):
        plan = tl.plan_orders(SYMS, sug, hold, day, cash)
        for j, s in enumerate(SYMS):
            so, sl, bq, bl, bs, sold = res[k * 12 + j].split()
            assert float(so) == plan["sell_open"].get(s, 0), (k, s)
            assert float(sl) == pytest.approx(plan["sell_all"].get(s, (0, 0.0))[1], abs=1e-4)
            q, lim, stop = plan["buys"].get(s, (0, 0.0, 0.0))
            assert float(bq) == q, (k, s, bq, q)
            assert float(bl) == pytest.approx(lim, abs=1e-4)
            assert float(bs) == pytest.approx(stop, abs=0.0100001)   # float vs double near a half cent
            assert bool(int(sold)) == (s in plan["sold"])
            n_buys += q > 0
    assert n_buys > 100, "the random books must exercise buys"
