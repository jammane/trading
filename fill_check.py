"""Did the trainer predict what the broker did? Yesterday's orders, simulated against reality.

Each paper session trains the previous decision day on the account's real book
(`training_v4_cpp --upkeep`). Slot 0 of that training day is the model production traded with, on
the same book, so it re-makes production's orders and the fill model plays them against the bar
that has since arrived. The trainer writes what it did, order by order, to `orders_sim.csv`; this
module sets that beside what Alpaca actually did, at three levels:

  intent   StockNN's raw outputs, production (Python/torch) against the trainer (C++). These must
           agree: a difference means the two built different inputs from the same day, and
           everything downstream inherits it.
  orders   the limit prices and quantities each side derived from those outputs.
  fills    for every order: did the fill model fill it, and did the market? At what price? And
           the result -- each symbol's holding after the session, simulated against real.

A mismatch at a level is reported with the first level it shows up at, since a different intent
explains every difference below it. The rows go to fill_check.csv (one per symbol with anything
to compare) and a summary is printed; nothing here touches the broker except to read.

There are no deliberate differences left at the order level: both sides take their orders from
training_lib.plan_orders (the trainer from its C++ twin), so a difference in orders means a
difference in intent. The fill level is where the simulator and the broker can still disagree.
"""
import csv
import os

INTENT_TOL = 1e-3          # relative difference between the two forward passes still "the same"
PRICE_TOL = 0.011          # a cent of rounding plus a little: production rounds limits to cents


def load_sim(path):
    """orders_sim.csv -> {symbol: row (numbers as floats)}."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["symbol"]] = {k: (v if k in ("date", "fill_date", "industry", "symbol")
                                    else float(v)) for k, v in r.items()}
    return out


def rel(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def real_by_symbol(orders):
    """Broker orders -> {symbol: {'buy': o, 'sell': [o..], 'stop': [o..]}}. orders: dicts with
    symbol, side, type, qty, limit, stop, filled_qty, filled_px, status, legs [same shape]."""
    out = {}
    for o in orders:
        d = out.setdefault(o["symbol"], {"buy": None, "sell": [], "stop": []})
        if o["side"] == "buy":
            d["buy"] = o
            for leg in o.get("legs") or []:
                d["stop"].append(leg)
        elif o["type"] == "stop":
            d["stop"].append(o)
        else:
            d["sell"].append(o)
    return out


def compare(pending, sim, broker_orders, positions):
    """-> (rows, summary). pending: production's recorded decision; sim: load_sim(); broker_orders:
    see real_by_symbol; positions: {symbol: shares held now}."""
    real = real_by_symbol(broker_orders)
    intents = pending.get("intents") or {}
    rows, s = [], dict(symbols=0, intent_checked=0, intent_mismatch=0, buys_real=0,
                       buy_both=0, buy_sim_only=0, buy_real_only=0, buy_neither=0,
                       buy_px_diff=[], hold_mismatch=0, notes=[])
    for sym, sr in sim.items():
        r = real.get(sym, {"buy": None, "sell": [], "stop": []})
        held_before = float(pending.get("holdings", {}).get(sym, 0.0))
        row = dict(date=sr["date"], fill_date=sr["fill_date"], industry=sr["industry"],
                   symbol=sym, held_before=held_before)
        # intent
        it = intents.get(sym)
        sim_it = [sr["bqty"], sr["bfrac"], sr["sfrac"], sr["sqty"]]
        if it is not None:
            d = max(rel(a, b) for a, b in zip(it, sim_it, strict=True))
            row.update(intent_diff=round(d, 6), intent_match=d <= INTENT_TOL)
            s["intent_checked"] += 1
            s["intent_mismatch"] += d > INTENT_TOL
        # buys
        b = r["buy"]
        sim_bought, sim_px = sr["bought_q"], sr["bought_px"]
        real_q = float(b["filled_qty"] or 0) if b else 0.0
        real_px = float(b["filled_px"] or 0) if b else 0.0
        row.update(real_buy_lim=b["limit"] if b else "", real_buy_qty=b["qty"] if b else "",
                   real_buy_status=b["status"] if b else "", sim_buy_lim=round(sr["buy_lim"], 4),
                   sim_bought=sim_bought, sim_buy_px=round(sim_px, 4), real_bought=real_q,
                   real_buy_px=real_px)
        if b:
            s["buys_real"] += 1
            if abs(float(b["limit"]) - sr["buy_lim"]) > PRICE_TOL:
                row["note"] = "buy limit differs"
            if sim_bought > 0 and real_q > 0:
                s["buy_both"] += 1
                s["buy_px_diff"].append(real_px - sim_px)
            elif sim_bought > 0:
                s["buy_sim_only"] += 1
                row["fill"] = "SIM filled, market did not"
            elif real_q > 0:
                s["buy_real_only"] += 1
                row["fill"] = "market filled, SIM did not"
            else:
                s["buy_neither"] += 1
        # sells and stops (real side summed; sim side split by how it fired)
        real_sold = sum(float(o["filled_qty"] or 0) for o in r["sell"])
        real_stopped = sum(float(o["filled_qty"] or 0) for o in r["stop"])
        sim_sold = sr["sold_open_q"] + sr["sold_all_q"]
        row.update(sim_sold=sim_sold, real_sold=real_sold, sim_stopped=sr["stopped_q"],
                   real_stopped=real_stopped)
        # result
        hold_sim, hold_real = sr["hold_after"], float(positions.get(sym, 0.0))
        row.update(sim_hold_after=hold_sim, real_hold_after=hold_real,
                   hold_match=abs(hold_sim - hold_real) < 0.5)
        if not row["hold_match"]:
            s["hold_mismatch"] += 1
        if b or held_before or hold_sim or hold_real or real_sold or sim_sold:
            rows.append(row)
            s["symbols"] += 1
    if s["intent_checked"] == 0:
        s["notes"].append("no production intents recorded for this decision (before fill_check)")
    return rows, s


def summary_lines(s, date):
    px = s["buy_px_diff"]
    out = [f"Fill check {date}: {s['symbols']} symbol(s) compared"]
    if s["intent_checked"]:
        out.append(f"  intent (Python vs C++ StockNN outputs): {s['intent_checked'] - s['intent_mismatch']}"
                   f"/{s['intent_checked']} match"
                   + ("" if not s["intent_mismatch"] else "  <-- MISMATCH: the two sides built "
                      "different inputs; every difference below inherits it"))
    out.append(f"  buys placed {s['buys_real']}: both filled {s['buy_both']}, sim only "
               f"{s['buy_sim_only']}, market only {s['buy_real_only']}, neither {s['buy_neither']}")
    if px:
        out.append(f"  fill price, market - sim: mean {sum(px) / len(px):+.4f}, worst "
                   f"{max(px, key=abs):+.4f} per share")
    out.append(f"  holdings after the session: {s['symbols'] - s['hold_mismatch']}/{s['symbols']} "
               f"as simulated")
    out += [f"  note: {n}" for n in s["notes"]]
    return out


def append_rows(path, rows):
    if not rows:
        return
    cols = ["date", "fill_date", "industry", "symbol", "held_before", "intent_diff",
            "intent_match", "real_buy_lim", "sim_buy_lim", "real_buy_qty", "real_buy_status",
            "sim_bought", "real_bought", "sim_buy_px", "real_buy_px", "sim_sold", "real_sold",
            "sim_stopped", "real_stopped", "sim_hold_after", "real_hold_after", "hold_match",
            "fill", "note"]
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def broker_order_dict(o):
    """An alpaca-py Order -> the plain dict compare() reads."""
    def f(x):
        return None if x is None else float(x)
    def v(e):
        return getattr(e, "value", e)
    return dict(symbol=o.symbol, side=v(o.side), type=v(o.order_type), qty=f(o.qty),
                limit=f(o.limit_price), stop=f(o.stop_price), filled_qty=f(o.filled_qty),
                filled_px=f(o.filled_avg_price), status=v(o.status),
                legs=[broker_order_dict(leg) for leg in (o.legs or [])])

