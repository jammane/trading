#!/usr/bin/env python3
"""
fill_reconciliation.py — measure what the fill simulator gets wrong.

StockNN is selected on *simulated* fills, so the fill model IS the training signal. If it is
wrong, every trained model is wrong. This compares, order by order, what the simulator would have
predicted against what Alpaca actually did.

Runs RETROACTIVELY: Alpaca retains order history, so no paper sessions are lost by adding this
after trading has already started. Safe to run any time; read-only against the broker.

What the simulator assumes (training_v4.cpp / training_lib.py):
  BUY limit at L on fill-day bar (O,H,L,C):
      open <= L            -> fills at `open`            (gap, no slippage)
      low  <  L  <  high   -> fills at L*(1+SLIPPAGE)    (intraday touch, ALWAYS fills)
      otherwise            -> no fill
  SELL/stop mirror it with (1-SLIPPAGE).

The three questions:
  1. FILL RATE  — of orders the simulator says fill, what fraction actually did?
     (Suspected main defect: a price touched once intraday does not guarantee a fill.)
  2. FILL PRICE — actual filled_avg_price vs the simulated fill price.
  3. FALSE NEG  — did anything fill that the simulator said would not?

Usage:
  export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
  python fill_reconciliation.py --paper --days 30
  python fill_reconciliation.py --paper --days 30 --csv logs/acct0/fill_recon.csv
"""
import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict

SLIPPAGE_RATE = 0.001          # must mirror training_v4.cpp / training_lib.py
STOCK_DATA_DIR = 'stock_data'


def load_bars(symbols):
    """symbol -> {date: bar}. Uses the same stock_data the trainer reads."""
    out = {}
    for s in symbols:
        p = os.path.join(STOCK_DATA_DIR, f'{s}.json')
        if not os.path.exists(p):
            continue
        try:
            out[s] = {d['date']: d for d in json.load(open(p))['days']}
        except Exception:
            pass
    return out


def simulate(side, limit, bar):
    """What the trainer's fill model would do. Returns (fills, price, reason)."""
    o, h, low = bar['open'], bar['high'], bar['low']
    if limit is None:
        return True, o, 'market@open'
    if side == 'buy':
        if o <= limit:
            return True, o, 'gap@open'
        if low < limit < h:
            return True, limit * (1.0 + SLIPPAGE_RATE), 'touch@limit'
        return False, None, 'no-touch'
    if o >= limit:
        return True, o, 'gap@open'
    if low < limit < h:
        return True, limit * (1.0 - SLIPPAGE_RATE), 'touch@limit'
    return False, None, 'no-touch'


def next_session(bars_any, after_date):
    """First trading date strictly after `after_date` present in the data."""
    for d in bars_any:
        if d > after_date:
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paper', action='store_true', help='paper account (default: live)')
    ap.add_argument('--days', type=int, default=30, help='lookback window')
    ap.add_argument('--csv', default=None, help='write per-order rows here')
    args = ap.parse_args()

    key, sec = os.environ.get('ALPACA_API_KEY'), os.environ.get('ALPACA_SECRET_KEY')
    if not key or not sec:
        print('ERROR: set ALPACA_API_KEY and ALPACA_SECRET_KEY')
        return 2

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest

    client = TradingClient(key, sec, paper=args.paper)
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)
    req = GetOrdersRequest(status='all', after=since, limit=500, nested=False)
    orders = client.get_orders(req)
    print(f'fetched {len(orders)} order(s) since {since.date()} '
          f'({"paper" if args.paper else "LIVE"})')
    if not orders:
        print('nothing to reconcile yet — run again after some sessions have traded.')
        return 0

    syms = sorted({o.symbol for o in orders})
    bars = load_bars(syms)
    if not bars:
        print(f'ERROR: no {STOCK_DATA_DIR}/*.json for these symbols — run from the repo root')
        return 2
    alldates = sorted({d for m in bars.values() for d in m})

    rows = []
    for o in orders:
        sym = o.symbol
        if sym not in bars:
            continue
        # Orders are submitted after the close, so the fill session is the NEXT trading day.
        sub = o.submitted_at.date().isoformat()
        fd = sub if sub in bars[sym] and o.submitted_at.hour < 16 else next_session(alldates, sub)
        bar = bars[sym].get(fd)
        if not bar:
            continue
        limit = float(o.limit_price) if getattr(o, 'limit_price', None) else None
        stop = float(o.stop_price) if getattr(o, 'stop_price', None) else None
        trigger = limit if limit is not None else stop
        side = str(o.side).lower().split('.')[-1]
        pred_fill, pred_price, reason = simulate(side, trigger, bar)
        filled_qty = float(o.filled_qty or 0)
        act_fill = filled_qty > 0
        act_price = float(o.filled_avg_price) if o.filled_avg_price else None
        rows.append(dict(
            date=fd, symbol=sym, side=side, type=str(o.order_type).split('.')[-1],
            trigger=trigger, qty=float(o.qty or 0), filled_qty=filled_qty,
            status=str(o.status).split('.')[-1],
            open=bar['open'], high=bar['high'], low=bar['low'], close=bar['close'],
            pred_fill=int(pred_fill), pred_price=pred_price, pred_reason=reason,
            act_fill=int(act_fill), act_price=act_price,
            price_err=(act_price - pred_price) if (act_price and pred_price) else None))

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {len(rows)} rows -> {args.csv}')

    # ---- the three questions ----------------------------------------------------------
    print()
    print('=' * 72)
    print('1. FILL RATE — simulator says fill; did it?')
    for reason in ('gap@open', 'touch@limit', 'market@open'):
        g = [r for r in rows if r['pred_fill'] and r['pred_reason'] == reason]
        if g:
            hit = sum(r['act_fill'] for r in g)
            print(f'   {reason:<12} {hit:4d}/{len(g):<4d} = {100*hit/len(g):5.1f}% actually filled')
    g = [r for r in rows if r['pred_fill']]
    if g:
        hit = sum(r['act_fill'] for r in g)
        print(f'   {"OVERALL":<12} {hit:4d}/{len(g):<4d} = {100*hit/len(g):5.1f}%'
              '   (simulator assumes 100%)')

    print()
    print('2. FILL PRICE — actual minus simulated, on orders that filled')
    errs = [r['price_err'] / r['pred_price'] for r in rows
            if r['price_err'] is not None and r['pred_price']]
    if errs:
        errs.sort()
        n = len(errs)
        print(f'   n={n}  mean={1e4*sum(errs)/n:+.1f} bp  median={1e4*errs[n//2]:+.1f} bp'
              f'  p10={1e4*errs[n//10]:+.1f}  p90={1e4*errs[9*n//10]:+.1f}')
        print(f'   (simulator applies a flat {1e4*SLIPPAGE_RATE:.0f} bp; positive = we paid worse)')
    else:
        print('   no filled orders with a comparable simulated price yet')

    print()
    print('3. FALSE NEGATIVES — simulator says no fill, but it filled anyway')
    g = [r for r in rows if not r['pred_fill']]
    if g:
        bad = [r for r in g if r['act_fill']]
        print(f'   {len(bad)}/{len(g)} = {100*len(bad)/len(g):.1f}%')
        for r in bad[:5]:
            print(f'     {r["date"]} {r["symbol"]:<6} {r["side"]:<4} trig={r["trigger"]} '
                  f'bar L={r["low"]} H={r["high"]}')
    else:
        print('   none predicted')

    print()
    print('4. WHOLE-SHARE CHECK — the simulator trades fractional shares, production cannot')
    frac = [r for r in rows if r['qty'] and abs(r['qty'] - round(r['qty'])) > 1e-9]
    print(f'   orders with fractional qty submitted: {len(frac)} (expect 0)')
    byday = defaultdict(int)
    for r in rows:
        byday[r['date']] += 1
    print(f'   sessions covered: {len(byday)}  (need ~30 for a usable fill-rate sample)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
