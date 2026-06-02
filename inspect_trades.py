"""
inspect_trades.py — Trade audit tool for trained StockNN / MasterNN elite models.

Usage (by calendar date):
  python inspect_trades.py --industry tech_hardware --date 2024-11-15 --models-dir ./models
  python inspect_trades.py --industry energy --date 2024-11-15 --models-dir ./models --top-n 5

Usage (by training day index, matching training_v2.py --start-day):
  python inspect_trades.py --industry tech_hardware --day-index 17 --models-dir ./models
  python inspect_trades.py --industry energy --day-index 17 --models-dir ./models --top-n 5

What it does
------------
  1. Loads (or fetches) the 15-day price window for the requested date / day-index.
  2. Reads the top-N elite model slots from <prefix>_top10_meta.json.
  3. Runs each elite model through one forward pass (no mutation, no evolution).
  4. Prints a market context table for all 12 stocks showing day-N and fill-day OHLCV,
     price change %, and intraday range width.
  5. Computes an equal-weight passive baseline for comparison.
  6. For each elite model:
       • Sell and buy simulation with full fill-condition explanation
       • Cash utilisation (how much of the portfolio was actually deployed)
       • Concentration warning if any single stock > 50% of holdings
       • Gain attribution per executed buy
       • Portfolio value at fill-day close vs starting cash

Arguments
---------
  --industry     One of the 12 industry keys (e.g. tech_hardware, financials, energy …)
  --date         The trading day to inspect (YYYY-MM-DD).  Fills simulate against the
                 *next* available trading day, exactly as training does.
  --day-index    Alternative to --date: the day number as printed in the training log
                 (e.g. --day-index 17 inspects the day logged as "Day 17/1256").
  --models-dir   Directory containing the trained .pt files and top10_meta.json
                 (the --output directory you passed to training_v2.py).
  --top-n        How many elite models to show (default 3, max 10).
  --stock-data   Path to stock_data/ directory (default: ./stock_data).
  --starting-cash  Starting cash per industry portfolio (default: 1666.67, matches training).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf

from fees import BUY_FILL, FINRA_TAF_MAX, FINRA_TAF_PER_SHARE, SEC_FEE_RATE, SELL_FILL, SLIPPAGE_RATE, _sell_net
from models import StockNN
from universe import INDUSTRIES


# ── Data helpers ───────────────────────────────────────────────────────────────
def _model_path(prefix, directory, slot):
    """Return .pt file path for the given slot."""
    return os.path.join(directory, f"{prefix}_model_{slot}.pt")

def _meta_path(prefix, directory):
    """Return top10_meta.json path for the given prefix."""
    return os.path.join(directory, f"{prefix}_top10_meta.json")


def load_top10_meta(prefix, directory):
    """Load top-N elite metadata list from disk; returns [] if missing."""
    path = _meta_path(prefix, directory)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_model(prefix, directory, slot):
    """Load one StockNN from disk into eval mode; uses random weights if missing."""
    model = StockNN()
    path  = _model_path(prefix, directory, slot)
    if not os.path.exists(path):
        print(f"  [WARN] Model file not found: {path} — using random weights")
        return model
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_all_local_data(stock_data_dir, symbols):
    """Load all local stock_data JSON files and merge into a {date: {sym: ohlcv}} dict."""
    by_date = {}
    for sym in symbols:
        path = os.path.join(stock_data_dir, f"{sym}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        for e in payload.get('days', []):
            dt = e['date']
            if dt not in by_date:
                by_date[dt] = {}
            by_date[dt][sym] = {k: float(e[k]) for k in ('open', 'high', 'low', 'close', 'volume')}
    return by_date


def fetch_history(symbols, target_date_str, stock_data_dir, fetch_days=17):
    """
    Returns (history_dict, day_data, next_day_data, all_dates, target_idx).

    history_dict  : {sym: list of [open,close,high,low,vol, d_open,d_close,d_high,d_low,d_vol]}
                    the last entry is target_date_str's data (up to 15-day window)
    day_data      : {sym: {open,high,low,close,volume}}  for target_date
    next_day_data : same for the *next* available trading day (fill prices)
    all_dates     : sorted list of all available trading dates
    target_idx    : index of target_date_str within all_dates
    """
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")

    local_by_date = load_all_local_data(stock_data_dir, symbols)

    # Fall back to yfinance for any missing symbols
    covered = {sym for sym in symbols
               if any(sym in local_by_date.get(d, {}) for d in local_by_date)}
    missing = [s for s in symbols if s not in covered]
    if missing:
        print(f"  Fetching {len(missing)} symbols from yfinance …")
        start = (target_dt - timedelta(days=fetch_days + 10)).strftime('%Y-%m-%d')
        end   = (target_dt + timedelta(days=5)).strftime('%Y-%m-%d')
        try:
            raw = yf.download(missing, start=start, end=end, progress=False, auto_adjust=True)
            for sym in missing:
                try:
                    sym_df = raw if len(missing) == 1 else raw.xs(sym, axis=1, level=1)
                    for ts, row in sym_df.iterrows():
                        ds = ts.strftime('%Y-%m-%d')
                        if ds not in local_by_date:
                            local_by_date[ds] = {}
                        local_by_date[ds][sym] = {
                            'open':   float(row['Open']),
                            'high':   float(row['High']),
                            'low':    float(row['Low']),
                            'close':  float(row['Close']),
                            'volume': float(row['Volume']),
                        }
                except Exception as e:
                    print(f"  [WARN] Could not process yfinance data for {sym}: {e}")
        except Exception as e:
            print(f"  [WARN] yfinance batch download failed: {e}")

    all_dates = sorted(local_by_date.keys())
    if target_date_str not in all_dates:
        nearby = [d for d in all_dates
                  if d >= (target_dt - timedelta(days=5)).strftime('%Y-%m-%d')][:10]
        raise ValueError(
            f"No data found for {target_date_str}. "
            f"Available dates near that range: {nearby}")

    target_idx   = all_dates.index(target_date_str)
    window_dates = all_dates[max(0, target_idx - 14): target_idx + 1]   # up to 15 days
    next_idx     = target_idx + 1
    next_date    = all_dates[next_idx] if next_idx < len(all_dates) else None

    day_data      = local_by_date[target_date_str]
    next_day_data = local_by_date[next_date] if next_date else day_data

    # Build rolling histories (open,close,high,low,vol + 5 deltas)
    history = {sym: [] for sym in symbols}
    for d in window_dates:
        for sym in symbols:
            if sym not in local_by_date.get(d, {}):
                continue
            e    = local_by_date[d][sym]
            raw  = [e['open'], e['close'], e['high'], e['low'], e['volume']]
            prev = history[sym][-1][:5] if history[sym] else None
            deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
            history[sym].append(raw + deltas)

    return history, day_data, next_day_data, all_dates, target_idx


def resolve_date_from_index(day_index, stock_data_dir, all_symbols):
    """
    Map a training log day number to a calendar date.

    training_v2.py logs "Day N/..." where N = actual_day + 1 and actual_day = all_days index.
    So log day N → all_dates[N-1].  Pass the number you see in the training log here.
    """
    by_date = load_all_local_data(stock_data_dir, all_symbols)
    all_dates = sorted(by_date.keys())
    idx = day_index - 1   # log is 1-based: "Day 17" → all_dates[16]
    if idx < 0 or idx >= len(all_dates):
        raise ValueError(
            f"Day index {day_index} out of range — "
            f"valid range is 1–{len(all_dates)} ({len(all_dates)} trading days available).")
    return all_dates[idx], all_dates


# ── Feature builder (mirrors step_industry in training_v2.py) ─────────────────
def build_input(symbols, histories, day_data, cash, holdings):
    """Build history_t (1,15,60) and today_t (1,208) tensors for one forward pass.
    Mirrors the feature engineering in training_v2.step_industry exactly.
    Returns (history_t, today_t, sym_stats)."""
    sym_stats = {}
    for sym in symbols:
        h = histories[sym]
        if len(h) >= 2:
            closes = [r[1] for r in h]
            vols   = [r[4] for r in h]
            hi15   = max(r[2] for r in h)
            lo15   = min(r[3] for r in h)
            avg_c  = sum(closes) / len(closes)
            avg_v  = sum(vols)   / len(vols)   if sum(vols)   > 0 else 1.0
            dvols  = [r[0] * r[4] for r in h]
            avg_dv = sum(dvols)  / len(dvols)  if sum(dvols)  > 0 else 1.0
            std_c  = (sum((c - avg_c)**2 for c in closes) / len(closes)) ** 0.5
            sym_stats[sym] = {
                'hi15': hi15, 'lo15': lo15,
                'avg_c':  avg_c  if avg_c  > 0 else 1.0,
                'avg_v':  avg_v,
                'avg_dv': avg_dv,
                'volatility': std_c / avg_c if avg_c > 0 else 0.0,
            }
        else:
            sym_stats[sym] = {'hi15': 1.0, 'lo15': 0.0, 'avg_c': 1.0,
                              'avg_v': 1.0, 'avg_dv': 1.0, 'volatility': 0.0}

    # history_t: (1, 15, 60) — OHLCV × 12 stocks, oldest day first
    history_rows = []
    for t in range(14, -1, -1):          # t=14 oldest (h[-15]), t=0 most recent (h[-1])
        row = []
        for sym in symbols:
            h = histories[sym]
            row += list(h[-(t + 1)][:5]) if len(h) > t else [0.0] * 5
        history_rows.append(row)
    history_t = torch.tensor(history_rows, dtype=torch.float32).unsqueeze(0)  # (1,15,60)

    # today_t: (1, 208) — full feature set for current day
    state_vec = [cash] + [holdings.get(sym, 0.0) for sym in symbols]
    today_row = []
    today_dl  = []
    for sym in symbols:
        st    = sym_stats[sym]
        d     = day_data.get(sym, {})
        raw_t = [d.get('open', 0.0), d.get('close', 0.0),
                 d.get('high', 0.0), d.get('low', 0.0), d.get('volume', 0.0)]
        prev  = histories[sym][-1] if histories[sym] else None
        dlt_t = [raw_t[i] - prev[i] for i in range(5)] if prev else [0.0] * 5
        rng   = max(st['hi15'] - st['lo15'], 1e-9)
        today_row += (raw_t + dlt_t + [
            (raw_t[1] - st['lo15']) / rng,
            raw_t[1] / st['avg_c'],
            st['volatility'],
            raw_t[4] / st['avg_v'],
            (raw_t[0] * raw_t[4]) / st['avg_dv'],
        ])
        today_dl.append(dlt_t)
    if today_dl:
        tr = list(zip(*today_dl))
        for tp in tr:
            today_row += [max(tp), min(tp), sum(tp) / len(tp)]
    else:
        today_row += [0.0] * 15
    today_row += state_vec
    today_t = torch.tensor(today_row, dtype=torch.float32).unsqueeze(0)    # (1,208)

    return history_t, today_t, sym_stats


# ── Trade simulation ───────────────────────────────────────────────────────────
def simulate_trades(symbols, out_tensor, day_data, next_day_data, cash, holdings):
    """
    Replay the exact trade logic from step_industry.
    Returns (trade_records, final_cash, final_holdings, portfolio_value).

    portfolio_value uses next_day_data close (fill_data), matching how training_v2
    scores portfolios — both baseline and post-trade value are on the fill day so
    the delta reflects only the trading decision, not overnight price movement.
    """
    out  = out_tensor.view(12, 4)
    nd   = next_day_data

    cur_holdings = {s: holdings.get(s, 0.0) for s in symbols}
    cur_cash     = cash
    stop_prices  = {s: 0.0 for s in symbols}

    sell_records = {}

    # ── SELLS ──────────────────────────────────────────────────────────────
    for j, sym in enumerate(symbols):
        if sym not in day_data:
            sell_records[sym] = {'action': 'SKIP', 'reason': 'symbol missing from day_data'}
            continue

        buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = out[j].tolist()

        low_t  = day_data[sym]['low']
        high_t = day_data[sym]['high']
        span_t = max(high_t - low_t, 1e-9)
        sell_all_price = low_t + sell_all_price_frac * span_t

        nd_sym  = nd.get(sym, day_data[sym])
        nd_open = nd_sym.get('open',  day_data[sym]['close'])
        nd_low  = nd_sym.get('low',   day_data[sym]['low'])
        nd_high = nd_sym.get('high',  day_data[sym]['high'])

        cur_qty = cur_holdings[sym]
        rec = {
            'sym': sym,
            'raw_buy_qty':        buy_qty,
            'raw_buy_price_frac': buy_price_frac,
            'raw_sell_all_frac':  sell_all_price_frac,
            'raw_sell_qty':       sell_qty,
            'sell_all_limit':     sell_all_price,
            'day_low':            low_t,
            'day_high':           high_t,
            'nd_open':            nd_open,
            'nd_low':             nd_low,
            'nd_high':            nd_high,
            'holdings_before':    cur_qty,
            'sell_actions':       [],
            'buy_action':         None,
        }

        # Partial sell (sell_qty)
        if sell_qty > 1e-6 and cur_qty > 1e-6:
            sell_amount = min(sell_qty, cur_qty)
            net = _sell_net(sell_amount, nd_open)
            cur_holdings[sym] -= sell_amount
            cur_cash          += net
            rec['sell_actions'].append({
                'type':       'PARTIAL_SELL',
                'shares':     sell_amount,
                'fill_price': nd_open,
                'net_cash':   net,
                'reason':     (f"Sell-qty {sell_qty:.3f} sh requested; "
                               f"had {cur_qty:.4f} sh → selling {sell_amount:.4f} sh "
                               f"at next-day open ${nd_open:.2f}."),
            })
        elif sell_qty > 1e-6 and cur_qty <= 1e-6:
            rec['sell_actions'].append({
                'type':   'SELL_NO_EXEC',
                'reason': f"Model requested {sell_qty:.3f} sh sell but portfolio holds 0 shares.",
            })

        # Sell-all limit
        cur_qty_after = cur_holdings[sym]
        if cur_qty_after > 1e-6:
            if nd_open >= sell_all_price:
                net = _sell_net(cur_qty_after, nd_open)
                cur_holdings[sym] = 0.0
                cur_cash += net
                rec['sell_actions'].append({
                    'type':       'SELL_ALL_EXEC',
                    'shares':     cur_qty_after,
                    'fill_price': nd_open,
                    'net_cash':   net,
                    'reason':     (f"Sell-all limit ${sell_all_price:.2f}; "
                                   f"next-day open ${nd_open:.2f} ≥ limit → gapped above, filled at open."),
                })
            elif nd_low < sell_all_price < nd_high:
                slipped = sell_all_price * (1.0 - SLIPPAGE_RATE)
                net = _sell_net(cur_qty_after, slipped)
                cur_holdings[sym] = 0.0
                cur_cash += net
                rec['sell_actions'].append({
                    'type':       'SELL_ALL_EXEC',
                    'shares':     cur_qty_after,
                    'fill_price': slipped,
                    'net_cash':   net,
                    'reason':     (f"Sell-all limit ${sell_all_price:.2f}; "
                                   f"next-day range ${nd_low:.2f}–${nd_high:.2f} crossed limit → "
                                   f"filled at ${slipped:.2f} (−{SLIPPAGE_RATE*100:.2f}% slip)."),
                })
            else:
                rec['sell_actions'].append({
                    'type':   'SELL_ALL_NO_EXEC',
                    'reason': (f"Sell-all limit ${sell_all_price:.2f}; "
                               f"next-day range ${nd_low:.2f}–${nd_high:.2f} never reached limit."),
                })
        elif cur_qty_after <= 1e-6 and sell_all_price_frac > 0.05:
            rec['sell_actions'].append({
                'type':   'SELL_ALL_NO_EXEC',
                'reason': f"Sell-all limit ${sell_all_price:.2f} set but no holdings remain.",
            })

        sell_records[sym] = rec

    # ── BUYS ───────────────────────────────────────────────────────────────
    for j, sym in enumerate(symbols):
        if sym not in day_data:
            sell_records[sym]['buy_action'] = {'type': 'SKIP', 'reason': 'symbol missing'}
            continue

        buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = out[j].tolist()
        low_t  = day_data[sym]['low']
        high_t = day_data[sym]['high']
        span_t = max(high_t - low_t, 1e-9)
        buy_price = low_t + buy_price_frac * span_t
        stop_loss = buy_price * 0.9

        nd_sym  = nd.get(sym, day_data[sym])
        nd_open = nd_sym.get('open',  day_data[sym]['close'])
        nd_low  = nd_sym.get('low',   day_data[sym]['low'])
        nd_high = nd_sym.get('high',  day_data[sym]['high'])

        rec = sell_records[sym]

        if buy_qty <= 1e-6:
            rec['buy_action'] = {
                'type':   'NO_BUY',
                'reason': f"buy_qty={buy_qty:.6f} ≤ threshold — no buy order placed.",
            }
            continue

        if buy_price <= 0:
            rec['buy_action'] = {
                'type':   'NO_BUY',
                'reason': f"Computed buy limit ${buy_price:.4f} ≤ 0 — invalid price.",
            }
            continue

        if nd_open <= buy_price:
            fill_price  = nd_open
            fill_reason = f"Next-day open ${nd_open:.2f} ≤ buy limit ${buy_price:.2f} → gapped below, filled at open."
        elif nd_low < buy_price < nd_high:
            fill_price  = buy_price * (1.0 + SLIPPAGE_RATE)
            fill_reason = (f"Buy limit ${buy_price:.2f} hit intraday "
                           f"(range ${nd_low:.2f}–${nd_high:.2f}); "
                           f"filled at ${fill_price:.2f} (+{SLIPPAGE_RATE*100:.2f}% slip).")
        else:
            rec['buy_action'] = {
                'type':      'BUY_NO_EXEC',
                'buy_limit': buy_price,
                'buy_qty':   buy_qty,
                'reason':    (f"Buy limit ${buy_price:.2f} not reached; "
                              f"next-day range ${nd_low:.2f}–${nd_high:.2f} stayed above limit."),
            }
            continue

        affordable = cur_cash / (fill_price * BUY_FILL)
        buy_amount = min(buy_qty, affordable)

        if buy_amount <= 1e-6:
            rec['buy_action'] = {
                'type':      'BUY_NO_EXEC',
                'buy_limit': buy_price,
                'buy_qty':   buy_qty,
                'reason':    (f"Fill condition met (${fill_price:.2f}) but cash "
                              f"${cur_cash:.2f} insufficient for even 1 share."),
            }
            continue

        cost = buy_amount * fill_price * BUY_FILL
        cur_holdings[sym] += buy_amount
        cur_cash          -= cost
        stop_prices[sym]   = stop_loss

        rec['buy_action'] = {
            'type':         'BUY_EXEC',
            'buy_limit':    buy_price,
            'buy_qty_req':  buy_qty,
            'buy_qty_fill': buy_amount,
            'fill_price':   fill_price,
            'cost':         cost,
            'stop_loss':    stop_loss,
            'reason':       (fill_reason +
                             f" Requested {buy_qty:.4f} sh, "
                             f"affordable {affordable:.4f} sh → bought {buy_amount:.4f} sh for ${cost:.2f}. "
                             f"Stop-loss set at ${stop_loss:.2f}."),
        }

    # Portfolio value at fill-day close (matches training_v2 scoring convention)
    final_value = cur_cash + sum(
        cur_holdings[s] * next_day_data.get(s, day_data.get(s, {})).get('close', 0.0)
        for s in symbols
    )

    return list(sell_records.values()), cur_cash, cur_holdings, final_value, stop_prices


# ── Market context table ───────────────────────────────────────────────────────
def print_market_context(symbols, day_data, next_day_data, starting_cash,
                         date_str, fill_date_str):
    """
    Print a table of all 12 stocks showing day-N OHLCV and fill-day OHLCV.
    Also computes an equal-weight passive baseline.
    """
    sep  = "─" * 90
    sep2 = "═" * 90

    print(f"\n{sep2}")
    print(f"  MARKET CONTEXT  |  decision date: {date_str}  →  fill date: {fill_date_str}")
    print(sep2)
    print(f"  {'Symbol':6s}  {'Day-Close':>9s}  {'Fill-Open':>9s}  {'Fill-Low':>8s}  "
          f"{'Fill-High':>9s}  {'Fill-Close':>10s}  {'Chg%':>6s}  {'Range%':>7s}")
    print(f"  {sep}")

    eq_end_value  = 0.0
    n_eq          = sum(1 for s in symbols if s in day_data and day_data[s]['close'] > 0)
    per_sym_cash  = starting_cash / n_eq if n_eq else 0.0

    for sym in symbols:
        if sym not in day_data:
            print(f"  {sym:6s}  (no data)")
            continue
        d  = day_data[sym]
        nd = next_day_data.get(sym, d)

        d_close  = d['close']
        nd_open  = nd.get('open',  d_close)
        nd_low   = nd.get('low',   d['low'])
        nd_high  = nd.get('high',  d['high'])
        nd_close = nd.get('close', d_close)

        chg_pct  = (nd_close - d_close) / d_close * 100 if d_close > 0 else 0.0
        rng_pct  = (nd_high - nd_low)   / nd_close * 100 if nd_close > 0 else 0.0

        # Equal-weight baseline: buy shares at nd_open, value at nd_close
        if d_close > 0 and nd_open > 0:
            eq_shares = per_sym_cash / nd_open
            eq_end_value += eq_shares * nd_close
        else:
            eq_end_value += per_sym_cash   # no trade, cash unchanged

        chg_sign = "+" if chg_pct >= 0 else ""
        print(f"  {sym:6s}  "
              f"${d_close:8.2f}  "
              f"${nd_open:8.2f}  "
              f"${nd_low:7.2f}  "
              f"${nd_high:8.2f}  "
              f"${nd_close:9.2f}  "
              f"{chg_sign}{chg_pct:5.2f}%  "
              f"{rng_pct:6.2f}%")

    eq_delta = eq_end_value - starting_cash
    eq_pct   = eq_delta / starting_cash * 100 if starting_cash > 0 else 0.0
    sign     = "+" if eq_delta >= 0 else ""
    print(f"  {sep}")
    print(f"  Equal-weight passive baseline (buy all at fill-open, hold to fill-close):  "
          f"${starting_cash:.2f} → ${eq_end_value:.2f}  "
          f"({sign}${eq_delta:.2f} / {sign}{eq_pct:.2f}%)")
    print()


# ── Gain attribution ──────────────────────────────────────────────────────────
def compute_attribution(symbols, trade_records, next_day_data, day_data):
    """
    For each filled buy, compute: qty * (fill_close - fill_price) = unrealised P&L.
    Returns list of (sym, shares_bought, fill_price, fill_close, gain) sorted by |gain|.
    """
    rows = []
    for rec in trade_records:
        sym = rec['sym']
        ba  = rec.get('buy_action', {})
        if ba and ba.get('type') == 'BUY_EXEC':
            nd    = next_day_data.get(sym, day_data.get(sym, {}))
            close = nd.get('close', 0.0)
            qty   = ba['buy_qty_fill']
            fp    = ba['fill_price']
            gain  = qty * (close - fp)
            rows.append((sym, qty, fp, close, gain))
    rows.sort(key=lambda r: abs(r[4]), reverse=True)
    return rows


# ── Per-model report ───────────────────────────────────────────────────────────
def print_model_report(model_rank, slot, score, symbols, day_data, next_day_data,
                        trade_records, final_cash, final_holdings, portfolio_value,
                        starting_cash, date_str, fill_date_str):
    """Print the per-model trade detail block for one elite slot."""
    sep  = "─" * 90
    sep2 = "═" * 90

    deployed      = starting_cash - final_cash
    util_pct      = deployed / starting_cash * 100 if starting_cash > 0 else 0.0
    holdings_val  = portfolio_value - final_cash
    delta         = portfolio_value - starting_cash
    delta_sign    = "+" if delta >= 0 else ""

    print(f"\n{sep2}")
    print(f"  ELITE MODEL #{model_rank}  |  slot={slot}  |  training score=${score:.2f}")
    print(f"  Decision date : {date_str}   Fill date : {fill_date_str}")
    print(f"  Starting cash : ${starting_cash:.2f}")
    print(sep2)

    active_buys  = [r for r in trade_records if r.get('buy_action', {}).get('type') == 'BUY_EXEC']
    SELL_EXEC_TYPES = {'PARTIAL_SELL', 'SELL_ALL_EXEC'}
    active_sells = [r for r in trade_records
                    if any(a.get('type', '') in SELL_EXEC_TYPES for a in r.get('sell_actions', []))]

    print(f"\n  {len(active_buys)} buy(s) executed   |   {len(active_sells)} sell(s) executed   |   "
          f"cash utilisation: ${deployed:.2f} of ${starting_cash:.2f}  ({util_pct:.1f}%)")

    # Concentration check
    if holdings_val > 0:
        for sym in symbols:
            qty   = final_holdings.get(sym, 0.0)
            nd    = next_day_data.get(sym, day_data.get(sym, {}))
            close = nd.get('close', 0.0)
            val   = qty * close
            conc  = val / holdings_val * 100
            if conc > 50:
                print(f"  *** CONCENTRATION WARNING: {conc:.1f}% of holdings in {sym} alone ***")

    # Gain attribution table
    attrib = compute_attribution(symbols, trade_records, next_day_data, day_data)
    if attrib:
        print("\n  Gain attribution (fill price → fill-day close):")
        print(f"  {'Symbol':6s}  {'Shares':>8s}  {'Fill@':>8s}  {'Close':>8s}  {'Gain':>9s}  {'Gain%':>7s}")
        print(f"  {'─'*54}")
        total_attrib = 0.0
        for sym, qty, fp, close, gain in attrib:
            g_pct = (close - fp) / fp * 100 if fp > 0 else 0.0
            sign  = "+" if gain >= 0 else ""
            total_attrib += gain
            print(f"  {sym:6s}  {qty:8.4f}  ${fp:7.2f}  ${close:7.2f}  "
                  f"{sign}${gain:7.2f}  {sign}{g_pct:5.2f}%")
        sign = "+" if total_attrib >= 0 else ""
        print(f"  {'─'*54}")
        print(f"  {'Total':6s}  {'':8s}  {'':8s}  {'':8s}  {sign}${total_attrib:7.2f}")

    # Per-symbol detail
    print("\n  Per-symbol details:")
    print(f"  {sep}")

    for rec in trade_records:
        sym = rec['sym']
        if sym not in day_data:
            continue

        cd = day_data[sym]
        nd = next_day_data.get(sym, cd)

        print(f"  {sym:6s}  day-close=${cd['close']:.2f}  "
              f"day-range ${cd['low']:.2f}–${cd['high']:.2f}  "
              f"fill-range ${nd.get('low', cd['low']):.2f}–${nd.get('high', cd['high']):.2f}  "
              f"held-before={rec['holdings_before']:.4f} sh")
        print(f"         outputs → buy_qty={rec['raw_buy_qty']:.4f}  "
              f"buy_frac={rec['raw_buy_price_frac']:.4f}  "
              f"sell_all_frac={rec['raw_sell_all_frac']:.4f}  "
              f"sell_qty={rec['raw_sell_qty']:.4f}")

        for sa in rec.get('sell_actions', []):
            filled = sa['type'] in ('PARTIAL_SELL', 'SELL_ALL_EXEC')
            tag    = "✓ SOLD" if filled else "✗ SELL"
            print(f"         {tag}   {sa['reason']}")
            if filled:
                print(f"                  → {sa['shares']:.4f} sh @ ${sa['fill_price']:.2f} = +${sa['net_cash']:.2f} cash")

        ba = rec.get('buy_action')
        if ba:
            if ba['type'] == 'BUY_EXEC':
                print(f"         ✓ BUY    {ba['reason']}")
                print(f"                  → now holds {final_holdings.get(sym, 0):.4f} sh  |  stop @ ${ba['stop_loss']:.2f}")
            elif ba['type'] == 'BUY_NO_EXEC':
                print(f"         ✗ BUY    {ba['reason']}")
            elif ba['type'] == 'NO_BUY':
                print(f"         — NO BUY {ba['reason']}")

    print(f"\n  {sep}")
    print(f"  Portfolio value (fill-day close) : ${portfolio_value:.2f}  "
          f"({delta_sign}${delta:.2f} vs starting cash / {delta_sign}{delta/starting_cash*100:.2f}%)")
    print(f"  Cash remaining                   : ${final_cash:.2f}")
    print(f"  Holdings value                   : ${holdings_val:.2f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Parse CLI args, resolve the target date, and run the trade audit report."""
    parser = argparse.ArgumentParser(
        description="Inspect elite model trade decisions for a given industry + date.")
    parser.add_argument('--industry',      required=True,
                        help=f"Industry key. One of: {', '.join(INDUSTRIES)}")

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date',      help="Trading day to inspect (YYYY-MM-DD).")
    date_group.add_argument('--day-index', type=int,
                            help="Training log day number (the N in 'Day N/...' printed during training). "
                                 "E.g. --day-index 17 inspects the day logged as 'Day 17'.")

    parser.add_argument('--models-dir',    required=True,
                        help="Directory containing trained .pt files and top10_meta.json")
    parser.add_argument('--top-n',         type=int, default=3,
                        help="Number of top elite models to show (default 3, max 10)")
    parser.add_argument('--stock-data',    default='./stock_data',
                        help="Path to stock_data/ directory (default: ./stock_data)")
    parser.add_argument('--starting-cash', type=float, default=1666.67,
                        help="Starting cash per portfolio (default 1666.67)")
    args = parser.parse_args()

    if args.industry not in INDUSTRIES:
        sys.exit(f"Unknown industry '{args.industry}'. Valid: {list(INDUSTRIES)}")

    symbols = INDUSTRIES[args.industry]
    top_n   = max(1, min(args.top_n, 10))

    # Resolve date
    if args.day_index is not None:
        all_symbols_flat = [s for syms in INDUSTRIES.values() for s in syms]
        try:
            target_date, all_dates_global = resolve_date_from_index(
                args.day_index, args.stock_data, all_symbols_flat)
        except ValueError as e:
            sys.exit(f"ERROR: {e}")
        print(f"\n  Day index {args.day_index} → calendar date {target_date}")
    else:
        target_date = args.date

    print(f"\n{'═'*90}")
    print(f"  Trade Audit — {args.industry.upper()}  |  date: {target_date}")
    print(f"  Models dir : {args.models_dir}   |   Top {top_n} elite model(s)")
    print(f"{'═'*90}\n")

    # Load price data
    print("Loading market data …")
    try:
        histories, day_data, next_day_data, all_dates, target_idx = fetch_history(
            symbols, target_date, args.stock_data)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    # Determine fill date label
    fill_idx      = target_idx + 1
    fill_date_str = all_dates[fill_idx] if fill_idx < len(all_dates) else f"{target_date} (same day)"

    num_warm     = min((len(histories[s]) for s in symbols if histories[s]), default=0)
    missing_syms = [s for s in symbols if s not in day_data]
    print(f"  History window : {num_warm} day(s) warm (need 15 for full accuracy)")
    if missing_syms:
        print(f"  Missing symbols: {missing_syms}")

    # Market context table
    print_market_context(symbols, day_data, next_day_data, args.starting_cash,
                         target_date, fill_date_str)

    # Load elite meta
    meta = load_top10_meta(args.industry, args.models_dir)
    if not meta:
        print(f"  [WARN] No top10_meta.json found in {args.models_dir}. "
              f"Using slots 0–{top_n-1} with random/untrained weights.")
        meta = [{'slot': i, 'score': 0.0} for i in range(top_n)]

    # Run each elite model
    for rank, entry in enumerate(meta[:top_n], start=1):
        slot  = entry['slot']
        score = entry.get('score', 0.0)

        print(f"Loading elite model #{rank} (slot {slot}, training score ${score:.2f}) …")
        model = load_model(args.industry, args.models_dir, slot)

        cash     = args.starting_cash
        holdings = {sym: 0.0 for sym in symbols}

        history_t, today_t, _ = build_input(symbols, histories, day_data, cash, holdings)

        with torch.inference_mode():
            out = model(history_t, today_t)

        trade_records, final_cash, final_holdings, portfolio_value, _ = simulate_trades(
            symbols, out, day_data, next_day_data, cash, holdings)

        print_model_report(
            rank, slot, score, symbols,
            day_data, next_day_data,
            trade_records, final_cash, final_holdings, portfolio_value,
            args.starting_cash, target_date, fill_date_str)

    print("Done.\n")


if __name__ == '__main__':
    main()
