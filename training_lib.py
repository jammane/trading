"""
training_lib.py — Shared evolutionary training functions used by
upkeep.py and production_v2.py.  Not a standalone training script.
"""

import copy
import gc
import json
import math
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fees import BUY_FILL, FINRA_TAF_MAX, FINRA_TAF_PER_SHARE, SEC_FEE_RATE, SELL_FILL, SLIPPAGE_RATE, _sell_net
from models import MasterNN, StockNN
from universe import INDUSTRIES


DUMP_DIR = 'data_dump'

N_SLOTS               = 200         # total model slots per pool (elites + mutations)
ELITE_COUNT           = 17          # direct elite slots (0 … ELITE_COUNT-1)
WAVG_COUNT            = 3           # weighted-average slots (w5, w10, w15)
ELITE_POOL            = ELITE_COUNT + WAVG_COUNT   # 20 — all parent slots
MUTATIONS_PER_PARENT  = 9           # each of the ELITE_POOL parents gets this many children
# Layout: 0–16 direct elites | 17 w5 | 18 w10 | 19 w15 | 20–199 mutations (9 per parent)

IND_STARTING_CASH     = 25_000.0    # per-industry portfolio starting capital
MST_STARTING_CASH     = 300_000.0   # master starting capital (12 × IND_STARTING_CASH)
IND_UNIT_PRICE        = 25_000.0    # fixed price per industry "unit" in master's portfolio
MAX_SINGLE_STOCK_PCT  = 0.60        # no single stock may exceed 60% of portfolio value

MASTER_LOOKBACKS     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 90]
MASTER_POLY3_WINDOWS = [10, 30, 60, 90]
MASTER_START_DAY     = 30
TIER_WEIGHTS         = {1: 1.0, 2: 1.5, 3: 2.25}
_NULL_DENOM          = TIER_WEIGHTS[1] + TIER_WEIGHTS[2] + TIER_WEIGHTS[3]  # 4.75

HIST_DAYS    = 5
HIST_PER_DAY = 10
HIST_ELITE   = 7   # top-7 direct elite slots saved per day
HIST_WAVG    = 3   # wavg slots (17,18,19) saved per day


class HardFlagError(Exception):
    """Raised when a HARD-flagged day's gain exceeds the configured threshold."""


def _dump_day(actual_day, prefix, flag_type, baseline, best_delta, pct_gain, scores, day_data, fill_data, models=None):
    """Write diagnostic JSON for a flagged day to data_dump/day_{N}/."""
    day_dir = os.path.join(DUMP_DIR, f"day_{actual_day + 1}")
    os.makedirs(day_dir, exist_ok=True)
    payload = {
        'industry':    prefix,
        'day':         actual_day + 1,
        'flag':        flag_type,
        'baseline':    round(baseline, 4),
        'best_delta':  round(best_delta, 4),
        'pct_gain':    round(pct_gain, 4),
        'scores':      [[s, round(v, 4)] for s, v in sorted(scores, key=lambda x: x[1], reverse=True)],
        'day_data':    day_data,
        'fill_data':   fill_data,
    }
    if models is not None:
        payload['models'] = models
    path = os.path.join(day_dir, f"{prefix}.json")
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def _fmt_slot(slot):
    """Convert a slot index to the elite.mutation display label."""
    if slot < ELITE_COUNT:
        return f"{slot}.0"
    if slot == ELITE_COUNT:
        return "w5"
    if slot == ELITE_COUNT + 1:
        return "w10"
    if slot == ELITE_COUNT + 2:
        return "w15"
    parent   = (slot - ELITE_POOL) // MUTATIONS_PER_PARENT
    mutation = (slot - ELITE_POOL) % MUTATIONS_PER_PARENT + 1
    return f"{parent}.{mutation}"


def _build_models_section(
    yesterday_elite_slots, today_elite_slots,
    ref_cash, ref_hold,
    portfolios, scores,
    baseline_score,
    symbols,
    hard_threshold, soft_threshold,
    price_start,   # {sym: start-of-day price}
    price_end,     # {sym: end-of-day price}
    actual_perf=None,  # master only: {ind: perf_ratio} used to back-calc pre-perf holdings
):
    """
    Build the 'models' section for the dump covering yesterday's and today's elites.

    Industry: symbols = stock tickers; prices from day_data/fill_data['close'].
    Master:   symbols = industry keys;  price_start = industry baseline value,
              price_end = industry slot-0 end value; actual_perf required.
    """
    scores_dict = {s: v for s, v in scores}
    all_slots   = sorted(set(yesterday_elite_slots) | set(today_elite_slots))

    models = {}
    for slot in all_slots:
        port       = portfolios[slot]
        end_cash   = port['cash']
        slot_score = scores_dict.get(slot, baseline_score)

        pct_increase = (slot_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0.0

        if pct_increase >= hard_threshold:
            is_flagged = 'hard'
        elif pct_increase >= soft_threshold:
            is_flagged = 'soft'
        else:
            is_flagged = 'false'

        # For master, actual_perf is baked into holdings — back-calculate trade-only holdings.
        if actual_perf is not None:
            trade_hold = {
                ind: port['holdings'].get(ind, 0.0) / max(1.0 + actual_perf.get(ind, 0.0), 1e-9)
                for ind in symbols
            }
        else:
            trade_hold = port['holdings']

        unchanged_syms = {}
        purchases_syms = {}
        sales_syms     = {}

        for sym in symbols:
            start_sh = ref_hold.get(sym, 0.0)
            end_sh   = trade_hold.get(sym, 0.0)
            delta    = end_sh - start_sh
            if abs(delta) < 1e-6:
                if start_sh > 1e-6:
                    unchanged_syms[sym] = round(start_sh, 6)
            elif delta > 1e-6:
                purchases_syms[sym] = round(delta, 6)
            else:
                sales_syms[sym] = round(abs(delta), 6)

        def _grp_vals(sym_shares):
            sv = sum(sh * price_start.get(sym, 0.0) for sym, sh in sym_shares.items())
            ev = sum(sh * price_end.get(sym, 0.0)   for sym, sh in sym_shares.items())
            return sv, ev

        unch_sv, unch_ev   = _grp_vals(unchanged_syms)
        purch_sv, purch_ev = _grp_vals(purchases_syms)
        sales_sv, sales_ev = _grp_vals(sales_syms)

        def _pct(chg, sv):
            return chg / sv * 100 if abs(sv) > 1e-9 else 0.0

        unch_dict = dict(unchanged_syms)
        unch_dict['unchanged_holdings_pct_value_change_today'] = f"{_pct(unch_ev - unch_sv, unch_sv):+.2f}%"
        unch_dict['unchanged_holdings_value_change_today']     = f"${unch_ev - unch_sv:+.2f}"

        purch_dict = dict(purchases_syms)
        purch_dict['purchases_pct_value_change_today'] = f"{_pct(purch_ev - purch_sv, purch_sv):+.2f}%"
        purch_dict['purchases_value_change_today']     = f"${purch_ev - purch_sv:+.2f}"

        sales_dict = dict(sales_syms)
        sales_dict['sales_pct_value_change_today'] = f"{_pct(sales_ev - sales_sv, sales_sv):+.2f}%"
        sales_dict['sales_value_change_today']     = f"${sales_ev - sales_sv:+.2f}"

        models[_fmt_slot(slot)] = {
            'isFlagged':             is_flagged,
            'total_pct_increase':    f"{pct_increase:+.2f}%",
            'starting_cash_on_hand': f"${ref_cash:.2f}",
            'ending_cash_on_hand':   f"${end_cash:.2f}",
            'unchanged_holdings':    unch_dict,
            'purchases':             purch_dict,
            'sales':                 sales_dict,
        }

    return models


# ── Daily burst helpers ────────────────────────────────────────────────────────

def _fill_sym_from_bars(sym, port, bars, buy_qty, buy_price, sell_all_price,
                        sell_qty, stop_loss, symbols, fill_data, day_data):
    """
    Walk 1-minute intraday bars for one symbol's fill simulation.
    Replaces the OHLC + seq_flags approximation with chronological bar ordering.
    Within each bar: low-based events (stop, limit buy) checked before high-based (sell_all).
    Modifies port in place. Returns (buys, sells).
    """
    buys = sells = 0.0
    if not bars:
        return buys, sells

    nd_open = bars[0]['open']

    # Open: partial sell
    cur_qty = port['holdings'].get(sym, 0.0)
    if sell_qty > 1e-6 and cur_qty > 1e-6:
        sell_amount = min(sell_qty, cur_qty)
        port['holdings'][sym] -= sell_amount
        port['cash'] += _sell_net(sell_amount, nd_open)
        sells += sell_amount

    # Open: gap sell_all
    cur_qty = port['holdings'].get(sym, 0.0)
    if cur_qty > 1e-6 and nd_open >= sell_all_price:
        port['holdings'][sym] = 0.0
        port['cash'] += _sell_net(cur_qty, nd_open)
        sells += cur_qty

    # Open: gap stop
    stop_p = port['stop_prices'].get(sym, 0.0)
    remaining = port['holdings'].get(sym, 0.0)
    if stop_p > 0 and remaining > 1e-6 and nd_open <= stop_p:
        port['holdings'][sym] = 0.0
        port['cash'] += _sell_net(remaining, nd_open)
        sells += remaining

    # Open: gap buy
    already_bought = False
    if buy_qty > 1e-6 and buy_price > 0 and nd_open <= buy_price:
        fill_price = nd_open
        affordable = port['cash'] / (fill_price * BUY_FILL)
        buy_amount = min(buy_qty, affordable)
        if buy_amount > 1e-6:
            port_value = port['cash'] + sum(
                port['holdings'].get(s, 0.0)
                * fill_data.get(s, day_data.get(s, {})).get('close', 0.0)
                for s in symbols)
            cur_sym_val = port['holdings'].get(sym, 0.0) * fill_price
            max_spend = max(0.0, MAX_SINGLE_STOCK_PCT * port_value - cur_sym_val)
            buy_amount = min(buy_amount, max_spend / (fill_price * BUY_FILL))
        if buy_amount > 1e-6:
            port['holdings'][sym] = port['holdings'].get(sym, 0.0) + buy_amount
            port['cash'] -= buy_amount * fill_price * BUY_FILL
            port['stop_prices'][sym] = stop_loss
            buys += buy_amount
            already_bought = True

    # Intraday: walk remaining bars chronologically
    # Within each bar: check low (stop, buy) before high (sell_all)
    for bar in bars[1:]:
        bar_low  = bar['low']
        bar_high = bar['high']

        stop_p    = port['stop_prices'].get(sym, 0.0)
        remaining = port['holdings'].get(sym, 0.0)
        if stop_p > 0 and remaining > 1e-6 and bar_low <= stop_p:
            port['holdings'][sym] = 0.0
            port['cash'] += _sell_net(remaining, stop_p * (1.0 - SLIPPAGE_RATE))
            sells += remaining
            break

        if not already_bought and buy_qty > 1e-6 and buy_price > 0 and bar_low <= buy_price:
            fill_price = buy_price * (1.0 + SLIPPAGE_RATE)
            affordable = port['cash'] / (fill_price * BUY_FILL)
            buy_amount = min(buy_qty, affordable)
            if buy_amount > 1e-6:
                port_value = port['cash'] + sum(
                    port['holdings'].get(s, 0.0)
                    * fill_data.get(s, day_data.get(s, {})).get('close', 0.0)
                    for s in symbols)
                cur_sym_val = port['holdings'].get(sym, 0.0) * fill_price
                max_spend = max(0.0, MAX_SINGLE_STOCK_PCT * port_value - cur_sym_val)
                buy_amount = min(buy_amount, max_spend / (fill_price * BUY_FILL))
            if buy_amount > 1e-6:
                port['holdings'][sym] = port['holdings'].get(sym, 0.0) + buy_amount
                port['cash'] -= buy_amount * fill_price * BUY_FILL
                port['stop_prices'][sym] = stop_loss
                buys += buy_amount
                already_bought = True

        cur_qty = port['holdings'].get(sym, 0.0)
        if cur_qty > 1e-6 and bar_high >= sell_all_price:
            slipped = sell_all_price * (1.0 - SLIPPAGE_RATE)
            port['holdings'][sym] = 0.0
            port['cash'] += _sell_net(cur_qty, slipped)
            sells += cur_qty
            break

    return buys, sells


def _simulate_one_model(model, ref_cash, ref_hold, ref_stop, symbols,
                        day_data, fill_data, history_t, today_t, seq_flags=None,
                        intraday_bars=None):
    """Run one model for one day starting from the reference portfolio. Returns (score, port_dict)."""
    port = {'cash': ref_cash, 'holdings': dict(ref_hold), 'stop_prices': dict(ref_stop)}
    with torch.inference_mode():
        out = model(history_t, today_t)
    out = out.view(len(symbols), 4)

    # ── Phase 1: open + low phase ────────────────────────────────────────────
    # Partial sells at open, gap sell_all, high-first intraday sell_all, stops, limit buys.
    for j, sym in enumerate(symbols):
        if sym not in day_data:
            continue
        buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = out[j].tolist()
        cur_qty        = port['holdings'].get(sym, 0.0)
        low_t          = day_data[sym]['low']
        high_t         = day_data[sym]['high']
        span_t         = max(high_t - low_t, 1e-9)
        sell_all_price = low_t + sell_all_price_frac * span_t
        buy_price      = low_t + buy_price_frac * span_t
        stop_loss      = buy_price * 0.9
        low_first      = seq_flags.get(sym, True) if seq_flags else True
        nd             = fill_data.get(sym, day_data.get(sym, {}))
        nd_open        = nd.get('open',  day_data[sym]['close'])
        nd_low         = nd.get('low',   day_data[sym]['low'])
        nd_high        = nd.get('high',  day_data[sym]['high'])

        sym_bars = (intraday_bars or {}).get(sym)
        if sym_bars:
            b, s = _fill_sym_from_bars(sym, port, sym_bars, buy_qty, buy_price,
                                       sell_all_price, sell_qty, stop_loss,
                                       symbols, fill_data, day_data)
            continue

        # Partial sell at open
        if sell_qty > 1e-6 and cur_qty > 1e-6:
            sell_amount            = min(sell_qty, cur_qty)
            port['holdings'][sym] -= sell_amount
            port['cash']          += _sell_net(sell_amount, nd_open)

        # Gap-up sell_all at open (both sequences)
        cur_qty_after = port['holdings'].get(sym, 0.0)
        if cur_qty_after > 1e-6 and nd_open >= sell_all_price:
            port['holdings'][sym] = 0.0
            port['cash']         += _sell_net(cur_qty_after, nd_open)

        # High-first: intraday sell_all fires before the buy
        if not low_first:
            intraday_qty = port['holdings'].get(sym, 0.0)
            if intraday_qty > 1e-6 and nd_low < sell_all_price < nd_high:
                slipped = sell_all_price * (1.0 - SLIPPAGE_RATE)
                port['holdings'][sym] = 0.0
                port['cash']         += _sell_net(intraday_qty, slipped)

        # Stop loss
        stop_p    = port['stop_prices'].get(sym, 0.0)
        remaining = port['holdings'].get(sym, 0.0)
        if stop_p > 0 and remaining > 1e-6:
            if nd_open <= stop_p:
                port['holdings'][sym] = 0.0
                port['cash']         += _sell_net(remaining, nd_open)
            elif nd_low <= stop_p:
                port['holdings'][sym] = 0.0
                port['cash']         += _sell_net(remaining, stop_p * (1.0 - SLIPPAGE_RATE))

        # Limit buy
        if buy_qty > 1e-6 and buy_price > 0:
            if nd_open <= buy_price:
                fill_price = nd_open
            elif nd_low < buy_price < nd_high:
                fill_price = buy_price * (1.0 + SLIPPAGE_RATE)
            else:
                fill_price = 0.0
            if fill_price > 0:
                affordable = port['cash'] / (fill_price * BUY_FILL)
                buy_amount = min(buy_qty, affordable)
                if buy_amount > 1e-6:
                    port_value    = port['cash'] + sum(
                        port['holdings'].get(s, 0.0)
                        * fill_data.get(s, day_data.get(s, {})).get('close', 0.0)
                        for s in symbols)
                    cur_sym_value = port['holdings'].get(sym, 0.0) * fill_price
                    max_sym_spend = max(0.0, MAX_SINGLE_STOCK_PCT * port_value - cur_sym_value)
                    buy_amount    = min(buy_amount, max_sym_spend / (fill_price * BUY_FILL))
                if buy_amount > 1e-6:
                    port['holdings'][sym]     = port['holdings'].get(sym, 0.0) + buy_amount
                    port['cash']             -= buy_amount * fill_price * BUY_FILL
                    port['stop_prices'][sym]  = stop_loss

    # ── Phase 2: high phase — low-first intraday sell_all ───────────────────
    for j, sym in enumerate(symbols):
        if sym not in day_data:
            continue
        if (intraday_bars or {}).get(sym):
            continue  # bar-walk already handled this symbol
        if seq_flags and not seq_flags.get(sym, True):
            continue  # high-first: sell_all already handled in phase 1
        _, _, sell_all_price_frac, _ = out[j].tolist()
        low_t          = day_data[sym]['low']
        high_t         = day_data[sym]['high']
        span_t         = max(high_t - low_t, 1e-9)
        sell_all_price = low_t + sell_all_price_frac * span_t
        nd      = fill_data.get(sym, day_data.get(sym, {}))
        nd_low  = nd.get('low',  day_data[sym]['low'])
        nd_high = nd.get('high', day_data[sym]['high'])
        cur_qty = port['holdings'].get(sym, 0.0)
        if cur_qty > 1e-6 and nd_low < sell_all_price < nd_high:
            slipped = sell_all_price * (1.0 - SLIPPAGE_RATE)
            port['holdings'][sym] = 0.0
            port['cash']         += _sell_net(cur_qty, slipped)

    return compute_value(port, fill_data, symbols), port


def _run_daily_burst(prefix, output_dir, symbols, burst_sigma,
                     ref_cash, ref_hold, ref_stop,
                     day_data, fill_data, history_t, today_t,
                     portfolios, actual_day, total_avail, seq_flags=None,
                     intraday_bars=None):
    """
    Generate 200 burst mutants (10 per elite from ELITE_POOL parents), merge
    top-ELITE_COUNT with current elites, save winners to disk, update portfolios.
    Each burst explores at a finer sigma than the main pool.
    Cap: at most 2 burst candidates replace existing elites per call (prevents
    wholesale displacement of the trained set).
    Returns new best score after merge.
    """
    burst_candidates = []
    for elite_slot in range(ELITE_POOL):
        parent = load_slot_model(prefix, output_dir, elite_slot, StockNN)
        for _ in range(10):
            child       = mutate(parent, sigma=burst_sigma)
            score, port = _simulate_one_model(
                child, ref_cash, ref_hold, ref_stop,
                symbols, day_data, fill_data, history_t, today_t,
                seq_flags=seq_flags, intraday_bars=intraday_bars)
            burst_candidates.append((child, score, port))
        del parent
    burst_candidates.sort(key=lambda x: x[1], reverse=True)
    top_burst = burst_candidates[:ELITE_COUNT]

    current_elites = []
    for rank in range(ELITE_COUNT):
        model = load_slot_model(prefix, output_dir, rank, StockNN)
        score = compute_value(portfolios[rank], fill_data, symbols)
        current_elites.append((model, score, copy.deepcopy(portfolios[rank])))

    # Merge burst top-ELITE_COUNT with current top-ELITE_COUNT; pick best ELITE_COUNT
    # Cap burst replacements at 2 per call to prevent wholesale elite displacement.
    burst_ids = {id(m) for m, _, _ in top_burst}
    all_candidates = current_elites + top_burst
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    new_elites  = []
    burst_count = 0
    for candidate in all_candidates:
        if len(new_elites) >= ELITE_COUNT:
            break
        if id(candidate[0]) in burst_ids:
            if burst_count >= 2:
                continue
            burst_count += 1
        new_elites.append(candidate)

    prev_best = current_elites[0][1]
    new_best  = new_elites[0][1]
    if new_best > prev_best:
        log(f"[{sn(prefix)}]   Burst σ={burst_sigma:.6f}: "
            f"{burst_count} burst model(s) entered elite, best ${new_best:.2f} (+${new_best - prev_best:.2f})")
    else:
        log(f"[{sn(prefix)}]   Burst σ={burst_sigma:.6f}: best ${new_best:.2f} — no improvement")

    for rank, (model, score, port) in enumerate(new_elites):
        save_slot_model(prefix, output_dir, rank, model)
        portfolios[rank] = port

    # Regenerate weighted-average slots 17..19 from updated elites
    new_scores = [s for _, s, _ in new_elites]
    for n_avg, wavg_slot in [(5, ELITE_COUNT), (10, ELITE_COUNT + 1), (15, ELITE_COUNT + 2)]:
        k      = min(n_avg, ELITE_COUNT)
        wavg_m = compute_weighted_avg_model(prefix, output_dir, list(range(k)), new_scores[:k], StockNN)
        wavg_p = compute_weighted_avg_portfolio([portfolios[i] for i in range(k)], new_scores[:k])
        save_slot_model(prefix, output_dir, wavg_slot, wavg_m)
        portfolios[wavg_slot] = wavg_p
        del wavg_m

    del burst_candidates, current_elites, top_burst, all_candidates, new_elites
    gc.collect()
    return new_best


# ── Post-training environment management ──────────────────────────────────────

def cleanup_dev_mutations(prefix, dev_dir):
    """Delete mutation slot files (slots ELITE_POOL..N_SLOTS-1) from dev, keeping only elites."""
    removed = 0
    for slot in range(ELITE_POOL, N_SLOTS):
        path = _model_path(prefix, dev_dir, slot)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
    if removed:
        log(f"[{sn(prefix)}] Dev cleanup: removed {removed} mutation file(s) from {dev_dir}")


def promote_models(from_dir, to_dir, prefixes):
    """Copy elite slot files, best model, and metadata from from_dir to to_dir."""
    import shutil
    os.makedirs(to_dir, exist_ok=True)
    for prefix in prefixes:
        for slot in range(ELITE_POOL):
            src = _model_path(prefix, from_dir, slot)
            if os.path.exists(src):
                shutil.copy2(src, _model_path(prefix, to_dir, slot))
        for fname in [f"{prefix}_best.pt", f"{prefix}_top10_meta.json"]:
            src = os.path.join(from_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(to_dir, fname))
    log(f"Promoted models: {from_dir} → {to_dir}")


# ── Industry display name mapping (log output only) ───────────────────────────
SHORT_NAMES = {
    'tech_hardware':          'hardware',
    'tech_software_ai':       'software',
    'financials':             'financial',
    'consumer_discretionary': 'discret',
    'consumer_services':      'services',
    'health_care':            'health',
    'industrials':            'industrl',
    'consumer_staples':       'staples',
    'energy':                 'energy',
    'utilities':              'utilitie',
    'real_estate':            'land',
    'materials':              'materials',
}


def sn(industry):
    """Return the short display name for an industry key, padded to exactly 9 chars."""
    return (SHORT_NAMES.get(industry) or industry[:9]).ljust(9)[:9]


_console_log_lines = []
_console_log_path  = None   # set at startup by main()
_CONSOLE_LOG_MAX   = 200


def log(msg):
    """Timestamped, immediately-flushed console output — rolling 200-line file buffer."""
    global _console_log_lines
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _console_log_lines.append(line)
    if len(_console_log_lines) > _CONSOLE_LOG_MAX:
        _console_log_lines = _console_log_lines[-_CONSOLE_LOG_MAX:]
    if _console_log_path:
        try:
            with open(_console_log_path, 'w') as f:
                f.write('\n'.join(_console_log_lines) + '\n')
        except Exception:
            pass


# ── Evolution helpers ──────────────────────────────────────────────────────────

def mutate(model, sigma=0.01):
    """Return a new model with Gaussian noise added to all weights and biases."""
    state = copy.deepcopy(model.state_dict())
    for key in state:
        if 'weight' in key or 'bias' in key:
            state[key] += torch.randn_like(state[key]) * sigma
    new_model = StockNN() if isinstance(model, StockNN) else MasterNN()
    new_model.load_state_dict(state)
    del state
    return new_model


# ── Portfolio helpers ──────────────────────────────────────────────────────────

def compute_value(portfolio, day_data, symbols):
    """Return total portfolio value: cash plus all holdings at day_data close prices."""
    val = portfolio['cash']
    for sym in symbols:
        qty = portfolio['holdings'].get(sym, 0.0)
        if sym in day_data and day_data[sym]['close'] > 0:
            val += qty * day_data[sym]['close']
    return val


def _mst_hist_at(history, lookback):
    """Return the history value at `lookback` steps ago; clamps to oldest entry."""
    idx = len(history) - 1 - lookback
    return history[max(0, idx)]

def _mst_window(history, n_days):
    """Return a list of exactly n_days values, padding with the oldest entry if history is short."""
    oldest = history[0] if history else 0.0
    pad    = max(0, n_days - len(history))
    return [oldest] * pad + list(history[-n_days:])

def build_master_features(ind_value_history, industry_list):
    """Build (1, 444) input tensor. 37 features × 12 industries = 444."""
    features = []
    for ind in industry_list:
        hist = ind_value_history.get(ind, [])
        for t in MASTER_LOOKBACKS:
            v_now  = _mst_hist_at(hist, t)
            v_prev = _mst_hist_at(hist, t + 1)
            denom  = abs(v_prev) if abs(v_prev) > 1e-9 else 1e-9
            features.append((v_now - v_prev) / denom)
        vals5 = _mst_window(hist, 5)
        x5    = np.linspace(0.0, 1.0, 5)
        features.extend(np.polyfit(x5, vals5, 2).tolist())
        for n in MASTER_POLY3_WINDOWS:
            vals = _mst_window(hist, n)
            xn   = np.linspace(0.0, 1.0, n)
            features.extend(np.polyfit(xn, vals, 3).tolist())
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

def decode_master_tiers(out_logits, industry_list):
    """Convert raw (1,48) MasterNN logits to {ind: tier} where tier ∈ {0,1,2,3}."""
    logits = out_logits.view(12, 4)
    probs  = F.softmax(logits, dim=1)
    tiers  = probs.argmax(dim=1).tolist()
    return {ind: tiers[i] for i, ind in enumerate(industry_list)}

def tiers_to_alloc(tier_map, industry_list, available_cash):
    """
    Convert a {ind: tier} map to {ind: dollar_amount} allocation.

    Industries with tier 0 receive $0. Positive-tier industries are divided
    into three terciles (lowest→tier 1, mid→tier 2, top→tier 3) and weighted
    by TIER_WEIGHTS. Returns {ind: 0.0} for all if no positive-tier industries.
    """
    positives = sorted([ind for ind in industry_list if tier_map[ind] > 0],
                       key=lambda ind: tier_map[ind])
    n_pos = len(positives)
    alloc = {ind: 0.0 for ind in industry_list}
    if n_pos == 0:
        return alloc
    if n_pos == 1:
        alloc[positives[0]] = (TIER_WEIGHTS[3] / _NULL_DENOM) * available_cash
    elif n_pos == 2:
        alloc[positives[0]] = (TIER_WEIGHTS[2] / _NULL_DENOM) * available_cash
        alloc[positives[1]] = (TIER_WEIGHTS[3] / _NULL_DENOM) * available_cash
    else:
        base = n_pos // 3; rem = n_pos % 3
        n1 = base + (1 if rem >= 1 else 0)
        n2 = base + (1 if rem >= 2 else 0)
        tier_assignments = {}
        for rank, ind in enumerate(positives):
            if rank < n1:          tier_assignments[ind] = 1
            elif rank < n1 + n2:   tier_assignments[ind] = 2
            else:                  tier_assignments[ind] = 3
        total_w = sum(TIER_WEIGHTS[tier_assignments[ind]] for ind in positives)
        for ind in positives:
            alloc[ind] = (TIER_WEIGHTS[tier_assignments[ind]] / total_w) * available_cash
    return alloc


def _optimal_tiers(actual_perf, industry_list):
    """
    Compute the ideal tier assignment given actual industry returns.

    Divides positive-return industries into terciles (same logic as tiers_to_alloc)
    and assigns tiers 1/2/3 bottom-to-top. Zero/negative-return industries get tier 0.
    Used by MT2 scoring to build the oracle target for _master_points.
    """
    positives = sorted(
        [ind for ind in industry_list if actual_perf.get(ind, 0.0) >= 0],
        key=lambda ind: actual_perf.get(ind, 0.0)
    )
    n_pos = len(positives)
    opt = {ind: 0 for ind in industry_list}
    if n_pos == 1:
        opt[positives[0]] = 3
    elif n_pos == 2:
        opt[positives[0]] = 2
        opt[positives[1]] = 3
    elif n_pos >= 3:
        base = n_pos // 3; rem = n_pos % 3
        n1 = base + (1 if rem >= 1 else 0)
        n2 = base + (1 if rem >= 2 else 0)
        for rank, ind in enumerate(positives):
            if rank < n1:          opt[ind] = 1
            elif rank < n1 + n2:   opt[ind] = 2
            else:                  opt[ind] = 3
    return opt


def _master_points(pred, opt):
    """
    Score one industry's tier prediction against the optimal tier.

    Correct predictions score positive (equal to the tier).
    Predicting non-zero when optimal is 0 scores -2 minus a penalty.
    Predicting 0 when optimal is positive scores -(optimal).
    Over-predicting the tier penalises at 0.25 per tier of overshoot.
    """
    if opt == 0:
        if pred == 0:
            return 0.0
        return -2.0 - 0.25 * pred
    if pred == 0:
        return -float(opt)
    if pred <= opt:
        return float(pred)
    return float(opt) - 0.25 * (pred - opt)


def compute_alloc_from_predicted(predicted, industry_list):
    """
    Decode MasterNN output (1,36) or (36,) → (alloc_prop, liq_depth, liq_trigger).
    alloc_prop: {ind: fraction}  40% cap, 2% floor enforced
    liq_depth:  {ind: 0-1}
    liq_trigger:{ind: 0-1}

    Hybrid allocation: the top-ranked industry receives the full cap (40%).
    All remaining industries are filled proportionally from the leftover budget,
    with floor guaranteed and cap as an absolute ceiling.  This prevents the
    greedy fill from silently maxing two industries while starving the rest.
    """
    n     = len(industry_list)
    floor = 0.02
    cap   = 0.40

    p        = predicted.squeeze()
    vals     = p.tolist() if hasattr(p, 'tolist') else list(p)
    weights  = vals[:12]   if len(vals) >= 12 else vals + [1.0/n]*(12-len(vals))
    depths   = vals[12:24] if len(vals) >= 24 else [0.5]*12
    triggers = vals[24:36] if len(vals) >= 36 else [0.5]*12

    w_map   = dict(zip(industry_list, weights))
    top_ind = max(w_map, key=lambda k: w_map[k])

    # Top industry gets full cap; all others start at floor
    alloc_prop          = {ind: floor for ind in industry_list}
    alloc_prop[top_ind] = cap

    # Proportional fill for the remaining industries
    others  = [ind for ind in industry_list if ind != top_ind]
    premium = 1.0 - cap - floor * len(others)   # budget above floors for others

    if premium > 1e-9 and others:
        uncapped = others[:]
        while premium > 1e-9 and uncapped:
            tw = sum(w_map[ind] for ind in uncapped)
            if tw <= 0:
                break
            proposed    = {ind: alloc_prop[ind] + premium * w_map[ind] / tw for ind in uncapped}
            newly_capped = [ind for ind in uncapped if proposed[ind] >= cap]
            if not newly_capped:
                for ind in uncapped:
                    alloc_prop[ind] = proposed[ind]
                break
            for ind in newly_capped:
                alloc_prop[ind] = cap
                uncapped.remove(ind)
            premium = 1.0 - sum(alloc_prop.values())

    liq_depth   = {ind: depths[i]   for i, ind in enumerate(industry_list)}
    liq_trigger = {ind: triggers[i] for i, ind in enumerate(industry_list)}
    return alloc_prop, liq_depth, liq_trigger


# ── Per-slot model I/O ─────────────────────────────────────────────────────────

def _model_path(prefix, directory, slot):
    """Return the .pt file path for a given prefix, directory, and slot index."""
    return os.path.join(directory, f"{prefix}_model_{slot}.pt")

def _meta_path(prefix, directory):
    """Return the top10_meta.json path for a given prefix and directory."""
    return os.path.join(directory, f"{prefix}_top10_meta.json")

def _hist_model_path(industry, directory, day_slot, pos):
    """Return path for a history model file at circular buffer slot day_slot, position pos."""
    return os.path.join(directory, f"{industry}_hist_{day_slot}_{pos}.pt")

def _hist_meta_path(industry, directory):
    """Return path for the history circular buffer metadata JSON file."""
    return os.path.join(directory, f"{industry}_hist_meta.json")

def _load_hist_meta(industry, directory):
    """Load (head, count) for the history circular buffer; returns (0, 0) if missing or corrupt."""
    path = _hist_meta_path(industry, directory)
    if os.path.exists(path):
        try:
            with open(path) as f:
                meta = json.load(f)
            return max(0, min(meta.get('head', 0), HIST_DAYS - 1)), max(0, min(meta.get('count', 0), HIST_DAYS))
        except Exception:
            pass
    return 0, 0

def _save_hist_meta(industry, directory, head, count):
    """Persist the history circular buffer (head, count) to disk."""
    with open(_hist_meta_path(industry, directory), 'w') as f:
        json.dump({'head': head, 'count': count}, f)


def save_slot_model(prefix, directory, slot, model):
    """Save model weights to disk for the given slot, logging on failure."""
    try:
        torch.save(model.state_dict(), _model_path(prefix, directory, slot))
    except Exception as e:
        log(f"ERROR saving {prefix} slot {slot}: {e}")


def load_slot_model(prefix, directory, slot, model_class):
    """Load one model from disk; returns an untrained default if the file is missing."""
    path  = _model_path(prefix, directory, slot)
    model = model_class()
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
        except Exception as e:
            log(f"WARNING: {prefix} slot {slot} load failed ({e}) — using random weights")
    return model


def save_top10_meta(prefix, directory, top10_meta):
    """Persist the top-10 elite metadata list to <prefix>_top10_meta.json."""
    try:
        with open(_meta_path(prefix, directory), 'w') as f:
            json.dump(top10_meta, f, indent=2)
    except Exception as e:
        log(f"WARNING: could not save top10 meta: {e}")


def load_top10_meta(prefix, directory):
    """Load top-10 metadata from disk; returns [] if the file is missing or corrupt."""
    path = _meta_path(prefix, directory)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log(f"WARNING: could not load top10 meta: {e}")
    return []


def copy_best_model(prefix, directory, top10_meta):
    """Copy slot 0's weights to <prefix>_best.pt for production use."""
    if not top10_meta:
        return
    best_slot = top10_meta[0]['slot']
    src = _model_path(prefix, directory, best_slot)
    dst = os.path.join(directory, f"{prefix}_best.pt")
    if os.path.exists(src):
        try:
            import shutil
            shutil.copy2(src, dst)
        except Exception as e:
            log(f"WARNING: could not copy best model: {e}")


# ── Weighted average ───────────────────────────────────────────────────────────

def _normalize_weights(values):
    """Clip negatives to zero and normalise *values* to sum to 1.0; returns equal weights on all-zero."""
    clipped = [max(float(v), 0.0) for v in values]
    total   = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [v / total for v in clipped]


def compute_weighted_avg_model(prefix, directory, slots, values, model_class):
    """Streaming weighted average: loads exactly one model at a time."""
    weights   = _normalize_weights(values)
    avg_state = None
    int_state = {}
    with torch.no_grad():
        for slot, weight in zip(slots, weights):
            m     = load_slot_model(prefix, directory, slot, model_class)
            state = m.state_dict()
            if avg_state is None:
                avg_state = {k: v.clone().float() * weight
                             for k, v in state.items() if torch.is_floating_point(v)}
                int_state = {k: v.clone()
                             for k, v in state.items() if not torch.is_floating_point(v)}
            else:
                for k, v in state.items():
                    if torch.is_floating_point(v) and k in avg_state:
                        avg_state[k] = avg_state[k] + v.float() * weight
            del m
    if avg_state is None:
        return model_class()
    result = model_class()
    result.load_state_dict({**avg_state, **int_state})
    return result


def compute_weighted_avg_portfolio(portfolios, values):
    """Return a new portfolio dict that is the performance-weighted average of *portfolios*."""
    weights = _normalize_weights(values)
    result  = {}
    for key in portfolios[0]:
        if isinstance(portfolios[0][key], dict):
            result[key] = {
                sym: sum(float(p[key].get(sym, 0.0)) * w for p, w in zip(portfolios, weights))
                for sym in portfolios[0][key]
            }
        elif isinstance(portfolios[0][key], (int, float)):
            result[key] = sum(float(p[key]) * w for p, w in zip(portfolios, weights))
        else:
            result[key] = copy.deepcopy(portfolios[0][key])
    return result


def blend_model_halfway(elite_model, model_class):
    """Halfway blend of elite with a fresh random model for diversity injection."""
    random_model = model_class()
    elite_state  = elite_model.state_dict()
    random_state = random_model.state_dict()
    blended = {}
    for k in elite_state:
        if torch.is_floating_point(elite_state[k]):
            blended[k] = 0.5 * elite_state[k] + 0.5 * random_state[k]
        else:
            blended[k] = elite_state[k].clone()
    m = model_class()
    m.load_state_dict(blended)
    return m


# ── Pool initialisation ────────────────────────────────────────────────────────

def initialise_pool(prefix, directory, load_models_dir, model_class, n=N_SLOTS):
    """Create or load all N model slot files; copies elite slots from load_models_dir if provided."""
    os.makedirs(directory, exist_ok=True)
    log(f"[{sn(prefix)}] Initialising {n} model slots in {directory}")

    for slot in range(ELITE_POOL):
        target = _model_path(prefix, directory, slot)
        if os.path.exists(target):
            continue
        sourced = False
        if load_models_dir:
            src = _model_path(prefix, load_models_dir, slot)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, target)
                log(f"[{sn(prefix)}]   Slot {slot:3d}: copied from {src}")
                sourced = True
        if not sourced:
            m = model_class()
            save_slot_model(prefix, directory, slot, m)
            del m
            log(f"[{sn(prefix)}]   Slot {slot:3d}: created random model")

    missing = [s for s in range(ELITE_POOL, n) if not os.path.exists(_model_path(prefix, directory, s))]
    if missing:
        log(f"[{sn(prefix)}]   Generating {len(missing)} mutation slots ...")
        for slot in missing:
            parent_slot = random.randint(0, ELITE_POOL - 1)
            parent      = load_slot_model(prefix, directory, parent_slot, model_class)
            child       = mutate(parent)
            del parent
            save_slot_model(prefix, directory, slot, child)
            del child

    log(f"[{sn(prefix)}] All {n} model slots ready")


# ── Selection & regeneration ───────────────────────────────────────────────────

def selection_and_mutation(
    prefix, directory, model_class,
    scores, portfolios, survival_floor,
    elite_count=ELITE_COUNT, inactive_slots=None,
    actual_day=None, total_avail=None,
    sigma=None,
    n_slots=N_SLOTS,
):
    """
    Fixed slot layout after selection:
      0–16        : top 17 performers in rank order (best at 0)
      17          : weighted average of top 5  (w5)
      18          : weighted average of top 10 (w10)
      19          : weighted average of top 15 (w15)
      20..(n-1)  : mutations — distributed evenly across ELITE_POOL parents
    """
    if inactive_slots is None:
        inactive_slots = set()

    surviving = [(s, v) for s, v in scores if v >= survival_floor and s not in inactive_slots]
    if not surviving:
        surviving = [(s, v) for s, v in scores if v >= survival_floor]
    if not surviving:
        log(f"[{sn(prefix)}]   No survivors — skipping selection this day")
        return portfolios

    surviving.sort(key=lambda x: x[1], reverse=True)
    top_elite   = surviving[:min(elite_count, len(surviving))]
    elite_slots = [s for s, _ in top_elite]
    elite_vals  = [v for _, v in top_elite]

    # Compute weighted averages before touching any slot files
    def _wavg(n):
        slots = elite_slots[:min(n, len(elite_slots))]
        vals  = elite_vals[:min(n, len(elite_vals))]
        return (
            compute_weighted_avg_model(prefix, directory, slots, vals, model_class),
            compute_weighted_avg_portfolio([portfolios[s] for s in slots], vals),
        )

    w5_model,  w5_port  = _wavg(5)
    w10_model, w10_port = _wavg(10)
    w15_model, w15_port = _wavg(15)

    # Load all elite models into memory before writing to avoid clobber
    elite_models = [load_slot_model(prefix, directory, s, model_class) for s in elite_slots]
    elite_ports  = [copy.deepcopy(portfolios[s]) for s in elite_slots]

    # Write elites in rank order to slots 0–16
    for rank, (model, port) in enumerate(zip(elite_models, elite_ports)):
        save_slot_model(prefix, directory, rank, model)
        portfolios[rank] = port
        del model

    # Write weighted averages to slots 17–19
    save_slot_model(prefix, directory, ELITE_COUNT,     w5_model);  portfolios[ELITE_COUNT]     = w5_port;  del w5_model
    save_slot_model(prefix, directory, ELITE_COUNT + 1, w10_model); portfolios[ELITE_COUNT + 1] = w10_port; del w10_model
    save_slot_model(prefix, directory, ELITE_COUNT + 2, w15_model); portfolios[ELITE_COUNT + 2] = w15_port; del w15_model

    # Mutations fill slots 20..(n_slots-1): distributed evenly across ELITE_POOL parents
    n_mutations = n_slots - ELITE_POOL
    muts_per_parent = max(1, n_mutations // ELITE_POOL)
    parent_assignments = defaultdict(list)
    for i, slot in enumerate(range(ELITE_POOL, n_slots)):
        parent_assignments[i // muts_per_parent].append(slot)

    for parent_rank, child_slots in parent_assignments.items():
        parent = load_slot_model(prefix, directory, parent_rank, model_class)
        for child_slot in child_slots:
            child = mutate(parent) if sigma is None else mutate(parent, sigma=sigma)
            save_slot_model(prefix, directory, child_slot, child)
            del child
            portfolios[child_slot] = copy.deepcopy(portfolios[parent_rank])
        del parent

    elite_display = [_fmt_slot(s) for s in elite_slots]
    log(f"[{sn(prefix)}]   Selection done | elite={elite_display} | "
        f"top=${elite_vals[0]:.2f} | "
        f"8th=${elite_vals[-1]:.2f}")

    return portfolios


# ── Industry training ──────────────────────────────────────────────────────────

def step_industry(industry, symbols, output_dir, portfolios, histories,
                  day, actual_day, total_avail, day_num, total_days,
                  next_day=None, all_zero_streak=0, daily_sigma=None,
                  pool_size=N_SLOTS, intraday_bars=None,
                  freeze=False, seq_flags=None):
    """
    Single trading day for one industry.  Sequential: load→infer→trade→evict per slot.
    Returns (baseline_score, top_slot_value, best_delta, top_hold_val, top_cash, new_streak).
    """
    day_data  = day['data']
    fill_data = next_day['data'] if next_day is not None else day_data
    hist_lengths = [len(histories[sym]) for sym in symbols]
    num_past     = min(hist_lengths) if hist_lengths else 0

    if day_num % 10 == 0 or day_num == total_days - 1:
        n_hist_warm = (0 if day_num == 0 else _load_hist_meta(industry, output_dir)[1]) * HIST_PER_DAY
        log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} — running {pool_size} models "
            f"+ {n_hist_warm} history (ohlcv={num_past}/15 days warm) ...")

    # ── Step 1: reset all slots to slot 0's portfolio (level playing field) ─────
    # Every slot starts each day from model 0.0's current cash + holdings.
    # baseline_score is slot 0's portfolio valued at fill_data prices with no
    # trading (i.e. "do nothing from today's production state").
    # best_delta therefore = honest single-day gain above holding slot 0's position.
    #
    # Valuation uses fill_data prices: fills execute at next-day prices so both
    # baseline and post-trade scoring must use the same day.
    val_data = fill_data  # next_day['data'] if available, else day_data

    # Load or reset per-industry history meta
    if day_num == 0:
        hist_head, hist_count = 0, 0
    else:
        hist_head, hist_count = _load_hist_meta(industry, output_dir)

    ref_cash  = portfolios[0]['cash']
    ref_hold  = {sym: portfolios[0]['holdings'].get(sym, 0.0) for sym in symbols}
    ref_stop  = {sym: portfolios[0]['stop_prices'].get(sym, 0.0) for sym in symbols}
    baseline_score = compute_value({'cash': ref_cash, 'holdings': ref_hold},
                                   val_data, symbols)

    for slot in range(pool_size):
        portfolios[slot]['cash']        = ref_cash
        portfolios[slot]['holdings']    = dict(ref_hold)
        portfolios[slot]['stop_prices'] = dict(ref_stop)

    # ── Pre-compute rolling stats (shared across all slots) ───────────────────
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

    # ── Build shared input tensors (same for all slots) ──────────────────────
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
    port0     = portfolios[0]
    state_vec = [port0['cash']] + [port0['holdings'].get(sym, 0.0) for sym in symbols]
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

    # ── Step 2: sequential load → infer → trade → evict ──────────────────────
    buy_exec_count   = 0
    sell_exec_count  = 0
    slot_trade_count = [0] * pool_size

    for slot in range(pool_size):
        port        = portfolios[slot]
        stop_prices = port['stop_prices']
        model       = load_slot_model(industry, output_dir, slot, StockNN)

        with torch.inference_mode():
            out = model(history_t, today_t)
        del model

        out = out.view(12, 4)
        local_buys = local_sells = 0

        # ── Open + low phase: partial sells, gap sell_all, stops, limit buys ───
        for j, sym in enumerate(symbols):
            if sym not in day_data:
                continue
            buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = out[j].tolist()
            cur_qty        = port['holdings'][sym]
            low_t          = day_data[sym]['low']
            high_t         = day_data[sym]['high']
            span_t         = max(high_t - low_t, 1e-9)
            sell_all_price = low_t + sell_all_price_frac * span_t
            buy_price      = low_t + buy_price_frac * span_t
            stop_loss      = buy_price * 0.9
            low_first      = seq_flags.get(sym, True) if seq_flags else True

            nd      = fill_data.get(sym, day_data.get(sym, {}))
            nd_open = nd.get('open',  day_data[sym]['close'])
            nd_low  = nd.get('low',   day_data[sym]['low'])
            nd_high = nd.get('high',  day_data[sym]['high'])

            sym_bars = (intraday_bars or {}).get(sym)
            if sym_bars:
                b, s = _fill_sym_from_bars(sym, port, sym_bars, buy_qty, buy_price,
                                           sell_all_price, sell_qty, stop_loss,
                                           symbols, fill_data, day_data)
                local_buys  += b
                local_sells += s
                continue

            # Partial sell at open
            if sell_qty > 1e-6 and cur_qty > 1e-6:
                sell_amount               = min(sell_qty, cur_qty)
                port['holdings'][sym]    -= sell_amount
                port['cash']             += _sell_net(sell_amount, nd_open)
                local_sells              += sell_amount

            # Gap-up sell_all at open (both sequences)
            cur_qty_after = port['holdings'][sym]
            if cur_qty_after > 1e-6 and nd_open >= sell_all_price:
                port['holdings'][sym]  = 0.0
                port['cash']          += _sell_net(cur_qty_after, nd_open)
                local_sells           += cur_qty_after

            # High-first: intraday sell_all fires at the high before the low
            if not low_first:
                intraday_qty = port['holdings'][sym]
                if intraday_qty > 1e-6 and nd_low < sell_all_price < nd_high:
                    slipped = sell_all_price * (1.0 - SLIPPAGE_RATE)
                    port['holdings'][sym]  = 0.0
                    port['cash']          += _sell_net(intraday_qty, slipped)
                    local_sells           += intraday_qty

            # Stop loss: gap-down at open, or intraday low
            stop_p    = stop_prices.get(sym, 0.0)
            remaining = port['holdings'][sym]
            if stop_p > 0 and remaining > 1e-6:
                if nd_open <= stop_p:
                    port['holdings'][sym]  = 0.0
                    port['cash']          += _sell_net(remaining, nd_open)
                    local_sells           += remaining
                elif nd_low <= stop_p:
                    port['holdings'][sym]  = 0.0
                    port['cash']          += _sell_net(remaining, stop_p * (1.0 - SLIPPAGE_RATE))
                    local_sells           += remaining

            # Buy at open (gap-down) or intraday low
            if buy_qty > 1e-6 and buy_price > 0:
                if nd_open <= buy_price:
                    fill_price = nd_open
                elif nd_low < buy_price < nd_high:   # strict: not right at day extreme
                    fill_price = buy_price * (1.0 + SLIPPAGE_RATE)
                else:
                    fill_price = 0.0

                if fill_price > 0:
                    affordable = port['cash'] / (fill_price * BUY_FILL)
                    buy_amount = min(buy_qty, affordable)
                    if buy_amount > 1e-6:
                        # 60% single-stock concentration cap
                        port_value    = port['cash'] + sum(
                            port['holdings'].get(s, 0.0)
                            * fill_data.get(s, day_data.get(s, {})).get('close', 0.0)
                            for s in symbols)
                        cur_sym_value = port['holdings'].get(sym, 0.0) * fill_price
                        max_sym_spend = max(0.0, MAX_SINGLE_STOCK_PCT * port_value - cur_sym_value)
                        buy_amount    = min(buy_amount, max_sym_spend / (fill_price * BUY_FILL))
                    if buy_amount > 1e-6:
                        port['holdings'][sym]  += buy_amount
                        port['cash']           -= buy_amount * fill_price * BUY_FILL
                        stop_prices[sym]        = stop_loss
                        local_buys             += buy_amount

        # ── High phase: intraday sell_all for low-first symbols only ─────────
        for j, sym in enumerate(symbols):
            if sym not in day_data:
                continue
            if (intraday_bars or {}).get(sym):
                continue  # bar-walk already handled this symbol
            if seq_flags and not seq_flags.get(sym, True):
                continue  # high-first: sell_all already fired before the low
            _, _, sell_all_price_frac, _ = out[j].tolist()
            low_t          = day_data[sym]['low']
            high_t         = day_data[sym]['high']
            span_t         = max(high_t - low_t, 1e-9)
            sell_all_price = low_t + sell_all_price_frac * span_t

            nd      = fill_data.get(sym, day_data.get(sym, {}))
            nd_low  = nd.get('low',  day_data[sym]['low'])
            nd_high = nd.get('high', day_data[sym]['high'])

            cur_qty = port['holdings'][sym]
            if cur_qty > 1e-6 and nd_low < sell_all_price < nd_high:   # strict: not at day extreme
                slipped = sell_all_price * (1.0 - SLIPPAGE_RATE)
                port['holdings'][sym]  = 0.0
                port['cash']          += _sell_net(cur_qty, slipped)
                local_sells           += cur_qty

        slot_trade_count[slot] = local_buys + local_sells
        buy_exec_count        += local_buys
        sell_exec_count       += local_sells

    # ── Step 3: update shared market-history window ───────────────────────────
    for sym in symbols:
        if sym in day_data:
            d      = day_data[sym]
            raw    = [d['open'], d['close'], d['high'], d['low'], d['volume']]
            prev   = histories[sym][-1][:5] if histories[sym] else None
            deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
            histories[sym].append(raw + deltas)
            if len(histories[sym]) > 15:
                histories[sym].pop(0)

    # ── Step 4: score & report ────────────────────────────────────────────────
    # Value at fill_data prices: fills executed at next-day prices must be marked
    # at those same prices, otherwise buys look artificially cheap.
    scores        = [(s, compute_value(portfolios[s], fill_data, symbols)) for s in range(pool_size)]
    best_score    = max(v for _, v in scores)
    worst_score   = min(v for _, v in scores)
    best_delta    = best_score  - baseline_score
    worst_delta   = worst_score - baseline_score

    # ── Score history candidates ───────────────────────────────────────────────
    # Each history candidate is a model from a previous day's elite pool.
    # We re-simulate from slot 0's starting state so scores are comparable.
    n_hist      = hist_count * HIST_PER_DAY
    hist_scored = []  # list of (h_index, score, port)
    for h in range(n_hist):
        day_slot = h // HIST_PER_DAY
        pos      = h % HIST_PER_DAY
        hpath    = _hist_model_path(industry, output_dir, day_slot, pos)
        if not os.path.exists(hpath):
            continue
        try:
            hmodel = StockNN()
            hmodel.load_state_dict(torch.load(hpath, weights_only=True))
            hscore, hport = _simulate_one_model(
                hmodel, ref_cash, ref_hold, ref_stop,
                symbols, day_data, fill_data, history_t, today_t,
                seq_flags=seq_flags)
            del hmodel
            hist_scored.append((h, hscore, hport))
        except Exception as e:
            log(f"[{sn(industry)}] WARNING: failed to load history {day_slot}.{pos}: {e}")

    # Elite portfolio value stats (slots 0..ELITE_COUNT-1)
    elite_vals     = [v for s, v in scores if s < ELITE_COUNT]
    elite_max_val  = max(elite_vals)
    elite_min_val  = min(elite_vals)
    elite_mean_val = sum(elite_vals) / len(elite_vals)
    # ── Gain flags (skip on buy-only days: no holdings at start = first day or post-reset) ──
    buy_only_day = all(v == 0.0 for v in ref_hold.values())
    if not buy_only_day and baseline_score > 0:
        pct_gain = best_delta / baseline_score * 100
        models_section = None
        if pct_gain >= 10.0:
            # Build per-elite model breakdown for the dump.
            ranked_for_dump       = sorted(scores, key=lambda x: x[1], reverse=True)
            yesterday_elites_ind  = list(range(ELITE_POOL))
            today_elites_ind      = [s for s, _ in ranked_for_dump[:ELITE_POOL]]
            price_start_ind = {sym: day_data.get(sym, {}).get('close', 0.0)  for sym in symbols}
            price_end_ind   = {sym: fill_data.get(sym, {}).get('close', 0.0) for sym in symbols}
            models_section = _build_models_section(
                yesterday_elites_ind, today_elites_ind,
                ref_cash, ref_hold,
                portfolios, scores,
                baseline_score,
                symbols,
                hard_threshold=12.5, soft_threshold=10.0,
                price_start=price_start_ind,
                price_end=price_end_ind,
            )
        if pct_gain >= 12.5:
            _dump_day(actual_day, industry, 'HARD', baseline_score, best_delta, pct_gain,
                      scores, day_data, fill_data, models=models_section)
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}  "
                f"*** HARD FLAG: +{pct_gain:.2f}% gain — data dumped to {DUMP_DIR}/day_{actual_day + 1}/ ***")
            # raise HardFlagError(f"{industry} day {actual_day + 1}: +{pct_gain:.2f}%")
        elif pct_gain >= 10.0:
            _dump_day(actual_day, industry, 'SOFT', baseline_score, best_delta, pct_gain,
                      scores, day_data, fill_data, models=models_section)
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}  "
                f"** SOFT FLAG: +{pct_gain:.2f}% gain — data dumped to {DUMP_DIR}/day_{actual_day + 1}/ **")

    # survival_floor is relative to yesterday's best (baseline_score):
    # individual slots scoring below 90% of baseline are excluded from selection.
    # The hard reset triggers if yesterday's best model itself falls below $1,500.
    survival_floor = baseline_score * 0.9
    abs_floor      = IND_STARTING_CASH * 0.9
    ranked_scores  = sorted(scores, key=lambda x: x[1], reverse=True)
    log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} | "
        f"best Δ${best_delta:+.2f}  worst Δ${worst_delta:+.2f} | "
        f"shares(buy/sell)={buy_exec_count:.0f}/{sell_exec_count:.0f} | "
        f"prod=${baseline_score:.2f}")

    if baseline_score < abs_floor:
        log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} "
            f"Production model (${baseline_score:.2f}) fell below floor "
            f"(${abs_floor:.2f}) — resetting and skipping selection")
        for slot in range(pool_size):
            portfolios[slot]['cash']        = IND_STARTING_CASH
            portfolios[slot]['holdings']    = {sym: 0.0 for sym in symbols}
            portfolios[slot]['stop_prices'] = {sym: 0.0 for sym in symbols}
        return baseline_score, baseline_score, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0

    # ── Zero-trade inaction filter ────────────────────────────────────────────
    inactive_slots = set()
    new_streak     = 0
    all_filtered   = False
    if num_past >= 15 and day_num > 0:
        inactive_slots = {s for s in range(pool_size) if slot_trade_count[s] == 0}
        if len(inactive_slots) == pool_size:
            all_filtered = True
            new_streak   = all_zero_streak + 1
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}   "
                f"Zero-trade filter: ALL {pool_size} slots inactive (streak={new_streak})")
        elif inactive_slots:
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}   "
                f"Zero-trade filter: {len(inactive_slots)} slot(s) excluded")

    # ── Delta-based selection scores with invested_pct multiplier ────────────
    # Selection score = raw delta adjusted by invested_pct:
    # positive-delta slots are scaled down proportional to cash held,
    # so a slot that earned $50 by deploying 80% outranks one that
    # earned $50 while sitting 90% in cash.
    scores_dict = dict(scores)
    below_floor = {s for s, v in scores if v < survival_floor}
    sel_scores  = []
    for s, raw in scores:
        raw_delta = raw - baseline_score
        if raw_delta > 0:
            invested_pct = max(0.0, 1.0 - portfolios[s]['cash'] / raw)
            sel_scores.append((s, raw_delta * invested_pct))
        else:
            sel_scores.append((s, raw_delta))

    # Append history candidates as virtual slots pool_size..pool_size+n_hist-1.
    # Copy each history .pt to its virtual slot path so selection_and_mutation
    # can load it by slot index without modification.
    hist_below_floor = set()
    for h, hscore, hport in hist_scored:
        vslot     = pool_size + h
        raw_delta = hscore - baseline_score
        if raw_delta > 0:
            invested_pct = max(0.0, 1.0 - hport['cash'] / hscore) if hscore > 0 else 0.0
            sel_scores.append((vslot, raw_delta * invested_pct))
        else:
            sel_scores.append((vslot, raw_delta))
        portfolios.append(hport)  # portfolios[vslot]
        hpath = _hist_model_path(industry, output_dir, h // HIST_PER_DAY, h % HIST_PER_DAY)
        shutil.copy2(hpath, _model_path(industry, output_dir, vslot))
        if hscore < survival_floor:
            hist_below_floor.add(vslot)

    # ── Under-investment soft flag: only fires if an elite candidate is affected ──
    # Suppressed when fill day closed down (close < open for majority of symbols):
    # holding cash was rational on a declining fill day, not a model deficiency.
    fill_down_day = sum(
        1 for sym in symbols
        if sym in fill_data
        and fill_data[sym].get('close', 0.0) < fill_data[sym].get('open', 0.0)
    ) > len(symbols) / 2
    excluded         = inactive_slots | below_floor | hist_below_floor
    elite_candidates = set(
        [s for s, _ in sorted(sel_scores, key=lambda x: x[1], reverse=True)
         if s not in excluded][:ELITE_COUNT]
    )
    under_invested   = {
        s for s in elite_candidates
        if scores_dict.get(s, 0) > 0
        and portfolios[s]['cash'] / scores_dict[s] >= 0.5
    }
    if under_invested and not fill_down_day:
        n_ui    = len(under_invested)
        avg_inv = 1.0 - sum(portfolios[s]['cash'] / scores_dict[s] for s in under_invested) / n_ui
        log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}  "
            f"** SOFT FLAG: under-investment — {n_ui} elite(s) ≥50% cash "
            f"(avg invested={avg_inv:.1%}) — data dumped to {DUMP_DIR}/day_{actual_day + 1}/ **")
        _dump_day(actual_day, industry, 'UNDER_INVEST',
                  baseline_score, best_delta, avg_inv * 100,
                  scores, day_data, fill_data)

    # Preserve slot 0's own post-trading result before selection overwrites it.
    # Selection replaces portfolios[0] with the winner's portfolio; restoring here
    # ensures prod tracks a single consistent deployed portfolio rather than the
    # best-of-200 cherry-picked each day.
    slot0_own_port = copy.deepcopy(portfolios[0])

    # ── Step 5: selection + mutation ─────────────────────────────────────────
    if not freeze:
        selection_and_mutation(
            industry, output_dir, StockNN,
            sel_scores, portfolios,
            survival_floor=-(baseline_score * 0.1),
            inactive_slots=inactive_slots | below_floor | hist_below_floor,
            actual_day=actual_day, total_avail=total_avail,
            n_slots=pool_size,
        )

    # Remove virtual slot files and trim extended portfolios list.
    for h, _, _ in hist_scored:
        vpath = _model_path(industry, output_dir, pool_size + h)
        try:
            os.remove(vpath)
        except FileNotFoundError:
            pass
    del portfolios[pool_size:]

    # ── Diversity injection if all-zero streak ≥ 2 ───────────────────────────
    # After selection, slots 0–16 = top 17 elites, 17–19 = weighted averages.
    if not freeze and all_filtered and new_streak >= 2:
        half         = ELITE_COUNT // 2
        ranked_for_inj = sorted(scores, key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
        source_slots = [s for s, _ in ranked_for_inj[:half]]
        inject_slots = [s for s, _ in ranked_for_inj[half:]]
        log(f"[{sn(industry)}]   Diversity injection: replacing bottom {len(inject_slots)} elites with half-random blends")
        for inject_slot, source_slot in zip(inject_slots, source_slots):
            elite = load_slot_model(industry, output_dir, source_slot, StockNN)
            blend = blend_model_halfway(elite, StockNN)
            save_slot_model(industry, output_dir, inject_slot, blend)
            portfolios[inject_slot] = copy.deepcopy(portfolios[source_slot])
            del elite, blend
        new_streak = 0
        log(f"[{sn(industry)}]   Diversity injection complete — streak reset")

    # ── Daily burst refinement (after selection + diversity injection) ───────────
    # Each burst generates 200 mutants (10 per elite) at a finer sigma, merges the
    # top-ELITE_COUNT with current elites, and pushes losers out.  Sequential so only
    # ELITE_COUNT models live in memory at once per burst.
    if daily_sigma is not None:
        log(f"[{sn(industry)}] Daily mode: running 4 refinement bursts ...")
        for burst_num in range(4):
            burst_sigma = daily_sigma / (2 ** (burst_num + 1))
            _run_daily_burst(
                industry, output_dir, symbols, burst_sigma,
                ref_cash, ref_hold, ref_stop,
                day_data, fill_data, history_t, today_t,
                portfolios, actual_day, total_avail,
                seq_flags=seq_flags, intraday_bars=intraday_bars)

    # ── History push ──────────────────────────────────────────────────────────
    # Save the day's top-7 direct elites + 3 wavg blends into the circular
    # history buffer so they can compete as re-selection candidates tomorrow.
    if not freeze:
        hd = hist_head
        for k in range(HIST_ELITE):
            src = _model_path(industry, output_dir, k)
            dst = _hist_model_path(industry, output_dir, hd, k)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        for k in range(HIST_WAVG):
            src = _model_path(industry, output_dir, ELITE_COUNT + k)
            dst = _hist_model_path(industry, output_dir, hd, HIST_ELITE + k)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        hist_head  = (hist_head + 1) % HIST_DAYS
        if hist_count < HIST_DAYS:
            hist_count += 1
        _save_hist_meta(industry, output_dir, hist_head, hist_count)

    # Restore slot 0's own portfolio so the next day's baseline carries forward
    # this model's actual result, not the winner's cherry-picked result.
    # Model weights in slot 0 remain updated from selection/bursts.
    portfolios[0] = slot0_own_port

    # slot0_score = slot 0's own end-of-day value.  Master uses this to price
    # industry units, so it reflects what the deployed model actually earned.
    slot0_score        = dict(scores).get(0, baseline_score)
    if daily_sigma is not None:
        slot0_score = compute_value(portfolios[0], fill_data, symbols)
        best_delta  = max(best_delta, slot0_score - baseline_score)
    top_holdings_value = sum(
        portfolios[0]['holdings'].get(sym, 0.0) * fill_data[sym]['close']
        for sym in symbols if sym in fill_data and fill_data[sym]['close'] > 0)
    top_cash_value     = portfolios[0]['cash']

    return baseline_score, slot0_score, best_delta, top_holdings_value, top_cash_value, new_streak, elite_max_val, elite_min_val, elite_mean_val

def train_master_one_day(industries, primed_portfolio, model_dir,
                         ind_value_history, industry_top_scores=None):
    """
    Legacy MasterNN upkeep stub — superseded by MT1/MT2 upkeep in upkeep.py.

    Called only during the ≤15-real-day transition window in production_v2.py
    while MT history is still too short for MT1/MT2. step_master was removed
    with training_v2/v3; this now returns (None, None) safely.
    """
    # step_master was removed when training_v2/v3 were retired; legacy master
    # upkeep is a no-op until MT1/MT2 have enough history to take over.
    log("[master] Legacy upkeep skipped — step_master removed; MT1/MT2 will take over at 15 real days")
    return None, None


# ── Data loading ───────────────────────────────────────────────────────────────
