"""
training_v4.py — Parallel evolutionary training: v2 per-slot on-demand loading +
dynamic industry process pool (ProcessPoolExecutor, max_workers=NUM_WORKERS).

Each industry slot is loaded from disk, inferred, traded, and evicted before the
next slot loads (v2 methodology — no persistent cache).  Industries are processed
concurrently via a dynamic work queue: worker processes pull the next available
industry from the pool as soon as they finish their current one, keeping all
available CPUs fully utilized without hard-coding industry-to-worker assignments.
Using processes (not threads) sidesteps the Python GIL, enabling genuine parallel
execution of the trade simulation loop.  Master runs after all industries complete
(sequential, single-process).

Usage:
    python training_v4.py --output models [--load-dir models] [--start-day N] [--stop-day N]
                          [--passes N] [--sigma 0.01] [--daily] [--promote uat,prod]
"""

import argparse
import copy
import gc
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf

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

NUM_WORKERS           = 2           # worker processes for industry parallelism; raise to 4 on a 4-vCPU host
DAY_TIMEOUT_SECS      = 420         # 7 min wall-clock limit per training day; v2 baseline ~5 min — exit if exceeded

N_IND                 = len(INDUSTRIES)  # 12 industry sectors
IND_SYMS              = 12               # symbols per industry (all sectors have exactly 12)
HIST_WINDOW           = 15               # rolling history window length

IND_STARTING_CASH     = 25_000.0    # per-industry portfolio starting capital
MST_STARTING_CASH     = 300_000.0   # master starting capital (12 × IND_STARTING_CASH)
IND_UNIT_PRICE        = 25_000.0    # fixed price per industry "unit" in master's portfolio
MAX_SINGLE_STOCK_PCT  = 0.60        # no single stock may exceed 60% of portfolio value


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

def _simulate_one_model(model, ref_cash, ref_hold, ref_stop, symbols,
                        day_data, fill_data, history_t, today_t, seq_flags=None):
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
                     portfolios, actual_day, total_avail, seq_flags=None):
    """
    Generate 200 burst mutants (10 per elite from ELITE_POOL parents), merge
    top-ELITE_COUNT with current elites, save winners to disk, update portfolios.
    Each burst explores at a finer sigma than the main pool.
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
                seq_flags=seq_flags)
            burst_candidates.append((child, score, port))
        del parent
    burst_candidates.sort(key=lambda x: x[1], reverse=True)
    top_burst = burst_candidates[:ELITE_COUNT]

    current_elites = []
    for rank in range(ELITE_COUNT):
        model = load_slot_model(prefix, output_dir, rank, StockNN)
        score = compute_value(portfolios[rank], fill_data, symbols)
        current_elites.append((model, score, copy.deepcopy(portfolios[rank])))

    # Merge burst top-ELITE_COUNT with current top-ELITE_COUNT; pick best ELITE_COUNT.
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
    """Timestamped, immediately-flushed console output — rolling 200-line file buffer.

    File writes are suppressed in worker processes to avoid concurrent write conflicts;
    workers print to stdout only (their output appears in the same terminal session).
    """
    global _console_log_lines
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if multiprocessing.current_process().name == 'MainProcess':
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


# ── Master decode / allocation helpers ────────────────────────────────────────

MASTER_LOOKBACKS     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 90]
MASTER_POLY3_WINDOWS = [10, 30, 60, 90]
MASTER_START_DAY     = 30   # first actual_day on which master trains (0-indexed)
FN_PENALTY           = 0.006  # score multiplier penalty per false-negative industry
FP_PENALTY           = 0.004  # score multiplier penalty per false-positive industry
TIER_WEIGHTS         = {1: 1.0, 2: 1.5, 3: 2.25}


def _mst_hist_at(history, lookback):
    """Industry portfolio value at `lookback` days ago; clamps to oldest on record."""
    idx = len(history) - 1 - lookback
    return history[max(0, idx)]


def _mst_window(history, n_days):
    """Last n_days industry values, left-padded with oldest on record."""
    oldest = history[0] if history else 0.0
    pad    = max(0, n_days - len(history))
    return [oldest] * pad + list(history[-n_days:])


def build_master_features(ind_value_history, industry_list):
    """Build (1, 444) input tensor for MasterNN.

    Per industry (37 features × 12 = 444):
      18 fractional deltas at MASTER_LOOKBACKS
       3 poly-2 coefs over 5-day window
      16 poly-3 coefs over 4 windows (10/30/60/90 days)
    """
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
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 444)


def decode_master_tiers(out_logits, industry_list):
    """Convert (1,48) raw logits → {ind: tier} where tier ∈ {0,1,2,3}."""
    logits = out_logits.view(12, 4)
    probs  = F.softmax(logits, dim=1)
    tiers  = probs.argmax(dim=1).tolist()
    return {ind: tiers[i] for i, ind in enumerate(industry_list)}


def tiers_to_alloc(tier_map, industry_list, available_cash):
    """Convert tier predictions → {ind: dollar_amount}.

    Positives (tier > 0) ranked lowest→highest, split into thirds:
      tier3 = floor(N/3) (smallest), remainders go to tier1 first then tier2.
    Capital per tier weighted geometrically: tier1=1.0, tier2=1.5, tier3=2.25.
    """
    positives = sorted(
        [ind for ind in industry_list if tier_map[ind] > 0],
        key=lambda ind: tier_map[ind],
    )
    N = len(positives)
    if N == 0:
        return {ind: 0.0 for ind in industry_list}

    base = N // 3
    rem  = N % 3
    n3   = base
    n2   = base + (1 if rem >= 2 else 0)

    tier_assignments = {}
    for rank, ind in enumerate(positives):
        if rank < n3:
            tier_assignments[ind] = 1
        elif rank < n3 + n2:
            tier_assignments[ind] = 2
        else:
            tier_assignments[ind] = 3

    total_w = sum(TIER_WEIGHTS[tier_assignments[ind]] for ind in positives)
    alloc   = {ind: 0.0 for ind in industry_list}
    for ind in positives:
        alloc[ind] = (TIER_WEIGHTS[tier_assignments[ind]] / total_w) * available_cash
    return alloc


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
):
    """
    Fixed slot layout after selection:
      0–16 : top 17 performers in rank order (best at 0)
      17   : weighted average of top 5  (w5)
      18   : weighted average of top 10 (w10)
      19   : weighted average of top 15 (w15)
      20–199: mutations — 9 per parent, deterministic, parents drawn from slots 0–19
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

    # Mutations fill slots 20–199: 9 per parent, deterministic, parents are slots 0–19
    parent_assignments = defaultdict(list)
    for i, slot in enumerate(range(ELITE_POOL, N_SLOTS)):
        parent_assignments[i // MUTATIONS_PER_PARENT].append(slot)

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

def init_industry(industry, symbols, output_dir, load_models_dir, all_days, day_start):
    """Returns (portfolios, histories).  stop_prices live inside each portfolio dict."""
    initialise_pool(industry, output_dir, load_models_dir, StockNN)

    portfolios = [
        {'cash': IND_STARTING_CASH,
         'holdings':    {sym: 0.0 for sym in symbols},
         'stop_prices': {sym: 0.0 for sym in symbols}}
        for _ in range(N_SLOTS)
    ]
    histories = {sym: [] for sym in symbols}

    if day_start > 0:
        pre_slice = all_days[max(0, day_start - 15):day_start]
        log(f"[{sn(industry)}] Pre-loading {len(pre_slice)} days of history before day {day_start}")
        for pre_day in pre_slice:
            pre_data = pre_day['data']
            for sym in symbols:
                if sym in pre_data:
                    d      = pre_data[sym]
                    raw    = [d['open'], d['close'], d['high'], d['low'], d['volume']]
                    prev   = histories[sym][-1][:5] if histories[sym] else None
                    deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
                    histories[sym].append(raw + deltas)
                    if len(histories[sym]) > 15:
                        histories[sym].pop(0)

    return portfolios, histories


def step_industry(industry, symbols, output_dir, portfolios, histories,
                  day, actual_day, total_avail, day_num, total_days,
                  next_day=None, all_zero_streak=0, daily_sigma=None,
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
        log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} — running 100 models "
            f"(history={num_past}/15 days warm) ...")

    # ── Step 1: reset all slots to slot 0's portfolio (level playing field) ─────
    # Every slot starts each day from model 0.0's current cash + holdings.
    # baseline_score is slot 0's portfolio valued at fill_data prices with no
    # trading (i.e. "do nothing from today's production state").
    # best_delta therefore = honest single-day gain above holding slot 0's position.
    #
    # Valuation uses fill_data prices: fills execute at next-day prices so both
    # baseline and post-trade scoring must use the same day.
    val_data = fill_data  # next_day['data'] if available, else day_data

    ref_cash  = portfolios[0]['cash']
    ref_hold  = {sym: portfolios[0]['holdings'].get(sym, 0.0) for sym in symbols}
    ref_stop  = {sym: portfolios[0]['stop_prices'].get(sym, 0.0) for sym in symbols}
    baseline_score = compute_value({'cash': ref_cash, 'holdings': ref_hold},
                                   val_data, symbols)

    for slot in range(N_SLOTS):
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
    slot_trade_count = [0] * N_SLOTS

    for slot in range(N_SLOTS):
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
    scores        = [(s, compute_value(portfolios[s], fill_data, symbols)) for s in range(N_SLOTS)]
    best_score    = max(v for _, v in scores)
    worst_score   = min(v for _, v in scores)
    best_delta    = best_score  - baseline_score
    worst_delta   = worst_score - baseline_score
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
        for slot in range(N_SLOTS):
            portfolios[slot]['cash']        = IND_STARTING_CASH
            portfolios[slot]['holdings']    = {sym: 0.0 for sym in symbols}
            portfolios[slot]['stop_prices'] = {sym: 0.0 for sym in symbols}
        return baseline_score, baseline_score, 0.0, 0.0, 0.0, 0

    # ── Zero-trade inaction filter ────────────────────────────────────────────
    inactive_slots = set()
    new_streak     = 0
    all_filtered   = False
    if num_past >= 15 and day_num > 0:
        inactive_slots = {s for s in range(N_SLOTS) if slot_trade_count[s] == 0}
        if len(inactive_slots) == N_SLOTS:
            all_filtered = True
            new_streak   = all_zero_streak + 1
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}   "
                f"Zero-trade filter: ALL N_SLOTS slots inactive (streak={new_streak})")
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

    # ── Under-investment soft flag: only fires if an elite candidate is affected ──
    # Suppressed when fill day closed down (close < open for majority of symbols):
    # holding cash was rational on a declining fill day, not a model deficiency.
    fill_down_day = sum(
        1 for sym in symbols
        if sym in fill_data
        and fill_data[sym].get('close', 0.0) < fill_data[sym].get('open', 0.0)
    ) > len(symbols) / 2
    excluded         = inactive_slots | below_floor
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
            inactive_slots=inactive_slots | below_floor,
            actual_day=actual_day, total_avail=total_avail,
        )

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
                seq_flags=seq_flags)

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

    return baseline_score, slot0_score, best_delta, top_holdings_value, top_cash_value, new_streak


# ── Master training ────────────────────────────────────────────────────────────

def init_master(output_dir, load_models_dir, industries):
    """Returns (portfolios, ind_value_history).

    portfolios: 200 dicts with {cash, holdings:{ind: float}, zero_counts:{ind: int}}
    ind_value_history: {ind: []} — per-industry portfolio value series, appended by main loop
    """
    industry_list = list(industries.keys())
    initialise_pool('master', output_dir, load_models_dir, MasterNN)

    portfolios = [
        {'cash':        MST_STARTING_CASH,
         'holdings':    {ind: 0.0 for ind in industry_list},
         'zero_counts': {ind: 0   for ind in industry_list}}
        for _ in range(N_SLOTS)
    ]
    ind_value_history = {ind: [] for ind in industry_list}
    return portfolios, ind_value_history


def compute_master_liquidation(
    industry_list, ind_capital_state, target_alloc, master_pool,
    ind_price, master_floor_pct=0.02
):
    """
    Compute retrospective liquidation amounts master would order.
    Returns (liquidation_costs, freed_cash, liq_orders).
    """
    floor_value       = master_pool * master_floor_pct
    liq_orders        = {}
    freed_cash        = 0.0
    liquidation_costs = 0.0

    ind_current_value = {
        ind: sum(ind_capital_state.get(ind, (0.0, 0.0)))
        for ind in industry_list
    }
    ind_target_value = {ind: target_alloc.get(ind, 0.0) * master_pool
                        for ind in industry_list}

    shrink_inds = [
        (ind, ind_current_value[ind], ind_target_value[ind])
        for ind in industry_list
        if ind_current_value[ind] > ind_target_value[ind]
        and ind_current_value[ind] > floor_value
    ]
    shrink_inds.sort(key=lambda x: (target_alloc.get(x[0], 0.0), -(x[1] - x[2])))

    for ind, current_v, target_v in shrink_inds:
        hold_v, _ = ind_capital_state.get(ind, (0.0, 0.0))
        needed     = current_v - target_v
        max_liq    = max(0.0, min(hold_v, current_v - floor_value))
        liq_amount = min(needed, max_liq)
        if liq_amount > 1e-6:
            liq_orders[ind]    = liq_amount
            # Cost approximation: SEC fee on dollar amount (FINRA TAF needs share count)
            cost               = liq_amount * SEC_FEE_RATE
            freed_cash        += liq_amount - cost
            liquidation_costs += cost

    return liquidation_costs, freed_cash, liq_orders


def step_master(output_dir, portfolios, ind_value_history, industries,
                actual_day, total_avail, day_num, total_days,
                industry_top_scores=None, sigma=None, no_save_master=False):
    """
    Single trading day for master. Skips if actual_day < MASTER_START_DAY.
    ind_value_history must already contain today's values before this is called.
    Returns (best_adj_score, prod_val, slot0_adj_score) in dollars, or (None,None,None).
    """
    if actual_day < MASTER_START_DAY:
        return None, None, None

    industry_list = list(industries.keys())

    if day_num % 10 == 0 or day_num == total_days - 1:
        history_len = min((len(v) for v in ind_value_history.values()), default=0)
        log(f"[master] Day {actual_day + 1}/{total_avail} — running {N_SLOTS} models "
            f"(value_history={history_len} days) ...")

    # ── Resolve industry returns from slot 0's actual performance ────────────────
    # ── Resolve actual industry performance ───────────────────────────────────
    actual_perf = {}
    for ind in industry_list:
        if industry_top_scores and ind in industry_top_scores:
            baseline_ind, slot0_val = industry_top_scores[ind]
            actual_perf[ind] = (slot0_val / baseline_ind - 1.0) if baseline_ind > 0 else 0.0
        else:
            actual_perf[ind] = 0.0

    # Fixed unit price keeps master's portfolio value changes market-driven only.
    ind_price = {ind: IND_UNIT_PRICE for ind in industry_list}

    # ── Build shared input tensor ─────────────────────────────────────────────
    today_t = build_master_features(ind_value_history, industry_list)

    # ── Step 1: reset all slots to slot 0's portfolio ────────────────────────
    def _port_val(p):
        return p['cash'] + sum(
            p['holdings'].get(ind, 0.0) * ind_price.get(ind, 0.0)
            for ind in industry_list)

    pool           = MST_STARTING_CASH
    ref_cash       = portfolios[0]['cash']
    ref_hold       = {ind: portfolios[0]['holdings'].get(ind, 0.0)    for ind in industry_list}
    ref_zeros      = {ind: portfolios[0]['zero_counts'].get(ind, 0)   for ind in industry_list}
    baseline_score = _port_val({'cash': ref_cash, 'holdings': ref_hold})

    for slot in range(N_SLOTS):
        portfolios[slot] = {
            'cash':        ref_cash,
            'holdings':    dict(ref_hold),
            'zero_counts': dict(ref_zeros),
        }

    # ── Step 2: infer → tier decode → liquidate → deploy → apply returns ─────
    slot_tier_maps  = {}
    buy_exec_count  = 0
    sell_exec_count = 0

    for slot in range(N_SLOTS):
        port  = portfolios[slot]
        model = load_slot_model('master', output_dir, slot, MasterNN)
        with torch.inference_mode():
            out = model(today_t)
        del model

        tier_map               = decode_master_tiers(out, industry_list)
        slot_tier_maps[slot]   = tier_map

        # Update consecutive-zero counts; liquidate at 3 in a row
        for ind in industry_list:
            if tier_map[ind] == 0:
                port['zero_counts'][ind] += 1
            else:
                port['zero_counts'][ind] = 0

        for ind in industry_list:
            if port['zero_counts'][ind] >= 3:
                held_units = port['holdings'].get(ind, 0.0)
                if held_units > 1e-9:
                    port['cash']          += _sell_net(held_units, ind_price[ind])
                    port['holdings'][ind]  = 0.0
                    sell_exec_count       += held_units

        # Deploy capital — tier 0 industries receive no new buys
        alloc = tiers_to_alloc(tier_map, industry_list, pool)
        for ind in industry_list:
            if tier_map[ind] == 0:
                continue
            price = ind_price.get(ind, 0.0)
            if price <= 0:
                continue
            diff = alloc[ind] - port['holdings'][ind] * price
            if diff > 1e-6:
                units = min(diff, port['cash']) / price
                if units > 1e-9:
                    port['holdings'][ind] += units
                    port['cash']          -= units * price * BUY_FILL
                    buy_exec_count        += units

        # Apply daily market returns
        for ind in industry_list:
            port['holdings'][ind] *= (1.0 + actual_perf.get(ind, 0.0))

    # ── Step 3: score with FN/FP penalties ───────────────────────────────────
    pred_scores = []
    for slot in range(N_SLOTS):
        port_val   = _port_val(portfolios[slot])
        tier_map   = slot_tier_maps[slot]
        false_negs = sum(1 for ind in industry_list
                         if actual_perf.get(ind, 0.0) < 0 and tier_map[ind] > 0)
        false_pos  = sum(1 for ind in industry_list
                         if actual_perf.get(ind, 0.0) >= 0 and tier_map[ind] == 0)
        adj = port_val * (1.0 - FN_PENALTY * false_negs - FP_PENALTY * false_pos)
        pred_scores.append((slot, adj))

    best_adj   = max(v for _, v in pred_scores)
    slot0_val  = _port_val(portfolios[0])
    slot0_adj  = pred_scores[0][1]

    s0_tiers    = slot_tier_maps.get(0, {})
    tier_counts = [sum(1 for ind in industry_list if s0_tiers.get(ind) == t) for t in range(4)]

    log(f"[master] Day {actual_day + 1}/{total_avail} | "
        f"best=${best_adj:.0f} prod=${baseline_score:.0f} | "
        f"tiers(0/1/2/3)={tier_counts[0]}/{tier_counts[1]}/{tier_counts[2]}/{tier_counts[3]} | "
        f"shares(buy/sell)={buy_exec_count:.0f}/{sell_exec_count:.0f}")

    # TODO: guard rails when all 12 industries are tier 0 for extended periods

    if baseline_score < MST_STARTING_CASH * 0.9:
        log(f"[master] Day {actual_day + 1}/{total_avail} "
            f"Production model (${baseline_score:.2f}) fell below floor — resetting")
        for slot in range(N_SLOTS):
            portfolios[slot] = {
                'cash':        MST_STARTING_CASH,
                'holdings':    {ind: 0.0 for ind in industry_list},
                'zero_counts': {ind: 0   for ind in industry_list},
            }
        return best_adj, slot0_val, slot0_adj

    slot0_own_port = copy.deepcopy(portfolios[0])

    # ── Step 4: selection + mutation (skipped for diagnostic no_save_master runs) ──
    if not no_save_master:
        if best_adj > baseline_score:
            score_vals = [v for _, v in pred_scores]
            mean_ps    = sum(score_vals) / len(score_vals)
            std_ps     = (sum((v - mean_ps) ** 2 for v in score_vals) / len(score_vals)) ** 0.5
            selection_and_mutation(
                'master', output_dir, MasterNN,
                pred_scores, portfolios,
                survival_floor=mean_ps - std_ps,
                inactive_slots=set(),
                actual_day=actual_day, total_avail=total_avail,
                sigma=sigma,
            )
        else:
            half = ELITE_COUNT // 2
            log(f"[master] Day {actual_day + 1}/{total_avail}   "
                f"best={best_adj:.0f} <= baseline={baseline_score:.0f} — "
                f"injecting diversity into bottom {ELITE_COUNT - half} elite slots")
            for inject_rank in range(half, ELITE_COUNT):
                source_rank = inject_rank - half
                elite = load_slot_model('master', output_dir, source_rank, MasterNN)
                blend = blend_model_halfway(elite, MasterNN)
                save_slot_model('master', output_dir, inject_rank, blend)
                portfolios[inject_rank] = copy.deepcopy(portfolios[source_rank])
                del elite, blend

    portfolios[0] = slot0_own_port
    return best_adj, slot0_val, slot0_adj




# ── Single-day wrappers (called from production_v2.py) ────────────────────────

def train_industry_one_day(industry, symbols, day_data, primed_portfolio, model_dir,
                           next_day_data=None, seq_flags=None):
    """
    Run one evolution step seeded from real Alpaca state.

    day_data:      Yesterday's OHLCV — used as model input features and baseline scoring.
    next_day_data: Today's actual OHLCV — used as fill prices (matching offline training
                   where models predict on day N and fills happen on day N+1).
                   If None, fills simulate against day_data (same-day, less accurate).
    seq_flags:     {sym: bool} actual intraday high/low sequence derived from today's
                   1-minute bars (True=low-first, False=high-first).  If None, each
                   symbol gets an independent random 50/50 draw.

    Returns (baseline_score, top_slot_value) or None on error.
    """
    portfolios = [copy.deepcopy(primed_portfolio) for _ in range(N_SLOTS)]
    for p in portfolios:
        p.setdefault('stop_prices', {sym: 0.0 for sym in symbols})
        for sym in symbols:
            p['holdings'].setdefault(sym, 0.0)

    # Load up to 15 days of history from stock_data/.
    # These files should contain data up to and including day_data's date —
    # the last entry IS day_data, so the LSTM sees a full window ending yesterday.
    histories = {}
    for sym in symbols:
        path = os.path.join('stock_data', f"{sym}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                entries = data.get('days', [])[-15:]
                hist = []
                for i, entry in enumerate(entries):
                    raw  = [entry['open'], entry['close'], entry['high'],
                            entry['low'], float(entry['volume'])]
                    prev = [entries[i-1]['open'], entries[i-1]['close'], entries[i-1]['high'],
                            entries[i-1]['low'], float(entries[i-1]['volume'])] if i > 0 else None
                    deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
                    hist.append(raw + deltas)
                histories[sym] = hist
            except Exception as e:
                log(f"WARNING: could not load history for {sym}: {e}")
                histories[sym] = []
        else:
            histories[sym] = []

    day      = {'data': day_data}
    next_day = {'data': next_day_data} if next_day_data else None
    result = step_industry(industry, symbols, model_dir, portfolios, histories,
                           day, actual_day=0, total_avail=1, day_num=0, total_days=1,
                           next_day=next_day, seq_flags=seq_flags)
    if result is None:
        return None
    return result[0], result[1]   # (baseline_score, top_slot_value)


def train_master_one_day(industries, primed_portfolio, model_dir,
                         ind_value_history, industry_top_scores=None):
    """
    Production upkeep wrapper — one evolution step for the master model.
    ind_value_history must already contain today's values before calling.
    Returns (best_adj, slot0_val) from step_master, or (None, None) if skipped.
    """
    industry_list = list(industries.keys())
    portfolios = [copy.deepcopy(primed_portfolio) for _ in range(N_SLOTS)]
    for p in portfolios:
        p.setdefault('zero_counts', {ind: 0 for ind in industry_list})
        for ind in industry_list:
            p['holdings'].setdefault(ind, 0.0)

    result = step_master(model_dir, portfolios, ind_value_history, industries,
                         actual_day=MASTER_START_DAY, total_avail=1,
                         day_num=0, total_days=1,
                         industry_top_scores=industry_top_scores)
    best_adj, slot0_val, _ = result
    return best_adj, slot0_val


# ── Data loading ───────────────────────────────────────────────────────────────

def load_stock_data_from_files(all_symbols, stock_data_dir):
    """Load all JSON files from stock_data_dir and merge into {date: {sym: ohlcv}} dict."""
    all_data = {}
    loaded   = 0
    for sym in all_symbols:
        path = os.path.join(stock_data_dir, f"{sym}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                sym_data = json.load(f)
            for entry in sym_data.get('days', []):
                date_str = entry['date']
                if date_str not in all_data:
                    all_data[date_str] = {}
                all_data[date_str][sym] = {
                    'open':   entry['open'],
                    'high':   entry['high'],
                    'low':    entry['low'],
                    'close':  entry['close'],
                    'volume': float(entry['volume']),
                }
            loaded += 1
        except Exception as e:
            log(f"WARNING: could not load {path}: {e}")
    log(f"Loaded local data for {loaded}/{len(all_symbols)} symbols")
    return all_data


def fetch_stock_data_from_yfinance(all_symbols):
    """Fetch the last 15 days of daily OHLCV for all symbols via yfinance; returns {date: {sym: ohlcv}}."""
    all_data = {}
    try:
        tickers = yf.Tickers(' '.join(all_symbols))
        for sym in all_symbols:
            try:
                hist = tickers.tickers[sym].history(period='1mo', interval='1d')
                if hist.empty:
                    log(f"No data returned for {sym}")
                    continue
                for date, row in hist.tail(15).iterrows():
                    dt       = datetime.fromisoformat(str(date).split('+')[0])
                    date_str = dt.strftime('%Y-%m-%d')
                    if date_str not in all_data:
                        all_data[date_str] = {}
                    all_data[date_str][sym] = {
                        'open':   float(row['Open']),
                        'high':   float(row['High']),
                        'low':    float(row['Low']),
                        'close':  float(row['Close']),
                        'volume': float(row['Volume']),
                    }
            except Exception as e:
                log(f"WARNING: could not fetch {sym} from yfinance: {e}")
    except Exception as e:
        log(f"ERROR fetching from yfinance: {e}")
    return all_data


def trim_stock_data_files(all_symbols, stock_data_dir, keep_days=15):
    """Trim each symbol's JSON file to the last *keep_days* entries to save disk space."""
    trimmed = 0
    for sym in all_symbols:
        path = os.path.join(stock_data_dir, f"{sym}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                payload = json.load(f)
            days = payload.get('days', [])
            if len(days) > keep_days:
                payload['days'] = days[-keep_days:]
                with open(path, 'w') as f:
                    json.dump(payload, f)
                trimmed += 1
        except Exception as e:
            log(f"WARNING: could not trim {path}: {e}")
    if trimmed:
        log(f"Trimmed {trimmed} stock data file(s) to the last {keep_days} days.")


# ── Day-timeout watchdog ───────────────────────────────────────────────────────

def _dump_day_timeout(actual_day, total_days, elapsed, completed_inds, hung_inds, output_dir):
    """Write a JSON diagnostic file when a day exceeds DAY_TIMEOUT_SECS.

    Records elapsed time, which industries completed vs hung, and RSS memory
    of the main process.  Written to data_dump/timeout_day_N.json so it
    survives after the process exits for post-mortem analysis.
    """
    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f'timeout_day_{actual_day + 1}.json')
    payload = {
        'day':                 actual_day + 1,
        'total_days':          total_days,
        'elapsed_seconds':     round(elapsed, 1),
        'timeout_seconds':     DAY_TIMEOUT_SECS,
        'completed_industries': completed_inds,
        'hung_industries':     hung_inds,
        'main_rss_mb':         round(rss_mb, 1),
        'timestamp':           datetime.now().isoformat(),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


# ── numpy IPC helpers ──────────────────────────────────────────────────────────
# Portfolio layout per slot: [cash, holdings[sym0..N-1], stop_prices[sym0..N-1]]
# History layout: (IND_SYMS, HIST_WINDOW, 10) — oldest entries first, zero-padded.
#
# Per-industry files are used instead of shared memmaps to avoid concurrent-
# write conflicts on copy-on-write filesystems (btrfs, ZFS).  Each worker
# reads and writes only its own industry's files; main reads portfolio files
# after all workers complete.

_PORT_COLS = 1 + IND_SYMS * 2   # 25: cash + 12 holdings + 12 stop_prices


def _ipc_paths(ipc_dir, ind_idx):
    """Return (port_path, hist_path, lens_path) for one industry index."""
    base = os.path.join(ipc_dir, f'_ipc_{ind_idx}')
    return f'{base}_port.npy', f'{base}_hist.npy', f'{base}_lens.npy'


def portfolio_to_arr(portfolios, syms):
    """Flatten list-of-dicts to ndarray (N_SLOTS, _PORT_COLS)."""
    arr = np.empty((N_SLOTS, _PORT_COLS), dtype=np.float64)
    for s, p in enumerate(portfolios):
        arr[s, 0] = p['cash']
        for i, sym in enumerate(syms):
            arr[s, 1 + i]            = p['holdings'].get(sym, 0.0)
            arr[s, 1 + IND_SYMS + i] = p['stop_prices'].get(sym, 0.0)
    return arr


def arr_to_portfolio(arr, syms):
    """Rebuild list-of-dicts from ndarray (N_SLOTS, _PORT_COLS)."""
    portfolios = []
    for s in range(N_SLOTS):
        portfolios.append({
            'cash':        float(arr[s, 0]),
            'holdings':    {sym: float(arr[s, 1 + i])            for i, sym in enumerate(syms)},
            'stop_prices': {sym: float(arr[s, 1 + IND_SYMS + i]) for i, sym in enumerate(syms)},
        })
    return portfolios


def history_to_arr(histories, syms):
    """Pack history dict into ndarray (IND_SYMS, HIST_WINDOW, 10) and lengths (IND_SYMS,)."""
    data = np.zeros((IND_SYMS, HIST_WINDOW, 10), dtype=np.float64)
    lens = np.zeros(IND_SYMS, dtype=np.int32)
    for i, sym in enumerate(syms):
        entries = histories.get(sym, [])
        n = len(entries)
        lens[i] = n
        if n:
            data[i, :n] = entries   # oldest first
    return data, lens


def arr_to_history(hist_arr, lens_arr, syms):
    """Reconstruct history dict from ndarray and lengths arrays.  Returns Python floats."""
    histories = {}
    for i, sym in enumerate(syms):
        n = int(lens_arr[i])
        histories[sym] = [[float(x) for x in hist_arr[i, j]] for j in range(n)]
    return histories


# ── Process-pool worker ────────────────────────────────────────────────────────

def _run_industry_proc(ind, syms, output_dir,
                       ipc_dir, ind_idx,
                       day, actual_day, total_days, day_num, num_days,
                       next_day, all_zero_streak, daily_sigma, freeze, seq_flags):
    """Worker entrypoint for ProcessPoolExecutor — reads/writes state via per-industry files.

    Must be a module-level function (not a closure) so that multiprocessing can
    pickle it.  Portfolio and history are exchanged via per-industry numpy binary
    files; each worker reads and writes only its own index, so there are no
    concurrent-write conflicts regardless of the underlying filesystem.
    """
    port_path, hist_path, lens_path = _ipc_paths(ipc_dir, ind_idx)

    portfolios = arr_to_portfolio(np.load(port_path), syms)
    history    = arr_to_history(np.load(hist_path), np.load(lens_path), syms)

    result = step_industry(
        ind, syms, output_dir, portfolios, history,
        day, actual_day, total_days, day_num, num_days,
        next_day=next_day,
        all_zero_streak=all_zero_streak,
        daily_sigma=daily_sigma,
        freeze=freeze,
        seq_flags=seq_flags,
    )

    np.save(port_path, portfolio_to_arr(portfolios, syms))

    return ind, result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Parse CLI args and run multi-pass evolutionary training across all industries and master."""
    parser = argparse.ArgumentParser(
        description="Train neural networks for stock trading (v4: parallel industry thread pool).")
    parser.add_argument('--output',    required=True,  help='Output directory for models')
    parser.add_argument('--load-dir',                  help='Seed top-10 models from this directory')
    parser.add_argument('--start-day', type=int, default=None)
    parser.add_argument('--stop-day',  type=int, default=None)
    parser.add_argument('--preserve-stock-data', action='store_true',
                        help='Do not trim stock_data/ files after training')
    parser.add_argument('--passes',      type=int,   default=1,
                        help='Number of passes over the day range (default: 1)')
    parser.add_argument('--sigma',        type=float, default=0.01,
                        help='Initial mutation sigma for industries (default: 0.01)')
    parser.add_argument('--master-sigma', type=float, default=None,
                        help='Mutation sigma for master (default: same as --sigma)')
    parser.add_argument('--sigma-decay',  type=float, default=0.5,
                        help='Sigma decay per pass (default: 0.5)')
    parser.add_argument('--daily',  action='store_true',
                        help='Run 4 finer-sigma refinement bursts per day after normal selection')
    parser.add_argument('--promote', default='',
                        help='Comma-separated sibling dirs to promote best models to after training '
                             '(e.g. "uat" or "uat,prod" when --output is models/dev)')
    parser.add_argument('--master-only', action='store_true',
                        help='Freeze industry models (no selection/mutation); train master only')
    parser.add_argument('--no-save-master', action='store_true',
                        help='Skip writing master model files (diagnostic runs only)')
    args = parser.parse_args()

    os.makedirs(args.output,  exist_ok=True)
    os.makedirs('stock_data', exist_ok=True)

    if os.path.isdir(DUMP_DIR):
        import shutil
        shutil.rmtree(DUMP_DIR)

    global _console_log_path, _console_log_lines
    _console_log_path  = os.path.join(args.output, 'console_log.txt')
    _console_log_lines = []
    try:
        open(_console_log_path, 'w').close()
    except Exception:
        pass

    industries = INDUSTRIES

    all_symbols = [sym for syms in industries.values() for sym in syms]

    sample_path = os.path.join('stock_data', f"{all_symbols[0]}.json")
    if os.path.exists(sample_path):
        log("Loading stock data from stock_data/ ...")
        all_data = load_stock_data_from_files(all_symbols, 'stock_data')
    else:
        log("No local stock data — fetching from yfinance ...")
        all_data = fetch_stock_data_from_yfinance(all_symbols)

    if not all_data:
        log("ERROR: No market data available. Exiting.")
        return

    all_days   = [{'data': all_data[date]} for date in sorted(all_data.keys())]
    total_days = len(all_days)
    day_start  = args.start_day if args.start_day is not None else 0
    day_end    = args.stop_day  if args.stop_day  is not None else total_days

    log(f"Total trading days available: {total_days}")
    log(f"Training window: days {day_start} to {day_end} ({day_end - day_start} days)")
    log(f"Passes: {args.passes} | sigma: {args.sigma} | sigma-decay: {args.sigma_decay}")
    log(f"Industry process pool: NUM_WORKERS={NUM_WORKERS}")

    csv_path    = os.path.join(args.output, 'training_log.csv')
    ind_names   = list(industries.keys())
    csv_headers = ['pass', 'day'] + ind_names + ['master_best_adj', 'master_slot0_val']
    with open(csv_path, 'w') as csv_f:
        csv_f.write(','.join(csv_headers) + '\n')
    log(f"Training log: {csv_path}")

    def _fmt(v, width=10):
        cap = 10 ** (width - 4) - 0.01   # e.g. width=10 → 999999.99, width=12 → 99999999.99
        return f"{max(-cap, min(cap, float(v))):+0{width}.2f}"

    # ── Multi-pass loop ───────────────────────────────────────────────────────
    orig_mutate = mutate.__code__

    # Tracking for end-of-run industry ranking
    ind_start_scores = {}   # {ind: first-day baseline}
    ind_end_scores   = {}   # {ind: last-day slot0_score} — overwritten each day
    ind_end_prices   = {}   # {ind: {sym: close}} — overwritten each day
    ind_pos_days     = {}   # {ind: count of days slot-0 delta > 0}
    ind_zero_days    = {}   # {ind: count of days slot-0 delta == 0}
    ind_neg_days     = {}   # {ind: count of days slot-0 delta < 0}

    for pass_num in range(args.passes):
        current_sigma = args.sigma * (args.sigma_decay ** pass_num)
        base_master_sigma = args.master_sigma if args.master_sigma is not None else args.sigma
        master_sigma = base_master_sigma * (args.sigma_decay ** pass_num)
        log(f"===== PASS {pass_num + 1}/{args.passes} | sigma={current_sigma:.6f} | master_sigma={master_sigma:.6f} =====")

        import training_v4 as _self
        _orig_mutate = _self.mutate
        def _patched_mutate(model, sigma=current_sigma):
            return _orig_mutate(model, sigma=sigma)
        _self.mutate = _patched_mutate

        log(f"Initialising {len(industries)} industry pools + master ...")
        ind_portfolios = {}
        ind_histories  = {}
        for ind, syms in industries.items():
            log(f"[{sn(ind)}] ===== PASS {pass_num+1} BEGIN | days {day_start}–{day_end} of {total_days} =====")
            ind_portfolios[ind], ind_histories[ind] = init_industry(
                ind, syms, args.output, args.load_dir, all_days, day_start)

        log(f"[master] ===== PASS {pass_num+1} BEGIN | days {day_start}–{day_end} of {total_days} =====")
        mst_portfolios, mst_ind_value_history = init_master(
            args.output, args.load_dir, industries)

        days_slice = all_days[day_start:day_end]
        num_days   = len(days_slice)
        log(f"Starting training: {len(industries)} industries + master, {num_days} days, {NUM_WORKERS} workers")

        ind_streaks = {}   # {ind: consecutive all-zero-trade days}

        # Per-industry IPC files — written by main before each day's workers start;
        # each worker reads its own 3 files and writes back only the portfolio file.
        # Using per-industry files (not a single shared file) avoids concurrent-
        # write conflicts on copy-on-write filesystems (btrfs).
        _ipc_dir   = args.output
        ind_list   = list(industries.keys())
        ind_to_idx = {ind: i for i, ind in enumerate(ind_list)}

        # The executor is created once per pass and kept alive for all days.
        # Workers are forked here — before step_master() ever runs in the main
        # process.  Forking after step_master() would inherit PyTorch's
        # OpenMP/BLAS thread-pool state with locked mutexes but without the
        # threads that own them, causing a futex deadlock on every day after
        # the first.  Keeping workers alive across days also eliminates
        # per-day fork overhead.
        # Python 3.12 spawns workers lazily (on first submit), so fork happens
        # after PyTorch's thread pool is initialized — causing a futex deadlock.
        # 'spawn' starts fresh worker processes that re-import torch cleanly.
        executor = ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            mp_context=__import__('multiprocessing').get_context('spawn'),
        )
        hard_flag_hit = False
        for day_num, day in enumerate(days_slice):
            if hard_flag_hit:
                break
            actual_day          = day_start + day_num
            next_day            = days_slice[day_num + 1] if day_num + 1 < len(days_slice) else None
            industry_top_scores = {}
            ind_best_deltas     = {}
            ind_capital_state   = {}

            # Per-symbol intraday sequence for this day: True=low-first, False=high-first (~50/50)
            seq_flags = {sym: (random.random() < 0.5)
                         for syms in industries.values() for sym in syms}

            day_wall_start = time.monotonic()
            completed_inds = []

            try:
                # ── Write state to per-industry files before launching workers ─
                # Workers read their 3 files (portfolio, history, lens) and write
                # back only the portfolio file.  No pickle round-trip for large data.
                for _ind, _syms in industries.items():
                    _idx = ind_to_idx[_ind]
                    _pp, _hp, _lp = _ipc_paths(_ipc_dir, _idx)
                    np.save(_pp, portfolio_to_arr(ind_portfolios[_ind], _syms))
                    _h_arr, _lens = history_to_arr(ind_histories[_ind], _syms)
                    np.save(_hp, _h_arr)
                    np.save(_lp, _lens)

                # ── Parallel industry phase: dynamic work queue ───────────────
                # Each future returns (ind, result) — portfolio and history are
                # exchanged via per-industry numpy files, not pickle.
                # HardFlagError raised in any worker propagates via future.result()
                # and is caught by the outer handler.
                futures = {
                    executor.submit(
                        _run_industry_proc,
                        ind, syms, args.output,
                        _ipc_dir, ind_to_idx[ind],
                        day, actual_day, total_days, day_num, num_days,
                        next_day,
                        ind_streaks.get(ind, 0),
                        current_sigma if args.daily else None,
                        args.master_only,
                        seq_flags,
                    ): ind
                    for ind, syms in industries.items()
                }
                for future in as_completed(futures, timeout=DAY_TIMEOUT_SECS):
                    ind, result = future.result()
                    completed_inds.append(ind)
                    if result is not None:
                        baseline, top_val, best_delta, top_hold, top_cash, new_streak = result
                        industry_top_scores[ind] = (baseline, top_val)
                        ind_best_deltas[ind]     = top_val - baseline
                        ind_capital_state[ind]   = (top_hold, top_cash)
                        ind_streaks[ind]         = new_streak
                        if ind not in ind_start_scores:
                            ind_start_scores[ind] = baseline
                        ind_end_scores[ind]  = top_val
                        ind_end_prices[ind]  = {
                            sym: day['data'].get(sym, {}).get('close', 0.0)
                            for sym in syms
                        }
                        day_delta = top_val - baseline
                        if day_delta > 0:
                            ind_pos_days[ind]  = ind_pos_days.get(ind, 0) + 1
                        elif day_delta < 0:
                            ind_neg_days[ind]  = ind_neg_days.get(ind, 0) + 1
                        else:
                            ind_zero_days[ind] = ind_zero_days.get(ind, 0) + 1
                    else:
                        ind_best_deltas[ind]   = 0.0
                        ind_capital_state[ind] = (0.0, 0.0)

                # ── Read updated portfolios from per-industry files ───────────
                for _ind, _syms in industries.items():
                    _pp, _, _ = _ipc_paths(_ipc_dir, ind_to_idx[_ind])
                    ind_portfolios[_ind] = arr_to_portfolio(np.load(_pp), _syms)

                # ── Append today's OHLCV to histories in main process ─────────
                # Workers discarded their local history updates; main is canonical.
                for _ind, _syms in industries.items():
                    for _sym in _syms:
                        _d = day['data'].get(_sym)
                        if _d:
                            _raw    = [_d['open'], _d['close'], _d['high'], _d['low'], _d['volume']]
                            _prev   = ind_histories[_ind][_sym][-1][:5] if ind_histories[_ind][_sym] else None
                            _deltas = [r - p for r, p in zip(_raw, _prev)] if _prev else [0.0] * 5
                            ind_histories[_ind][_sym].append(_raw + _deltas)
                            if len(ind_histories[_ind][_sym]) > HIST_WINDOW:
                                ind_histories[_ind][_sym].pop(0)

                # ── Append today's industry values to master history ───────────
                # Uses slot-0's top portfolio value from each industry's worker result.
                for _ind in industries:
                    _top_val = industry_top_scores.get(_ind, (IND_STARTING_CASH, IND_STARTING_CASH))[1]
                    mst_ind_value_history[_ind].append(_top_val)

                master_best_adj, master_slot0_val, _ = step_master(
                    args.output, mst_portfolios, mst_ind_value_history, industries,
                    actual_day, total_avail=(day_end - day_start), day_num=day_num, total_days=num_days,
                    industry_top_scores=industry_top_scores,
                    sigma=master_sigma,
                    no_save_master=args.no_save_master)

                day_elapsed = time.monotonic() - day_wall_start
                if day_elapsed > DAY_TIMEOUT_SECS:
                    hung = [i for i in industries if i not in completed_inds]
                    dump_path = _dump_day_timeout(
                        actual_day, total_days, day_elapsed, completed_inds, hung, args.output)
                    log(f"*** DAY TIMEOUT: Day {actual_day + 1}/{total_days} took "
                        f"{day_elapsed:.0f}s (limit={DAY_TIMEOUT_SECS}s) — "
                        f"master phase overran — data → {dump_path} ***")
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)

            except FuturesTimeoutError:
                day_elapsed = time.monotonic() - day_wall_start
                hung = [i for i in industries if i not in completed_inds]
                dump_path = _dump_day_timeout(
                    actual_day, total_days, day_elapsed, completed_inds, hung, args.output)
                log(f"*** DAY TIMEOUT: Day {actual_day + 1}/{total_days} took "
                    f">{day_elapsed:.0f}s (limit={DAY_TIMEOUT_SECS}s) — "
                    f"{len(hung)} hung industr{'y' if len(hung)==1 else 'ies'}: "
                    f"{[sn(i) for i in hung]} — data → {dump_path} ***")
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

            except HardFlagError as e:
                log(f"*** HARD FLAG — halting training: {e} ***")
                hard_flag_hit = True
                gc.collect()
                continue

            row  = [f"{pass_num + 1:<2}", f"{actual_day + 1:<4}"]
            row += [_fmt(ind_best_deltas.get(ind, 0.0)) for ind in ind_names]
            _mba = master_best_adj   if master_best_adj   is not None else 0.0
            _ms0 = master_slot0_val  if master_slot0_val  is not None else 0.0
            row += [f"{_mba:+.2f}", f"{_ms0:+.2f}"]
            with open(csv_path, 'a') as csv_f:
                csv_f.write(','.join(row) + '\n')

            gc.collect()

            # ── 255-day elite snapshot (one trading year) ─────────────────────
            if (day_num + 1) % 255 == 0:
                _elite_base = os.path.join(
                    os.path.dirname(os.path.normpath(args.output)), 'elite_training')
                _snap_dir = os.path.join(
                    _elite_base, f'pass{pass_num + 1}_day{actual_day + 1}')
                os.makedirs(_snap_dir, exist_ok=True)
                for _ind in industries:
                    _src = os.path.join(args.output, f'{_ind}_model_0.pt')
                    if os.path.exists(_src):
                        shutil.copy2(_src, os.path.join(_snap_dir, f'{_ind}_model_0.pt'))
                _src = os.path.join(args.output, 'master_model_0.pt')
                if os.path.exists(_src):
                    shutil.copy2(_src, os.path.join(_snap_dir, 'master_model_0.pt'))
                log(f"[checkpoint] 255-day elite snapshot saved → {_snap_dir}")

        executor.shutdown(wait=True)

        # ── End-of-pass checkpoints ───────────────────────────────────────────
        for ind, syms in industries.items():
            ports = ind_portfolios[ind]
            log(f"[{sn(ind)}] Pass {pass_num+1} complete — saving top-10 checkpoint ...")
            if days_slice:
                final_scores = [(s, compute_value(ports[s], days_slice[-1]['data'], syms))
                                for s in range(N_SLOTS)]
                final_scores.sort(key=lambda x: x[1], reverse=True)
                top10_meta = [{'slot': s, 'score': v} for s, v in final_scores[:10]]
                save_top10_meta(ind, args.output, top10_meta)
                copy_best_model(ind, args.output, top10_meta)
                scores_str = ', '.join(f"${v:.2f}" for _, v in final_scores[:10])
                log(f"[{sn(ind)}] ===== PASS {pass_num+1} COMPLETE | Top10: [{scores_str}] =====")

        log(f"[master] Pass {pass_num+1} complete — saving top-10 checkpoint ...")

        industry_list = list(industries.keys())
        def _mst_val(p):
            return p['cash'] + sum(
                p['holdings'].get(ind, 0.0) * IND_UNIT_PRICE
                for ind in industry_list
            )
        final_scores  = sorted([(s, _mst_val(mst_portfolios[s])) for s in range(N_SLOTS)],
                               key=lambda x: x[1], reverse=True)
        top10_meta = [{'slot': s, 'score': v} for s, v in final_scores[:10]]
        save_top10_meta('master', args.output, top10_meta)
        copy_best_model('master', args.output, top10_meta)
        scores_str = ', '.join(f"${v:.2f}" for _, v in final_scores[:10])
        log(f"[master] ===== PASS {pass_num+1} COMPLETE | Top10: [{scores_str}] =====")

        _self.mutate = _orig_mutate

    # ── End-of-run industry ranking ───────────────────────────────────────────
    if ind_end_scores:
        ranking = []
        for ind, syms in industries.items():
            start   = ind_start_scores.get(ind, IND_STARTING_CASH)
            end     = ind_end_scores.get(ind, start)
            delta   = end - start
            pct     = delta / start * 100 if start > 0 else 0.0
            pos     = ind_pos_days.get(ind, 0)
            zero    = ind_zero_days.get(ind, 0)
            neg     = ind_neg_days.get(ind, 0)
            prices  = ind_end_prices.get(ind, {})
            valid   = [(s, p) for s, p in prices.items() if p > 0]
            top1    = max(valid, key=lambda x: x[1]) if valid else None
            bottom2 = sorted(valid, key=lambda x: x[1])[:2]
            top3    = ([top1] + bottom2) if top1 else []
            min3    = top1[1] + sum(p for _, p in bottom2) if top1 else 0.0
            ranking.append((ind, start, end, delta, pct, pos, zero, neg, min3, top3))
        ranking.sort(key=lambda x: x[3], reverse=True)

        log("")
        log("=" * 100)
        log("  END-OF-RUN INDUSTRY RANKING  (slot-0 portfolio: start → end)")
        log(f"  {'Rank':<5} {'Industry':<12} {'Start':>9} {'End':>9} {'Delta':>9} {'Return':>7}  {'pos:0:neg':>11}  {'Min-3':>7}  Top-3 stocks")
        log("-" * 100)
        for rank, (ind, start, end, delta, pct, pos, zero, neg, min3, top3) in enumerate(ranking, 1):
            top3_str  = ', '.join(f"{s}=${p:.0f}" for s, p in top3)
            sign      = '+' if delta >= 0 else ''
            ratio_str = f"{pos}:{zero}:{neg}"
            log(f"  {rank:<5} {sn(ind):<12} ${start:>8,.0f} ${end:>8,.0f} {sign}${delta:>7,.0f} {sign}{pct:>5.1f}%  {ratio_str:>11}  ${min3:>6,.0f}  {top3_str}")
        log("=" * 95)
        log("")

    # ── Post-training cleanup ─────────────────────────────────────────────────
    if args.preserve_stock_data:
        log("--preserve-stock-data set: stock_data/ files left untouched.")
    else:
        log("Trimming stock_data/ files to last 15 days ...")
        trim_stock_data_files(all_symbols, 'stock_data')

    log("===== ALL TRAINING COMPLETE =====")

    # ── Dev cleanup: strip mutation slots, keep only elites (0..ELITE_POOL-1) ──
    all_prefixes = list(industries.keys()) + ['master']
    log("Cleaning up dev mutations ...")
    for prefix in all_prefixes:
        cleanup_dev_mutations(prefix, args.output)

    # ── Promote to target environments ────────────────────────────────────────
    if args.promote:
        parent_dir = os.path.dirname(os.path.abspath(args.output))
        for target in [t.strip() for t in args.promote.split(',') if t.strip()]:
            target_dir = os.path.join(parent_dir, target)
            log(f"Promoting to {target_dir} ...")
            promote_models(args.output, target_dir, all_prefixes)


if __name__ == '__main__':
    main()
