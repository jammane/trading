"""
training_v3.py — Parallel evolutionary training loop (2 worker threads, rolling model cache).

v3 vs v2 differences:
  - NUM_THREADS = 2: inference runs in parallel across all slots per industry (one per vCPU)
  - _model_cache: rolling cache — one prefix loaded at a time, evicted after each daily step
  - PortfolioArray: numpy-backed wrapper for vectorised cash/holdings access
  - No SLIPPAGE_RATE on limit fills (v3 trades without slippage)
  - _port_path: slot-level portfolio JSON persisted alongside model weights

Usage identical to training_v2.py.
"""

import argparse
import copy
import gc
import json
import math
import os
import random
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf

from fees import BUY_FILL, FINRA_TAF_MAX, FINRA_TAF_PER_SHARE, SEC_FEE_RATE, SELL_FILL, _sell_net
from models import MasterNN, StockNN
from universe import INDUSTRIES

# ── v3: parallel training — 7 worker threads, relaxed RAM constraints ──────────
NUM_THREADS = 2      # matches 2 vCPUs on droplet
_log_lock   = Lock() # serialise console output across threads

# In-memory model cache: {prefix: [model_0, ..., model_99]}
# Populated on first load; written back to disk only after selection mutates weights.
_model_cache: dict = {}


# ── Daily burst helpers ────────────────────────────────────────────────────────

def _simulate_one_model(model, ref_cash, ref_hold, ref_stop, symbols,
                        day_data, fill_data, history_t, today_t, seq_flags=None):
    """
    Run one model for one day starting from the reference portfolio (v3 trading logic:
    no slippage on limit fills, no concentration cap). Returns (score, port_dict).
    """
    port        = {'cash': ref_cash, 'holdings': dict(ref_hold)}
    stop_prices = dict(ref_stop)
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
            if intraday_qty > 1e-6 and nd_low <= sell_all_price <= nd_high:
                port['holdings'][sym] = 0.0
                port['cash']         += _sell_net(intraday_qty, sell_all_price)

        # Stop loss
        stop_p    = stop_prices.get(sym, 0.0)
        remaining = port['holdings'].get(sym, 0.0)
        if stop_p > 0 and remaining > 1e-6:
            if nd_open <= stop_p:
                port['holdings'][sym] = 0.0
                port['cash']         += _sell_net(remaining, nd_open)
            elif nd_low <= stop_p:
                port['holdings'][sym] = 0.0
                port['cash']         += _sell_net(remaining, stop_p)

        # Limit buy
        if buy_qty > 1e-6 and buy_price > 0:
            if nd_open <= buy_price:
                fill_price = nd_open
            elif nd_low <= buy_price <= nd_high:
                fill_price = buy_price
            else:
                fill_price = 0.0
            if fill_price > 0:
                affordable = port['cash'] / (fill_price * BUY_FILL)
                buy_amount = min(buy_qty, affordable)
                if buy_amount > 1e-6:
                    port['holdings'][sym]  = port['holdings'].get(sym, 0.0) + buy_amount
                    port['cash']          -= buy_amount * fill_price * BUY_FILL
                    stop_prices[sym]       = stop_loss

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
        if cur_qty > 1e-6 and nd_low <= sell_all_price <= nd_high:
            port['holdings'][sym] = 0.0
            port['cash']         += _sell_net(cur_qty, sell_all_price)

    score = port['cash'] + sum(
        port['holdings'].get(sym, 0.0) * fill_data.get(sym, {}).get('close', 0.0)
        for sym in symbols)
    return score, port


def _run_daily_burst(prefix, output_dir, symbols, burst_sigma,
                     ref_cash, ref_hold, ref_stop,
                     day_data, fill_data, history_t, today_t,
                     portfolios, actual_day, total_avail, seq_flags=None):
    """
    v3 variant: generate 200 burst mutants (10 per elite), merge top-ELITE_COUNT
    with current elites, save winners to disk, update PortfolioArray in place.
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
        model     = load_slot_model(prefix, output_dir, rank, StockNN)
        slot_dict = portfolios[rank]
        score     = slot_dict['cash'] + sum(
            slot_dict['holdings'].get(sym, 0.0) * fill_data.get(sym, {}).get('close', 0.0)
            for sym in symbols)
        current_elites.append((model, score, {'cash': slot_dict['cash'],
                                               'holdings': dict(slot_dict['holdings'])}))

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
        k       = min(n_avg, ELITE_COUNT)
        ports_k = [portfolios[i] for i in range(k)]
        wavg_m  = compute_weighted_avg_model(prefix, output_dir, list(range(k)), new_scores[:k], StockNN)
        wavg_p  = compute_weighted_avg_portfolio(ports_k, new_scores[:k])
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


# ── Industry display name mapping (log output only) ──────────────────────────
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
    return SHORT_NAMES.get(industry, industry[:9]).ljust(9)[:9]


_console_log_lines = []
_console_log_path  = None   # set at startup by main()
_CONSOLE_LOG_MAX   = 200


def log(msg):
    """Timestamped, immediately-flushed console output — thread-safe, rolling 200-line file buffer."""
    global _console_log_lines
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
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


# ── Pool constants ────────────────────────────────────────────────────────────
N_SLOTS              = 200
ELITE_COUNT          = 17
WAVG_COUNT           = 3
ELITE_POOL           = ELITE_COUNT + WAVG_COUNT   # 20
MUTATIONS_PER_PARENT = 9
IND_STARTING_CASH    = 25_000.0
MST_STARTING_CASH    = 300_000.0
MAX_SINGLE_STOCK_PCT = 0.60
IND_UNIT_PRICE       = 25_000.0
DUMP_DIR             = 'data_dump'

MASTER_LOOKBACKS     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 90]
MASTER_POLY3_WINDOWS = [10, 30, 60, 90]
MASTER_START_DAY     = 30
TIER_WEIGHTS         = {1: 1.0, 2: 1.5, 3: 2.25}
_NULL_DENOM          = TIER_WEIGHTS[1] + TIER_WEIGHTS[2] + TIER_WEIGHTS[3]  # 4.75


def _dump_day(actual_day, prefix, flag_type, baseline, best_delta, pct_gain, scores, day_data, fill_data, models=None):
    """Write diagnostic JSON for a flagged day to data_dump/day_{N}/."""
    day_dir = os.path.join(DUMP_DIR, f"day_{actual_day + 1}")
    os.makedirs(day_dir, exist_ok=True)
    payload = {
        'industry':   prefix,
        'day':        actual_day + 1,
        'flag':       flag_type,
        'baseline':   round(baseline, 4),
        'best_delta': round(best_delta, 4),
        'pct_gain':   round(pct_gain, 4),
        'scores':     [[s, round(v, 4)] for s, v in sorted(scores, key=lambda x: x[1], reverse=True)],
        'day_data':   day_data,
        'fill_data':  fill_data,
    }
    if models is not None:
        payload['models'] = models
    path = os.path.join(day_dir, f"{prefix}.json")
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def _fmt_slot(slot):
    """Convert a slot index to the elite.mutation display label."""
    if slot < ELITE_COUNT: return f"{slot}.0"
    if slot == ELITE_COUNT:     return "w5"
    if slot == ELITE_COUNT + 1: return "w10"
    if slot == ELITE_COUNT + 2: return "w15"
    parent   = (slot - ELITE_POOL) // MUTATIONS_PER_PARENT
    mutation = (slot - ELITE_POOL) % MUTATIONS_PER_PARENT + 1
    return f"{parent}.{mutation}"


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
        else:
            portfolio['holdings'][sym] = 0.0
    return val


def _mst_hist_at(history, lookback):
    idx = len(history) - 1 - lookback
    return history[max(0, idx)]

def _mst_window(history, n_days):
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
    logits = out_logits.view(12, 4)
    probs  = F.softmax(logits, dim=1)
    tiers  = probs.argmax(dim=1).tolist()
    return {ind: tiers[i] for i, ind in enumerate(industry_list)}

def tiers_to_alloc(tier_map, industry_list, available_cash):
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
    Decode MasterNN output (1, 36) or (36,) into allocation and liquidation signals.

    predicted: tensor
      [:12]  softmax allocation weights (sum to 1)
      [12:24] sigmoid liquidation depth   (0=hold, 1=liquidate to floor)
      [24:36] sigmoid liquidation trigger (0=skip, 1=execute)

    Returns:
      alloc_prop:  {ind: fraction}  40% cap, 2% floor enforced
      liq_depth:   {ind: float}     0-1
      liq_trigger: {ind: float}     0-1
    """
    n     = len(industry_list)
    floor = 0.02
    cap   = 0.40

    # Flatten to 1D if needed
    p = predicted.squeeze()
    vals     = p.tolist() if hasattr(p, 'tolist') else list(p)
    weights  = vals[:12]   if len(vals) >= 12 else vals + [1.0/n]*(12-len(vals))
    depths   = vals[12:24] if len(vals) >= 24 else [0.5]*12
    triggers = vals[24:36] if len(vals) >= 36 else [0.5]*12

    # Hybrid allocation: top industry gets full cap; rest filled proportionally.
    w_map   = dict(zip(industry_list, weights))
    top_ind = max(w_map, key=w_map.get)

    alloc_prop          = {ind: floor for ind in industry_list}
    alloc_prop[top_ind] = cap

    others  = [ind for ind in industry_list if ind != top_ind]
    premium = 1.0 - cap - floor * len(others)

    if premium > 1e-9 and others:
        uncapped = others[:]
        while premium > 1e-9 and uncapped:
            tw = sum(w_map[ind] for ind in uncapped)
            if tw <= 0:
                break
            proposed     = {ind: alloc_prop[ind] + premium * w_map[ind] / tw for ind in uncapped}
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


# ── PortfolioArray: vectorized wrapper for 100 portfolio dicts ────────────────

class _CashView:
    """Numpy-style accessor for the cash field across all slots."""
    __slots__ = ('_slots',)
    def __init__(self, slots):
        """Initialise view over slot cash fields."""
        self._slots = slots

    def __getitem__(self, idx):
        """Return cash for slot *idx*, or ndarray slice."""
        if isinstance(idx, slice):
            return np.array([s['cash'] for s in self._slots[idx]], dtype=np.float64)
        return self._slots[idx]['cash']

    def __setitem__(self, idx, val):
        """Set cash for slot *idx* or broadcast scalar to slice."""
        if isinstance(idx, slice):
            fv = float(val)
            for s in self._slots[idx]:
                s['cash'] = fv
        else:
            self._slots[idx]['cash'] = float(val)


class _HoldingsView:
    """Numpy-style accessor for the holdings array across all slots."""
    __slots__ = ('_slots', '_symbols')
    def __init__(self, slots, symbols):
        """Initialise view over slot holdings fields."""
        self._slots   = slots
        self._symbols = symbols

    def __getitem__(self, idx):
        """Return holdings for slot *idx* as a 1D ndarray, or slice as 2D."""
        if isinstance(idx, tuple):
            slot_idx, col_idx = idx
            row = np.array([self._slots[slot_idx]['holdings'].get(sym, 0.0)
                            for sym in self._symbols], dtype=np.float64)
            return row[col_idx]
        elif isinstance(idx, slice):
            return np.array([[s['holdings'].get(sym, 0.0) for sym in self._symbols]
                             for s in self._slots[idx]], dtype=np.float64)
        else:
            h = self._slots[idx]['holdings']
            return np.array([h.get(sym, 0.0) for sym in self._symbols], dtype=np.float64)

    def __setitem__(self, idx, val):
        """Set holdings for slot *idx* from a 1D array or broadcast to slice."""
        arr = np.asarray(val, dtype=np.float64).ravel()
        if isinstance(idx, tuple):
            slot_idx, _ = idx
            self._slots[slot_idx]['holdings'] = {
                sym: float(arr[j]) for j, sym in enumerate(self._symbols)}
        elif isinstance(idx, slice):
            hd = {sym: float(arr[j]) for j, sym in enumerate(self._symbols)}
            for s in self._slots[idx]:
                s['holdings'] = dict(hd)
        else:
            self._slots[idx]['holdings'] = {
                sym: float(arr[j]) for j, sym in enumerate(self._symbols)}


class PortfolioArray:
    """
    Vectorized wrapper for 100 portfolio dicts.
    Each slot: {'cash': float, 'holdings': {sym: float}}

    Exposes:
      .cash              → _CashView    (numpy-style [i] / [:]=scalar)
      .holdings          → _HoldingsView (numpy-style [i]→1D, [i,:]→1D, [:]=1D broadcast)
      .sym_prices(data)  → np.ndarray (n_syms,)
      .all_values(prices)→ np.ndarray (100,)
      .reset_all_to(cash, hold_array) → broadcast-reset all slots
    """
    def __init__(self, symbols, n=N_SLOTS, init_cash=IND_STARTING_CASH):
        """Allocate N portfolio dicts with init_cash each; wire up CashView and HoldingsView."""
        self.symbols  = list(symbols)
        self.n        = n
        self._slots   = [
            {'cash': float(init_cash), 'holdings': {sym: 0.0 for sym in symbols}}
            for _ in range(n)
        ]
        self.cash     = _CashView(self._slots)
        self.holdings = _HoldingsView(self._slots, self.symbols)

    def __getitem__(self, idx):
        """Return the raw slot dict at *idx*."""
        return self._slots[idx]

    def __setitem__(self, idx, val):
        """Replace slot at *idx* with *val* (dict or duck-type with cash/holdings)."""
        if isinstance(val, dict):
            self._slots[idx] = val
        else:
            self._slots[idx] = {'cash': float(val['cash']),
                                'holdings': dict(val['holdings'])}

    def sym_prices(self, day_data):
        """Return a (n_syms,) float64 array of close prices from day_data."""
        return np.array([
            day_data[sym]['close'] if sym in day_data else 0.0
            for sym in self.symbols
        ], dtype=np.float64)

    def all_values(self, prices):
        """Return a (N,) float64 array of total portfolio value per slot at *prices*."""
        p      = np.asarray(prices, dtype=np.float64)
        result = np.empty(self.n, dtype=np.float64)
        syms   = self.symbols
        for i, slot in enumerate(self._slots):
            v = slot['cash']
            h = slot['holdings']
            for j, sym in enumerate(syms):
                v += h.get(sym, 0.0) * float(p[j])
            result[i] = v
        return result

    def reset_all_to(self, baseline_cash, baseline_hold):
        """baseline_hold: 1D numpy array indexed by symbol position, or dict."""
        cash_f = float(baseline_cash)
        if isinstance(baseline_hold, dict):
            hd = {sym: float(baseline_hold.get(sym, 0.0)) for sym in self.symbols}
        else:
            arr = np.asarray(baseline_hold, dtype=np.float64)
            hd  = {sym: float(arr[j]) for j, sym in enumerate(self.symbols)}
        for slot in self._slots:
            slot['cash']     = cash_f
            slot['holdings'] = dict(hd)


# ── Per-slot model I/O (only 1 model weight tensor in RAM at a time) ──────────

def _model_path(prefix, directory, slot):
    """Return the .pt file path for a given prefix, directory, and slot index."""
    return os.path.join(directory, f"{prefix}_model_{slot}.pt")

def _port_path(prefix, directory, slot):
    """Return the per-slot portfolio JSON path (v3 only)."""
    return os.path.join(directory, f"{prefix}_port_{slot}.json")

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
    """Load one model from disk; returns an untrained default if file is missing."""
    path  = _model_path(prefix, directory, slot)
    model = model_class()
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, weights_only=True))
        except Exception as e:
            log(f"WARNING: {prefix} slot {slot} load failed ({e}) — using random weights")
    return model


def load_all_models(prefix, directory, model_class, n=N_SLOTS):
    """
    Return the in-memory model list for this prefix, loading from disk on first call.
    Subsequent calls return the cached list directly — no disk I/O.
    v3 only: prefix is evicted from cache after each daily step; peak = 1 prefix × 200 models.
    """
    if prefix not in _model_cache:
        models = []
        for slot in range(n):
            models.append(load_slot_model(prefix, directory, slot, model_class))
        _model_cache[prefix] = models
    return _model_cache[prefix]


def flush_model_cache(prefix, directory):
    """Write all cached models for this prefix back to disk after selection mutates them."""
    if prefix not in _model_cache:
        return
    for slot, model in enumerate(_model_cache[prefix]):
        save_slot_model(prefix, directory, slot, model)


def invalidate_cache(prefix):
    """Remove a prefix from the cache (forces reload from disk on next access)."""
    _model_cache.pop(prefix, None)


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


# ── Weighted average (streaming: load one model at a time) ────────────────────

def _normalize_weights(values):
    """Clip negatives to zero and normalise *values* to sum to 1.0; returns equal weights on all-zero."""
    clipped = [max(float(v), 0.0) for v in values]
    total   = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [v / total for v in clipped]


def compute_weighted_avg_model(prefix, directory, slots, values, model_class):
    """
    Build a weighted-average model from the given slots.
    Loads exactly one model at a time — never more than 1 model tensor in RAM.
    """
    weights    = _normalize_weights(values)
    avg_state  = None
    int_state  = {}

    with torch.no_grad():
        for slot, weight in zip(slots, weights):
            m     = load_slot_model(prefix, directory, slot, model_class)
            state = m.state_dict()
            if avg_state is None:
                avg_state = {
                    k: v.clone().float() * weight
                    for k, v in state.items()
                    if torch.is_floating_point(v)
                }
                int_state = {
                    k: v.clone()
                    for k, v in state.items()
                    if not torch.is_floating_point(v)
                }
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
    """
    Create a new random model then blend its weights halfway with elite_model.
    Result is a model that starts from a random position but is pulled toward
    the elite's learned direction — diversity injection without full reset.
    """
    random_model = model_class()
    elite_state  = elite_model.state_dict()
    random_state = random_model.state_dict()
    blended = {}
    for k in elite_state:
        if torch.is_floating_point(elite_state[k]):
            blended[k] = 0.5 * elite_state[k] + 0.5 * random_state[k]
        else:
            blended[k] = elite_state[k].clone()
    blended_model = model_class()
    blended_model.load_state_dict(blended)
    return blended_model


# ── Pool initialisation ────────────────────────────────────────────────────────

def initialise_pool(prefix, directory, load_models_dir, model_class, n=N_SLOTS):
    """
    Ensure all n slot files exist in `directory`.
    Slots 0–(ELITE_POOL-1) → loaded from load_models_dir (or random if absent).
    Slots ELITE_POOL–(n-1) → mutations of elite slots, one at a time.
    Already-present files are kept as-is.
    """
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
    scores,          # list of (slot, value) for all N_SLOTS models
    portfolios,      # PortfolioArray or list indexed by slot
    survival_floor,  # minimum portfolio value to be considered a survivor
    inactive_slots=None,  # set of slot indices excluded from elite candidacy (zero-trade filter)
    actual_day=None, total_avail=None,
    sigma=None,
):
    """
    Keeps top ELITE_COUNT direct elites (slots 0..16), plus three weighted-average
    keepers (w5/w10/w15 at slots 17/18/19 = ELITE_COUNT..ELITE_COUNT+2).
    Fills slots ELITE_POOL..N_SLOTS-1 (180 slots) with 9 mutations per parent,
    assigned deterministically: parent = (slot - ELITE_POOL) // MUTATIONS_PER_PARENT.
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
    top_elite   = surviving[:min(ELITE_COUNT, len(surviving))]
    elite_slots = [s for s, _ in top_elite]
    elite_vals  = [v for _, v in top_elite]

    # Collect elite models/portfolios first (avoid read-after-write if slots swap)
    elite_models = []
    elite_ports  = []
    for old_slot, _ in top_elite:
        m = (_model_cache[prefix][old_slot]
             if prefix in _model_cache
             else load_slot_model(prefix, directory, old_slot, model_class))
        elite_models.append(m)
        if hasattr(portfolios, 'reset_all_to'):
            elite_ports.append((float(portfolios.cash[old_slot]),
                                portfolios.holdings[old_slot].copy()))
        else:
            elite_ports.append(copy.deepcopy(portfolios[old_slot]))

    # Copy direct elites to slots 0..ELITE_COUNT-1
    for new_slot, (model, port) in enumerate(zip(elite_models, elite_ports)):
        if prefix in _model_cache:
            _model_cache[prefix][new_slot] = model
        save_slot_model(prefix, directory, new_slot, model)
        if hasattr(portfolios, 'reset_all_to'):
            portfolios.cash[new_slot]        = port[0]
            portfolios.holdings[new_slot, :] = port[1]
        else:
            portfolios[new_slot] = port

    # Weighted averages: w5 (top-5), w10 (top-10), w15 (top-15) at slots 17/18/19
    wavg_configs = [
        (ELITE_COUNT,     min(5,  len(surviving))),
        (ELITE_COUNT + 1, min(10, len(surviving))),
        (ELITE_COUNT + 2, min(15, len(surviving))),
    ]
    for wavg_slot, k in wavg_configs:
        top_k       = surviving[:k]
        top_k_slots = [s for s, _ in top_k]
        top_k_vals  = [v for _, v in top_k]
        wm          = compute_weighted_avg_model(prefix, directory, top_k_slots, top_k_vals, model_class)
        _dicts      = ([portfolios[s].to_dict() for s in top_k_slots]
                       if hasattr(portfolios[0], 'to_dict') else [portfolios[s] for s in top_k_slots])
        wp          = compute_weighted_avg_portfolio(_dicts, top_k_vals)
        if prefix in _model_cache:
            _model_cache[prefix][wavg_slot] = wm
        save_slot_model(prefix, directory, wavg_slot, wm)
        portfolios[wavg_slot] = wp

    # Deterministic mutations: 9 per parent, parent = (slot - ELITE_POOL) // 9
    parent_assignments = defaultdict(list)
    for i, slot in enumerate(range(ELITE_POOL, N_SLOTS)):
        parent_assignments[i // MUTATIONS_PER_PARENT].append(slot)

    for parent_idx, child_slots in parent_assignments.items():
        parent_slot = parent_idx  # direct elite slot index
        parent = (_model_cache[prefix][parent_slot]
                  if prefix in _model_cache
                  else load_slot_model(prefix, directory, parent_slot, model_class))
        for child_slot in child_slots:
            child = mutate(parent, sigma=sigma) if sigma is not None else mutate(parent)
            if prefix in _model_cache:
                _model_cache[prefix][child_slot] = child
            save_slot_model(prefix, directory, child_slot, child)
            if hasattr(portfolios, 'reset_all_to'):
                portfolios.cash[child_slot]        = portfolios.cash[parent_slot]
                portfolios.holdings[child_slot, :] = portfolios.holdings[parent_slot, :]
            else:
                portfolios[child_slot] = copy.deepcopy(portfolios[parent_slot])

    elite_display = [_fmt_slot(s) for s in range(len(top_elite))]
    log(f"[{sn(prefix)}]   Selection done | elite={elite_display} | "
        f"top score=${max(elite_vals):.2f} | mutations={N_SLOTS - ELITE_POOL}")

    # Seed dominance check: if 13+ of 17 direct elites are original seed slots (<ELITE_POOL),
    # inject diversity by replacing bottom half of elites with halfway blends from top half.
    seed_elite_count = sum(1 for s in elite_slots if s < ELITE_POOL)
    if seed_elite_count >= ELITE_COUNT - 4:
        day_tag = f" Day {actual_day + 1}/{total_avail}" if actual_day is not None else ""
        log(f"[{sn(prefix)}]{day_tag}   Seed dominance ({seed_elite_count}/{ELITE_COUNT} elites are seeds) — injecting diversity")
        half        = ELITE_COUNT // 2
        inject_slots = elite_slots[half:]
        source_slots = elite_slots[:half]
        for inject_slot, source_slot in zip(inject_slots, source_slots):
            elite  = load_slot_model(prefix, directory, source_slot, model_class)
            blend  = blend_model_halfway(elite, model_class)
            if prefix in _model_cache:
                _model_cache[prefix][inject_slot] = blend
            save_slot_model(prefix, directory, inject_slot, blend)
            portfolios[inject_slot] = copy.deepcopy(portfolios[source_slot])
            del elite, blend
        log(f"[{sn(prefix)}]{day_tag}   Seed dominance injection complete")

    return portfolios


# ── Industry training ──────────────────────────────────────────────────────────

def init_industry(industry, symbols, output_dir, load_models_dir, all_days, day_start):
    """
    One-time setup: initialise pool, portfolios, histories.
    Pre-loads up to 15 days of history when day_start > 0.
    Returns (portfolios, histories, stop_prices_all).
    """
    initialise_pool(industry, output_dir, load_models_dir, StockNN)

    portfolios      = PortfolioArray(symbols, n=N_SLOTS, init_cash=IND_STARTING_CASH)
    stop_prices_all = [{sym: 0.0 for sym in symbols} for _ in range(N_SLOTS)]
    histories       = {sym: [] for sym in symbols}

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

    return portfolios, histories, stop_prices_all


def step_industry(industry, symbols, output_dir, portfolios, histories,
                  day, actual_day, total_avail, day_num, total_days,
                  stop_prices_all=None,
                  next_day=None, all_zero_streak=0, daily_sigma=None,
                  freeze=False, seq_flags=None):
    """
    Process a single trading day for one industry.
    Orders are placed using today's model predictions but filled against
    next_day's OHLCV (simulating post-close order submission).
    If next_day is None (last day of window), falls back to same-day fills.
    stop_prices_all: list of 100 {sym: stop_price} dicts (threaded from init_industry).
    all_zero_streak: consecutive days where all 100 models were filtered.
    Memory contract: exactly 1 model tensor in RAM at any point.
    Mutates portfolios and histories in place.
    Returns: (baseline_score, top_slot_value, best_delta, top_holdings_value,
              top_cash_value, new_all_zero_streak)
    """
    if stop_prices_all is None:
        stop_prices_all = [{sym: 0.0 for sym in symbols} for _ in range(N_SLOTS)]

    day_data  = day['data']
    fill_data = next_day['data'] if next_day is not None else day_data
    hist_lengths = [len(histories[sym]) for sym in symbols]
    num_past = min(hist_lengths) if hist_lengths else 0

    if day_num % 10 == 0 or day_num == total_days - 1:
        log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} — running {N_SLOTS} models "
            f"(history={num_past}/15 days warm) ...")

    # ── Step 1: reset all slots to slot 0's portfolio (matches v2 logic) ───
    prices        = portfolios.sym_prices(day_data)
    baseline_cash = float(portfolios.cash[0])
    baseline_hold = portfolios.holdings[0].copy()          # shape (n_syms,)
    portfolios.reset_all_to(baseline_cash, baseline_hold)  # single broadcast op
    baseline_score = float(baseline_cash + baseline_hold @ prices)

    # ── Step 2: batched inference + parallel trade execution ────────────────
    # history_t built once (shared OHLCV); today_t built per-slot (adds state).
    # Trade execution (pure Python, per-slot state) runs in parallel threads.
    all_models       = load_all_models(industry, output_dir, StockNN)
    buy_exec_count   = 0
    sell_exec_count  = 0
    slot_trade_count = [0] * N_SLOTS

    # ── Pre-compute rolling stats and industry aggregates ───────────────────
    # Used to build the new 208-feature per-timestep input
    sym_stats = {}
    for sym in symbols:
        h = histories[sym]
        if len(h) >= 2:
            closes  = [r[1] for r in h]
            vols    = [r[4] for r in h]
            hi15    = max(r[2] for r in h)
            lo15    = min(r[3] for r in h)
            avg_c   = sum(closes) / len(closes)
            avg_v   = sum(vols)   / len(vols)   if sum(vols)   > 0 else 1.0
            dvols   = [r[0] * r[4] for r in h]
            avg_dv  = sum(dvols)  / len(dvols)  if sum(dvols)  > 0 else 1.0
            std_c   = (sum((c - avg_c)**2 for c in closes) / len(closes)) ** 0.5
            sym_stats[sym] = {
                'hi15': hi15, 'lo15': lo15, 'avg_c': avg_c if avg_c > 0 else 1.0,
                'avg_v': avg_v, 'avg_dv': avg_dv,
                'volatility': std_c / avg_c if avg_c > 0 else 0.0,
            }
        else:
            sym_stats[sym] = {'hi15': 1.0, 'lo15': 0.0, 'avg_c': 1.0,
                              'avg_v': 1.0, 'avg_dv': 1.0, 'volatility': 0.0}

    # history_t shared across all slots: (1,15,60) OHLCV only, oldest first
    history_rows = []
    for t in range(14, -1, -1):
        row = []
        for sym in symbols:
            h = histories[sym]
            row += list(h[-(t+1)][:5]) if len(h) > t else [0.0] * 5
        history_rows.append(row)
    history_t_shared = torch.tensor(history_rows, dtype=torch.float32).unsqueeze(0)  # (1,15,60)

    # today features shared (without state): raw_t + dlt_t + stats per sym + ind_aggs
    today_shared = []
    today_dl_shared = []
    for sym in symbols:
        st    = sym_stats[sym]
        d     = day_data.get(sym, {})
        raw_t = [d.get('open', 0), d.get('close', 0), d.get('high', 0),
                 d.get('low', 0), d.get('volume', 0)]
        h     = histories[sym]
        prev  = h[-1][:5] if h else None
        dlt_t = [raw_t[i] - prev[i] for i in range(5)] if prev else [0.0] * 5
        rng   = max(st['hi15'] - st['lo15'], 1e-9)
        today_shared += (raw_t + dlt_t +
                         [(raw_t[1] - st['lo15']) / rng,
                          raw_t[1] / st['avg_c'],
                          st['volatility'],
                          raw_t[4] / st['avg_v'],
                          (raw_t[0] * raw_t[4]) / st['avg_dv']])
        today_dl_shared.append(dlt_t)
    if today_dl_shared:
        tr = list(zip(*today_dl_shared))
        for tp in tr: today_shared += [max(tp), min(tp), sum(tp)/len(tp)]
    else:
        today_shared += [0.0] * 15
    # today_shared is now 195 values (12×15 + 15 ind_agg); state (13) appended per-slot

    def _build_stock_input(slot):
        """Build history_t (1,15,60) and today_t (1,208) for one slot."""
        port      = portfolios[slot]
        state_vec = [port['cash']] + [port['holdings'].get(sym, 0.0) for sym in symbols]
        today_t   = torch.tensor(today_shared + state_vec, dtype=torch.float32).unsqueeze(0)  # (1,208)
        return history_t_shared, today_t

    all_outputs = [None] * N_SLOTS   # (12, 4) per slot

    def _batch_infer(slot):
        history_t_s, today_t_s = _build_stock_input(slot)
        with torch.inference_mode():
            out = all_models[slot](history_t_s, today_t_s)
        all_outputs[slot] = out.view(12, 4)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        list(pool.map(_batch_infer, range(N_SLOTS)))

    # Trade execution — each thread owns its slot
    def _trade(slot):
        port = portfolios[slot]
        out  = all_outputs[slot]   # (12, 4) all ReLU
        local_buys = local_sells = 0
        stop_prices = stop_prices_all[slot]

        # ── Open + low phase: partial sells, gap sell_all, stops, limit buys ───
        for j, sym in enumerate(symbols):
            if sym not in day_data:
                continue
            buy_qty, buy_price_frac, sell_all_price_frac, sell_qty = out[j].tolist()
            cur_qty = port['holdings'][sym]
            low_t   = day_data[sym]['low']
            high_t  = day_data[sym]['high']
            span_t  = max(high_t - low_t, 1e-9)
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
                sell_amount = min(sell_qty, cur_qty)
                port['holdings'][sym] -= sell_amount
                port['cash']          += _sell_net(sell_amount, nd_open)
                local_sells           += sell_amount

            # Gap-up sell_all at open (both sequences)
            cur_qty_after = port['holdings'][sym]
            if cur_qty_after > 1e-6 and nd_open >= sell_all_price:
                port['holdings'][sym]  = 0.0
                port['cash']          += _sell_net(cur_qty_after, nd_open)
                local_sells           += cur_qty_after

            # High-first: intraday sell_all fires at the high before the low
            if not low_first:
                intraday_qty = port['holdings'][sym]
                if intraday_qty > 1e-6 and nd_low <= sell_all_price <= nd_high:
                    port['holdings'][sym]  = 0.0
                    port['cash']          += _sell_net(intraday_qty, sell_all_price)
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
                    port['cash']          += _sell_net(remaining, stop_p)
                    local_sells           += remaining

            # Buy at open (gap-down) or intraday low
            if buy_qty > 1e-6 and buy_price > 0:
                if nd_open <= buy_price:
                    fill_price = nd_open
                elif nd_low <= buy_price <= nd_high:
                    fill_price = buy_price
                else:
                    fill_price = 0.0

                if fill_price > 0:
                    affordable = port['cash'] / (fill_price * BUY_FILL)
                    buy_amount = min(buy_qty, affordable)
                    if buy_amount > 1e-6:
                        port['holdings'][sym]    += buy_amount
                        port['cash']             -= buy_amount * fill_price * BUY_FILL
                        stop_prices[sym]          = stop_loss
                        local_buys               += buy_amount

        # ── High phase: intraday sell_all for low-first symbols only ─────────
        for j, sym in enumerate(symbols):
            if sym not in day_data:
                continue
            if seq_flags and not seq_flags.get(sym, True):
                continue  # high-first: sell_all already fired before the low
            _, _, sell_all_price_frac, _ = out[j].tolist()
            low_t   = day_data[sym]['low']
            high_t  = day_data[sym]['high']
            span_t  = max(high_t - low_t, 1e-9)
            sell_all_price = low_t + sell_all_price_frac * span_t

            nd      = fill_data.get(sym, day_data.get(sym, {}))
            nd_low  = nd.get('low',  day_data[sym]['low'])
            nd_high = nd.get('high', day_data[sym]['high'])

            cur_qty = port['holdings'][sym]
            if cur_qty > 1e-6 and nd_low <= sell_all_price <= nd_high:
                port['holdings'][sym]  = 0.0
                port['cash']          += _sell_net(cur_qty, sell_all_price)
                local_sells           += cur_qty
        return slot, local_buys, local_sells

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        for slot, lb, ls in pool.map(_trade, range(N_SLOTS)):
            buy_exec_count        += lb
            sell_exec_count       += ls
            slot_trade_count[slot] = lb + ls

    # ── Step 3: update the shared market-history window ──────────────────────
    for sym in symbols:
        if sym in day_data:
            d      = day_data[sym]
            raw    = [d['open'], d['close'], d['high'], d['low'], d['volume']]
            prev   = histories[sym][-1][:5] if histories[sym] else None
            deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
            histories[sym].append(raw + deltas)
            if len(histories[sym]) > 15:
                histories[sym].pop(0)

    # ── Step 4: vectorized scoring ───────────────────────────────────────────
    prices_end     = portfolios.sym_prices(day_data)
    vals_end       = portfolios.all_values(prices_end)            # shape (100,)
    best_score     = float(vals_end.max())
    worst_score    = float(vals_end.min())
    best_delta     = best_score  - baseline_score
    worst_delta    = worst_score - baseline_score

    # Elite portfolio value stats (slots 0..ELITE_COUNT-1)
    _elite_arr     = vals_end[:ELITE_COUNT]
    elite_max_val  = float(_elite_arr.max())
    elite_min_val  = float(_elite_arr.min())
    elite_mean_val = float(_elite_arr.mean())

    survival_floor = IND_STARTING_CASH * 0.9
    ranked_idx_end = np.argsort(vals_end)[::-1]
    ranked_scores  = [(int(ranked_idx_end[i]), float(vals_end[ranked_idx_end[i]])) for i in range(N_SLOTS)]
    elite_slice    = ranked_scores[:min(ELITE_COUNT, len(ranked_scores))]
    worst_elite    = elite_slice[-1][1] if elite_slice else worst_score
    scores         = ranked_scores   # alias for selection_and_mutation
    log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} | "
        f"best Δ${best_delta:+.2f}  worst Δ${worst_delta:+.2f} | "
        f"shares(buy/sell)={buy_exec_count:.0f}/{sell_exec_count:.0f} | "
        f"worst_elite=${worst_elite:.2f}")

    if worst_elite < survival_floor:
        log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail} Worst elite (${worst_elite:.2f}) fell below floor (${survival_floor:.2f}) — "
            f"resetting all portfolios to defaults and skipping selection for this day")
        portfolios.cash[:]     = IND_STARTING_CASH
        portfolios.holdings[:] = 0.0
        for sp in stop_prices_all:
            sp.clear()
        return baseline_score, baseline_score, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0   # no signal this day

    # ── Zero-trade inaction filter ────────────────────────────────────────────
    inactive_slots  = set()
    new_streak      = 0
    all_filtered    = False
    if num_past >= 15 and day_num > 0:
        inactive_slots = {s for s in range(N_SLOTS) if slot_trade_count[s] == 0}
        if len(inactive_slots) == N_SLOTS:
            all_filtered = True
            new_streak   = all_zero_streak + 1
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}   Zero-trade filter: ALL {N_SLOTS} slots inactive (streak={new_streak})")
        elif inactive_slots:
            log(f"[{sn(industry)}] Day {actual_day + 1}/{total_avail}   Zero-trade filter: {len(inactive_slots)} slot(s) excluded from elite candidacy")

    # ── Delta-based selection scores with invested_pct multiplier ────────────
    # Selection score = raw delta adjusted by invested_pct:
    # positive-delta slots are scaled down proportional to cash held,
    # so a slot that earned $50 by deploying 80% outranks one that
    # earned $50 while sitting 90% in cash.
    scores_dict = dict(ranked_scores)
    below_floor = {s for s, v in ranked_scores if v < survival_floor}
    sel_scores  = []
    for s, raw in ranked_scores:
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
    under_invested = {
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
        fill_data_v3 = next_day['data'] if next_day is not None else {}
        _dump_day(actual_day, industry, 'UNDER_INVEST',
                  baseline_score, best_delta, avg_inv * 100,
                  ranked_scores, day_data, fill_data_v3)

    # Preserve slot 0's own post-trading result before selection overwrites it.
    slot0_own_port = copy.deepcopy(portfolios[0])

    # ── Step 5: selection + mutation ─────────────────────────────────────────
    if not freeze:
        selection_and_mutation(
            industry, output_dir, StockNN,
            sel_scores, portfolios,
            survival_floor=-(baseline_score * 0.1),
            inactive_slots=inactive_slots | below_floor,
            actual_day=actual_day, total_avail=total_avail,
        )  # sigma threaded via mutate() default — v3 keeps monkey-patch approach

    # ── Diversity injection if all-zero streak hits 2 ────────────────────────
    # Replace bottom 4 of top-8 elites with halfway-blended random models.
    # Gives evolution fresh variation anchored in the direction already learned.
    if not freeze and all_filtered and new_streak >= 2:
        log(f"[{sn(industry)}]   Diversity injection: replacing bottom {ELITE_COUNT - ELITE_COUNT // 2} elites with half-random blends")
        half         = ELITE_COUNT // 2
        top_slots    = [s for s, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:ELITE_COUNT]]
        inject_slots = top_slots[half:]
        source_slots = top_slots[:half]
        for inject_slot, source_slot in zip(inject_slots, source_slots):
            elite = load_slot_model(industry, output_dir, source_slot, StockNN)
            blend = blend_model_halfway(elite, StockNN)
            save_slot_model(industry, output_dir, inject_slot, blend)
            portfolios[inject_slot] = copy.deepcopy(portfolios[source_slot])
            del elite, blend
        new_streak = 0   # reset after injection
        log(f"[{sn(industry)}]   Diversity injection complete — streak reset")

    # ── Daily burst refinement (after selection + diversity injection) ───────────
    # Build fill_data and ref tensors once, then run 4 bursts at finer sigmas.
    if daily_sigma is not None:
        _fill_data     = next_day['data'] if next_day is not None else day_data
        _ref_hold_dict = {symbols[i]: float(baseline_hold[i]) for i in range(len(symbols))}
        _ref_stop      = dict(stop_prices_all[0]) if stop_prices_all else {sym: 0.0 for sym in symbols}
        _state_vec     = [baseline_cash] + [_ref_hold_dict.get(sym, 0.0) for sym in symbols]
        _today_t_burst = torch.tensor(today_shared + _state_vec, dtype=torch.float32).unsqueeze(0)
        log(f"[{sn(industry)}] Daily mode: running 4 refinement bursts ...")
        for burst_num in range(4):
            burst_sigma = daily_sigma / (2 ** (burst_num + 1))
            _run_daily_burst(
                industry, output_dir, symbols, burst_sigma,
                baseline_cash, _ref_hold_dict, _ref_stop,
                day_data, _fill_data, history_t_shared, _today_t_burst,
                portfolios, actual_day, total_avail,
                seq_flags=seq_flags)

    # Restore slot 0's own portfolio so the next day's baseline carries forward
    # this model's actual result, not the winner's cherry-picked result.
    # Model weights in slot 0 remain updated from selection/bursts.
    portfolios[0] = slot0_own_port

    slot0_score = float(vals_end[0])
    if daily_sigma is not None:
        _fd         = next_day['data'] if next_day is not None else day_data
        slot0_score = portfolios[0]['cash'] + sum(
            portfolios[0]['holdings'].get(sym, 0.0) * _fd.get(sym, {}).get('close', 0.0)
            for sym in symbols)
        best_delta  = max(best_delta, slot0_score - baseline_score)
    top_holdings_value = sum(
        portfolios[0]['holdings'].get(sym, 0.0) * day_data[sym]['close']
        for sym in symbols if sym in day_data and day_data[sym]['close'] > 0
    )
    top_cash_value = portfolios[0]['cash']
    return baseline_score, slot0_score, best_delta, top_holdings_value, top_cash_value, new_streak, elite_max_val, elite_min_val, elite_mean_val


# ── Master training ────────────────────────────────────────────────────────────

def init_master(output_dir, load_models_dir, industries):
    """Returns (portfolios, ind_value_history).
    portfolios: N_SLOTS dicts {cash, holdings:{ind: float}, zero_counts:{ind: int}}
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
    Compute retrospective liquidation amounts master would have sent to industries.

    For industries where master wants LESS capital than they currently hold,
    compute how much to liquidate (down to floor) to free up cash for redeployment.

    Returns:
        liquidation_costs: float  — total 1.25% spread cost master absorbs
        freed_cash:        float  — cash recovered from liquidations
        liq_orders:        dict   — {ind: dollar_amount_liquidated}
    """
    floor_value    = master_pool * master_floor_pct
    liq_orders     = {}
    freed_cash     = 0.0
    liquidation_costs = 0.0

    # Current value of each industry from master's perspective
    ind_current_value = {}
    for ind in industry_list:
        hold_v, cash_v = ind_capital_state.get(ind, (0.0, 0.0))
        ind_current_value[ind] = hold_v + cash_v

    # Target value per industry
    ind_target_value = {ind: target_alloc.get(ind, 0.0) * master_pool
                        for ind in industry_list}

    # Industries that need to shrink — sorted weakest predicted first
    # (target_alloc already ranked by master prediction, lowest alloc = weakest)
    shrink_inds = [
        (ind, ind_current_value[ind], ind_target_value[ind])
        for ind in industry_list
        if ind_current_value[ind] > ind_target_value[ind]
        and ind_current_value[ind] > floor_value
    ]
    # Sort by target alloc ascending (weakest first), then by excess descending for ties
    shrink_inds.sort(key=lambda x: (target_alloc.get(x[0], 0.0), -(x[1] - x[2])))

    for ind, current_v, target_v in shrink_inds:
        # Only liquidate holdings portion (not cash — cash is already liquid)
        hold_v, cash_v = ind_capital_state.get(ind, (0.0, 0.0))
        # How much do we need from this industry?
        needed       = current_v - target_v
        # Can't go below floor, can't liquidate more than holdings
        max_liq      = max(0.0, min(hold_v, current_v - floor_value))
        liq_amount   = min(needed, max_liq)
        if liq_amount > 1e-6:
            liq_orders[ind]    = liq_amount
            proceeds           = liq_amount * (1.0 - SEC_FEE_RATE)
            freed_cash        += proceeds
            liquidation_costs += liq_amount * SEC_FEE_RATE

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
        return None, None, None, None, None, None, None

    industry_list = list(industries.keys())

    if day_num % 10 == 0 or day_num == total_days - 1:
        history_len = min((len(v) for v in ind_value_history.values()), default=0)
        log(f"[master] Day {actual_day + 1}/{total_avail} — running {N_SLOTS} models "
            f"(value_history={history_len} days) ...")

    actual_perf = {}
    for ind in industry_list:
        if industry_top_scores and ind in industry_top_scores:
            baseline_ind, slot0_val = industry_top_scores[ind]
            actual_perf[ind] = (slot0_val / baseline_ind - 1.0) if baseline_ind > 0 else 0.0
        else:
            actual_perf[ind] = 0.0

    ind_price = {ind: IND_UNIT_PRICE for ind in industry_list}

    today_t = build_master_features(ind_value_history, industry_list)

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

        for ind in industry_list:
            port['holdings'][ind] *= (1.0 + actual_perf.get(ind, 0.0))

    opt_tiers  = _optimal_tiers(actual_perf, industry_list)
    ideal_pts  = sum(opt_tiers.values())
    slot_pts   = []
    pred_scores = []
    for slot in range(N_SLOTS):
        port_val = _port_val(portfolios[slot])
        tier_map = slot_tier_maps[slot]
        pts = sum(_master_points(tier_map[ind], opt_tiers[ind]) for ind in industry_list)
        slot_pts.append(pts)
        pred_scores.append((slot, pts * 1e9 + port_val))

    best_pts  = max(slot_pts)
    slot0_val = _port_val(portfolios[0])
    slot0_pts = slot_pts[0]

    # Elite pts stats (slots 0..ELITE_COUNT-1)
    elite_pts      = slot_pts[:ELITE_COUNT]
    elite_max_pts  = max(elite_pts)
    elite_min_pts  = min(elite_pts)
    elite_mean_pts = sum(elite_pts) / len(elite_pts)

    s0_tiers    = slot_tier_maps.get(0, {})
    tier_counts = [sum(1 for ind in industry_list if s0_tiers.get(ind) == t) for t in range(4)]

    log(f"[master] Day {actual_day + 1}/{total_avail} | "
        f"best_pts={best_pts:+.2f} slot0_pts={slot0_pts:+.2f} | "
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
        return best_pts, slot0_val, slot0_pts, elite_max_pts, elite_min_pts, elite_mean_pts, ideal_pts

    slot0_own_port = copy.deepcopy(portfolios[0])

    if not no_save_master:
        if best_pts >= 0.0:
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
                f"best_pts={best_pts:.2f} < 0 — "
                f"injecting diversity into bottom {ELITE_COUNT - half} elite slots")
            for inject_rank in range(half, ELITE_COUNT):
                source_rank = inject_rank - half
                elite = load_slot_model('master', output_dir, source_rank, MasterNN)
                blend = blend_model_halfway(elite, MasterNN)
                save_slot_model('master', output_dir, inject_rank, blend)
                portfolios[inject_rank] = copy.deepcopy(portfolios[source_rank])
                del elite, blend

    portfolios[0] = slot0_own_port
    return best_pts, slot0_val, slot0_pts, elite_max_pts, elite_min_pts, elite_mean_pts, ideal_pts


# ── Single-day training wrappers (called from production_v2.py) ───────────────

def train_industry_one_day(industry, symbols, day_data, primed_portfolio, model_dir,
                           next_day_data=None):
    """
    Run one evolution step seeded from real Alpaca state.

    day_data:      Yesterday's OHLCV — model input features and baseline scoring.
    next_day_data: Today's actual OHLCV — fill prices (matching offline training where
                   models predict on day N and fills execute on day N+1).
                   If None, fills simulate against day_data (same-day, less accurate).
    primed_portfolio: {'cash': float, 'holdings': {sym: float}}
        Seeded from Alpaca — real cash allocation + actual positions.
    """
    # Build 100-slot PortfolioArray seeded from primed portfolio
    portfolios      = PortfolioArray(symbols, n=N_SLOTS, init_cash=primed_portfolio.get('cash', IND_STARTING_CASH))
    stop_prices_all = [{sym: 0.0 for sym in symbols} for _ in range(N_SLOTS)]
    seed_hold       = primed_portfolio.get('holdings', {})
    for slot in range(N_SLOTS):
        for sym in symbols:
            portfolios[slot]['holdings'][sym] = seed_hold.get(sym, 0.0)

    # Load histories from stock_data/ (last 15 days)
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
                    raw    = [entry['open'], entry['close'], entry['high'],
                              entry['low'], float(entry['volume'])]
                    prev   = [entries[i-1]['open'], entries[i-1]['close'],
                              entries[i-1]['high'], entries[i-1]['low'],
                              float(entries[i-1]['volume'])] if i > 0 else None
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
    # total_avail=1, actual_day=0, day_num=0, total_days=1 — single step
    result = step_industry(industry, symbols, model_dir, portfolios, histories,
                           day, actual_day=0, total_avail=1, day_num=0, total_days=1,
                           stop_prices_all=stop_prices_all,
                           next_day=next_day)
    return result   # (baseline_score, top_slot_value, ...) or None


def train_master_one_day(industries, primed_portfolio, model_dir,
                         ind_value_history, industry_top_scores=None):
    """
    Production upkeep wrapper — one evolution step for the master model.
    ind_value_history must already contain today's values before calling.
    Returns (best_pts, slot0_val) from step_master, or (None, None) if skipped.
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
    best_pts, slot0_val, *_ = result
    return best_pts, slot0_val


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
                    dt = datetime.fromisoformat(str(date).split('+')[0])
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


# ── Main ───────────────────────────────────────────────────────────────────────

def trim_stock_data_files(all_symbols, stock_data_dir, keep_days=15):
    """Trim each stock data file down to the most recent keep_days entries.

    Called after training completes in normal (daily) operation so the files
    don't grow unboundedly.  Skipped entirely when --preserve-stock-data is set.
    """
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


def main():
    """Parse CLI args and run multi-pass evolutionary training across all industries and master."""
    parser = argparse.ArgumentParser(description="Train neural networks for stock trading v3.")
    parser.add_argument('--output',    required=True,       help='Output directory for models and checkpoints')
    parser.add_argument('--load-dir',                       help='Directory to seed top-10 models from (optional)')
    parser.add_argument('--start-day', type=int, default=None,
                        help='First day index (0-based, inclusive). Default: 0.')
    parser.add_argument('--stop-day',  type=int, default=None,
                        help='Last day index (exclusive). Default: all available days.')
    parser.add_argument('--preserve-stock-data', action='store_true',
                        help='Do not trim stock_data/ files after training.')
    parser.add_argument('--passes', type=int, default=1,
                        help='Number of passes over the day range (default: 1). '
                             'Sigma decays by --sigma-decay between passes.')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='Initial mutation sigma (default: 0.01).')
    parser.add_argument('--master-sigma', type=float, default=None,
                        help='Independent mutation sigma for master model (defaults to --sigma)')
    parser.add_argument('--sigma-decay', type=float, default=0.5,
                        help='Multiplicative sigma decay between passes (default: 0.5). '
                             'Pass 1: sigma, Pass 2: sigma*decay, Pass 3: sigma*decay^2 ...')
    parser.add_argument('--daily',  action='store_true',
                        help='Run 4 finer-sigma refinement bursts per day after normal selection')
    parser.add_argument('--master-only', action='store_true',
                        help='Freeze industry models (no selection/mutation); train master only')
    parser.add_argument('--no-save-master', action='store_true',
                        help='Skip writing master model files (diagnostic runs only)')
    parser.add_argument('--promote', default='',
                        help='Comma-separated sibling dirs to promote best models to after training '
                             '(e.g. "uat" or "uat,prod" when --output is models/dev)')
    args = parser.parse_args()

    os.makedirs(args.output,  exist_ok=True)
    os.makedirs('stock_data', exist_ok=True)

    # Initialise rolling console log — overwrites on each run
    global _console_log_path, _console_log_lines
    _console_log_path  = os.path.join(args.output, 'console_log.txt')
    _console_log_lines = []
    try:
        open(_console_log_path, 'w').close()   # truncate / create
    except Exception:
        pass

    industries = INDUSTRIES

    all_symbols = [sym for syms in industries.values() for sym in syms]

    # Prefer pre-downloaded local JSON files; fall back to live yfinance
    sample_path = os.path.join('stock_data', f"{all_symbols[0]}.json")
    if os.path.exists(sample_path):
        log("Loading stock data from stock_data/ ...")
        all_data = load_stock_data_from_files(all_symbols, 'stock_data')
    else:
        log("No local stock data found — fetching from yfinance (last 1 month) ...")
        all_data = fetch_stock_data_from_yfinance(all_symbols)

    if not all_data:
        log("ERROR: No market data available. Exiting.")
        return

    all_days   = [{'data': all_data[date]} for date in sorted(all_data.keys())]
    total_days = len(all_days)
    day_start  = args.start_day if args.start_day is not None else 0
    day_end    = args.stop_day  if args.stop_day  is not None else total_days

    log(f"Total trading days available: {total_days}")
    log(f"Training window: days {day_start} to {day_end} ({day_end - day_start} days in window)")
    log(f"Passes: {args.passes} | sigma: {args.sigma} | sigma-decay: {args.sigma_decay}")

    # ── Initialise training CSV (overwrite on each script execution) ──────────
    csv_path    = os.path.join(args.output, 'training_log.csv')
    ind_names   = list(industries.keys())
    _ind_stat_cols = [f"{n}_{s}" for n in ind_names for s in ("elite_max", "elite_min", "elite_mean")]
    csv_headers = ['pass', 'day'] + _ind_stat_cols + ['master_elite_max_pts', 'master_elite_min_pts', 'master_elite_mean_pts', 'master_ideal_pts']
    with open(csv_path, 'w') as csv_f:
        csv_f.write(','.join(csv_headers) + '\n')
    log(f"Training log: {csv_path}")

    def _fmt(v):
        return f"{max(-99999.99, min(99999.99, float(v))):+09.2f}"


    # ── Multi-pass loop ───────────────────────────────────────────────────────
    for pass_num in range(args.passes):
        current_sigma        = args.sigma * (args.sigma_decay ** pass_num)
        base_master_sigma    = args.master_sigma if args.master_sigma is not None else args.sigma
        current_master_sigma = base_master_sigma * (args.sigma_decay ** pass_num)
        log(f"===== PASS {pass_num + 1}/{args.passes} | sigma={current_sigma:.6f} | master_sigma={current_master_sigma:.6f} =====")

        # Patch mutate() default sigma for industry models
        import training_v3 as _self
        _orig_mutate = _self.mutate
        def _patched_mutate(model, sigma=current_sigma):
            return _orig_mutate(model, sigma=sigma)
        _self.mutate = _patched_mutate

        # Clear model cache between passes so updated weights are re-evaluated fresh
        _model_cache.clear()

        # Reinitialise pools and histories for this pass
        log(f"Initialising {len(industries)} industry pools + master ...")
        ind_portfolios   = {}
        ind_histories    = {}
        ind_stop_prices  = {}
        for ind, syms in industries.items():
            log(f"[{sn(ind)}] ===== PASS {pass_num+1} BEGIN | days {day_start}–{day_end} of {total_days} =====")
            ind_portfolios[ind], ind_histories[ind], ind_stop_prices[ind] = init_industry(
                ind, syms, args.output, args.load_dir, all_days, day_start)

        log(f"[master] ===== PASS {pass_num+1} BEGIN | days {day_start}–{day_end} of {total_days} =====")
        mst_portfolios, mst_ind_value_history = init_master(
            args.output, args.load_dir, industries)

        days_slice = all_days[day_start:day_end]
        num_days   = len(days_slice)
        log(f"Starting training: {len(industries)} industries + master, {num_days} days")

        ind_streaks = {ind: 0 for ind in industries}

        for day_num, day in enumerate(days_slice):
            next_day = days_slice[day_num + 1] if day_num + 1 < num_days else None
            actual_day          = day_start + day_num
            industry_top_scores = {}
            ind_best_deltas     = {}
            ind_elite_stats     = {}

            # Per-symbol intraday sequence for this day: True=low-first, False=high-first (~50/50)
            seq_flags = {sym: (random.random() < 0.5)
                         for syms in industries.values() for sym in syms}

            ind_capital_state = {}
            for ind, syms in industries.items():
                result = step_industry(ind, syms, args.output,
                                       ind_portfolios[ind], ind_histories[ind],
                                       day, actual_day, total_days, day_num, num_days,
                                       stop_prices_all=ind_stop_prices[ind],
                                       next_day=next_day,
                                       all_zero_streak=ind_streaks.get(ind, 0),
                                       daily_sigma=current_sigma if args.daily else None,
                                       freeze=args.master_only,
                                       seq_flags=seq_flags)
                if result is not None:
                    baseline, top_val, best_delta, top_hold, top_cash, new_streak, \
                        e_max, e_min, e_mean = result
                    industry_top_scores[ind] = (baseline, top_val)
                    ind_best_deltas[ind]     = top_val - baseline
                    ind_capital_state[ind]   = (top_hold, top_cash)
                    ind_elite_stats[ind]     = (e_max, e_min, e_mean)
                    ind_streaks[ind]         = new_streak
                else:
                    ind_best_deltas[ind]     = 0.0
                    ind_capital_state[ind]   = (0.0, 0.0)
                    ind_elite_stats[ind]     = (0.0, 0.0, 0.0)
                invalidate_cache(ind)

            # Append today's industry values to master history before step_master
            for _ind in industries:
                _top_val = industry_top_scores.get(_ind, (IND_STARTING_CASH, IND_STARTING_CASH))[1]
                mst_ind_value_history[_ind].append(_top_val)

            master_best_pts, master_slot0_val, _, \
                master_elite_max_pts, master_elite_min_pts, master_elite_mean_pts, \
                master_ideal_pts = step_master(
                args.output, mst_portfolios, mst_ind_value_history, industries,
                actual_day, total_avail=(day_end - day_start), day_num=day_num, total_days=num_days,
                industry_top_scores=industry_top_scores,
                sigma=current_master_sigma,
                no_save_master=args.no_save_master)
            invalidate_cache('master')

            # Append one row to the training CSV
            row  = [f"{pass_num + 1:<7}", f"{actual_day + 1:<4}"]
            for ind in ind_names:
                e_max, e_min, e_mean = ind_elite_stats.get(ind, (0.0, 0.0, 0.0))
                row += [f"{e_max:+10.2f}", f"{e_min:+10.2f}", f"{e_mean:+10.2f}"]
            _mmx  = master_elite_max_pts  if master_elite_max_pts  is not None else 0.0
            _mmn  = master_elite_min_pts  if master_elite_min_pts  is not None else 0.0
            _mmn2 = master_elite_mean_pts if master_elite_mean_pts is not None else 0.0
            _mid  = master_ideal_pts      if master_ideal_pts      is not None else 0.0
            row += [f"{_mmx:+.2f}", f"{_mmn:+.2f}", f"{_mmn2:+.2f}", f"{_mid:+.2f}"]
            with open(csv_path, 'a') as csv_f:
                csv_f.write(','.join(row) + '\n')

            gc.collect()

        # ── End-of-pass checkpoints ───────────────────────────────────────────
        for ind, syms in industries.items():
            ports = ind_portfolios[ind]
            log(f"[{sn(ind)}] Pass {pass_num+1} complete — saving top-10 checkpoint ...")
            if days_slice:
                _p_end  = ports.sym_prices(days_slice[-1]['data'])
                _v_end  = ports.all_values(_p_end)
                final_scores = [(s, float(_v_end[s])) for s in range(N_SLOTS)]
                final_scores.sort(key=lambda x: x[1], reverse=True)
                top10_meta = [{'slot': s, 'score': v} for s, v in final_scores[:10]]
                save_top10_meta(ind, args.output, top10_meta)
                copy_best_model(ind, args.output, top10_meta)
                scores_str = ', '.join(f"${v:.2f}" for _, v in final_scores[:10])
                log(f"[{sn(ind)}] ===== PASS {pass_num+1} COMPLETE | Top10: [{scores_str}] =====")

        log(f"[master] Pass {pass_num+1} complete — saving top-10 checkpoint ...")
        industry_list_k = list(industries.keys())
        def _mst_value(p):
            v = p['cash']
            for ind in industry_list_k:
                v += p['holdings'].get(ind, 0.0)
            return v
        final_scores = sorted([(s, _mst_value(mst_portfolios[s])) for s in range(N_SLOTS)],
                              key=lambda x: x[1], reverse=True)
        top10_meta = [{'slot': s, 'score': v} for s, v in final_scores[:10]]
        save_top10_meta('master', args.output, top10_meta)
        copy_best_model('master', args.output, top10_meta)
        scores_str = ', '.join(f"${v:.2f}" for _, v in final_scores[:10])
        log(f"[master] ===== PASS {pass_num+1} COMPLETE | Top10: [{scores_str}] =====")

        # Restore original mutate for next pass (will be re-patched)
        _self.mutate = _orig_mutate

    # ── Post-training cleanup ─────────────────────────────────────────────────
    if args.preserve_stock_data:
        log("--preserve-stock-data set: stock_data/ files left untouched.")
    else:
        log("Trimming stock_data/ files to last 15 days (use --preserve-stock-data to skip) ...")
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
