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
import shutil
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
        return baseline_score, baseline_score, 0.0, 0.0, 0.0   # no signal this day

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
    return baseline_score, slot0_score, best_delta, top_holdings_value, top_cash_value, new_streak


# ── Master training ────────────────────────────────────────────────────────────

def init_master(output_dir, load_models_dir, industries, all_days, day_start):
    """
    One-time setup for master: initialise pool, portfolios, histories.
    Pre-loads up to 15 days of history when day_start > 0.
    Returns (portfolios, histories).
    """
    all_symbols = [sym for syms in industries.values() for sym in syms]
    initialise_pool('master', output_dir, load_models_dir, MasterNN)
    industry_list = list(industries.keys())
    portfolios    = PortfolioArray(industry_list, n=N_SLOTS, init_cash=MST_STARTING_CASH)
    histories     = {sym: [] for sym in all_symbols}

    if day_start > 0:
        pre_slice = all_days[max(0, day_start - 15):day_start]
        log(f"[master] Pre-loading {len(pre_slice)} days of history before day {day_start}")
        for pre_day in pre_slice:
            pre_data = pre_day['data']
            for sym in all_symbols:
                if sym in pre_data:
                    d      = pre_data[sym]
                    raw    = [d['open'], d['close'], d['high'], d['low'], d['volume']]
                    prev   = histories[sym][-1][:5] if histories[sym] else None
                    deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
                    histories[sym].append(raw + deltas)
                    if len(histories[sym]) > 15:
                        histories[sym].pop(0)

    return portfolios, histories


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



def step_master(output_dir, portfolios, histories, industries,
                day, actual_day, total_avail, day_num, total_days,
                flat_cos_history=None,
                industry_top_scores=None, ind_capital_state=None, sigma=None):
    """
    Process a single trading day for the master model.

    industry_top_scores: dict {industry: (baseline_value, top_slot_value)}
        Provided by the main loop after industries have run.  Master scores
        its allocation prediction against the actual top-slot P&L per industry.
        If None (cold start / no industry data), falls back to raw price returns.

    Portfolio structure per slot:
        {
          'cash':     float,            # unallocated cash
          'holdings': {ind: float},     # units held per industry (price = baseline_value)
        }
    Industry "price" each day = baseline_value for that industry (slot 0 value).
    Daily return on a holding = (top_slot_value - baseline_value) / baseline_value.

    Memory contract: exactly 1 model tensor in RAM at any point.
    Mutates portfolios and histories in place.
    """
    day_data      = day['data']
    industry_list = list(industries.keys())
    all_symbols   = [sym for syms in industries.values() for sym in syms]

    num_past = min(len(histories[sym]) for sym in all_symbols) if all_symbols else 0

    if day_num % 10 == 0 or day_num == total_days - 1:
        log(f"[master] Day {actual_day + 1}/{total_avail} — running {N_SLOTS} models "
            f"(history={num_past}/15 days warm) ...")

    # ── Resolve industry returns for this day ─────────────────────────────────
    # Prefer top-slot P&L signal; fall back to raw close-to-close price return.
    actual_perf = {}
    for ind, ind_syms in industries.items():
        if industry_top_scores and ind in industry_top_scores:
            baseline_v, top_v = industry_top_scores[ind]
            actual_perf[ind] = (top_v - baseline_v) / baseline_v if baseline_v > 0 else 0.0
        else:
            # Fallback: mean close-to-close return across industry symbols
            returns = []
            for sym in ind_syms:
                if sym in day_data and len(histories[sym]) > 1:
                    prev_close = histories[sym][-2][1]
                    cur_close  = histories[sym][-1][1]
                    if prev_close > 0:
                        returns.append((cur_close - prev_close) / prev_close)
            actual_perf[ind] = statistics.mean(returns) if returns else 0.0

    # ── Industry "prices" for buy/sell simulation ─────────────────────────────
    # Use the industry baseline (slot 0 value) as the unit price so
    # master can buy/sell industry positions the same way StockNN buys stocks.
    ind_price = {}
    for ind in industry_list:
        if industry_top_scores and ind in industry_top_scores:
            ind_price[ind] = industry_top_scores[ind][0]   # baseline value = "open price"
        else:
            ind_price[ind] = IND_STARTING_CASH

    # ── Pre-compute per-industry stats for 216-feature shared input ──────────
    ind_stats = {}
    for ind, ind_syms in industries.items():
        mean_deltas = []
        for t in range(15):
            dl = [histories[sym][-(t+1)][5:] for sym in ind_syms if len(histories[sym]) > t]
            if dl:
                mean_deltas.append([sum(col)/len(col) for col in zip(*dl)])
            else:
                mean_deltas.append([0.0] * 5)
        vals    = [d[0] for d in mean_deltas]
        mean_v  = sum(vals) / len(vals) if vals else 0.0
        std_v   = (sum((v - mean_v)**2 for v in vals) / len(vals))**0.5 if vals else 0.0
        mean_5d = sum(vals[:5]) / 5 if len(vals) >= 5 else mean_v
        ind_stats[ind] = {
            'volatility':  std_v  / abs(mean_v) if abs(mean_v) > 1e-9 else 0.0,
            'momentum':    mean_5d / mean_v      if abs(mean_v) > 1e-9 else 1.0,
            'mean_deltas': mean_deltas,
        }

    # ── Step 1: reset all slots to slot 0's portfolio (matches v2 logic) ───
    def _portfolio_value(p):
        v = p['cash']
        for ind in industry_list:
            v += p['holdings'].get(ind, 0.0) * ind_price.get(ind, 0.0)
        return v

    ind_price_arr    = np.array([ind_price.get(ind, 0.0) for ind in industry_list],
                                dtype=np.float64)
    baseline_cash    = float(portfolios.cash[0])
    baseline_hold    = portfolios.holdings[0].copy()
    pool             = float(baseline_cash + baseline_hold @ ind_price_arr)
    # Reset all slots to slot 0's portfolio (matches v2 logic)
    portfolios.cash[:]     = baseline_cash
    portfolios.holdings[:] = baseline_hold
    baseline_score         = pool
    floor_pct              = 0.02
    floor_value            = pool * floor_pct

    # ── Step 2: parallel inference + liquidate + deploy ──────────────────────
    all_master_models = load_all_models('master', output_dir, MasterNN)
    buy_exec_count    = 0
    sell_exec_count   = 0
    master_allocs     = [None] * N_SLOTS

    # history_t shared: (1,15,61) mean OHLCV per industry + flat_cos, oldest first
    fc_hist        = list(flat_cos_history or [])
    fc_hist_padded = ([0.0] * max(0, 15 - len(fc_hist))) + fc_hist[-15:]
    mst_history_rows = []
    for idx, t in enumerate(range(14, -1, -1)):   # idx=0 oldest, idx=14 most recent
        row = []
        for ind in industry_list:
            dl = [histories[sym][-(t+1)][:5] for sym in industries[ind]
                  if len(histories[sym]) > t]
            if dl:
                row += [sum(col)/len(col) for col in zip(*dl)]
            else:
                row += [0.0] * 5
        row.append(fc_hist_padded[idx])            # flat_cos regime signal
        mst_history_rows.append(row)
    mst_history_t = torch.tensor(mst_history_rows, dtype=torch.float32).unsqueeze(0)  # (1,15,61)

    # today_t shared features (without state): ind_aggs + vol/mom/corr per industry
    today_ind_data = {}
    for ind in industry_list:
        dl = []
        for sym in industries[ind]:
            h = histories[sym]
            if len(h) >= 2:
                raw_t = h[-1][:5]; prev = h[-2][:5]
                dl.append([raw_t[i] - prev[i] for i in range(5)])
            elif len(h) == 1:
                dl.append([0.0] * 5)
        if dl:
            tr = list(zip(*dl)); aggs = []
            for tp in tr: aggs += [max(tp), min(tp), sum(tp)/len(tp)]
            ind_mean_today = sum(row[1] for row in dl) / len(dl)
        else:
            aggs = [0.0] * 15; ind_mean_today = 0.0
        today_ind_data[ind] = (aggs, ind_mean_today)

    all_mean_today = (sum(v[1] for v in today_ind_data.values()) / len(today_ind_data)
                      if today_ind_data else 0.0)
    mst_today_shared = []
    for ind in industry_list:
        st = ind_stats[ind]; aggs, ind_mean_today = today_ind_data[ind]
        corr = ind_mean_today / all_mean_today if abs(all_mean_today) > 1e-9 else 1.0
        mst_today_shared += aggs + [st['volatility'], st['momentum'], corr]
    # mst_today_shared is 216 values; state (13) appended per-slot

    def _build_master_input(slot):
        """Build history_t (1,15,60) and today_t (1,229) for one slot."""
        port      = portfolios[slot]
        cash_norm = port['cash'] / max(pool, 1.0)
        hold_vals = [port['holdings'].get(ind, 0.0) * ind_price.get(ind, 0.0)
                     for ind in industry_list]
        state_vec = [cash_norm] + hold_vals  # 13 values
        today_t   = torch.tensor(mst_today_shared + state_vec, dtype=torch.float32).unsqueeze(0)  # (1,229)
        return mst_history_t, today_t

    def _master_infer(slot):
        history_t_s, today_t_s = _build_master_input(slot)
        with torch.inference_mode():
            out = all_master_models[slot](history_t_s, today_t_s)
        master_allocs[slot] = compute_alloc_from_predicted(out.squeeze(), industry_list)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool_ex:
        list(pool_ex.map(_master_infer, range(N_SLOTS)))

    def _master_trade(slot):
        port  = portfolios[slot]
        alloc, liq_depth, liq_trigger = master_allocs[slot]
        local_buys = local_sells = 0

        # Liquidation pass — trigger gates, depth controls amount
        sorted_inds = sorted(industry_list, key=lambda ind: alloc.get(ind, 0.0))
        for ind in sorted_inds:
            price = ind_price.get(ind, 0.0)
            if price <= 0:
                continue
            if liq_trigger.get(ind, 0.0) <= 0.5:
                continue
            current_hold_v = port['holdings'][ind] * price
            if current_hold_v <= floor_value:
                continue
            target_hold_v = alloc[ind] * pool
            depth         = liq_depth.get(ind, 0.0)
            effective_tgt = target_hold_v + (1.0 - depth) * max(0.0, current_hold_v - target_hold_v)
            effective_tgt = max(effective_tgt, floor_value)
            liq_v         = max(0.0, min(current_hold_v - effective_tgt,
                                         current_hold_v - floor_value))
            liq_units     = liq_v / price
            if liq_units > 1e-9:
                port['holdings'][ind] -= liq_units
                port['cash']          += _sell_net(liq_units, price)
                local_sells           += liq_units

        # Deployment pass
        for ind in industry_list:
            price = ind_price.get(ind, 0.0)
            if price <= 0:
                continue
            current_hold_v = port['holdings'][ind] * price
            target_hold_v  = alloc[ind] * pool
            diff = target_hold_v - current_hold_v
            if diff <= 1e-6:
                continue
            affordable = min(diff, port['cash'])
            units      = affordable / price
            if units > 1e-9:
                port['holdings'][ind] += units
                port['cash']          -= units * price
                local_buys            += units

        for ind in industry_list:
            port['holdings'][ind] *= (1.0 + actual_perf.get(ind, 0.0))
        return slot, local_buys, local_sells

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool_ex:
        for slot, lb, ls in pool_ex.map(_master_trade, range(N_SLOTS)):
            buy_exec_count  += lb
            sell_exec_count += ls

    del all_master_models

    # ── Step 3: update shared symbol history ─────────────────────────────────
    for sym in all_symbols:
        if sym in day_data:
            d      = day_data[sym]
            raw    = [d['open'], d['close'], d['high'], d['low'], d['volume']]
            prev   = histories[sym][-1][:5] if histories[sym] else None
            deltas = [r - p for r, p in zip(raw, prev)] if prev else [0.0] * 5
            histories[sym].append(raw + deltas)
            if len(histories[sym]) > 15:
                histories[sym].pop(0)

    # ── Step 3b: compute prediction-accuracy scores ───────────────────────────
    def _cos_sim(a, b, keys):
        pv  = [a.get(k, 0.0) for k in keys]
        tv  = [b.get(k, 0.0) for k in keys]
        dot = sum(p * t for p, t in zip(pv, tv))
        n_p = math.sqrt(sum(p * p for p in pv))
        n_t = math.sqrt(sum(t * t for t in tv))
        return dot / (n_p * n_t) if n_p > 1e-9 and n_t > 1e-9 else 0.0

    target_pct = {}
    if industry_top_scores:
        raw_deltas = {ind: industry_top_scores[ind][1] - industry_top_scores[ind][0]
                      for ind in industry_list if ind in industry_top_scores}
        if raw_deltas:
            min_d   = min(raw_deltas.values())
            shifted = {ind: d - min_d + 1e-9 for ind, d in raw_deltas.items()}
            total   = sum(shifted.values())
            target_pct = {ind: shifted[ind] / total for ind in shifted}

    # ── Step 4: score & report ────────────────────────────────────────────────
    flat_alloc = {ind: 1.0 / len(industry_list) for ind in industry_list}
    if target_pct:
        slot_preds  = {s: master_allocs[s][0] for s in range(N_SLOTS)}
        pred_scores = [(s, _cos_sim(slot_preds[s], target_pct, industry_list))
                       for s in range(N_SLOTS)]
        best_pred   = max(v for _, v in pred_scores)
        flat_cos    = _cos_sim(flat_alloc, target_pct, industry_list)
    else:
        pred_scores = [(s, 0.0) for s in range(N_SLOTS)]
        best_pred   = 0.0
        flat_cos    = 0.0

    slot0_pred = next((v for s, v in pred_scores if s == 0), 0.0)

    log(f"[master] Day {actual_day + 1}/{total_avail} | "
        f"pred={best_pred:.4f}  flat={flat_cos:.4f} | "
        f"shares(buy/sell)={buy_exec_count:.0f}/{sell_exec_count:.0f} | "
        f"prod=${baseline_score:.2f}")

    if baseline_score < MST_STARTING_CASH * 0.9:
        log(f"[master] Day {actual_day + 1}/{total_avail} "
            f"Production model (${baseline_score:.2f}) fell below floor "
            f"(${MST_STARTING_CASH * 0.9:.2f}) — resetting portfolios and skipping selection")
        for slot in range(N_SLOTS):
            portfolios[slot] = {'cash': MST_STARTING_CASH,
                                'holdings': {ind: 0.0 for ind in industry_list}}
        return best_pred, flat_cos, slot0_pred

    # Preserve slot 0's own post-trading portfolio before selection overwrites it.
    slot0_own_port = copy.deepcopy(portfolios[0])

    # ── Step 5: selection + mutation (or injection if no one beats flat) ────────
    if best_pred >= 0.50:
        pred_vals  = [v for _, v in pred_scores]
        mean_ps    = sum(pred_vals) / len(pred_vals)
        std_ps     = (sum((v - mean_ps) ** 2 for v in pred_vals) / len(pred_vals)) ** 0.5
        pool_floor = mean_ps - std_ps
        selection_and_mutation(
            'master', output_dir, MasterNN,
            pred_scores, portfolios,
            survival_floor=pool_floor,
            inactive_slots=set(),
            actual_day=actual_day, total_avail=total_avail,
            sigma=sigma,
        )
    else:
        half = ELITE_COUNT // 2
        log(f"[master] Day {actual_day + 1}/{total_avail}   best_pred={best_pred:.4f} below floor — injecting diversity into bottom {ELITE_COUNT - half} elite slots")
        for inject_rank in range(half, ELITE_COUNT):
            source_rank = inject_rank - half
            elite = load_slot_model('master', output_dir, source_rank, MasterNN)
            blend = blend_model_halfway(elite, MasterNN)
            save_slot_model('master', output_dir, inject_rank, blend)
            portfolios[inject_rank] = copy.deepcopy(portfolios[source_rank])
            del elite, blend
        log(f"[master] Day {actual_day + 1}/{total_avail}   Master diversity injection complete")

    # Restore slot 0's own portfolio so prod tracks the deployed model's
    # actual accumulated result, not the winner's cherry-picked result.
    portfolios[0] = slot0_own_port

    return best_pred, flat_cos, slot0_pred


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


def train_master_one_day(day_data, industries, primed_portfolio, model_dir,
                         industry_top_scores=None):
    """
    Run one evolution step for the master model using today's data.
    Called from production after all industry one-day steps have run.

    primed_portfolio: {'cash': float, 'holdings': {ind: float}}
        Seeded from Alpaca — master's real cash + industry allocation units.

    industry_top_scores: dict {industry: (baseline_value, top_slot_value)}
        Pass the return values from train_industry_one_day calls.
    """
    industry_list = list(industries.keys())
    all_symbols   = [sym for syms in industries.values() for sym in syms]

    portfolios = [copy.deepcopy(primed_portfolio) for _ in range(N_SLOTS)]
    for p in portfolios:
        for ind in industry_list:
            p['holdings'].setdefault(ind, 0.0)

    # Load shared symbol histories from stock_data/
    histories = {}
    for sym in all_symbols:
        path = os.path.join('stock_data', f"{sym}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                entries = data.get('days', [])[-15:]
                hist = []
                for i, entry in enumerate(entries):
                    raw  = [entry['open'], entry['close'], entry['high'], entry['low'], float(entry['volume'])]
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

    day = {'data': day_data}
    step_master(model_dir, portfolios, histories, industries,
                day, actual_day=0, total_avail=1, day_num=0, total_days=1,
                industry_top_scores=industry_top_scores)


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
    csv_headers = ['pass', 'day'] + ind_names + ['master', 'alpha', 'flat']
    with open(csv_path, 'w') as csv_f:
        csv_f.write(','.join(csv_headers) + '\n')
    log(f"Training log: {csv_path}")

    def _fmt(v):
        return f"{max(-99999.99, min(99999.99, float(v))):+09.2f}"

    mst_pred_history     = []   # rolling window of the last 15 slot0 pred values
    mst_flat_cos_history = []   # rolling window of the last 15 flat_cos values (fed into MasterNN)

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
        mst_portfolios, mst_histories = init_master(
            args.output, args.load_dir, industries, all_days, day_start)

        days_slice = all_days[day_start:day_end]
        num_days   = len(days_slice)
        log(f"Starting training: {len(industries)} industries + master, {num_days} days")

        ind_streaks = {ind: 0 for ind in industries}

        for day_num, day in enumerate(days_slice):
            next_day = days_slice[day_num + 1] if day_num + 1 < num_days else None
            actual_day          = day_start + day_num
            industry_top_scores = {}
            ind_best_deltas     = {}

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
                    baseline, top_val, best_delta, top_hold, top_cash, new_streak = result
                    industry_top_scores[ind] = (baseline, top_val)
                    ind_best_deltas[ind]     = top_val - baseline
                    ind_capital_state[ind]   = (top_hold, top_cash)
                    ind_streaks[ind]         = new_streak
                else:
                    ind_best_deltas[ind]     = 0.0
                    ind_capital_state[ind]   = (0.0, 0.0)
                invalidate_cache(ind)

            master_best_delta, master_flat_cos, master_slot0_pred = step_master(
                args.output, mst_portfolios, mst_histories, industries,
                day, actual_day, total_days, day_num, num_days,
                flat_cos_history=mst_flat_cos_history,
                industry_top_scores=industry_top_scores,
                ind_capital_state=ind_capital_state,
                sigma=current_master_sigma)
            invalidate_cache('master')

            # Append one row to the training CSV
            row  = [f"{pass_num + 1:<7}", f"{actual_day + 1:<4}"]
            row += [_fmt(ind_best_deltas.get(ind, 0.0)) for ind in ind_names]
            _mbd = master_best_delta if master_best_delta is not None else 0.0
            _mfc = master_flat_cos   if master_flat_cos   is not None else 0.0
            row += [f"{_mbd:+.4f}", f"{_mbd - _mfc:+.4f}", f"{_mfc:+.4f}"]
            with open(csv_path, 'a') as csv_f:
                csv_f.write(','.join(row) + '\n')

            mst_flat_cos_history.append(master_flat_cos if master_flat_cos is not None else 0.0)
            if len(mst_flat_cos_history) > 15:
                mst_flat_cos_history.pop(0)

            mst_pred_history.append(master_slot0_pred)
            if len(mst_pred_history) > 15:
                mst_pred_history.pop(0)
            if len(mst_pred_history) >= 6:
                prior     = mst_pred_history[:-1]
                mean_pred = sum(prior) / len(prior)
                std_pred  = (sum((v - mean_pred) ** 2 for v in prior) / len(prior)) ** 0.5
                if std_pred > 0.01 and master_slot0_pred < mean_pred - std_pred:
                    log(f"[master] Day {actual_day + 1}/{total_days}  "
                        f"** SOFT FLAG: slot0 pred={master_slot0_pred:.4f} significantly below "
                        f"{len(prior)}-day avg={mean_pred:.4f} (1σ={std_pred:.4f}) **")

            gc.collect()

            # ── Incremental elite snapshot every 255 days and at pass end ──────
            if (day_num + 1) % 255 == 0 or day_num == num_days - 1:
                _inc_dir = 'models/incremental'
                os.makedirs(_inc_dir, exist_ok=True)
                for _ind in industries:
                    for _slot in range(ELITE_POOL):
                        _src = _model_path(_ind, args.output, _slot)
                        if os.path.exists(_src):
                            shutil.copy2(_src, _model_path(_ind, _inc_dir, _slot))
                for _slot in range(ELITE_POOL):
                    _src = _model_path('master', args.output, _slot)
                    if os.path.exists(_src):
                        shutil.copy2(_src, _model_path('master', _inc_dir, _slot))
                log(f"[checkpoint] incremental elite snapshot saved → {_inc_dir}")

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
