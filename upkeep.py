"""
upkeep.py — Single-day evolution step for production upkeep.

Imported by production_v2.py after each trading day. Provides three upkeep functions:

    upkeep_industry()      — one evolution step for a StockNN industry pool, with burst
                             refinement now correctly enabled (UPKEEP_SIGMA passed through).

    upkeep_mt1_industry()  — one evolution step for one MT1NN industry pool (12 pools total),
                             followed by 4 burst refinement passes at sigma/2, /4, /8, /16.
                             MT1 is scored on 5 components: direction, range (calibration
                             method), accuracy (dollar-denominated), and confidence (out[3]).
                             Composite = 0.50×dir + 0.33×rng + 0.17×acc (out[0–2] only).
                             10-day rolling floor prevents divide-by-zero when actual ≈ 0.

    upkeep_mt2()           — one evolution step for the MT2NN cross-industry allocator pool.
                             Diversity injection fires when ≥75% of the 200-slot pool scores
                             below MT2_INJ_THRESHOLD (-7.0 pts), with a 10-day hold after each.
                             Uses raw MT1 slot0 activations (4 per industry, no normalization)
                             as input — the cross-industry dollar difference is the signal.

Constants:
    UPKEEP_SIGMA      = 0.004   — base sigma for burst refinement (half of full-train 0.008)
    MT1_SCALE_DOLLARS = 10000.0 — tanh ceiling for dollar P&L prediction
    MT1_FLOOR_COLD    = 250.0   — rolling-floor cold-start value ($25K × 0.01)
    MT1_ROLLING_DAYS  = 10      — days in rolling |actual_d| buffer
"""

import copy
import gc
import json
import math
import os
import random
import shutil
from collections import defaultdict

import torch
import torch.nn.functional as F

from models import MT1NN, MT2NN, StockNN
from training_lib import (
    ELITE_COUNT,
    ELITE_POOL,
    HIST_DAYS,
    HIST_ELITE,
    HIST_PER_DAY,
    HIST_WAVG,
    IND_STARTING_CASH,
    MT1_BLEND_SLOTS,
    MT1_COMP_ELITE,
    MT1_COMP_ELITE_EXT,
    MT1_COMP_INJECT,
    MT1_COMP_MUTS,
    MT1_COMP_SLOTS,
    MT1_COMP_SLOTS_EXT,
    MT1_DIR_BACKFILL,
    MT1_DIR_DAYS,
    MT1_POOL_NAMES,
    MT1_RANGE_CEIL_MULT,
    MT1_RANGE_FLOOR,
    MT1_RANGE_INJECT,
    MT1_REINJECT,
    N_SLOTS,
    WAVG_COUNT,
    _master_points,
    _model_path,
    _optimal_tiers,
    blend_model_halfway,
    build_master_features,
    compute_weighted_avg_model,
    load_slot_model,
    log,
    save_slot_model,
    sn,
    step_industry,
)

# ── Constants ──────────────────────────────────────────────────────────────────

UPKEEP_SIGMA      = 0.004
MT1_SCALE_DOLLARS = 10000.0   # tanh ceiling for dollar P&L prediction
MT1_FLOOR_COLD    = 250.0     # acc_floor cold-start (÷2 = $125)
MT1_ROLLING_DAYS  = 10        # days in rolling buffers
MT2_INJ_THRESHOLD = -7.0      # injection fires when ≥75% of pool below this
MT2_INJ_MIN_BELOW = int(N_SLOTS * 0.75)  # 150 of 200



# ── Generic helpers ────────────────────────────────────────────────────────────

def _mutate_generic(model, model_class, sigma):
    """Return a new model instance with Gaussian noise added to all weights/biases."""
    state = copy.deepcopy(model.state_dict())
    for k in state:
        if 'weight' in k or 'bias' in k:
            state[k] += torch.randn_like(state[k]) * sigma
    m = model_class()
    m.load_state_dict(state)
    del state
    return m


def _normalize_weights(values):
    """Clip negatives to zero and normalise values to sum to 1.0; equal weights on all-zero input."""
    clipped = [max(float(v), 0.0) for v in values]
    total   = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [v / total for v in clipped]


def _select_and_mutate(prefix, model_dir, model_class, scores, sigma):
    """
    Selection and mutation for any model class.

    Mirrors training_v2.selection_and_mutation but uses _mutate_generic so MT1NN
    and MT2NN are created correctly (the training_v2 version hardcodes StockNN/MasterNN).

    scores: [(slot, score), ...] for all N_SLOTS slots.
    Returns (elite_slots, elite_vals) after writing updated files to model_dir.
    """
    score_vals     = [s for _, s in scores]
    mean_s         = sum(score_vals) / len(score_vals)
    std_s          = (sum((v - mean_s) ** 2 for v in score_vals) / len(score_vals)) ** 0.5
    survival_floor = mean_s - std_s

    surviving = sorted(
        [(s, v) for s, v in scores if v >= survival_floor],
        key=lambda x: x[1], reverse=True,
    )
    if not surviving:
        surviving = sorted(scores, key=lambda x: x[1], reverse=True)

    top_elite   = surviving[:min(ELITE_COUNT, len(surviving))]
    elite_slots = [s for s, _ in top_elite]
    elite_vals  = [v for _, v in top_elite]

    def _wavg(n):
        k = min(n, len(elite_slots))
        return compute_weighted_avg_model(prefix, model_dir, elite_slots[:k], elite_vals[:k], model_class)

    w5, w10, w15 = _wavg(5), _wavg(10), _wavg(15)

    elite_models = [load_slot_model(prefix, model_dir, s, model_class) for s in elite_slots]
    for rank, m in enumerate(elite_models):
        save_slot_model(prefix, model_dir, rank, m)
        del m

    save_slot_model(prefix, model_dir, ELITE_COUNT,     w5);  del w5
    save_slot_model(prefix, model_dir, ELITE_COUNT + 1, w10); del w10
    save_slot_model(prefix, model_dir, ELITE_COUNT + 2, w15); del w15

    n_mut      = N_SLOTS - ELITE_POOL
    muts_per   = max(1, n_mut // ELITE_POOL)
    child_map  = defaultdict(list)
    for i, slot in enumerate(range(ELITE_POOL, N_SLOTS)):
        child_map[i // muts_per].append(slot)

    for parent_rank, child_slots in child_map.items():
        parent = load_slot_model(prefix, model_dir, parent_rank, model_class)
        for child_slot in child_slots:
            child = _mutate_generic(parent, model_class, sigma)
            save_slot_model(prefix, model_dir, child_slot, child)
            del child
        del parent

    return elite_slots, elite_vals


def _select_and_mutate_mt1_component(prefix, model_dir, scores, sigma,
                                      reinject_paths=None):
    """
    Elite selection + mutation for one MT1 component pool (230 slots).

    scores:          list of (slot, cat_score) sorted or unsorted.
    reinject_paths:  list of up to MT1_REINJECT .pt file paths for re-injection models
                     (written into elite slots 20–22 before mutation).

    Slot layout after selection:
      0–16   direct elites (ELITE_COUNT)
      17–19  wavg blends   (WAVG_COUNT: top-5, top-10, top-15)
      20–22  re-injection slots (MT1_REINJECT) — copied from reinject_paths unchanged
      23–229 mutations (MT1_COMP_MUTS=207, round-robin over all 23 parents)
    """
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_slots = [s for s, _ in sorted_scores[:ELITE_COUNT]]
    while len(top_slots) < ELITE_COUNT:
        top_slots.append(top_slots[0])

    cache = {s: load_slot_model(prefix, model_dir, s, MT1NN) for s in set(top_slots)}

    for rank, s in enumerate(top_slots):
        save_slot_model(prefix, model_dir, rank, cache[s])

    # Wavg blends: equal-weight average of top 5, 10, 15 direct elites
    for b, k in enumerate([5, 10, 15]):
        inv_k  = 1.0 / k
        avg_st = None
        for rank in range(k):
            m     = load_slot_model(prefix, model_dir, rank, MT1NN)
            state = m.state_dict()
            del m
            if avg_st is None:
                avg_st = {key: (v.clone().float() * inv_k if torch.is_floating_point(v) else v.clone())
                          for key, v in state.items()}
            else:
                for key, v in state.items():
                    if torch.is_floating_point(v) and key in avg_st:
                        avg_st[key] = avg_st[key] + v.float() * inv_k
        wm = MT1NN(); wm.load_state_dict(avg_st)
        save_slot_model(prefix, model_dir, ELITE_COUNT + b, wm)
        del wm, avg_st

    # Re-injection slots 20–22
    ri_paths = reinject_paths or []
    for k, ri_path in enumerate(ri_paths[:MT1_REINJECT]):
        if os.path.exists(ri_path):
            try:
                ri_m = MT1NN()
                ri_m.load_state_dict(torch.load(ri_path, weights_only=True))
                save_slot_model(prefix, model_dir, ELITE_COUNT + WAVG_COUNT + k, ri_m)
                del ri_m
            except Exception:
                pass

    del cache

    # Mutations: round-robin over all MT1_COMP_ELITE (23) parents
    for i, slot in enumerate(range(MT1_COMP_ELITE, MT1_COMP_SLOTS)):
        parent_rank = i % MT1_COMP_ELITE
        parent = load_slot_model(prefix, model_dir, parent_rank, MT1NN)
        child  = _mutate_generic(parent, MT1NN, sigma)
        save_slot_model(prefix, model_dir, slot, child)
        del parent, child


# ── MT1 rolling floor ─────────────────────────────────────────────────────────

def load_mt1_rolling_state(model_dir):
    """Load per-industry rolling |actual_d| buffers. Returns empty state on missing/error."""
    path = os.path.join(model_dir, 'mt1_rolling_state.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_mt1_rolling_state(model_dir, state):
    """Persist per-industry rolling |actual_d| buffers."""
    try:
        with open(os.path.join(model_dir, 'mt1_rolling_state.json'), 'w') as f:
            json.dump(state, f)
    except Exception as e:
        log(f"WARNING: could not save mt1_rolling_state: {e}")


def _rolling_acc_floor(ind_state):
    """Acc floor = mean(last 10 |actual_d|) / 2. Returns MT1_FLOOR_COLD/2 on cold start."""
    buf = ind_state.get('actual_buf', [])
    if not buf:
        return MT1_FLOOR_COLD / 2.0
    return sum(buf) / len(buf) / 2.0


def _rolling_range_ceiling(ind_state):
    """Range ceiling = MT1_RANGE_CEIL_MULT × mean(last 10 |actual_d - comp0_delta_d|).
    Returns None (no ceiling) on cold start."""
    buf = ind_state.get('residual_buf', [])
    if not buf:
        return None
    return MT1_RANGE_CEIL_MULT * sum(buf) / len(buf)


def _rolling_update(ind_state, abs_actual_d, comp0_residual):
    """Update both rolling circular buffers (max MT1_ROLLING_DAYS entries each)."""
    for key, val in (('actual_buf', abs_actual_d), ('residual_buf', comp0_residual)):
        buf = ind_state.get(key, [])
        buf.append(val)
        if len(buf) > MT1_ROLLING_DAYS:
            buf = buf[-MT1_ROLLING_DAYS:]
        ind_state[key] = buf


# ── MT1 scoring ────────────────────────────────────────────────────────────────

def _mt1_score_breakdown(out4, actual_d, acc_floor, range_ceiling=None):
    """
    Score one MT1 slot against the industry's actual dollar P&L.

    out4:          raw logit tensor shape (4,)
    actual_d:      float — actual dollar P&L (actual_frac × portfolio_value)
    acc_floor:     float — per-industry adaptive floor for accuracy denom
    range_ceiling: float or None — cap on range r (None = no ceiling)
    Returns (composite, direction, range_, accuracy, confidence) all in [0.0, 1.0].

    Composite = 0.50×direction + 0.33×range + 0.17×accuracy (out[0–2] only).
    Confidence (out[3]) is scored separately and does not enter composite.
    """
    conf     = torch.sigmoid(out4[0]).item()
    delta_t  = torch.tanh(out4[1]).item()
    delta_d  = delta_t * MT1_SCALE_DOLLARS
    rng_pct  = F.softplus(out4[2]).item()
    conf4    = torch.sigmoid(out4[3]).item()

    score_dir = conf if actual_d >= 0.0 else (1.0 - conf)

    eff_delta = max(abs(delta_d), MT1_RANGE_FLOOR)
    r         = rng_pct * eff_delta
    if range_ceiling is not None:
        r = min(r, range_ceiling)
    err       = abs(actual_d - delta_d)
    m         = err / r if r > 0.0 else float('inf')
    score_rng = m if m < 1.0 else 0.0

    denom     = max(abs(actual_d), acc_floor)
    score_acc = max(0.0, 1.0 - err / denom)

    d         = err
    dor       = (d / r) if r > 1e-9 else (1e9 if d > 0.0 else 0.0)
    ideal     = 1.0 / (1.0 + dor * dor)
    diff      = conf4 - ideal
    score_conf = 1.0 - diff * diff
    if err > r:
        score_conf = 0.5 + 0.25 * score_conf  # compress outside-range to [0.5, 0.75]

    composite = 0.50 * score_dir + 0.33 * score_rng + 0.17 * score_acc
    return composite, score_dir, score_rng, score_acc, score_conf


def _mt1_decode(model, in37_t):
    """Run MT1 inference and return raw activations for MT2 input and logging.

    Returns (conf, delta_t, range_pct, conf4):
      conf     = sigmoid(out[0]) ∈ [0,1]  — direction confidence
      delta_t  = tanh(out[1])   ∈ [-1,1] — bounded P&L (× MT1_SCALE_DOLLARS for dollars)
      range_pct = softplus(out[2]) > 0    — range as % of effective delta
      conf4    = sigmoid(out[3]) ∈ [0,1]  — calibrated confidence
    """
    model.eval()
    with torch.inference_mode():
        out4 = model(in37_t).squeeze(0)
    conf      = torch.sigmoid(out4[0]).item()
    delta_t   = torch.tanh(out4[1]).item()
    range_pct = F.softplus(out4[2]).item()
    conf4     = torch.sigmoid(out4[3]).item()
    return conf, delta_t, range_pct, conf4


# ── MT1 burst refinement ───────────────────────────────────────────────────────

def _mt1_burst_component(prefix, model_dir, dir_hist, actual_d,
                          acc_floor, range_ceiling, burst_sigma, score_idx):
    """
    One burst refinement pass for one MT1 component pool.

    score_idx: 0=direction 1=accuracy 2=range 3=confidence
    Uses 5-day summed scoring from dir_hist; today (last entry) is doubled.
    Generates 10 mutants per elite (23 parents → 230 mutants), scores on the
    component-specific score, merges top-ELITE_COUNT with current elites
    (cap: 2 burst replacements). Regenerates wavg blend slots 17–19.
    """
    # breakdown tuple: (composite, dir, rng, acc, conf) → pick component score
    _sc_idx = [1, 3, 2, 4][score_idx]

    def _score_multiday(m):
        total = 0.0
        with torch.inference_mode():
            for di, (feat_t, ad) in enumerate(dir_hist):
                is_today = (di == len(dir_hist) - 1)
                out4 = m(feat_t).squeeze(0)
                if score_idx == 0:
                    conf = torch.sigmoid(out4[0]).item()
                    actual_pos = (ad >= 0.0)
                    day_score = conf if actual_pos else (1.0 - conf)
                else:
                    bd = _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)
                    day_score = bd[_sc_idx]
                total += day_score * 2.0 if is_today else day_score
        return total

    burst_candidates = []
    for elite_slot in range(MT1_COMP_ELITE):
        parent = load_slot_model(prefix, model_dir, elite_slot, MT1NN)
        for _ in range(10):
            child = _mutate_generic(parent, MT1NN, burst_sigma)
            child.eval()
            cat_sc = _score_multiday(child)
            burst_candidates.append((child, cat_sc))
        del parent
    burst_candidates.sort(key=lambda x: x[1], reverse=True)
    top_burst = burst_candidates[:ELITE_COUNT]

    current_elites = []
    for rank in range(ELITE_COUNT):
        m = load_slot_model(prefix, model_dir, rank, MT1NN)
        m.eval()
        cat_sc = _score_multiday(m)
        current_elites.append((m, cat_sc))

    burst_ids  = {id(m) for m, _ in top_burst}
    all_cands  = current_elites + top_burst
    all_cands.sort(key=lambda x: x[1], reverse=True)

    new_elites  = []
    burst_count = 0
    for cand in all_cands:
        if len(new_elites) >= ELITE_COUNT:
            break
        if id(cand[0]) in burst_ids:
            if burst_count >= 2:
                continue
            burst_count += 1
        new_elites.append(cand)

    prev_best = current_elites[0][1]
    new_best  = new_elites[0][1]
    label     = MT1_POOL_NAMES[score_idx]
    if new_best > prev_best:
        log(f"[mt1/{sn(prefix[4:])}:{label}] Burst σ={burst_sigma:.5f}: {burst_count} replacement(s), best={new_best:.4f}")
    else:
        log(f"[mt1/{sn(prefix[4:])}:{label}] Burst σ={burst_sigma:.5f}: best={new_best:.4f} — no improvement")

    for rank, (m, _) in enumerate(new_elites):
        save_slot_model(prefix, model_dir, rank, m)

    # Regenerate wavg blend slots 17, 18, 19 (top-5, top-10, top-15)
    for b, k in enumerate([5, 10, 15]):
        inv_k  = 1.0 / k
        avg_st = None
        for rank in range(k):
            m     = load_slot_model(prefix, model_dir, rank, MT1NN)
            state = m.state_dict()
            del m
            if avg_st is None:
                avg_st = {kk: (v.clone().float() * inv_k if torch.is_floating_point(v) else v.clone())
                          for kk, v in state.items()}
            else:
                for kk, v in state.items():
                    if torch.is_floating_point(v) and kk in avg_st:
                        avg_st[kk] = avg_st[kk] + v.float() * inv_k
        wm = MT1NN(); wm.load_state_dict(avg_st)
        save_slot_model(prefix, model_dir, ELITE_COUNT + b, wm)
        del wm, avg_st

    del burst_candidates, current_elites, top_burst, all_cands, new_elites
    gc.collect()
    return new_best


# ── Industry upkeep ────────────────────────────────────────────────────────────

def upkeep_industry(industry, symbols, model_dir, primed_portfolio,
                    day_data, next_day_data=None, seq_flags=None,
                    intraday_bars=None, pool_size=100, sigma=UPKEEP_SIGMA):
    """
    One evolution step for a StockNN industry pool with burst refinement enabled.

    Equivalent to training_v2.train_industry_one_day() but passes daily_sigma so
    the 4 burst refinement passes in step_industry actually run. The original wrapper
    never forwarded sigma, silently skipping all burst work in production upkeep.

    Returns (baseline_score, slot0_score) or None on error.
    """
    slot0_path = _model_path(industry, model_dir, 0)
    best_path  = os.path.join(model_dir, f"{industry}_best.pt")
    if not os.path.exists(slot0_path) and os.path.exists(best_path):
        log(f"[{sn(industry)}] Bootstrapping {pool_size}-slot upkeep pool from {industry}_best.pt")
        shutil.copy2(best_path, slot0_path)
        base = load_slot_model(industry, model_dir, 0, StockNN)
        for slot in range(1, pool_size):
            child = _mutate_generic(base, StockNN, sigma)
            save_slot_model(industry, model_dir, slot, child)
            del child
        del base

    portfolios = [copy.deepcopy(primed_portfolio) for _ in range(pool_size)]
    for p in portfolios:
        p.setdefault('stop_prices', {sym: 0.0 for sym in symbols})
        for sym in symbols:
            p['holdings'].setdefault(sym, 0.0)

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
                    prev   = ([entries[i-1]['open'], entries[i-1]['close'],
                                entries[i-1]['high'], entries[i-1]['low'],
                                float(entries[i-1]['volume'])] if i > 0 else None)
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
    result   = step_industry(
        industry, symbols, model_dir, portfolios, histories,
        day, actual_day=0, total_avail=1, day_num=0, total_days=1,
        next_day=next_day, seq_flags=seq_flags, pool_size=pool_size,
        intraday_bars=intraday_bars,
        daily_sigma=sigma,   # KEY FIX: burst refinement now runs
    )
    if result is None:
        return None
    return result[0], result[1]   # (baseline_score, slot0_score)


# ── MT1 upkeep ─────────────────────────────────────────────────────────────────

def upkeep_mt1_industry(industry, model_dir, in37_t, actual_perf_i, sigma=UPKEEP_SIGMA,
                         portfolio_value=None, rolling_state=None):
    """
    One evolution step for one industry's MT1NN 5-pool system.

    Runs 4 component pools (dir, acc, rng, cfd) then a composite blend pool.
    Each component pool has MT1_COMP_SLOTS=230 slots; the composite pool generates
    MT1_BLEND_SLOTS=200 fresh blends from component direct elites each day, scored by
    composite score. Top MT1_REINJECT=3 composite models are re-injected into component
    pool elite slots 20-22 the following day.

    industry:        industry key (e.g. 'energy')
    model_dir:       directory containing model files
    in37_t:          (1, 37) tensor — this industry's slice of build_master_features output
    actual_perf_i:   float — fractional return (slot0_score / baseline - 1) for today
    sigma:           base mutation sigma
    portfolio_value: current slot0 industry portfolio value (dollars); defaults to IND_STARTING_CASH
    rolling_state:   mutable dict with per-industry rolling buffer; updated in place.

    File naming:
      Component pools: mt1_{ind}_{dir|acc|rng|cfd}_model_{slot}.pt
      Composite best:  mt1_{ind}_best.pt
      Reinject:        mt1_{ind}_comp_reinject_{0..2}.pt
      Comp history:    mt1_{ind}_comp_hist_{day}_{pos}.pt + mt1_{ind}_comp_hist_meta.json

    Returns (best_comp_score, best_comp_score, slot0_conf, slot0_delta_t, slot0_range_pct,
             slot0_conf4) — slot0 is the winning composite model.
    """
    if portfolio_value is None:
        portfolio_value = IND_STARTING_CASH
    if rolling_state is None:
        rolling_state = {}

    actual_d      = actual_perf_i * portfolio_value
    ind_rs        = rolling_state.setdefault(industry, {})
    acc_floor     = _rolling_acc_floor(ind_rs)
    range_ceiling = _rolling_range_ceiling(ind_rs)

    in37_t = in37_t.detach()

    # score_idx into _mt1_score_breakdown() tuple (composite, dir, rng, acc, conf)
    # MT1_POOL_NAMES = ('dir', 'acc', 'rng', 'cfd')
    pool_score_idx = [1, 3, 2, 4]  # dir→[1], acc→[3], rng→[2], cfd→[4]

    comp_prefix    = f'mt1_{industry}_comp'
    reinject_paths = [os.path.join(model_dir, f'{comp_prefix}_reinject_{k}.pt')
                      for k in range(MT1_REINJECT)]

    # ── Direction day buffer (multi-day scoring) ─────────────────────────────────
    dir_hist_path = os.path.join(model_dir, f'mt1_{industry}_dir_hist.json')
    dir_hist_raw: list[dict] = []
    if os.path.exists(dir_hist_path):
        with open(dir_hist_path) as _f:
            dir_hist_raw = json.load(_f)
    # Append today and trim to last MT1_DIR_DAYS entries
    dir_hist_raw.append({'feat37': in37_t.squeeze(0).tolist(), 'actual_d': float(actual_d)})
    dir_hist_raw = dir_hist_raw[-MT1_DIR_DAYS:]
    with open(dir_hist_path, 'w') as _f:
        json.dump(dir_hist_raw, _f)
    # Pre-convert to tensors for scoring
    dir_hist = [(torch.tensor(e['feat37'], dtype=torch.float32).unsqueeze(0), e['actual_d'])
                for e in dir_hist_raw]

    # ── Component pools ──────────────────────────────────────────────────────────
    for pool_id, pool_name in enumerate(MT1_POOL_NAMES):
        prefix     = f'mt1_{industry}_{pool_name}'
        slot0_path = _model_path(prefix, model_dir, 0)
        best_path  = os.path.join(model_dir, f'mt1_{industry}_best.pt')

        if not os.path.exists(slot0_path):
            base = MT1NN()
            if os.path.exists(best_path):
                try:
                    base.load_state_dict(torch.load(best_path, weights_only=True))
                    log(f"[mt1/{sn(industry)}:{pool_name}] Bootstrapping from mt1_{industry}_best.pt")
                except Exception:
                    log(f"[mt1/{sn(industry)}:{pool_name}] Bootstrap failed — random weights")
            else:
                log(f"[mt1/{sn(industry)}:{pool_name}] Initializing with random weights")
            pool_slots = MT1_COMP_SLOTS_EXT if pool_id != 3 else MT1_COMP_SLOTS
            save_slot_model(prefix, model_dir, 0, base)
            for slot in range(1, pool_slots):
                child = _mutate_generic(base, MT1NN, sigma)
                save_slot_model(prefix, model_dir, slot, child)
                del child
            del base

        pool_slots = MT1_COMP_SLOTS_EXT if pool_id != 3 else MT1_COMP_SLOTS
        cat_idx = pool_score_idx[pool_id]
        scores = []

        # All pools: 5-day summed scoring using shared dir_hist; today's score is doubled.
        # Direction pool (pool_id==0) also tracks binary correct count for backfill check.
        dir_max_correct = 0
        if dir_hist:
            for slot in range(pool_slots):
                m = load_slot_model(prefix, model_dir, slot, MT1NN)
                m.eval()
                total_sum = 0.0
                n_correct = 0
                with torch.inference_mode():
                    for di, (feat_t, ad) in enumerate(dir_hist):
                        is_today = (di == len(dir_hist) - 1)
                        out4 = m(feat_t).squeeze(0)
                        if pool_id == 0:
                            conf = torch.sigmoid(out4[0]).item()
                            actual_pos = (ad >= 0.0)
                            day_score = conf if actual_pos else (1.0 - conf)
                            n_correct += 1 if (conf >= 0.5) == actual_pos else 0
                        else:
                            breakdown = _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)
                            day_score = breakdown[cat_idx]
                        total_sum += day_score * 2.0 if is_today else day_score
                scores.append((slot, total_sum))
                if n_correct > dir_max_correct:
                    dir_max_correct = n_correct
                del m

        best_cat  = max(sc for _, sc in scores)
        slot0_cat = scores[0][1]
        log(f"[mt1/{sn(industry)}:{pool_name}] best={best_cat:.4f} slot0={slot0_cat:.4f} "
            f"actual_d=${actual_d:+.1f}")

        if pool_id == 0 and dir_hist and dir_max_correct < 3:
            log(f"[mt1/{sn(industry)}:dir] max_correct={dir_max_correct}/{len(dir_hist)} < 3 — backfill: keeping yesterday's elites")
            continue

        _select_and_mutate_mt1_component(prefix, model_dir, scores, sigma, reinject_paths)

        for burst_num in range(4):
            _mt1_burst_component(prefix, model_dir, dir_hist, actual_d,
                                  acc_floor, range_ceiling,
                                  sigma / (2 ** (burst_num + 1)), pool_id)

        # Range pool: snapshot top elites for confidence cross-injection (anti-gaming)
        if pool_id == 2:
            rng_prefix = prefix
            for k in range(MT1_RANGE_INJECT):
                src  = os.path.join(model_dir, f'{rng_prefix}_elite_{k}.pt')
                dst  = os.path.join(model_dir, f'mt1_{industry}_rng_inject_{k}.pt')
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        # Confidence pool: overwrite bottom 5 direct elites with top range elites
        if pool_id == 3:
            start = ELITE_COUNT - MT1_RANGE_INJECT  # slot 12
            for k in range(MT1_RANGE_INJECT):
                src = os.path.join(model_dir, f'mt1_{industry}_rng_inject_{k}.pt')
                if os.path.exists(src):
                    dst_slot = start + k
                    shutil.copy2(src, os.path.join(model_dir, f'{prefix}_elite_{dst_slot}.pt'))

        # Dir/acc/rng: populate additive elite slots 23–27 with yesterday's composite top-5
        if pool_id != 3:
            for k in range(MT1_COMP_INJECT):
                src = os.path.join(model_dir, f'mt1_{industry}_comp_inject_{k}.pt')
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(model_dir, f'{prefix}_elite_{MT1_COMP_ELITE + k}.pt'))

    # ── Composite pool ───────────────────────────────────────────────────────────
    # Load ELITE_COUNT direct elites from each component pool (ranks 0–16)
    elites_by_pool = []
    for pool_name in MT1_POOL_NAMES:
        prefix      = f'mt1_{industry}_{pool_name}'
        pool_elites = [load_slot_model(prefix, model_dir, rank, MT1NN)
                       for rank in range(ELITE_COUNT)]
        elites_by_pool.append(pool_elites)

    # Generate MT1_BLEND_SLOTS composite blends via position-weighted averaging
    blend_scored = []
    for _ in range(MT1_BLEND_SLOTS):
        sources = []
        for pool_elites in elites_by_pool:
            r1, r2 = sorted(random.sample(range(ELITE_COUNT), 2))
            sources.append((r1, pool_elites[r1]))
            sources.append((r2, pool_elites[r2]))
        weights = [20 - r for r, _ in sources]
        w_sum   = sum(weights)
        weights = [w / w_sum for w in weights]
        avg_st  = None
        for wi, (_, m) in zip(weights, sources):
            state = m.state_dict()
            if avg_st is None:
                avg_st = {k: (v.clone().float() * wi if torch.is_floating_point(v) else v.clone())
                          for k, v in state.items()}
            else:
                for k, v in state.items():
                    if torch.is_floating_point(v) and k in avg_st:
                        avg_st[k] = avg_st[k] + v.float() * wi
        bm = MT1NN()
        bm.load_state_dict(avg_st)
        bm.eval()
        total_sum = 0.0
        with torch.inference_mode():
            for di, (feat_t, ad) in enumerate(dir_hist):
                is_today = (di == len(dir_hist) - 1)
                out4 = bm(feat_t).squeeze(0)
                day_sc = _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)[0]
                total_sum += day_sc * 2.0 if is_today else day_sc
        comp_sc = total_sum
        blend_scored.append((bm, comp_sc))
        del avg_st

    for pool_elites in elites_by_pool:
        del pool_elites
    del elites_by_pool

    # Load composite history candidates and score against today's actual_d
    hist_meta_path  = os.path.join(model_dir, f'{comp_prefix}_hist_meta.json')
    hist_candidates = []
    if os.path.exists(hist_meta_path):
        try:
            with open(hist_meta_path) as f:
                hist_meta = json.load(f)
            h_head = hist_meta.get('head', 0)
            h_count = hist_meta.get('count', 0)
            n_days  = min(h_count, HIST_DAYS)
            for d in range(n_days):
                day_slot = (h_head - n_days + d) % HIST_DAYS
                for pos in range(HIST_PER_DAY):
                    hp = os.path.join(model_dir, f'{comp_prefix}_hist_{day_slot}_{pos}.pt')
                    if os.path.exists(hp):
                        try:
                            hm = MT1NN()
                            hm.load_state_dict(torch.load(hp, weights_only=True))
                            hm.eval()
                            total_sum = 0.0
                            with torch.inference_mode():
                                for di, (feat_t, ad) in enumerate(dir_hist):
                                    is_today = (di == len(dir_hist) - 1)
                                    out4 = hm(feat_t).squeeze(0)
                                    day_sc = _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)[0]
                                    total_sum += day_sc * 2.0 if is_today else day_sc
                            comp_sc = total_sum
                            hist_candidates.append((hm, comp_sc))
                        except Exception:
                            pass
        except Exception:
            pass

    all_candidates = blend_scored + hist_candidates
    all_candidates.sort(key=lambda x: x[1], reverse=True)

    best_comp_model = all_candidates[0][0]
    best_comp_score = all_candidates[0][1]

    log(f"[mt1/{sn(industry)}:comp] best={best_comp_score:.4f} actual_d=${actual_d:+.1f} "
        f"acc_floor=${acc_floor:.1f} n_hist={len(hist_candidates)}")

    slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4 = _mt1_decode(best_comp_model, in37_t)
    comp0_delta_d  = slot0_delta_t * MT1_SCALE_DOLLARS
    comp0_residual = abs(actual_d - comp0_delta_d)

    # Save best composite model as _best.pt for production inference
    best_path = os.path.join(model_dir, f'mt1_{industry}_best.pt')
    try:
        torch.save(best_comp_model.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save mt1_{industry}_best.pt: {e}")

    # Save top MT1_REINJECT models for next-day component re-injection (slots 20–22)
    for k in range(min(MT1_REINJECT, len(all_candidates))):
        rp = os.path.join(model_dir, f'{comp_prefix}_reinject_{k}.pt')
        try:
            torch.save(all_candidates[k][0].state_dict(), rp)
        except Exception as e:
            log(f"WARNING: could not save {comp_prefix}_reinject_{k}: {e}")

    # Save top MT1_COMP_INJECT models for next-day dir/acc/rng additive injection (slots 23–27)
    for k in range(min(MT1_COMP_INJECT, len(all_candidates))):
        cp = os.path.join(model_dir, f'mt1_{industry}_comp_inject_{k}.pt')
        try:
            torch.save(all_candidates[k][0].state_dict(), cp)
        except Exception as e:
            log(f"WARNING: could not save mt1_{industry}_comp_inject_{k}: {e}")

    # Save top HIST_PER_DAY composite models to circular history (5 days × 10/day)
    if os.path.exists(hist_meta_path):
        try:
            with open(hist_meta_path) as f:
                hist_meta = json.load(f)
        except Exception:
            hist_meta = {'head': 0, 'count': 0}
    else:
        hist_meta = {'head': 0, 'count': 0}
    h_head  = hist_meta.get('head', 0)
    h_count = hist_meta.get('count', 0)
    day_slot = h_head % HIST_DAYS
    for pos, (hm, _) in enumerate(all_candidates[:HIST_PER_DAY]):
        hp = os.path.join(model_dir, f'{comp_prefix}_hist_{day_slot}_{pos}.pt')
        try:
            torch.save(hm.state_dict(), hp)
        except Exception as e:
            log(f"WARNING: could not save comp hist {day_slot}_{pos}: {e}")
    try:
        with open(hist_meta_path, 'w') as f:
            json.dump({'head': (h_head + 1) % HIST_DAYS,
                       'count': min(h_count + 1, HIST_DAYS)}, f)
    except Exception as e:
        log(f"WARNING: could not save {comp_prefix}_hist_meta: {e}")

    _rolling_update(ind_rs, abs(actual_d), comp0_residual)

    del blend_scored, hist_candidates, all_candidates
    gc.collect()

    return best_comp_score, best_comp_score, slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4


# ── MT2 upkeep ─────────────────────────────────────────────────────────────────

def upkeep_mt2(model_dir, mt1_slot0_outputs, actual_perf, industry_list,
               sigma=UPKEEP_SIGMA):
    """
    One evolution step for the MT2NN cross-industry allocator pool.

    mt1_slot0_outputs: {ind: (conf, delta_t, range_pct, conf4)} — MT1 slot0 raw activations
                       for each industry, produced by upkeep_mt1_industry().
                       No normalization applied — raw activations passed directly to MT2.
    actual_perf:       {ind: float} fractional return per industry for scoring.
    industry_list:     list of industry keys in canonical INDUSTRY_NAMES order.

    File naming: mt2_model_{slot}.pt (prefix 'mt2').

    Returns (best_pts, slot0_pts, injected_flag).
    """
    prefix    = 'mt2'
    best_path = os.path.join(model_dir, 'mt2_best.pt')

    # Build in48: [conf, delta_t, range_pct, conf4] × 12 industries, no normalization
    in48 = []
    for ind in industry_list:
        conf, delta_t, range_pct, conf4 = mt1_slot0_outputs.get(ind, (0.5, 0.0, 0.01, 0.5))
        in48.extend([conf, delta_t, range_pct, conf4])
    in48_t = torch.tensor(in48, dtype=torch.float32).unsqueeze(0)   # (1, 48)

    # Bootstrap pool if slot files don't exist
    slot0_path = _model_path(prefix, model_dir, 0)
    if not os.path.exists(slot0_path):
        base = MT2NN()
        if os.path.exists(best_path):
            try:
                base.load_state_dict(torch.load(best_path, weights_only=True))
                log("[mt2] Bootstrapping pool from mt2_best.pt")
            except Exception:
                log("[mt2] Bootstrap failed — using random weights")
        else:
            log("[mt2] Initializing MT2 pool with random weights")
        save_slot_model(prefix, model_dir, 0, base)
        for slot in range(1, N_SLOTS):
            child = _mutate_generic(base, MT2NN, sigma)
            save_slot_model(prefix, model_dir, slot, child)
            del child
        del base

    opt_tiers  = _optimal_tiers(actual_perf, industry_list)
    ideal_pts  = sum(opt_tiers.values())

    # Score all N_SLOTS MT2 slots
    slot_pts    = []
    pred_scores = []
    for slot in range(N_SLOTS):
        m = load_slot_model(prefix, model_dir, slot, MT2NN)
        m.eval()
        with torch.inference_mode():
            out = m(in48_t)   # (1, 48)
        del m
        tier_preds = out.view(12, 4).argmax(dim=1).tolist()
        tier_map   = {ind: tier_preds[i] for i, ind in enumerate(industry_list)}
        pts        = sum(_master_points(tier_map[ind], opt_tiers[ind]) for ind in industry_list)
        slot_pts.append(pts)
        pred_scores.append((slot, pts * 1e9 + slot))   # tiebreak by slot

    best_pts  = max(slot_pts)
    slot0_pts = slot_pts[0]

    # Re-run slot 0 for tier logging
    m0 = load_slot_model(prefix, model_dir, 0, MT2NN)
    m0.eval()
    with torch.inference_mode():
        out0 = m0(in48_t)
    del m0
    tier0 = out0.view(12, 4).argmax(dim=1).tolist()
    tier_counts = [sum(1 for t in tier0 if t == c) for c in range(4)]

    log(f"[mt2] best_pts={best_pts:+.2f} slot0_pts={slot0_pts:+.2f} ideal={ideal_pts} "
        f"tiers(0/1/2/3)={tier_counts[0]}/{tier_counts[1]}/{tier_counts[2]}/{tier_counts[3]}")

    # Load 10-day post-injection hold counter
    inj_state_path = os.path.join(model_dir, 'mt2_inj_state.json')
    try:
        with open(inj_state_path) as _f:
            injection_hold = json.load(_f).get('injection_hold', 0)
    except Exception:
        injection_hold = 0
    if injection_hold > 0:
        injection_hold -= 1
    injection_suppressed = injection_hold > 0

    below_thresh = sum(1 for p in slot_pts if p < MT2_INJ_THRESHOLD)
    inject_triggered = below_thresh >= MT2_INJ_MIN_BELOW and not injection_suppressed

    injected = False
    if not inject_triggered:
        _select_and_mutate(prefix, model_dir, MT2NN, pred_scores, sigma)
    else:
        injected = True
        injection_hold = 10
        half = ELITE_COUNT // 2
        log(f"[mt2] {below_thresh}/{N_SLOTS} slots < {MT2_INJ_THRESHOLD:+.1f} — injecting diversity (hold=10)")
        for inject_rank in range(half, ELITE_COUNT):
            source_rank = inject_rank - half
            elite = load_slot_model(prefix, model_dir, source_rank, MT2NN)
            blend = blend_model_halfway(elite, MT2NN)
            save_slot_model(prefix, model_dir, inject_rank, blend)
            del elite, blend

    try:
        with open(inj_state_path, 'w') as _f:
            json.dump({'injection_hold': injection_hold}, _f)
    except Exception as e:
        log(f"WARNING: could not save mt2_inj_state.json: {e}")

    slot0_final = load_slot_model(prefix, model_dir, 0, MT2NN)
    try:
        torch.save(slot0_final.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save mt2_best.pt: {e}")
    del slot0_final

    return best_pts, slot0_pts, injected


# ── Production inference (MT1 → MT2) ──────────────────────────────────────────

def run_mt_inference(model_dir, industries, ind_value_history, zero_counts, total_cash):
    """
    MT1→MT2 inference chain for daily capital allocation in production.

    Loads mt1_{ind}_best.pt for each industry and mt2_best.pt, runs the full
    chain, and returns (allocations, tier_map, mt1_outputs).

    Caller should fall back to MasterNN/equal allocation if mt2_best.pt is absent.

    allocations:  {ind: dollar_amount}
    tier_map:     {ind: 0-3}
    mt1_outputs:  {ind: (conf, delta_t, range_pct, conf4)} — raw activations (no normalization)
    """
    from training_lib import build_master_features, tiers_to_alloc

    industry_list = list(industries.keys())
    today444 = build_master_features(ind_value_history, industry_list)

    mt1_outputs: dict = {}
    for i, ind in enumerate(industry_list):
        in37_t  = today444[:, i * 37:(i + 1) * 37]
        best_pt = os.path.join(model_dir, f"mt1_{ind}_best.pt")
        mt1_m   = MT1NN()
        if os.path.exists(best_pt):
            try:
                mt1_m.load_state_dict(torch.load(best_pt, weights_only=True))
            except Exception as e:
                print(f"Warning: could not load mt1_{ind}_best.pt: {e}")
        mt1_m.eval()
        with torch.no_grad():
            conf, delta_t, range_pct, conf4 = _mt1_decode(mt1_m, in37_t)
        mt1_outputs[ind] = (conf, delta_t, range_pct, conf4)
        del mt1_m

    # Build in48: raw activations, no normalization
    in48 = []
    for ind in industry_list:
        conf, delta_t, range_pct, conf4 = mt1_outputs[ind]
        in48.extend([conf, delta_t, range_pct, conf4])
    in48_t = torch.tensor(in48, dtype=torch.float32).unsqueeze(0)   # (1, 48)

    mt2_path = os.path.join(model_dir, 'mt2_best.pt')
    mt2_m    = MT2NN()
    if os.path.exists(mt2_path):
        try:
            mt2_m.load_state_dict(torch.load(mt2_path, weights_only=True))
        except Exception as e:
            print(f"Warning: could not load mt2_best.pt: {e}")
    mt2_m.eval()
    with torch.no_grad():
        out = mt2_m(in48_t)
    tier_preds = out.view(12, 4).argmax(dim=1).tolist()
    tier_map   = {ind: tier_preds[i] for i, ind in enumerate(industry_list)}
    del mt2_m

    for ind in industry_list:
        if tier_map[ind] == 0:
            zero_counts[ind] = zero_counts.get(ind, 0) + 1
        else:
            zero_counts[ind] = 0

    allocations = tiers_to_alloc(tier_map, industry_list, total_cash)
    return allocations, tier_map, mt1_outputs
