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
import shutil
from collections import defaultdict

import torch
import torch.nn.functional as F

from models import MT1NN, MT2NN, StockNN
from training_lib import (
    ELITE_COUNT,
    ELITE_POOL,
    IND_STARTING_CASH,
    MT1_DIRECT_ELITES,
    MT1_ELITES_PER_CAT,
    MT1_ELITE_POOL,
    MT1_N_CATS,
    MT1_WAVG_BLENDS,
    N_SLOTS,
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
MT1_SCALE_DOLLARS = 10000.0          # tanh ceiling for dollar P&L prediction
MT1_FLOOR_COLD    = 250.0            # rolling-floor cold-start (IND_STARTING_CASH × 0.01)
MT1_ROLLING_DAYS  = 10              # days in rolling |actual_d| buffer
MT2_INJ_THRESHOLD = -7.0            # injection fires when ≥75% of pool below this
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


def _select_and_mutate_mt1(prefix, model_dir, scores_breakdown, sigma,
                            skip_range_selection=False):
    """
    Multi-category elite selection for MT1 pool (28 total parent slots).

    scores_breakdown: list of (slot, composite, direction, range_, accuracy, confidence)
                      for N_SLOTS slots.
    skip_range_selection: if True, leave range elite slots (10–14) unchanged (p==a edge case).

    Slot layout after selection:
      0–4   composite top-5
      5–9   direction top-5
      10–14 range top-5
      15–19 accuracy top-5
      20–24 confidence top-5
      25    wavg of {0,5,10,15,20}         — top-1 from each category
      26    wavg of {0,1,5,6,10,11,15,16,20,21}  — top-2 from each
      27    wavg of {0,1,2,5,6,7,10,11,12,15,16,17,20,21,22}  — top-3 from each
      28–199 mutations, round-robin over 28 parents
    """
    n = len(scores_breakdown)

    # Build sorted index lists per category (position in scores_breakdown)
    by_comp = sorted(range(n), key=lambda i: scores_breakdown[i][1], reverse=True)
    by_dir  = sorted(range(n), key=lambda i: scores_breakdown[i][2], reverse=True)
    by_rng  = sorted(range(n), key=lambda i: scores_breakdown[i][3], reverse=True)
    by_acc  = sorted(range(n), key=lambda i: scores_breakdown[i][4], reverse=True)
    by_conf = sorted(range(n), key=lambda i: scores_breakdown[i][5], reverse=True)
    cats    = [by_comp, by_dir, by_rng, by_acc, by_conf]

    # Pick top MT1_ELITES_PER_CAT from each category (preserving per-category order)
    cat_elites = []
    for cat in cats:
        picked = []
        for i in cat:
            if len(picked) >= MT1_ELITES_PER_CAT:
                break
            picked.append(scores_breakdown[i][0])
        while len(picked) < MT1_ELITES_PER_CAT:
            picked.append(picked[0])
        cat_elites.append(picked)

    # elite_slots_ordered: [comp0..4, dir0..4, rng0..4, acc0..4, conf0..4]
    elite_slots_ordered = [s for cat in cat_elites for s in cat]

    # Blend lists use explicit slot indices after writing:
    # slot 25 → top-1 from each cat: slot indices 0,5,10,15,20
    # slot 26 → top-2 from each cat: 0,1,5,6,10,11,15,16,20,21
    # slot 27 → top-3 from each cat: 0,1,2,5,6,7,10,11,12,15,16,17,20,21,22
    blend_source_ranks = [
        [0, 5, 10, 15, 20],
        [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22],
    ]

    # Pre-load all needed originals before any writes to avoid clobber
    cache = {s: load_slot_model(prefix, model_dir, s, MT1NN) for s in set(elite_slots_ordered)}

    # Write direct elites 0–MT1_DIRECT_ELITES-1
    # If skip_range_selection, preserve slots 10–14 (range category) from disk unchanged
    for rank, s in enumerate(elite_slots_ordered):
        if skip_range_selection and 10 <= rank <= 14:
            continue
        save_slot_model(prefix, model_dir, rank, cache[s])

    # Compute equal-weight wavg blend states from written elite slots
    blend_states = []
    for source_ranks in blend_source_ranks:
        inv_n  = 1.0 / len(source_ranks)
        avg_st = None
        for rank in source_ranks:
            m     = load_slot_model(prefix, model_dir, rank, MT1NN)
            state = m.state_dict()
            del m
            if avg_st is None:
                avg_st = {k: (v.clone().float() * inv_n if torch.is_floating_point(v) else v.clone())
                          for k, v in state.items()}
            else:
                for k, v in state.items():
                    if torch.is_floating_point(v) and k in avg_st:
                        avg_st[k] = avg_st[k] + v.float() * inv_n
        blend_states.append(avg_st)

    # Write wavg blends MT1_DIRECT_ELITES – MT1_ELITE_POOL-1 (slots 25, 26, 27)
    for L, avg_st in enumerate(blend_states):
        m = MT1NN()
        m.load_state_dict(avg_st)
        save_slot_model(prefix, model_dir, MT1_DIRECT_ELITES + L, m)
        del m

    del cache, blend_states

    # Mutations: round-robin over MT1_ELITE_POOL parents (slots 28–199)
    for i, slot in enumerate(range(MT1_ELITE_POOL, N_SLOTS)):
        parent_rank = i % MT1_ELITE_POOL
        parent = load_slot_model(prefix, model_dir, parent_rank, MT1NN)
        child  = _mutate_generic(parent, MT1NN, sigma)
        save_slot_model(prefix, model_dir, slot, child)
        del parent, child

    return elite_slots_ordered


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


def _rolling_floor(ind_state):
    """Compute floor_d from per-industry rolling buffer dict. Returns MT1_FLOOR_COLD if empty."""
    buf = ind_state.get('buffer', [])
    if not buf:
        return MT1_FLOOR_COLD
    return max(sum(buf) / len(buf), MT1_FLOOR_COLD)


def _rolling_update(ind_state, abs_actual_d):
    """Append to rolling circular buffer (max MT1_ROLLING_DAYS entries)."""
    buf  = ind_state.get('buffer', [])
    buf.append(abs_actual_d)
    if len(buf) > MT1_ROLLING_DAYS:
        buf = buf[-MT1_ROLLING_DAYS:]
    ind_state['buffer'] = buf


# ── MT1 scoring ────────────────────────────────────────────────────────────────

def _mt1_score_breakdown(out4, actual_d, floor_d):
    """
    Score one MT1 slot against the industry's actual dollar P&L.

    out4:     raw logit tensor shape (4,)
    actual_d: float — actual dollar P&L for the day (actual_frac × portfolio_value)
    floor_d:  float — rolling floor from 10-day mean |actual_d|, min MT1_FLOOR_COLD
    Returns (composite, direction, range_, accuracy, confidence) all in [0.0, 1.0].

    Composite = 0.50×direction + 0.33×range + 0.17×accuracy (out[0–2] only).
    Confidence (out[3]) is scored separately and does not enter composite.
    """
    conf     = torch.sigmoid(out4[0]).item()
    delta_t  = torch.tanh(out4[1]).item()
    delta_d  = delta_t * MT1_SCALE_DOLLARS
    rng_pct  = F.softplus(out4[2]).item()
    conf4    = torch.sigmoid(out4[3]).item()

    # Direction
    score_dir = 1.0 if (conf >= 0.5) == (actual_d >= 0.0) else 0.0

    # Range (calibration method: reward range tightness that still covers actual)
    eff_delta = max(abs(delta_d), floor_d)
    r         = rng_pct * eff_delta          # range in dollars
    err       = abs(actual_d - delta_d)
    m         = err / r if r > 0.0 else float('inf')
    score_rng = m if m < 1.0 else 0.0

    # Accuracy (dollar-denominated, floor prevents div-by-zero when actual ≈ 0)
    denom     = max(abs(actual_d), floor_d)
    score_acc = max(0.0, 1.0 - err / denom)

    # Confidence (grades out[3] against ideal derived from range geometry)
    d         = err
    if d <= r:
        ideal = 1.0 - 0.5 * (d / r) if r > 0.0 else 1.0
    else:
        ideal = r / (d + r) if (d + r) > 0.0 else 0.0
    score_conf = 1.0 - abs(conf4 - ideal)

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

def _mt1_burst(prefix, model_dir, in37_t, actual_d, floor_d, burst_sigma):
    """
    One burst refinement pass for an MT1 pool.

    Generates 280 mutants (10 per MT1_ELITE_POOL=28 parents), scores via composite,
    merges top-MT1_DIRECT_ELITES with current elites (cap: 2 burst replacements),
    and regenerates wavg blend slots 25–27. Returns new best composite score.
    """
    burst_candidates = []
    for elite_slot in range(MT1_ELITE_POOL):
        parent = load_slot_model(prefix, model_dir, elite_slot, MT1NN)
        for _ in range(10):
            child = _mutate_generic(parent, MT1NN, burst_sigma)
            child.eval()
            with torch.inference_mode():
                out4 = child(in37_t).squeeze(0)
            comp, *_ = _mt1_score_breakdown(out4, actual_d, floor_d)
            burst_candidates.append((child, comp))
        del parent
    burst_candidates.sort(key=lambda x: x[1], reverse=True)
    top_burst = burst_candidates[:MT1_DIRECT_ELITES]

    current_elites = []
    for rank in range(MT1_DIRECT_ELITES):
        m = load_slot_model(prefix, model_dir, rank, MT1NN)
        m.eval()
        with torch.inference_mode():
            out4 = m(in37_t).squeeze(0)
        comp, *_ = _mt1_score_breakdown(out4, actual_d, floor_d)
        current_elites.append((m, comp))

    burst_ids  = {id(m) for m, _ in top_burst}
    all_cands  = current_elites + top_burst
    all_cands.sort(key=lambda x: x[1], reverse=True)

    new_elites  = []
    burst_count = 0
    for cand in all_cands:
        if len(new_elites) >= MT1_DIRECT_ELITES:
            break
        if id(cand[0]) in burst_ids:
            if burst_count >= 2:
                continue
            burst_count += 1
        new_elites.append(cand)

    prev_best = current_elites[0][1]
    new_best  = new_elites[0][1]
    if new_best > prev_best:
        log(f"[mt1/{sn(prefix[4:])}] Burst σ={burst_sigma:.5f}: {burst_count} replacement(s), best={new_best:.4f}")
    else:
        log(f"[mt1/{sn(prefix[4:])}] Burst σ={burst_sigma:.5f}: best={new_best:.4f} — no improvement")

    for rank, (m, _) in enumerate(new_elites):
        save_slot_model(prefix, model_dir, rank, m)

    # Regenerate wavg blend slots 25, 26, 27 using same source ranks as main selection
    blend_source_ranks = [
        [0, 5, 10, 15, 20],
        [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22],
    ]
    for L, source_ranks in enumerate(blend_source_ranks):
        n_src  = len(source_ranks)
        inv_n  = 1.0 / n_src
        avg_st = None
        for rank in source_ranks:
            m     = load_slot_model(prefix, model_dir, rank, MT1NN)
            state = m.state_dict()
            del m
            if avg_st is None:
                avg_st = {kk: (v.clone().float() * inv_n if torch.is_floating_point(v) else v.clone())
                          for kk, v in state.items()}
            else:
                for kk, v in state.items():
                    if torch.is_floating_point(v) and kk in avg_st:
                        avg_st[kk] = avg_st[kk] + v.float() * inv_n
        wavg_m = MT1NN()
        wavg_m.load_state_dict(avg_st)
        save_slot_model(prefix, model_dir, MT1_DIRECT_ELITES + L, wavg_m)
        del wavg_m, avg_st

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
    One evolution step for one industry's MT1NN pool, with 4 burst passes.

    industry:        industry key (e.g. 'energy')
    model_dir:       directory containing model files
    in37_t:          (1, 37) tensor — this industry's slice of build_master_features output
    actual_perf_i:   float — fractional return (slot0_score / baseline - 1) for today
    sigma:           base mutation sigma
    portfolio_value: current slot0 industry portfolio value (dollars); defaults to
                     IND_STARTING_CASH if not supplied
    rolling_state:   mutable dict with per-industry rolling buffer; updated in place.
                     Caller should load/save with load_mt1_rolling_state/save_mt1_rolling_state.

    File naming: mt1_{industry}_model_{slot}.pt (uses _model_path with prefix 'mt1_{industry}')

    Returns (best_mt1_score, slot0_mt1_score, slot0_conf, slot0_delta_t, slot0_range_pct,
             slot0_conf4).
    """
    if portfolio_value is None:
        portfolio_value = IND_STARTING_CASH
    if rolling_state is None:
        rolling_state = {}

    actual_d = actual_perf_i * portfolio_value
    ind_rs   = rolling_state.setdefault(industry, {})
    floor_d  = _rolling_floor(ind_rs)

    prefix     = f"mt1_{industry}"
    slot0_path = _model_path(prefix, model_dir, 0)
    best_path  = os.path.join(model_dir, f"{prefix}_best.pt")

    if not os.path.exists(slot0_path):
        base = MT1NN()
        if os.path.exists(best_path):
            try:
                base.load_state_dict(torch.load(best_path, weights_only=True))
                log(f"[mt1/{sn(industry)}] Bootstrapping pool from {prefix}_best.pt")
            except Exception:
                log(f"[mt1/{sn(industry)}] Bootstrap failed — using random weights")
        else:
            log(f"[mt1/{sn(industry)}] Initializing MT1 pool with random weights")
        save_slot_model(prefix, model_dir, 0, base)
        for slot in range(1, N_SLOTS):
            child = _mutate_generic(base, MT1NN, sigma)
            save_slot_model(prefix, model_dir, slot, child)
            del child
        del base
    elif not os.path.exists(_model_path(prefix, model_dir, N_SLOTS - 1)):
        # Partial pool: expand to full N_SLOTS round-robin from existing elites
        n_existing = sum(1 for s in range(MT1_ELITE_POOL)
                         if os.path.exists(_model_path(prefix, model_dir, s)))
        n_parents  = max(n_existing, 1)
        log(f"[mt1/{sn(industry)}] Expanding MT1 pool from {n_existing} elites to {N_SLOTS} slots")
        for s in range(n_existing, MT1_ELITE_POOL):
            base = load_slot_model(prefix, model_dir, 0, MT1NN)
            save_slot_model(prefix, model_dir, s, base)
            del base
        for i, slot in enumerate(range(MT1_ELITE_POOL, N_SLOTS)):
            parent_rank = i % n_parents
            parent = load_slot_model(prefix, model_dir, parent_rank, MT1NN)
            child  = _mutate_generic(parent, MT1NN, sigma)
            save_slot_model(prefix, model_dir, slot, child)
            del parent, child

    in37_t = in37_t.detach()

    # Score all N_SLOTS slots across all 5 MT1 components
    scores_breakdown = []   # [(slot, composite, dir, rng, acc, conf), ...]
    for slot in range(N_SLOTS):
        m = load_slot_model(prefix, model_dir, slot, MT1NN)
        m.eval()
        with torch.inference_mode():
            out4 = m(in37_t).squeeze(0)
        comp, dir_, rng_, acc_, conf_ = _mt1_score_breakdown(out4, actual_d, floor_d)
        scores_breakdown.append((slot, comp, dir_, rng_, acc_, conf_))
        del m

    # Record slot0 outputs before selection (selection may overwrite slot 0)
    slot0_m = load_slot_model(prefix, model_dir, 0, MT1NN)
    slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4 = _mt1_decode(slot0_m, in37_t)
    del slot0_m

    best_score  = max(bd[1] for bd in scores_breakdown)
    slot0_score = scores_breakdown[0][1]

    log(f"[mt1/{sn(industry)}] best={best_score:.4f} slot0={slot0_score:.4f} "
        f"actual_d=${actual_d:+.1f} floor_d=${floor_d:.1f}")

    # Check p==a edge case: if best range score is 0, skip range elite selection
    best_rng_score = max(bd[3] for bd in scores_breakdown)
    skip_range = best_rng_score == 0.0

    _select_and_mutate_mt1(prefix, model_dir, scores_breakdown, sigma,
                            skip_range_selection=skip_range)

    # Update rolling buffer AFTER scoring
    _rolling_update(ind_rs, abs(actual_d))

    # 4 burst refinement passes
    log(f"[mt1/{sn(industry)}] Running 4 burst passes ...")
    for burst_num in range(4):
        _mt1_burst(prefix, model_dir, in37_t, actual_d, floor_d,
                   sigma / (2 ** (burst_num + 1)))

    # Save slot 0 (post-selection best) as the best model for production inference
    slot0_final = load_slot_model(prefix, model_dir, 0, MT1NN)
    try:
        torch.save(slot0_final.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save {prefix}_best.pt: {e}")
    del slot0_final

    return best_score, slot0_score, slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4


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
