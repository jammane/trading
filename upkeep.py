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

from models import MT1NN, MT1Head, MT1Tail, MT2NN, StockNN
from training_lib import (
    ELITE_COUNT,
    ELITE_POOL,
    HIST_DAYS,
    HIST_ELITE,
    HIST_PER_DAY,
    HIST_WAVG,
    IND_STARTING_CASH,
    MT1_BLEND_SLOTS,
    MT1_COMP_CHILDREN,
    MT1_COMP_INJECT,
    MT1_COMP_PARENTS,
    MT1_COMP_SLOTS,
    MT1_DIR_BACKFILL,
    MT1_DIR_DAYS,
    MT1_DIR_SKILL_FLOOR,
    MT1_DIR_MIN_CORRECT,
    MT1_POOL_NAMES,
    MT1_RANGE_CEIL_MULT,
    MT1_RANGE_FLOOR,
    MT1_RANGE_INJECT,
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

UPKEEP_SIGMA      = 0.004   # StockNN (industry models)
UPKEEP_DIR_SIGMA  = 0.006   # MT1 direction pool
UPKEEP_RNG_SIGMA  = 0.004   # MT1 range pool (stable; ceiling cull handles exploration)
UPKEEP_ACC_SIGMA  = 0.006   # MT1 accuracy pool (noisy until industry stabilizes)
UPKEEP_CFD_SIGMA  = 0.003   # MT1 confidence pool (most stable; fine-tune only)
UPKEEP_HEAD_SIGMA = 0.004   # MT1 shared head/trunk (block-cycle head phase)
UPKEEP_MT2_SIGMA  = 0.001   # MT2 (v0.2.6.0: 0.002→0.001 to tighten the pool)
MT1_SCALE_DOLLARS = 10000.0   # tanh ceiling for dollar P&L prediction
MT1_FLOOR_COLD    = 250.0     # acc_floor cold-start (÷2 = $125)
MT1_ROLLING_DAYS  = 10        # days in rolling buffers
MT2_INJ_THRESHOLD = -7.0      # injection fires when ≥75% of pool below this
MT2_INJ_MIN_BELOW = int(N_SLOTS * 0.75)  # 150 of 200

# Weighted children per parent for (legacy) MT1 component pools (mirror of C++ kChildren): slot0=13,
# slots1-4=11/11/10/10, slots5-19=7, injected slots20-24=3. Sums to 175 = MT1_COMP_SLOTS-PARENTS.
_MT1_CHILDREN     = [13, 11, 11, 10, 10] + [7] * 12 + [7] * 3 + [3] * 5
MT1_PARENT_OF_MUT = [p for p, c in enumerate(_MT1_CHILDREN) for _ in range(c)]
assert len(MT1_PARENT_OF_MUT) == MT1_COMP_SLOTS - MT1_COMP_PARENTS, "MT1 children table must sum to 175"

# ── Heads/tails pool layout (Increment 4B) ──────────────────────────────────────
# One shared head pool + 4 specialized tail pools per industry. No injection slots — the shared
# head propagates cross-component learning. HT_PARENTS = 17 direct elites + 3 wavg (= 20); the
# reclaimed capacity goes into more elite mutations. Children table mirrors C++ step_mt1_pool.
MT1_BLOCK_DAYS = 25   # head cycle fires every MT1_BLOCK_DAYS upkeep runs; tails evolve every run
HT_PARENTS     = ELITE_COUNT + WAVG_COUNT   # 20
_HT_CHILDREN   = [16, 13, 13, 12, 12] + [8] * 12 + [6] * 3   # sums to 180
HT_PARENT_OF_MUT = [p for p, c in enumerate(_HT_CHILDREN) for _ in range(c)]
assert len(HT_PARENT_OF_MUT) == MT1_COMP_SLOTS - HT_PARENTS, "HT children table must sum to 180"



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
                                       hist_models=None):
    """
    Elite selection + mutation for one MT1 component pool (200 slots).

    scores:      list of (slot, cat_score) — culled slots have score=-1e30.
    hist_models: optional list of (MT1NN, cat_score) for history candidates
                 (no culling applied — they are unconditional safety-net entries).

    Slot layout after selection:
      0–16   direct elites (ELITE_COUNT)
      17–19  wavg blends   (WAVG_COUNT: top-5, top-10, top-15)
      20–24  injection slots (MT1_COMP_INJECT) — written by caller after this call
      25–199 mutations (MT1_COMP_CHILDREN=7 per parent, round-robin over 25 parents)
    Returns (new_elite_models, new_wavg_models) — lists of loaded MT1NN objects for
    the HIST_ELITE direct elites and HIST_WAVG wavg models (for history saving by caller).
    """
    # Build combined candidate list: live slots + history candidates
    # History candidates use sentinel slot indices >= MT1_COMP_SLOTS
    all_scores = list(scores)  # (slot_idx, score)
    hist_offset = MT1_COMP_SLOTS
    if hist_models:
        for h_idx, (hm, hsc) in enumerate(hist_models):
            all_scores.append((hist_offset + h_idx, hsc))

    sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    top_entries = sorted_scores[:ELITE_COUNT]
    while len(top_entries) < ELITE_COUNT:
        top_entries.append(top_entries[0])

    # Build cache: live slots loaded from disk, history models from hist_models list
    live_slots_needed = {s for s, _ in top_entries if s < hist_offset}
    cache = {s: load_slot_model(prefix, model_dir, s, MT1NN) for s in live_slots_needed}
    hist_cache = {}
    if hist_models:
        for h_idx, (hm, _) in enumerate(hist_models):
            hist_cache[hist_offset + h_idx] = hm

    for rank, (s, _) in enumerate(top_entries):
        m = cache[s] if s < hist_offset else hist_cache[s]
        save_slot_model(prefix, model_dir, rank, m)

    # Wavg blends: equal-weight average of top 5, 10, 15 direct elites
    new_wavg_models = []
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
        new_wavg_models.append(wm)
        del avg_st

    # Injection slots 20–24: written by caller after this call

    del cache

    # Mutations: weighted children per parent (mirror of C++ kChildren) — concentrate breeding on
    # proven elites (slot0=13, slots1-4=11/11/10/10, slots5-19=7) and starve unproven injected
    # immigrants (slots 20-24 = 3 each). Sums to 175 = MT1_COMP_SLOTS - MT1_COMP_PARENTS.
    for i, slot in enumerate(range(MT1_COMP_PARENTS, MT1_COMP_SLOTS)):
        parent_rank = MT1_PARENT_OF_MUT[i]
        parent = load_slot_model(prefix, model_dir, parent_rank, MT1NN)
        child  = _mutate_generic(parent, MT1NN, sigma)
        save_slot_model(prefix, model_dir, slot, child)
        del parent, child

    # Return top HIST_ELITE elites and HIST_WAVG wavg models for history saving
    new_elites = [load_slot_model(prefix, model_dir, k, MT1NN) for k in range(HIST_ELITE)]
    return new_elites, new_wavg_models[:HIST_WAVG]


def _load_comp_pool_hist_models(industry, pool_name, model_dir):
    """Load MT1 component pool history models. Returns list of MT1NN models (may be empty)."""
    meta_path = os.path.join(model_dir, f'mt1_{industry}_{pool_name}_hist_meta.json')
    models = []
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        head  = meta.get('head', 0)
        count = meta.get('count', 0)
        n_days = min(count, HIST_DAYS)
        for d in range(n_days):
            day_slot = (head - n_days + d) % HIST_DAYS
            for pos in range(HIST_PER_DAY):
                hp = os.path.join(model_dir,
                                  f'mt1_{industry}_{pool_name}_hist_{day_slot}_{pos}.pt')
                if os.path.exists(hp):
                    try:
                        m = MT1NN()
                        m.load_state_dict(torch.load(hp, weights_only=True))
                        models.append(m)
                    except Exception:
                        pass
    except Exception:
        pass
    return models


def _save_comp_pool_hist(industry, pool_name, model_dir, new_elites, new_wavgs):
    """Save top HIST_ELITE elites + HIST_WAVG wavg models to component pool history."""
    meta_path = os.path.join(model_dir, f'mt1_{industry}_{pool_name}_hist_meta.json')
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        head, count = meta.get('head', 0), meta.get('count', 0)
    except Exception:
        head, count = 0, 0
    for k, m in enumerate(new_elites[:HIST_ELITE]):
        hp = os.path.join(model_dir, f'mt1_{industry}_{pool_name}_hist_{head}_{k}.pt')
        try:
            torch.save(m.state_dict(), hp)
        except Exception:
            pass
    for k, m in enumerate(new_wavgs[:HIST_WAVG]):
        hp = os.path.join(model_dir,
                          f'mt1_{industry}_{pool_name}_hist_{head}_{HIST_ELITE + k}.pt')
        try:
            torch.save(m.state_dict(), hp)
        except Exception:
            pass
    head  = (head + 1) % HIST_DAYS
    count = min(count + 1, HIST_DAYS)
    try:
        with open(meta_path, 'w') as f:
            json.dump({'head': head, 'count': count}, f)
    except Exception:
        pass


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


def _rolling_range_ceiling(ind_state, today_residual=0.0):
    """Range ceiling = max(4 × mean_residual, 4 × today_residual).
    Returns None (no ceiling) on cold start."""
    buf = ind_state.get('residual_buf', [])
    if not buf:
        return None
    mean_r = sum(buf) / len(buf)
    return MT1_RANGE_CEIL_MULT * max(mean_r, today_residual)


def _rolling_update(ind_state, abs_actual_d, comp0_residual):
    """Update both rolling circular buffers (max MT1_ROLLING_DAYS entries each)."""
    for key, val in (('actual_buf', abs_actual_d), ('residual_buf', comp0_residual)):
        buf = ind_state.get(key, [])
        buf.append(val)
        if len(buf) > MT1_ROLLING_DAYS:
            buf = buf[-MT1_ROLLING_DAYS:]
        ind_state[key] = buf


# ── MT1 scoring ────────────────────────────────────────────────────────────────

def mt1_win_weight(di, day_count):
    """Linear scoring-window weight (mirror of C++ mt1_win_weight): today (di=day_count-1) = 2.0,
    a day MT1_DIR_DAYS-1 steps back = 1.0. Anchored to MT1_DIR_DAYS (absolute age)."""
    if MT1_DIR_DAYS <= 1:
        return 2.0
    age = day_count - 1 - di
    w = 2.0 - age / (MT1_DIR_DAYS - 1)
    return 1.0 if w < 1.0 else w


def _dir_day_weights(dir_hist):
    """Class-balanced direction-pool day weights (mirror of C++ step_mt1_component): up-days and
    down-days each carry half the window weight, so a constant-prediction model scores exactly the
    no-skill baseline dir_W/2 regardless of the absolute target's sign skew. Returns (weights, dir_W).
    Single-class window → plain recency weights. dir_hist entries are (feat_t, actual_d)."""
    n = len(dir_hist)
    ws = [mt1_win_weight(di, n) for di in range(n)]
    dir_W = sum(ws)
    w_up = sum(ws[di] for di in range(n) if dir_hist[di][1] >= 0.0)
    w_down = dir_W - w_up
    if w_up > 0.0 and w_down > 0.0:
        dw = [ws[di] * ((dir_W / (2.0 * w_up)) if dir_hist[di][1] >= 0.0 else (dir_W / (2.0 * w_down)))
              for di in range(n)]
    else:
        dw = list(ws)
    return dw, dir_W


def _mt1_score_breakdown(out4, actual_d, acc_floor, range_ceiling=None):
    """
    Score one MT1 slot against the industry's actual dollar P&L.

    out4:          raw logit tensor shape (4,)
    actual_d:      float — actual dollar P&L (actual_frac × portfolio_value)
    acc_floor:     float — per-industry adaptive floor for accuracy denom
    range_ceiling: float or None — cap on range r (None = no ceiling)
    Returns (composite, direction, range_, accuracy, confidence) all in [0.0, 1.0].

    Composite = equal-weight mean of the four components' secondary [0,1] normalizations
    against each one's naive baseline (mirror of C++ compute_mt1_scores). The returned
    direction/range/accuracy/confidence are the RAW component scores (used by the pools).
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
    err       = abs(abs(actual_d) - abs(delta_d))   # signless: magnitude graded independent of sign (direction owns sign)
    m         = err / r if r > 0.0 else float('inf')
    score_rng = m if m < 1.0 else 0.0

    denom     = max(abs(actual_d), acc_floor)
    score_acc = denom / (err + denom)

    d         = err
    dor       = (d / r) if r > 1e-9 else (1e9 if d > 0.0 else 0.0)
    ideal     = 1.0 / (1.0 + dor * dor)
    diff      = conf4 - ideal
    score_conf = 1.0 - diff * diff
    if err > r:
        score_conf = 0.5 + 0.25 * score_conf  # compress outside-range to [0.5, 0.75]

    # Secondary [0,1] normalization per component vs naive baseline B (ideal=1):
    #   sec = clamp((raw - B) / (1 - B), 0, 1)  → naive→0, ideal→1.  (mirror of C++)
    def _c01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    b_dir   = 0.5                                          # conf = 0.5, no view
    sec_dir = _c01((score_dir - b_dir) / max(1.0 - b_dir, 1e-6))

    b_acc   = denom / (abs(actual_d) + denom)              # predict-0 null model
    sec_acc = _c01((score_acc - b_acc) / max(1.0 - b_acc, 1e-6))

    m_naive = (err / acc_floor) if acc_floor > 1e-9 else (1e9 if err > 0.0 else 0.0)
    b_rng   = m_naive if m_naive < 1.0 else 0.0            # band = unconditional scale (acc_floor)
    sec_rng = _c01((score_rng - b_rng) / max(1.0 - b_rng, 1e-6))

    diff0   = 0.5 - ideal                                  # conf4 = 0.5
    b_cfd   = 1.0 - diff0 * diff0
    if err > r:
        b_cfd = 0.5 + 0.25 * b_cfd
    sec_cfd = _c01((score_conf - b_cfd) / max(1.0 - b_cfd, 1e-6))

    composite = 0.25 * (sec_dir + sec_acc + sec_rng + sec_cfd)
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
    Applies same culls as main scoring (range ceiling for pool 2/3,
    alignment for pool 1). Generates 10 mutants per MT1_COMP_PARENTS (25)
    parent elites, merges top-ELITE_COUNT with current elites (cap: 2 replacements).
    """
    # breakdown tuple: (composite, dir, rng, acc, conf) → pick component score
    _sc_idx = [1, 3, 2, 4][score_idx]
    dir_dw, _ = _dir_day_weights(dir_hist) if dir_hist else ([], 0.0)   # class-balanced direction weights

    def _score_multiday(m):
        total = 0.0
        culled = False
        prev_conf_pos, prev_act_pos = None, None
        conf_crossings, market_flips = 0, 0
        with torch.inference_mode():
            for di, (feat_t, ad) in enumerate(dir_hist):
                if culled:
                    break
                is_today = (di == len(dir_hist) - 1)
                out4 = m(feat_t).squeeze(0)
                conf    = torch.sigmoid(out4[0]).item()
                delta_d = torch.tanh(out4[1]).item() * MT1_SCALE_DOLLARS
                rng_pct = F.softplus(out4[2]).item()
                r_raw   = rng_pct * max(abs(delta_d), MT1_RANGE_FLOOR)
                if (score_idx == 2 or score_idx == 3) and range_ceiling is not None:
                    if r_raw > range_ceiling:
                        culled = True; break
                # (accuracy-pool sign-alignment cull removed — magnitude graded signless)
                if score_idx == 0:
                    conf_pos = conf >= 0.5
                    act_pos  = ad >= 0.0
                    day_score = conf if act_pos else (1.0 - conf)
                    if prev_conf_pos is not None and conf_pos != prev_conf_pos:
                        conf_crossings += 1
                    if prev_act_pos is not None and act_pos != prev_act_pos:
                        market_flips += 1
                    prev_conf_pos = conf_pos
                    prev_act_pos  = act_pos
                elif score_idx == 1:
                    err = abs(abs(ad) - abs(delta_d))   # signless: magnitude graded independent of sign
                    day_score = acc_floor / (err + acc_floor)
                else:
                    bd = _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)
                    day_score = bd[_sc_idx]
                total += day_score * (dir_dw[di] if score_idx == 0 else mt1_win_weight(di, len(dir_hist)))
        if not culled and score_idx == 0 and len(dir_hist) >= 2:
            required = market_flips // 2   # floor: no cull at market_flips<=1 (mirror of C++)
            if conf_crossings < required:
                culled = True
        return -1e30 if culled else total

    burst_candidates = []
    for elite_slot in range(MT1_COMP_PARENTS):
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

# ── Heads/tails pool history + selection helpers (Increment 4B) ──────────────────

def _ht_load_hist(prefix, model_dir, model_class):
    """Load a head/tail pool's 5-day elite history models. Returns a list (may be empty)."""
    meta_path = os.path.join(model_dir, f'{prefix}_hist_meta.json')
    models = []
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        head, count = meta.get('head', 0), meta.get('count', 0)
        n_days = min(count, HIST_DAYS)
        for d in range(n_days):
            day_slot = (head - n_days + d) % HIST_DAYS
            for pos in range(HIST_PER_DAY):
                hp = os.path.join(model_dir, f'{prefix}_hist_{day_slot}_{pos}.pt')
                if os.path.exists(hp):
                    try:
                        m = model_class()
                        m.load_state_dict(torch.load(hp, weights_only=True))
                        models.append(m)
                    except Exception:
                        pass
    except Exception:
        pass
    return models


def _ht_save_hist(prefix, model_dir, model_class, new_elites, new_wavgs):
    """Save top HIST_ELITE elites + HIST_WAVG wavg models to a head/tail pool's history."""
    meta_path = os.path.join(model_dir, f'{prefix}_hist_meta.json')
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        head, count = meta.get('head', 0), meta.get('count', 0)
    except Exception:
        head, count = 0, 0
    for k, m in enumerate(new_elites[:HIST_ELITE]):
        try:
            torch.save(m.state_dict(), os.path.join(model_dir, f'{prefix}_hist_{head}_{k}.pt'))
        except Exception:
            pass
    for k, m in enumerate(new_wavgs[:HIST_WAVG]):
        try:
            torch.save(m.state_dict(),
                       os.path.join(model_dir, f'{prefix}_hist_{head}_{HIST_ELITE + k}.pt'))
        except Exception:
            pass
    try:
        with open(meta_path, 'w') as f:
            json.dump({'head': (head + 1) % HIST_DAYS, 'count': min(count + 1, HIST_DAYS)}, f)
    except Exception:
        pass


def _ht_select_and_mutate(prefix, model_dir, model_class, scores, sigma, hist_models=None):
    """
    Elite selection + mutation for one head or tail pool (HT layout, mirror of C++ step_mt1_pool):
      0–16   direct elites (ELITE_COUNT)
      17–19  wavg blends   (WAVG_COUNT: top-5/10/15)
      20–…   mutations (HT_PARENT_OF_MUT children per parent; NO injection slots)
    scores:      [(slot, score)] with culled slots = -1e30.
    hist_models: [(model, score)] history candidates (no culling — safety-net entries).
    Returns (new_elites[:HIST_ELITE], new_wavgs[:HIST_WAVG]) for history saving.
    """
    all_scores = list(scores)
    hist_offset = MT1_COMP_SLOTS
    if hist_models:
        for h_idx, (hm, hsc) in enumerate(hist_models):
            all_scores.append((hist_offset + h_idx, hsc))
    sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    top_entries = sorted_scores[:ELITE_COUNT]
    while len(top_entries) < ELITE_COUNT:
        top_entries.append(top_entries[0])

    live_needed = {s for s, _ in top_entries if s < hist_offset}
    cache = {s: load_slot_model(prefix, model_dir, s, model_class) for s in live_needed}
    hist_cache = {}
    if hist_models:
        for h_idx, (hm, _) in enumerate(hist_models):
            hist_cache[hist_offset + h_idx] = hm
    for rank, (s, _) in enumerate(top_entries):
        save_slot_model(prefix, model_dir, rank, cache[s] if s < hist_offset else hist_cache[s])

    new_wavgs = []
    for b, k in enumerate([5, 10, 15]):
        inv_k = 1.0 / k
        avg_st = None
        for rank in range(k):
            m = load_slot_model(prefix, model_dir, rank, model_class)
            state = m.state_dict(); del m
            if avg_st is None:
                avg_st = {key: (v.clone().float() * inv_k if torch.is_floating_point(v) else v.clone())
                          for key, v in state.items()}
            else:
                for key, v in state.items():
                    if torch.is_floating_point(v) and key in avg_st:
                        avg_st[key] = avg_st[key] + v.float() * inv_k
        wm = model_class(); wm.load_state_dict(avg_st)
        save_slot_model(prefix, model_dir, ELITE_COUNT + b, wm)
        new_wavgs.append(wm); del avg_st
    del cache

    for i, slot in enumerate(range(HT_PARENTS, MT1_COMP_SLOTS)):
        parent_rank = HT_PARENT_OF_MUT[i] if i < len(HT_PARENT_OF_MUT) else (i % HT_PARENTS)
        parent = load_slot_model(prefix, model_dir, parent_rank, model_class)
        child = _mutate_generic(parent, model_class, sigma)
        save_slot_model(prefix, model_dir, slot, child)
        del parent, child

    new_elites = [load_slot_model(prefix, model_dir, k, model_class) for k in range(HIST_ELITE)]
    return new_elites, new_wavgs[:HIST_WAVG]


# ── MT1 industry upkeep (heads/tails block cycle) ────────────────────────────────

def upkeep_mt1_industry(industry, model_dir, in37_t, actual_d,
                         dir_sigma=UPKEEP_DIR_SIGMA, rng_sigma=UPKEEP_RNG_SIGMA,
                         acc_sigma=UPKEEP_ACC_SIGMA, cfd_sigma=UPKEEP_CFD_SIGMA,
                         head_sigma=UPKEEP_HEAD_SIGMA, rolling_state=None):
    """
    One production evolution step for one industry's heads/tails MT1 (Increment 4B).

    Daily: evolve the 4 tail pools (freeze the best head + the other best tails; mirror the C++
    step_mt1_tail — direction keeps class-balanced weights, two-half selection, flip cull and the
    correct-count collapse floor). Every MT1_BLOCK_DAYS runs (block counter in rolling_state):
    also evolve the head pool (freeze the best tails; mirror step_mt1_head, composite fitness).
    Production model = composed best head + best tails → mt1_{ind}_best.pt for inference.

    File naming: mt1_{ind}_head_model_{slot}.pt (MT1Head),
                 mt1_{ind}_tail_{dir|acc|rng|cfd}_model_{slot}.pt (MT1Tail),
                 mt1_{ind}_best.pt (composed MT1NN), + per-pool _hist_* and mt1_{ind}_dir_hist.json.

    Returns (best_score, best_score, conf, delta_t, range_pct, conf4, conf, delta_t, range_pct,
    conf4) — the composed production model's slot0 activations (the composite and "direction" MT2
    feeds are identical in the head/tail design, so MT2_FEED_DIRECTION is moot).
    """
    if rolling_state is None:
        rolling_state = {}
    ind_rs    = rolling_state.setdefault(industry, {})
    acc_floor = _rolling_acc_floor(ind_rs)
    in37_t    = in37_t.detach()

    head_prefix   = f'mt1_{industry}_head'
    tail_prefixes = [f'mt1_{industry}_tail_{n}' for n in MT1_POOL_NAMES]
    tail_sigmas   = [dir_sigma, acc_sigma, rng_sigma, cfd_sigma]
    best_path     = os.path.join(model_dir, f'mt1_{industry}_best.pt')

    # Bootstrap pools on first run (seed from best.pt head/tails if present, else fresh).
    if not os.path.exists(_model_path(head_prefix, model_dir, 0)):
        base = MT1NN()
        if os.path.exists(best_path):
            try:
                base.load_state_dict(torch.load(best_path, weights_only=True))
                log(f"[mt1/{sn(industry)}] Bootstrapping head/tail pools from mt1_{industry}_best.pt")
            except Exception:
                log(f"[mt1/{sn(industry)}] Bootstrap failed — random head/tail pools")
        else:
            log(f"[mt1/{sn(industry)}] Initializing random head/tail pools")
        hb = MT1Head(); hb.load_state_dict(base.head.state_dict())
        save_slot_model(head_prefix, model_dir, 0, hb)
        for slot in range(1, MT1_COMP_SLOTS):
            save_slot_model(head_prefix, model_dir, slot, _mutate_generic(hb, MT1Head, head_sigma))
        for c, tp in enumerate(tail_prefixes):
            tb = MT1Tail(); tb.load_state_dict(base.tails[c].state_dict())
            save_slot_model(tp, model_dir, 0, tb)
            for slot in range(1, MT1_COMP_SLOTS):
                save_slot_model(tp, model_dir, slot, _mutate_generic(tb, MT1Tail, tail_sigmas[c]))
        del base

    # Frozen production bests at phase start (head + 4 tails).
    head0  = load_slot_model(head_prefix, model_dir, 0, MT1Head); head0.eval()
    tails0 = [load_slot_model(tp, model_dir, 0, MT1Tail) for tp in tail_prefixes]
    for t in tails0:
        t.eval()

    # Today's residual (composed best) for the range ceiling.
    today_residual = 0.0
    if ind_rs.get('residual_buf'):
        with torch.inference_mode():
            c0 = head0(in37_t)
            comp0_delta_d = math.tanh(tails0[1](c0).reshape(-1)[0].item()) * MT1_SCALE_DOLLARS
        today_residual = abs(actual_d - comp0_delta_d)
    range_ceiling = _rolling_range_ceiling(ind_rs, today_residual)

    # Direction day buffer (append today, persist, load the trailing window).
    dir_hist_path = os.path.join(model_dir, f'mt1_{industry}_dir_hist.json')
    dir_hist_raw = []
    if os.path.exists(dir_hist_path):
        try:
            with open(dir_hist_path) as _f:
                dir_hist_raw = json.load(_f)
        except Exception:
            dir_hist_raw = []
    dir_hist_raw.append({'feat37': in37_t.squeeze(0).tolist(), 'actual_d': float(actual_d)})
    dir_hist_raw = dir_hist_raw[-MT1_DIR_DAYS:]
    with open(dir_hist_path, 'w') as _f:
        json.dump(dir_hist_raw, _f)
    dir_hist = [(torch.tensor(e['feat37'], dtype=torch.float32).unsqueeze(0), e['actual_d'])
                for e in dir_hist_raw]
    dir_dw, _ = _dir_day_weights(dir_hist) if dir_hist else ([], 0.0)
    n_win = len(dir_hist)

    # Precompute frozen-head concat + frozen tail logits per window day (shared across tail pools).
    concats, frozen_logits = [], []
    with torch.inference_mode():
        for feat_t, _ad in dir_hist:
            c = head0(feat_t)
            concats.append(c)
            frozen_logits.append([tails0[k](c).reshape(-1)[0].item() for k in range(4)])

    def _score_tail(cand, comp, apply_cull=True):
        cand.eval()
        total = 0.0; culled = False
        n_correct = n_correct_dbl = today_correct = 0
        prev_cp = prev_ap = None; crossings = flips = 0
        with torch.inference_mode():
            for di, (feat_t, ad) in enumerate(dir_hist):
                is_today = (di == n_win - 1)
                o = list(frozen_logits[di]); o[comp] = cand(concats[di]).reshape(-1)[0].item()
                out4 = torch.tensor(o)
                conf    = torch.sigmoid(out4[0]).item()
                delta_d = math.tanh(o[1]) * MT1_SCALE_DOLLARS
                rng_pct = F.softplus(out4[2]).item()
                r_raw   = rng_pct * max(abs(delta_d), MT1_RANGE_FLOOR)
                if (comp == 2 or comp == 3) and range_ceiling is not None and r_raw > range_ceiling:
                    culled = True
                if comp == 0:
                    cp = conf >= 0.5; ap = ad >= 0.0
                    day = conf if ap else (1.0 - conf); corr = cp == ap
                    n_correct += 1 if corr else 0
                    n_correct_dbl += (2 if corr else 0) if is_today else (1 if corr else 0)
                    if is_today:
                        today_correct = 1 if corr else 0
                    if prev_cp is not None and cp != prev_cp:
                        crossings += 1
                    if prev_ap is not None and ap != prev_ap:
                        flips += 1
                    prev_cp = cp; prev_ap = ap
                elif comp == 1:
                    err = abs(abs(ad) - abs(delta_d)); day = acc_floor / (err + acc_floor)
                else:
                    day = _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)[2 if comp == 2 else 4]
                total += day * (dir_dw[di] if comp == 0 else mt1_win_weight(di, n_win))
        if comp == 0 and n_win >= 2 and crossings < flips // 2:
            culled = True
        if apply_cull and culled:
            return -1e30, n_correct, n_correct_dbl, today_correct
        return total, n_correct, n_correct_dbl, today_correct

    # ── Tail phase (every run): freeze head0 + tail0, evolve the 4 tail pools ──
    for comp, prefix in enumerate(tail_prefixes):
        scores = []; dir_ncorr = {}; dir_today = {}; dir_max_correct = 0
        for slot in range(MT1_COMP_SLOTS):
            m = load_slot_model(prefix, model_dir, slot, MT1Tail)
            sc, nc, _ncd, td = _score_tail(m, comp, apply_cull=True)
            scores.append((slot, sc))
            if comp == 0 and sc > -1e29:
                dir_ncorr[slot] = nc; dir_today[slot] = td
                if nc > dir_max_correct:
                    dir_max_correct = nc
            del m
        hist_models = _ht_load_hist(prefix, model_dir, MT1Tail)
        hist_cands = []
        for hm in hist_models:
            sc, nc, _ncd, td = _score_tail(hm, comp, apply_cull=False)
            hist_cands.append((hm, sc, nc, td))

        # Direction collapse backfill: keep yesterday's elites on genuine collapse.
        if comp == 0 and dir_max_correct < MT1_DIR_MIN_CORRECT:
            log(f"[mt1/{sn(industry)}:dir] max_correct={dir_max_correct} < {MT1_DIR_MIN_CORRECT} "
                f"— backfill: keeping yesterday's tail elites")
            continue

        hist_offset = MT1_COMP_SLOTS
        if comp != 0:
            sort_scores = scores
            hist_with_scores = [(hm, sc) for (hm, sc, _, _) in hist_cands]
        else:
            # Direction two-half selection (mirror of C++ / old component path).
            cand = {}
            for slot, sc in scores:
                cand[slot] = (sc, dir_ncorr.get(slot, 0), dir_today.get(slot, 0))
            for i, (_hm, sc, nc, td) in enumerate(hist_cands):
                cand[hist_offset + i] = (sc, nc, td)
            ids = list(cand.keys())
            by_score   = sorted(ids, key=lambda k: cand[k][0], reverse=True)
            by_correct = sorted(ids, key=lambda k: (cand[k][1], cand[k][0]), reverse=True)
            order, taken = [], set()
            slot0 = by_score[0] if by_score else None
            for k in by_score:
                if cand[k][2]:
                    slot0 = k; break
            if slot0 is not None:
                order.append(slot0); taken.add(slot0)
            score_target = 1 + ELITE_COUNT // 2
            for k in by_score:
                if len(order) >= score_target: break
                if k not in taken: order.append(k); taken.add(k)
            for k in by_correct:
                if len(order) >= ELITE_COUNT: break
                if k not in taken: order.append(k); taken.add(k)
            for k in by_score:
                if len(order) >= ELITE_COUNT: break
                if k not in taken: order.append(k); taken.add(k)
            BIG = 1e9
            syn = {k: cand[k][0] for k in ids}
            for rank, k in enumerate(order):
                syn[k] = BIG - rank
            sort_scores = [(slot, syn[slot]) for slot, _ in scores]
            hist_with_scores = [(hm, syn[hist_offset + i])
                                for i, (hm, _s, _n, _t) in enumerate(hist_cands)]

        new_elites, new_wavgs = _ht_select_and_mutate(
            prefix, model_dir, MT1Tail, sort_scores, tail_sigmas[comp], hist_models=hist_with_scores)
        _ht_save_hist(prefix, model_dir, MT1Tail, new_elites, new_wavgs)
        del new_elites, new_wavgs, hist_cands, hist_models

    # ── Head phase (every MT1_BLOCK_DAYS runs): freeze the just-updated tails, evolve head ──
    block_ctr = int(ind_rs.get('block_ctr', 0))
    do_head = n_win > 0 and (block_ctr % MT1_BLOCK_DAYS == 0)
    if do_head:
        tails_frozen = [load_slot_model(tp, model_dir, 0, MT1Tail) for tp in tail_prefixes]
        for t in tails_frozen:
            t.eval()

        def _score_head(cand):
            cand.eval(); total = 0.0
            with torch.inference_mode():
                for di, (feat_t, ad) in enumerate(dir_hist):
                    c = cand(feat_t)
                    out4 = torch.tensor([tails_frozen[k](c).reshape(-1)[0].item() for k in range(4)])
                    total += _mt1_score_breakdown(out4, ad, acc_floor, range_ceiling)[0] \
                        * mt1_win_weight(di, n_win)
            return total

        scores = []
        for slot in range(MT1_COMP_SLOTS):
            m = load_slot_model(head_prefix, model_dir, slot, MT1Head)
            scores.append((slot, _score_head(m)))
            del m
        hist_models = _ht_load_hist(head_prefix, model_dir, MT1Head)
        hist_with_scores = [(hm, _score_head(hm)) for hm in hist_models]
        new_elites, new_wavgs = _ht_select_and_mutate(
            head_prefix, model_dir, MT1Head, scores, head_sigma, hist_models=hist_with_scores)
        _ht_save_hist(head_prefix, model_dir, MT1Head, new_elites, new_wavgs)
        best_head_sc = max((sc for _, sc in scores), default=0.0)
        log(f"[mt1/{sn(industry)}:head] block cycle (ctr={block_ctr}) — best={best_head_sc:.4f}")
        del new_elites, new_wavgs, hist_models, tails_frozen

    ind_rs['block_ctr'] = block_ctr + 1

    # ── Compose production best from the new head0 + new tail0 ──
    best = MT1NN()
    nh = load_slot_model(head_prefix, model_dir, 0, MT1Head)
    best.head.load_state_dict(nh.state_dict())
    for c, tp in enumerate(tail_prefixes):
        nt = load_slot_model(tp, model_dir, 0, MT1Tail)
        best.tails[c].load_state_dict(nt.state_dict())
    best.eval()
    try:
        torch.save(best.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save mt1_{industry}_best.pt: {e}")

    # Composed slot0 activations + windowed composite score + rolling update.
    best_score = 0.0
    with torch.inference_mode():
        out4 = best(in37_t).squeeze(0)
        for di, (feat_t, ad) in enumerate(dir_hist):
            o4d = best(feat_t).squeeze(0)
            best_score += _mt1_score_breakdown(o4d, ad, acc_floor, range_ceiling)[0] \
                * mt1_win_weight(di, n_win)
    slot0_conf      = torch.sigmoid(out4[0]).item()
    slot0_delta_t   = torch.tanh(out4[1]).item()
    slot0_range_pct = F.softplus(out4[2]).item()
    slot0_conf4     = torch.sigmoid(out4[3]).item()
    comp0_residual  = abs(actual_d - slot0_delta_t * MT1_SCALE_DOLLARS)
    _rolling_update(ind_rs, abs(actual_d), comp0_residual)

    log(f"[mt1/{sn(industry)}] best={best_score:.4f} actual_d=${actual_d:+.1f} "
        f"acc_floor=${acc_floor:.1f} head_cycle={'Y' if do_head else 'n'}")

    gc.collect()
    return (best_score, best_score, slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4,
            slot0_conf, slot0_delta_t, slot0_range_pct, slot0_conf4)


# ── MT2 upkeep ─────────────────────────────────────────────────────────────────

def upkeep_mt2(model_dir, mt1_slot0_outputs, actual_perf, industry_list,
               sigma=UPKEEP_MT2_SIGMA):
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
        # Signed-magnitude reassembly: sign from direction confidence, |size| from the prediction
        # (delta_t's own sign is ungraded under signless magnitude scoring).
        delta_signed = abs(delta_t) if conf >= 0.5 else -abs(delta_t)
        in48.extend([conf, delta_signed, range_pct, conf4])
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

    # Load and score MT2 history candidates (no pool_floor filter)
    mt2_hist_meta_path = os.path.join(model_dir, 'mt2_hist_meta.json')
    mt2_hist_models = []   # list of (MT2NN, score)
    try:
        with open(mt2_hist_meta_path) as _f:
            mt2_hist_meta = json.load(_f)
        h_head  = mt2_hist_meta.get('head', 0)
        h_count = mt2_hist_meta.get('count', 0)
        n_hist_days = min(h_count, HIST_DAYS)
        for d in range(n_hist_days):
            day_slot = (h_head - n_hist_days + d) % HIST_DAYS
            for pos in range(HIST_PER_DAY):
                hp = os.path.join(model_dir, f'mt2_hist_{day_slot}_{pos}.pt')
                if os.path.exists(hp):
                    try:
                        hm = MT2NN()
                        hm.load_state_dict(torch.load(hp, weights_only=True))
                        hm.eval()
                        with torch.inference_mode():
                            h_out = hm(in48_t)
                        h_tier_preds = h_out.view(12, 4).argmax(dim=1).tolist()
                        h_tier_map   = {ind: h_tier_preds[i] for i, ind in enumerate(industry_list)}
                        h_pts        = sum(_master_points(h_tier_map[ind], opt_tiers[ind])
                                           for ind in industry_list)
                        mt2_hist_models.append((hm, h_pts))
                    except Exception:
                        pass
    except Exception:
        pass

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
        if mt2_hist_models:
            # History-aware selection: merge history candidates (sentinel idx >= N_SLOTS)
            # unconditionally (no pool_floor), then run standard elite copy + mutation.
            score_vals     = [s for _, s in pred_scores]
            mean_s         = sum(score_vals) / len(score_vals)
            std_s          = (sum((v - mean_s) ** 2 for v in score_vals) / len(score_vals)) ** 0.5
            survival_floor = mean_s - std_s

            surviving = [(s, v) for s, v in pred_scores if v >= survival_floor]
            if not surviving:
                surviving = list(pred_scores)

            # Add history candidates unconditionally (no floor filter)
            hist_sentinel_base = N_SLOTS
            for h_idx, (_, h_pts) in enumerate(mt2_hist_models):
                surviving.append((hist_sentinel_base + h_idx, h_pts * 1e9 + hist_sentinel_base + h_idx))

            surviving.sort(key=lambda x: x[1], reverse=True)
            top_elite = surviving[:ELITE_COUNT]
            while len(top_elite) < ELITE_COUNT:
                top_elite.append(top_elite[0])

            # Build hist model cache indexed by sentinel slot
            hist_cache = {hist_sentinel_base + i: hm for i, (hm, _) in enumerate(mt2_hist_models)}

            for rank, (slot, _) in enumerate(top_elite):
                if slot < N_SLOTS:
                    m = load_slot_model(prefix, model_dir, slot, MT2NN)
                    save_slot_model(prefix, model_dir, rank, m)
                    del m
                else:
                    save_slot_model(prefix, model_dir, rank, hist_cache[slot])

            # Wavg blends (top-5, top-10, top-15) from newly-ranked elites on disk
            for b, k in enumerate([5, 10, 15]):
                k = min(k, ELITE_COUNT)
                inv_k  = 1.0 / k
                avg_st = None
                for rank in range(k):
                    wm     = load_slot_model(prefix, model_dir, rank, MT2NN)
                    state  = wm.state_dict()
                    del wm
                    if avg_st is None:
                        avg_st = {key: (v.clone().float() * inv_k if torch.is_floating_point(v) else v.clone())
                                  for key, v in state.items()}
                    else:
                        for key, v in state.items():
                            if torch.is_floating_point(v) and key in avg_st:
                                avg_st[key] = avg_st[key] + v.float() * inv_k
                wm_blend = MT2NN(); wm_blend.load_state_dict(avg_st)
                save_slot_model(prefix, model_dir, ELITE_COUNT + b, wm_blend)
                del wm_blend, avg_st

            del hist_cache

            # Mutations: same pattern as _select_and_mutate
            n_mut    = N_SLOTS - ELITE_POOL
            muts_per = max(1, n_mut // ELITE_POOL)
            child_map_mt2 = defaultdict(list)
            for i, slot in enumerate(range(ELITE_POOL, N_SLOTS)):
                child_map_mt2[i // muts_per].append(slot)
            for parent_rank, child_slots in child_map_mt2.items():
                parent = load_slot_model(prefix, model_dir, parent_rank, MT2NN)
                for child_slot in child_slots:
                    child = _mutate_generic(parent, MT2NN, sigma)
                    save_slot_model(prefix, model_dir, child_slot, child)
                    del child
                del parent
        else:
            _select_and_mutate(prefix, model_dir, MT2NN, pred_scores, sigma)

        # Save top HIST_ELITE elites + HIST_WAVG wavg models to MT2 history
        try:
            with open(mt2_hist_meta_path) as _f:
                mt2_hist_meta_out = json.load(_f)
            h_head_out  = mt2_hist_meta_out.get('head', 0)
            h_count_out = mt2_hist_meta_out.get('count', 0)
        except Exception:
            h_head_out, h_count_out = 0, 0
        for k in range(HIST_ELITE):
            hp = os.path.join(model_dir, f'mt2_hist_{h_head_out}_{k}.pt')
            try:
                hm_save = load_slot_model(prefix, model_dir, k, MT2NN)
                torch.save(hm_save.state_dict(), hp)
                del hm_save
            except Exception:
                pass
        for k in range(HIST_WAVG):
            hp = os.path.join(model_dir, f'mt2_hist_{h_head_out}_{HIST_ELITE + k}.pt')
            try:
                wm_save = load_slot_model(prefix, model_dir, ELITE_COUNT + k, MT2NN)
                torch.save(wm_save.state_dict(), hp)
                del wm_save
            except Exception:
                pass
        h_head_out  = (h_head_out + 1) % HIST_DAYS
        h_count_out = min(h_count_out + 1, HIST_DAYS)
        try:
            with open(mt2_hist_meta_path, 'w') as _f:
                json.dump({'head': h_head_out, 'count': h_count_out}, _f)
        except Exception as e:
            log(f"WARNING: could not save mt2_hist_meta.json: {e}")
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

    del mt2_hist_models

    slot0_final = load_slot_model(prefix, model_dir, 0, MT2NN)
    try:
        torch.save(slot0_final.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save mt2_best.pt: {e}")
    del slot0_final

    return best_pts, slot0_pts, injected


# ── Production inference (MT1 → MT2) ──────────────────────────────────────────

def run_mt_inference(model_dir, industries, mkt_val_history, zero_counts, total_cash):
    """
    MT1→MT2 inference chain for daily capital allocation in production.

    mkt_val_history: {ind: [cumulative_market_index]} — features for MT1/MT2 input.
    Loads mt1_{ind}_best.pt for each industry and mt2_best.pt, runs the full
    chain, and returns (allocations, tier_map, mt1_outputs).

    Caller should fall back to MasterNN/equal allocation if mt2_best.pt is absent.

    allocations:  {ind: dollar_amount}
    tier_map:     {ind: 0-3}
    mt1_outputs:  {ind: (conf, delta_t, range_pct, conf4)} — raw activations (no normalization)
    """
    from training_lib import build_master_features, tiers_to_alloc

    industry_list = list(industries.keys())
    today444 = build_master_features(mkt_val_history, industry_list)

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

    # Build in48 for MT2 from the composed production MT1 (head0+tail0). In the heads/tails design
    # there is a single production model, so the composite and direction feeds are identical and the
    # MT2_FEED_DIRECTION toggle is moot (Increment 4C).
    in48 = []
    for ind in industry_list:
        in48.extend(list(mt1_outputs[ind]))
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
