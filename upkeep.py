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
import pickle
import shutil
from collections import defaultdict

import torch

from models import MT1NN, MT2NN, MT1DualHead, MT1Tail, StockNN
from training_lib import (
    ELITE_COUNT,
    ELITE_POOL,
    HIST_DAYS,
    HIST_ELITE,
    HIST_PER_DAY,
    HIST_WAVG,
    MT1_COMP_SLOTS,
    MT1_DIR_CONST_TRIP,
    MT1_DIR_DAYS,
    MT1_DIR_INJ_BLEND,
    MT1_DIR_INJ_COOLDOWN,
    MT1_DIR_MIN_CORRECT,
    MT1_LOGIT_CAP,
    MT1_POOL_NAMES,
    MT1_SOFTPLUS_CLAMP,
    N_SLOTS,
    WAVG_COUNT,
    _master_points,
    _model_path,
    _optimal_tiers,
    blend_model_halfway,
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

# ── Heads/tails pool layout (Increment 4B) ──────────────────────────────────────
# One shared head pool + 4 specialized tail pools per industry. No injection slots — the shared
# head propagates cross-component learning. HT_PARENTS = 17 direct elites + 3 wavg (= 20); the
# reclaimed capacity goes into more elite mutations. Children table mirrors C++ step_mt1_pool.
MT1_BLOCK_DAYS = 25   # head cycle fires every MT1_BLOCK_DAYS upkeep runs; tails evolve every run
HT_PARENTS     = ELITE_COUNT + WAVG_COUNT   # 20
_HT_CHILDREN   = [16, 13, 13, 12, 12] + [8] * 12 + [6] * 3   # sums to 180
HT_PARENT_OF_MUT = [p for p, c in enumerate(_HT_CHILDREN) for _ in range(c)]
assert len(HT_PARENT_OF_MUT) == MT1_COMP_SLOTS - HT_PARENTS, "HT children table must sum to 180"

# ── Direction pool: forward accumulation (v0.5.0.0) ─────────────────────────────
# The acc/rng/cfd pools score a model by REPLAYING it over the trailing window, which regenerates
# 183 of 200 slots every run and gives ~1.6 independent observations at MT1_FWD_DAYS = 10. The
# v0.4.3.0 out-of-sample instrument measured direction at 49.54% OOS against 61.09% in-sample —
# all fit, no skill. Direction now keeps MT1_COMP_SLOTS persistent individuals, each carrying its
# own rolling 16-prediction record, culled gently on a percentile floor.
# These MUST match mt1_scoring.h; the C++ trainer and this path evolve the same pool files.
# Volatility gets its own scale, not MT1_SCALE_DOLLARS — the target averages $242, so at the
# $10,000 delta scale a model would need softplus in 0.005-0.06, deep in its flat tail where
# mutations barely move the output. Must match mt1_scoring.h.
MT1_VOL_SCALE       = 500.0
MT1_UNGRADED_FEED   = 0.5    # what the ungraded conf4 channel forwards to MT2
MT1_DIR_HIST_BITS   = 16
MT1_DIR_MIN_AGE     = 8       # predictions before a model may be culled OR breed
MT1_DIR_CULL_PCT    = 0.083   # fraction of MATURE models culled per run → ~60% mature, ~20-run life
MT1_DIR_ELITE_PCT   = 0.10    # top fraction of mature models used as parents
MT1_DIR_LINEAGE_CAP = 0.125   # lineage above this share of the pool stops breeding
MT1_DIR_LINEAGE_RESUME = 0.10  # ...and resumes only below this (hysteresis; see _upkeep_dir_pool)
MT1_DIR_RECENCY_W   = (1.0, 0.8, 0.6, 0.4)   # most-recent-first, blocks of 4; full record sums 11.2
MT1_DIR_WAVG_K      = (5, 10, 15)            # ephemeral blend parents (top-5/10/15)



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


# ── MT1 rolling floor ─────────────────────────────────────────────────────────

def load_mt1_rolling_state(model_dir):
    """Load per-industry rolling |actual_d| buffers. Returns empty state on missing/error."""
    path = os.path.join(model_dir, 'mt1_rolling_state.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log(f"WARNING: rolling state {path} unreadable, starting cold: {e}")
    return {}


def save_mt1_rolling_state(model_dir, state):
    """Persist per-industry rolling |actual_d| buffers."""
    try:
        with open(os.path.join(model_dir, 'mt1_rolling_state.json'), 'w') as f:
            json.dump(state, f)
    except OSError as e:
        log(f"WARNING: could not save mt1_rolling_state: {e}")


def _rolling_acc_floor(ind_state):
    """Acc floor = mean(last 10 |actual_d|) / 2. Returns MT1_FLOOR_COLD/2 on cold start."""
    buf = ind_state.get('actual_buf', [])
    if not buf:
        return MT1_FLOOR_COLD / 2.0
    return sum(buf) / len(buf) / 2.0


def _rolling_vol_floor(ind_state):
    """Vol floor = mean(last 10 vol targets) / 2, the same shape as the acc floor. 2x this is the
    naive "predict the rolling mean" answer that sec_vol normalizes against. Returns
    MT1_FLOOR_COLD/2 on cold start. Mirror of C++ run_mt1_block::floors()."""
    buf = ind_state.get('vol_buf', [])
    if not buf:
        return MT1_FLOOR_COLD / 2.0
    return sum(buf) / len(buf) / 2.0


def _rolling_update(ind_state, abs_actual_d, vol_actual):
    """Update both rolling circular buffers (max MT1_ROLLING_DAYS entries each).

    The second buffer held |actual_d - comp0_delta| for the retired band ceiling; it now holds the
    realized-vol target itself, since channel 2 predicts volatility as of v0.5.0.0.
    """
    for key, val in (('actual_buf', abs_actual_d), ('vol_buf', vol_actual)):
        buf = ind_state.get(key, [])
        buf.append(val)
        if len(buf) > MT1_ROLLING_DAYS:
            buf = buf[-MT1_ROLLING_DAYS:]
        ind_state[key] = buf


# ── MT1 output decode (single source of truth) ─────────────────────────────────
# Mirror of the C++ mt1_conf / mt1_conf4 / mt1_range_pct / mt1_delta_t helpers. Every consumer of an
# MT1 raw logit must go through these so the bounded activation cannot be bypassed at one site.
#
# conf/conf4 squash the logit through tanh before the sigmoid so the output can never reach the 0/1
# rails, where the derivative vanishes and Gaussian weight mutations stop moving the output at all.

def _mt1_conf(raw):
    return 1.0 / (1.0 + math.exp(-MT1_LOGIT_CAP * math.tanh(raw / MT1_LOGIT_CAP)))


def _mt1_conf4(raw):
    return _mt1_conf(raw)


def _mt1_range_pct(raw):
    c = max(-MT1_SOFTPLUS_CLAMP, min(MT1_SOFTPLUS_CLAMP, raw))
    return math.log1p(math.exp(c))          # softplus, guarded against overflow → inf


def _mt1_delta_t(raw):
    return math.tanh(raw)                   # already bounded; unchanged


# ── MT1 scoring ────────────────────────────────────────────────────────────────

def mt1_win_weight(di, day_count):
    """Linear scoring-window weight (mirror of C++ mt1_win_weight): today (di=day_count-1) = 2.0,
    a day MT1_DIR_DAYS-1 steps back = 1.0. Anchored to MT1_DIR_DAYS (absolute age)."""
    if MT1_DIR_DAYS <= 1:
        return 2.0
    age = day_count - 1 - di
    w = 2.0 - age / (MT1_DIR_DAYS - 1)
    return 1.0 if w < 1.0 else w


def _mt1_score_breakdown(out4, actual_d, acc_floor, vol_actual=0.0, vol_floor=None):
    """
    Score one MT1 slot against the industry's actual dollar P&L and realized volatility.

    out4:       raw logit tensor shape (4,)
    actual_d:   float — actual dollar P&L (actual_frac x portfolio_value)
    acc_floor:  float — per-industry adaptive floor for the accuracy denom
    vol_actual: float — realized vol over the next MT1_VOL_DAYS sessions, x MT1_SCALE_DOLLARS
    vol_floor:  float — half the rolling mean of vol_actual (None -> cold start)
    Returns (composite, direction, vol, accuracy, confidence), all in [0.0, 1.0].

    Composite = equal-weight mean of the THREE graded components' secondary [0,1] normalizations
    (mirror of C++ compute_mt1_scores). conf4 is present but ungraded and always returns 0.0.
    """
    if vol_floor is None:
        vol_floor = MT1_FLOOR_COLD / 2.0
    conf     = _mt1_conf(out4[0].item())
    delta_d  = _mt1_delta_t(out4[1].item()) * MT1_SCALE_DOLLARS
    vol_pred = _mt1_range_pct(out4[2].item()) * MT1_VOL_SCALE       # channel 2 is VOLATILITY now

    score_dir = conf if actual_d >= 0.0 else (1.0 - conf)

    # Magnitude: signless, so the delta head owns size and the direction head owns sign.
    err       = abs(abs(actual_d) - abs(delta_d))
    denom     = max(abs(actual_d), acc_floor)
    score_acc = denom / (err + denom)

    # Volatility. Channel 2 used to be a band width graded against the delta head's own residual —
    # against the part of the target it had just failed to predict, i.e. noise — so the pool
    # converged to a constant. Forward realized vol IS predictable (trailing vol reaches OOS
    # r = 0.444; the same pool machinery on a full window reaches 0.527).
    err_vol   = abs(vol_pred - vol_actual)
    # Guarded: a degenerate all-zero vol history leaves vol_actual and vol_floor both 0, and an
    # unguarded den_vol raises ZeroDivisionError (C++ would silently produce NaN instead).
    den_vol   = max(vol_actual, vol_floor, 1e-6)
    score_vol = den_vol / (err_vol + den_vol)

    # conf4 no longer scored: it graded against the band geometry that went away, and with
    # err > r always its optimum was the degenerate conf4 = 0.
    score_conf = 0.0

    # Secondary [0,1] normalization per component vs naive baseline B (ideal=1):
    #   sec = clamp((raw - B) / (1 - B), 0, 1)  -> naive->0, ideal->1.  (mirror of C++)
    def _c01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    b_dir   = 0.5                                          # conf = 0.5, no view
    sec_dir = _c01((score_dir - b_dir) / max(1.0 - b_dir, 1e-6))

    b_acc   = denom / (abs(actual_d) + denom)               # predict-0 null model
    sec_acc = _c01((score_acc - b_acc) / max(1.0 - b_acc, 1e-6))

    # Naive vol predictor = the rolling mean, which is 2 x vol_floor by construction.
    b_vol   = den_vol / (abs(2.0 * vol_floor - vol_actual) + den_vol)
    sec_vol = _c01((score_vol - b_vol) / max(1.0 - b_vol, 1e-6))

    composite = (sec_dir + sec_acc + sec_vol) / 3.0
    return composite, score_dir, score_vol, score_acc, score_conf

def _mt1_decode(model, in74_t):
    """Run MT1 inference and return raw activations for MT2 input and logging.

    Returns (conf, delta_t, range_pct, conf4) — all via the bounded decode helpers:
      conf      = _mt1_conf(out[0])      ∈ (0.018, 0.982) — direction confidence, never saturating
      delta_t   = tanh(out[1])           ∈ [-1,1] — bounded P&L (× MT1_SCALE_DOLLARS for dollars)
      range_pct = softplus(clamp(out[2])) > 0     — range as % of effective delta, finite
      conf4     = _mt1_conf4(out[3])     ∈ (0.018, 0.982) — calibrated confidence
    """
    model.eval()
    with torch.inference_mode():
        out4 = model(in74_t).squeeze(0)
    return (_mt1_conf(out4[0].item()), _mt1_delta_t(out4[1].item()),
            _mt1_range_pct(out4[2].item()), _mt1_conf4(out4[3].item()))


# ── MT1 burst refinement ───────────────────────────────────────────────────────

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
                    deltas = [r - p for r, p in zip(raw, prev, strict=True)] if prev else [0.0] * 5
                    hist.append(raw + deltas)
                histories[sym] = hist
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
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
                    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
                        # A dropped history model silently shrinks the candidate pool, which is
                        # indistinguishable from genuine convergence — never swallow this.
                        log(f"WARNING: could not load history model {hp}: {e}")
    except (OSError, json.JSONDecodeError) as e:
        log(f"WARNING: could not read history meta {meta_path}: {e}")
    return models


def _ht_save_hist(prefix, model_dir, model_class, new_elites, new_wavgs):
    """Save top HIST_ELITE elites + HIST_WAVG wavg models to a head/tail pool's history."""
    meta_path = os.path.join(model_dir, f'{prefix}_hist_meta.json')
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        head, count = meta.get('head', 0), meta.get('count', 0)
    except (OSError, json.JSONDecodeError) as e:
        log(f"WARNING: history meta {meta_path} unreadable, restarting ring at 0: {e}")
        head, count = 0, 0
    for k, m in enumerate(new_elites[:HIST_ELITE]):
        hp = os.path.join(model_dir, f'{prefix}_hist_{head}_{k}.pt')
        try:
            torch.save(m.state_dict(), hp)
        except (OSError, RuntimeError) as e:
            log(f"WARNING: could not save history elite {hp}: {e}")
    for k, m in enumerate(new_wavgs[:HIST_WAVG]):
        hp = os.path.join(model_dir, f'{prefix}_hist_{head}_{HIST_ELITE + k}.pt')
        try:
            torch.save(m.state_dict(), hp)
        except (OSError, RuntimeError) as e:
            log(f"WARNING: could not save history wavg {hp}: {e}")
    try:
        with open(meta_path, 'w') as f:
            json.dump({'head': (head + 1) % HIST_DAYS, 'count': min(count + 1, HIST_DAYS)}, f)
    except OSError as e:
        log(f"WARNING: could not write history meta {meta_path}: {e}")


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

def _mt1_dir_record_scores(hist, n_pred):
    """(primary, secondary, tertiary) for one rolling record. Exact mirror of
    mt1_dir_record_scores in mt1_scoring.h — the C++ trainer and this path evolve the SAME pool
    files, so any divergence here silently re-ranks models between a batch run and daily upkeep.

    n = min(n_pred, 16) occupied slots. A partial record normalizes by the weight actually
    occupied, so an 8-prediction model is judged on its 8 rather than diluted toward zero.
    """
    n = min(int(n_pred), MT1_DIR_HIST_BITS)
    if n <= 0:
        return 0.0, 0.0, 0.0
    correct = wsum = wcorrect = 0.0
    for i in range(n):
        w = MT1_DIR_RECENCY_W[i // 4]
        if (hist >> i) & 1:
            correct += 1.0
            wcorrect += w
        wsum += w
    primary = correct / n
    secondary = (wcorrect / wsum) if wsum > 0 else 0.0
    # Tertiary is a function of the other two, so as a THIRD sort key it can never break a tie they
    # left. Computed because the blend distribution is worth logging.
    return primary, secondary, 0.4 * primary + 0.6 * secondary


def _dir_sort_key(meta):
    """Descending (primary, secondary, tertiary) — use with reverse=True."""
    return _mt1_dir_record_scores(meta[0], meta[1])


def _load_dir_meta(model_dir, industry):
    """Per-slot direction metadata: [hist, n_pred, lineage] per slot. Missing file → every slot is
    a fresh individual with its own lineage, which is exactly the cold-start state (and how a pool
    converted from .bin by convert_weights.py arrives)."""
    path = os.path.join(model_dir, f'mt1_{industry}_tail_dir_meta.json')
    slots, next_lineage, best_slot = None, MT1_COMP_SLOTS, 0
    if os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            s = d.get('slots')
            if isinstance(s, list) and len(s) == MT1_COMP_SLOTS:
                slots = [[int(x[0]), int(x[1]), int(x[2])] for x in s]
                next_lineage = int(d.get('next_lineage', MT1_COMP_SLOTS))
                best_slot = int(d.get('best_slot', 0)) % MT1_COMP_SLOTS
        except Exception:
            slots = None
    if slots is None:
        slots = [[0, 0, i] for i in range(MT1_COMP_SLOTS)]
    return slots, next_lineage, best_slot


def _save_dir_meta(model_dir, industry, slots, next_lineage, best_slot):
    path = os.path.join(model_dir, f'mt1_{industry}_tail_dir_meta.json')
    with open(path, 'w') as f:
        json.dump({'slots': slots, 'next_lineage': int(next_lineage),
                   'best_slot': int(best_slot)}, f)


def _upkeep_dir_pool(prefix, model_dir, industry, concat_today, actual_d, sigma, ind_rs):
    """One forward-accumulation step for the direction pool. Mirrors step_mt1_dir_pool in
    training_v4.cpp: every slot predicts today, the outcome shifts into its record, the worst
    MT1_DIR_CULL_PCT of MATURE models are culled, and the freed slots are refilled round-robin
    from the top MT1_DIR_ELITE_PCT plus three ephemeral wavg blends.

    Returns (best_slot, stats dict).
    """
    slots, next_lineage, _prev_best = _load_dir_meta(model_dir, industry)
    target_up = bool(actual_d >= 0.0)

    # 1. Predict + record. This is the only place a model's evidence grows: there is no replay, so
    #    a model's score IS its own track record.
    for s in range(MT1_COMP_SLOTS):
        m = load_slot_model(prefix, model_dir, s, MT1Tail)
        with torch.no_grad():
            logit = m(concat_today).reshape(-1)[0].item()
        del m
        correct = (_mt1_conf(logit) >= 0.5) == target_up
        slots[s][0] = ((slots[s][0] << 1) | (1 if correct else 0)) & 0xFFFF
        if slots[s][1] < 0xFFFF:
            slots[s][1] += 1

    mature = [s for s in range(MT1_COMP_SLOTS) if slots[s][1] >= MT1_DIR_MIN_AGE]
    mature.sort(key=lambda s: _dir_sort_key(slots[s]), reverse=True)
    best_slot = mature[0] if mature else 0

    scores = [_mt1_dir_record_scores(slots[s][0], slots[s][1]) for s in range(MT1_COMP_SLOTS)]
    stats = {
        'mature': len(mature) / MT1_COMP_SLOTS,
        'mean_primary': sum(x[0] for x in scores) / MT1_COMP_SLOTS,
        'mean_secondary': sum(x[1] for x in scores) / MT1_COMP_SLOTS,
        'culled': 0,
    }
    if len(mature) < 4:
        _save_dir_meta(model_dir, industry, slots, next_lineage, best_slot)
        return best_slot, stats

    # 2. Lineage shares and the breeding bar.
    counts = {}
    for s in range(MT1_COMP_SLOTS):
        counts[slots[s][2]] = counts.get(slots[s][2], 0) + 1
    barred = set(ind_rs.get('dir_barred', []))
    for lin, c in counts.items():
        share = c / MT1_COMP_SLOTS
        # Hysteresis: bar above CAP, release only below RESUME. Equal thresholds make a lineage
        # hovering at the cap flip state constantly, and a bar that lasts one step suppresses
        # nothing (the first C++ run logged 1850 barrings / 1838 re-enables over 35 days).
        if lin not in barred and share > MT1_DIR_LINEAGE_CAP:
            barred.add(lin)
            log(f"[mt1/{sn(industry)}:dir] lineage {lin} BARRED from breeding "
                f"(share {100*share:.0f}%)")
        elif lin in barred and share <= MT1_DIR_LINEAGE_RESUME:
            barred.discard(lin)
            log(f"[mt1/{sn(industry)}:dir] lineage {lin} re-enabled (share {100*share:.0f}%)")
    barred &= set(counts)          # forget lineages that no longer exist
    ind_rs['dir_barred'] = sorted(barred)
    stats['lineage_max'] = max(counts.values()) / MT1_COMP_SLOTS
    stats['lineage_n'] = len(counts)

    # 3. Parents: top MT1_DIR_ELITE_PCT of mature with an unbarred lineage. The three wavg blends
    #    are breeding TEMPLATES rather than pool residents — a synthetic average has no track
    #    record, so it could never mature, and giving it a slot would park an unscoreable model
    #    in the pool.
    want = max(1, int(len(mature) * MT1_DIR_ELITE_PCT + 0.5))
    parents = [s for s in mature if slots[s][2] not in barred][:want]
    if not parents:
        parents = [mature[0]]                       # every lineage barred: breed anyway
    n_par = len(parents) + len(MT1_DIR_WAVG_K)

    n_cull = max(1, int(len(mature) * MT1_DIR_CULL_PCT + 0.5))
    n_cull = max(0, min(n_cull, len(mature) - len(parents)))
    stats['culled'] = n_cull

    for k in range(n_cull):
        dead = mature[len(mature) - 1 - k]
        pi = k % n_par
        if pi < len(parents):
            parent = load_slot_model(prefix, model_dir, parents[pi], MT1Tail)
            child = _mutate_generic(parent, MT1Tail, sigma)
            del parent
            child_lineage = slots[parents[pi]][2]
        else:
            kk = min(MT1_DIR_WAVG_K[pi - len(parents)], len(mature))
            inv = 1.0 / kk
            avg = None
            for t in range(kk):
                m = load_slot_model(prefix, model_dir, mature[t], MT1Tail)
                st = m.state_dict(); del m
                if avg is None:
                    avg = {key: (v.clone().float() * inv if torch.is_floating_point(v) else v.clone())
                           for key, v in st.items()}
                else:
                    for key, v in st.items():
                        if torch.is_floating_point(v) and key in avg:
                            avg[key] = avg[key] + v.float() * inv
            blend = MT1Tail(); blend.load_state_dict(avg)
            child = _mutate_generic(blend, MT1Tail, sigma)
            del blend, avg
            child_lineage = next_lineage           # a blend is a genuinely new genotype
            next_lineage += 1
        save_slot_model(prefix, model_dir, dead, child)
        del child
        slots[dead] = [0, 0, child_lineage]        # fresh individual: no record, immune until MIN_AGE

    _save_dir_meta(model_dir, industry, slots, next_lineage, best_slot)
    return best_slot, stats


def _inject_dir_tail(prefix, model_dir, industry, best_dir_tail):
    """Re-diversify the direction pool after a constant-collapse. Under forward accumulation this
    targets the WORST mature individuals rather than elite ranks — with persistent identity there
    are no rank slots to overwrite, and replacing the bottom of the pool is what "re-diversify
    without discarding the good models" actually means. Each replacement becomes a fresh individual
    (no record, own lineage, immune until MT1_DIR_MIN_AGE), exactly like a cull birth."""
    slots, next_lineage, _best = _load_dir_meta(model_dir, industry)
    mature = [s for s in range(MT1_COMP_SLOTS) if slots[s][1] >= MT1_DIR_MIN_AGE]
    if not mature:
        return 0
    mature.sort(key=lambda s: _dir_sort_key(slots[s]), reverse=True)
    best_state = best_dir_tail.state_dict()
    n_inj = max(1, len(mature) // 4)               # worst quartile of mature
    for k in range(n_inj):
        dead = mature[len(mature) - 1 - k]
        rand_state = MT1Tail().state_dict()        # PyTorch default (kaiming-uniform) random tail
        blended = {key: MT1_DIR_INJ_BLEND * best_state[key]
                        + (1.0 - MT1_DIR_INJ_BLEND) * rand_state[key]
                   for key in best_state}
        m = MT1Tail(); m.load_state_dict(blended)
        save_slot_model(prefix, model_dir, dead, m)
        del m
        slots[dead] = [0, 0, next_lineage]
        next_lineage += 1
    _save_dir_meta(model_dir, industry, slots, next_lineage, _best)
    return n_inj


def upkeep_mt1_industry(industry, model_dir, in74_t, actual_d, vol_actual=0.0,
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

    File naming: mt1_{ind}_head_model_{slot}.pt (MT1DualHead),
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
    in74_t    = in74_t.detach()

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
        hb = MT1DualHead(); hb.load_state_dict(base.head.state_dict())
        save_slot_model(head_prefix, model_dir, 0, hb)
        for slot in range(1, MT1_COMP_SLOTS):
            save_slot_model(head_prefix, model_dir, slot, _mutate_generic(hb, MT1DualHead, head_sigma))
        for c, tp in enumerate(tail_prefixes):
            tb = MT1Tail(); tb.load_state_dict(base.tails[c].state_dict())
            save_slot_model(tp, model_dir, 0, tb)
            for slot in range(1, MT1_COMP_SLOTS):
                save_slot_model(tp, model_dir, slot, _mutate_generic(tb, MT1Tail, tail_sigmas[c]))
        del base

    # Frozen production bests at phase start (head + 4 tails).
    head0  = load_slot_model(head_prefix, model_dir, 0, MT1DualHead); head0.eval()
    tails0 = [load_slot_model(tp, model_dir, 0, MT1Tail) for tp in tail_prefixes]
    for t in tails0:
        t.eval()

    # Vol floor from the trailing vol-target buffer (strictly backward-looking).
    vol_floor = _rolling_vol_floor(ind_rs)

    # Direction day buffer (append today, persist, load the trailing window).
    dir_hist_path = os.path.join(model_dir, f'mt1_{industry}_dir_hist.json')
    dir_hist_raw = []
    if os.path.exists(dir_hist_path):
        try:
            with open(dir_hist_path) as _f:
                dir_hist_raw = json.load(_f)
        except (OSError, json.JSONDecodeError) as e:
            # Losing the window silently resets MT1 scoring to a 1-day window without warning.
            log(f"WARNING: direction history {dir_hist_path} unreadable, restarting window: {e}")
            dir_hist_raw = []
    dir_hist_raw.append({'feat74': in74_t.squeeze(0).tolist(), 'actual_d': float(actual_d),
                         'vol_d': float(vol_actual)})
    dir_hist_raw = dir_hist_raw[-MT1_DIR_DAYS:]
    with open(dir_hist_path, 'w') as _f:
        json.dump(dir_hist_raw, _f)
    # vol_d defaults to 0.0 for windows written before v0.5.0.0, so an in-flight state file loads
    # rather than crashing; the entry ages out within MT1_DIR_DAYS runs.
    dir_hist = [(torch.tensor(e['feat74'], dtype=torch.float32).unsqueeze(0), e['actual_d'],
                 e.get('vol_d', 0.0))
                for e in dir_hist_raw]
    n_win = len(dir_hist)

    # Precompute frozen-head concat + frozen tail logits per window day (shared across tail pools).
    concats, frozen_logits = [], []
    with torch.inference_mode():
        for feat_t, _ad, _vd in dir_hist:
            c = head0(feat_t)
            concats.append(c)
            frozen_logits.append([tails0[k](c).reshape(-1)[0].item() for k in range(4)])

    def _score_tail(cand, comp, apply_cull=True):
        """Score one candidate tail over the trailing window (mirror of C++ step_mt1_tail).

        No culls. Both were removed in v0.4.1.0:
          * range-ceiling cull (comps 2/3) — score_rng is now single-peaked and self-limiting on the
            wide side, and the threshold was derived from today's target.
          * direction flip cull — it culled any model whose confidence never crossed 0.5, i.e. exactly
            the constant predictor that _dir_day_weights is DESIGNED to score at the no-skill baseline
            (dir_W/2 = 7.50). Removing that floor let the pool sit below it, and it did: ~35% balanced
            accuracy, systematically inverted, for all 5 passes of the v0.4.0.0 run.
        `apply_cull` is retained for signature compatibility with the history-candidate call sites.
        """
        cand.eval()
        total = 0.0
        n_correct = n_correct_dbl = today_correct = 0
        with torch.inference_mode():
            for di, (feat_t, ad, vd) in enumerate(dir_hist):
                is_today = (di == n_win - 1)
                o = list(frozen_logits[di]); o[comp] = cand(concats[di]).reshape(-1)[0].item()
                out4 = torch.tensor(o)
                if comp == 0:
                    conf = _mt1_conf(o[0])
                    cp = conf >= 0.5; ap = ad >= 0.0
                    day = conf if ap else (1.0 - conf); corr = cp == ap
                    n_correct += 1 if corr else 0
                    n_correct_dbl += (2 if corr else 0) if is_today else (1 if corr else 0)
                    if is_today:
                        today_correct = 1 if corr else 0
                elif comp == 1:
                    delta_d = _mt1_delta_t(o[1]) * MT1_SCALE_DOLLARS
                    err = abs(abs(ad) - abs(delta_d)); day = acc_floor / (err + acc_floor)
                else:
                    day = _mt1_score_breakdown(out4, ad, acc_floor, vd, vol_floor)[2 if comp == 2 else 4]
                total += day * mt1_win_weight(di, n_win)
        return total, n_correct, n_correct_dbl, today_correct

    # ── Direction pool (v0.5.0.0): forward accumulation, not replay ──
    # Runs before the replay pools so tail0[0] is the day's best when they freeze against it.
    dir_best_slot, dir_stats = 0, {}
    if n_win > 0:
        dir_best_slot, dir_stats = _upkeep_dir_pool(
            tail_prefixes[0], model_dir, industry, concats[-1], actual_d, dir_sigma, ind_rs)
        # Slot 0 is the deployed direction tail by convention everywhere downstream, but identity
        # is positional-free here: copy the ranked best into slot 0 rather than reordering the pool.
        if dir_best_slot != 0:
            best_m = load_slot_model(tail_prefixes[0], model_dir, dir_best_slot, MT1Tail)
            save_slot_model(tail_prefixes[0], model_dir, 0, best_m)
            del best_m

    # ── Tail phase (every run): freeze head0 + tail0, evolve the acc/rng/cfd replay pools ──
    for comp, prefix in enumerate(tail_prefixes):
        if comp == 0:
            continue          # direction handled above by _upkeep_dir_pool
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
            # Direction two-half selection.
            #
            # DELIBERATE DIVERGENCE FROM C++ (v0.4.1.0) — do NOT "re-sync" this away. The slot-0
            # "got today right" filter below (`if cand[k][2]`) was removed from the batch trainer
            # but is KEPT here. Rationale: getting the next call right is the real objective, and
            # the windowed score is only an estimator of it — so the live path keeps expressing it.
            # In the batch trainer `today` refers to the most recently MATURED day (MT1 lags the sim
            # by MT1_FWD_DAYS via the forward-buffer ring), i.e. a call made ten sessions ago that is
            # 9/10 overlapping with the rest of the window — a stale filter, so it was dropped there.
            # NOTE: production_v2.py:~207 buffers predictions the same MT1_FWD_DAYS, so today this
            # filter is equally lagged here and is effectively cosmetic; it is retained so the live
            # path is already correct if/when upkeep moves to a shorter horizon.
            # See the matching note in training_v4.cpp step_mt1_pool.
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
                for di, (feat_t, ad, vd) in enumerate(dir_hist):
                    c = cand(feat_t)
                    out4 = torch.tensor([tails_frozen[k](c).reshape(-1)[0].item() for k in range(4)])
                    total += _mt1_score_breakdown(out4, ad, acc_floor, vd, vol_floor)[0] \
                        * mt1_win_weight(di, n_win)
            return total

        scores = []
        for slot in range(MT1_COMP_SLOTS):
            m = load_slot_model(head_prefix, model_dir, slot, MT1DualHead)
            scores.append((slot, _score_head(m)))
            del m
        hist_models = _ht_load_hist(head_prefix, model_dir, MT1DualHead)
        hist_with_scores = [(hm, _score_head(hm)) for hm in hist_models]
        new_elites, new_wavgs = _ht_select_and_mutate(
            head_prefix, model_dir, MT1DualHead, scores, head_sigma, hist_models=hist_with_scores)
        _ht_save_hist(head_prefix, model_dir, MT1DualHead, new_elites, new_wavgs)
        best_head_sc = max((sc for _, sc in scores), default=0.0)
        log(f"[mt1/{sn(industry)}:head] block cycle (ctr={block_ctr}) — best={best_head_sc:.4f}")
        del new_elites, new_wavgs, hist_models, tails_frozen

    ind_rs['block_ctr'] = block_ctr + 1

    # ── Compose production best from the new head0 + new tail0 ──
    best = MT1NN()
    nh = load_slot_model(head_prefix, model_dir, 0, MT1DualHead)
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
        out4 = best(in74_t).squeeze(0)
        for di, (feat_t, ad, vd) in enumerate(dir_hist):
            o4d = best(feat_t).squeeze(0)
            best_score += _mt1_score_breakdown(o4d, ad, acc_floor, vd, vol_floor)[0] \
                * mt1_win_weight(di, n_win)
    slot0_conf      = _mt1_conf(out4[0].item())
    slot0_delta_t   = _mt1_delta_t(out4[1].item())
    slot0_range_pct = _mt1_range_pct(out4[2].item())
    slot0_conf4     = _mt1_conf4(out4[3].item())
    # Second buffer now feeds the vol floor, not the retired band ceiling.
    _rolling_update(ind_rs, abs(actual_d), abs(vol_actual))

    # ── Direction constant-collapse injection (mirror of C++ process_block detector) ──
    # Once the direction window is full, test the deployed model over it: constant = every window
    # day calls the same direction; imperfect = ≥1 window day wrong. MT1_DIR_CONST_TRIP consecutive
    # constant+imperfect checks re-diversifies the direction tail. Streak + cooldown persist in
    # ind_rs (mt1_rolling_state.json). Stochastic + daily-cadence, so NOT bit-parity with C++.
    if len(dir_hist) == MT1_DIR_DAYS:
        with torch.inference_mode():
            calls = [( _mt1_conf(best(feat_t).squeeze(0)[0].item()) >= 0.5, ad >= 0.0)
                     for feat_t, ad in dir_hist]
        constant  = all(up == calls[0][0] for up, _ in calls)
        imperfect = any(up != actual_up for up, actual_up in calls)
        cooldown  = max(0, ind_rs.get('dir_inj_cooldown', 0) - 1)
        streak    = ind_rs.get('dir_const_streak', 0) + 1 if (constant and imperfect) else 0
        if streak >= MT1_DIR_CONST_TRIP and cooldown == 0:
            best_dir_tail = load_slot_model(tail_prefixes[0], model_dir, 0, MT1Tail)
            n_inj = _inject_dir_tail(tail_prefixes[0], model_dir, industry, best_dir_tail)
            log(f"[mt1/{sn(industry)}:dir] constant-collapse — injected {n_inj} blended "
                f"individuals (½ best + ½ random)")
            del best_dir_tail
            streak, cooldown = 0, MT1_DIR_INJ_COOLDOWN
        ind_rs['dir_const_streak'] = streak
        ind_rs['dir_inj_cooldown'] = cooldown

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
        # Magnitude only. delta_t's own sign is ungraded (magnitude scoring is signless), and the
        # old signed reassembly folded conf's sign into the size channel — randomising it whenever
        # conf is uninformative, which the v0.4.2.0 run showed it is. conf is already channel 0, so
        # the two stay orthogonal and MT2 decides how to combine them. Mirrors training_v4.cpp.
        # conf4 is ungraded as of v0.5.0.0 — forward a constant, not a frozen arbitrary
        # function, which MT2 could otherwise fit as structured noise.
        in48.extend([conf, abs(delta_t), range_pct, MT1_UNGRADED_FEED])
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
                    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
                        log(f"WARNING: could not load MT2 history model {hp}: {e}")
    except (OSError, json.JSONDecodeError) as e:
        log(f"WARNING: could not read {mt2_hist_meta_path}: {e}")

    # Load 10-day post-injection hold counter
    inj_state_path = os.path.join(model_dir, 'mt2_inj_state.json')
    try:
        with open(inj_state_path) as _f:
            injection_hold = json.load(_f).get('injection_hold', 0)
    except (OSError, json.JSONDecodeError):
        injection_hold = 0   # absent on first run — expected, not an error
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
        except (OSError, json.JSONDecodeError) as e:
            log(f"WARNING: {mt2_hist_meta_path} unreadable, restarting ring at 0: {e}")
            h_head_out, h_count_out = 0, 0
        for k in range(HIST_ELITE):
            hp = os.path.join(model_dir, f'mt2_hist_{h_head_out}_{k}.pt')
            try:
                hm_save = load_slot_model(prefix, model_dir, k, MT2NN)
                torch.save(hm_save.state_dict(), hp)
                del hm_save
            except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
                log(f"WARNING: could not save MT2 history elite {hp}: {e}")
        for k in range(HIST_WAVG):
            hp = os.path.join(model_dir, f'mt2_hist_{h_head_out}_{HIST_ELITE + k}.pt')
            try:
                wm_save = load_slot_model(prefix, model_dir, ELITE_COUNT + k, MT2NN)
                torch.save(wm_save.state_dict(), hp)
                del wm_save
            except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as e:
                log(f"WARNING: could not save MT2 history wavg {hp}: {e}")
        h_head_out  = (h_head_out + 1) % HIST_DAYS
        h_count_out = min(h_count_out + 1, HIST_DAYS)
        try:
            with open(mt2_hist_meta_path, 'w') as _f:
                json.dump({'head': h_head_out, 'count': h_count_out}, _f)
        except OSError as e:
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
    except OSError as e:
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

def run_mt_inference(model_dir, industries, mkt_val_history, pf_val_history, zero_counts, total_cash):
    """
    MT1→MT2 inference chain for daily capital allocation in production.

    mkt_val_history: {ind: [cumulative_market_index]}  — market feature source.
    pf_val_history:  {ind: [cumulative_slot0_portfolio_index]} — portfolio feature source (Part C).
    MT1 input per industry = 74 = [37 market ‖ 37 portfolio] features. Loads mt1_{ind}_best.pt for
    each industry and mt2_best.pt, runs the chain, returns (allocations, tier_map, mt1_outputs).

    Caller should fall back to MasterNN/equal allocation if mt2_best.pt is absent.

    allocations:  {ind: dollar_amount}
    tier_map:     {ind: 0-3}
    mt1_outputs:  {ind: (conf, delta_t, range_pct, conf4)} — raw activations (no normalization)
    """
    from training_lib import build_master_features, tiers_to_alloc

    industry_list = list(industries.keys())
    today888 = build_master_features(mkt_val_history, pf_val_history, industry_list)

    mt1_outputs: dict = {}
    for i, ind in enumerate(industry_list):
        in74_t  = today888[:, i * 74:(i + 1) * 74]
        best_pt = os.path.join(model_dir, f"mt1_{ind}_best.pt")
        mt1_m   = MT1NN()
        if os.path.exists(best_pt):
            try:
                mt1_m.load_state_dict(torch.load(best_pt, weights_only=True))
            except Exception as e:
                print(f"Warning: could not load mt1_{ind}_best.pt: {e}")
        mt1_m.eval()
        with torch.no_grad():
            conf, delta_t, range_pct, conf4 = _mt1_decode(mt1_m, in74_t)
        mt1_outputs[ind] = (conf, delta_t, range_pct, conf4)
        del mt1_m

    # Build in48 for MT2 from the composed production MT1 (head0+tail0). In the heads/tails design
    # there is a single production model, so the composite and direction feeds are identical and the
    # MT2_FEED_DIRECTION toggle is moot (Increment 4C).
    in48 = []
    for ind in industry_list:
        conf, delta_t, range_pct, conf4 = mt1_outputs[ind]
        # Magnitude only, matching upkeep_mt2 and training_v4.cpp. This site previously forwarded
        # delta_t's raw (ungraded) sign while upkeep_mt2 forwarded conf's sign — the two MT2 feeds
        # disagreed on what channel 1 meant.
        # conf4 is ungraded as of v0.5.0.0 — forward a constant, not a frozen arbitrary
        # function, which MT2 could otherwise fit as structured noise.
        in48.extend([conf, abs(delta_t), range_pct, MT1_UNGRADED_FEED])
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
