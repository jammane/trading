"""
upkeep.py — Single-day evolution step for production upkeep.

Imported by production_v2.py after each trading day. Provides three upkeep functions:

    upkeep_industry()      — one evolution step for a StockNN industry pool, with burst
                             refinement now correctly enabled (UPKEEP_SIGMA passed through).

    upkeep_mt1_industry()  — one evolution step for one industry's MT1 pool (12 pools total),
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
    MT1_PRED_SCALE    = 10000.0 — tanh ceiling for the dollar P&L prediction
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

from models import MT1Net, MT2NN, StockNN
from training_lib import (
    ELITE_COUNT,
    ELITE_POOL,
    HIST_DAYS,
    HIST_ELITE,
    HIST_PER_DAY,
    HIST_WAVG,
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
UPKEEP_MT1_SIGMA  = 0.006   # MT1 (one pool per industry; was four per-channel sigmas)
UPKEEP_MT2_SIGMA  = 0.001   # MT2 (v0.2.6.0: 0.002→0.001 to tighten the pool)
MT2_INJ_THRESHOLD = -7.0      # injection fires when ≥75% of pool below this
MT2_INJ_MIN_BELOW = int(N_SLOTS * 0.75)  # 150 of 200

# ── MT1 pool (mirrors mt1_pool.h — the C++ trainer and this path evolve the SAME files) ──
# Every constant here has a twin in mt1_pool.h. They must match: the trainer writes the pool and
# the sidecar, production keeps evolving them, and a divergence shows up as models that mature or
# get culled on different schedules in the two halves of the same pool's life.
MT1_PRED_SCALE         = 10000.0   # tanh(out) x scale = predicted next-session P&L, in dollars
MT1_BASELINE_DAYS      = 20        # trailing mean of realised P&L — the predictor MT1 must beat
MT1_FLOOR_DAYS         = 10        # rolling window behind the score floor
MT1_FLOOR_FRAC         = 0.5
MT1_POOL_SLOTS         = 200
MT1_SCORE_HIST         = 16        # rolling per-model score register
MT1_POOL_MIN_AGE       = 8         # predictions before a model may be culled OR breed
MT1_POOL_CULL_PCT      = 0.083     # fraction of MATURE models culled per run
MT1_POOL_ELITE_PCT     = 0.10      # top fraction of mature models used as parents
MT1_POOL_LINEAGE_CAP   = 0.125     # lineage above this share of the pool stops breeding
MT1_POOL_LINEAGE_RESUME = 0.10     # ...and resumes only below this (hysteresis)
MT1_RECENCY_W          = (1.0, 0.8, 0.6, 0.4)   # newest-first, blocks of 4; full register sums 11.2

MT2_INJ_THRESHOLD = -7.0      # injection fires when >=75% of pool below this
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

    Mirrors training_v2.selection_and_mutation but uses _mutate_generic so MT1Net
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


# ── MT1 state, scoring and pool lifecycle ──────────────────────────────────────
#
# Replaces the rolling acc/vol floors, the four decode helpers, the four-component score
# breakdown, the recency-weighted replay window and the direction pool's parallel machinery.
# One target, one score, one pool.


def load_mt1_rolling_state(model_dir):
    """Per-industry MT1 state: trailing realised P&L, parked predictions, slot registers."""
    path = os.path.join(model_dir, 'mt1_rolling_state.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_mt1_rolling_state(model_dir, state):
    path = os.path.join(model_dir, 'mt1_rolling_state.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, path)


def mt1_pred(raw):
    """Decode one raw MT1Net output into a dollar prediction. Mirrors mt1_pred in mt1_pool.h."""
    return math.tanh(float(raw)) * MT1_PRED_SCALE


def mt1_baseline(actuals):
    """Trailing mean of realised P&L — the naive predictor a score of 0.5 means tying."""
    if not actuals:
        return 0.0
    return sum(actuals) / len(actuals)


def mt1_floor(actuals):
    """Score-denominator floor: half the recent mean absolute P&L, never below $1.

    Without it the score is hypersensitive on any day the trailing mean happens to land on the
    outcome, because the baseline's error goes to zero and every model looks infinitely better
    than it.
    """
    window = actuals[-MT1_FLOOR_DAYS:]
    if not window:
        return 1.0
    return max(sum(abs(a) for a in window) / len(window) * MT1_FLOOR_FRAC, 1.0)


def mt1_score(actual, predicted, baseline, floor_v):
    """d / (err + d) where d = max(baseline error, floor). Mirrors mt1_score in mt1_pool.h.

    0.5 = tied the trailing mean. Above 0.5 = beat it. Bounded in (0, 1], so no single day can
    dominate a register, and symmetric in the sign of the error.
    """
    err = abs(actual - predicted)
    base = abs(actual - baseline)
    d = max(base, max(floor_v, 1e-6))
    return d / (err + d)


def mt1_windows_are_causal(scored_day, baseline_last_day, floor_last_day):
    """Both windows must end strictly before the day being scored."""
    return baseline_last_day < scored_day and floor_last_day < scored_day


def _mt1_slot_rank_key(meta):
    """(recency-weighted mean, plain mean) descending — mirrors the C++ mt1_slot_better.

    ORDER MATTERS, and it is not the intuitive one. The primary key is the RECENCY-WEIGHTED mean
    (`mt1_slot_score` in mt1_pool.h), with the plain mean (`mt1_slot_mean`) only as a tie-break.
    Having the two the other way round still ranks identically whenever the plain means differ by
    less than nothing — which is most pairs — so it reads as correct and passes almost every
    spot check. It diverges on exactly the pairs that matter: a model improving over its last
    eight predictions loses to a flat mediocre one. Since training_v4.cpp and this path evolve the
    SAME pool, that means a different deployed model and different cull victims depending on which
    half of the pool's life is running, with no error anywhere.

    A partial register normalises by the weight actually occupied, so an 8-prediction model is
    judged on its 8 rather than penalised for the 8 it has not made yet.

    `scores` is oldest-first here; the C++ `score[0]` is the MOST RECENT. Reverse before weighting.
    """
    scores = meta.get('scores', [])
    n = min(len(scores), MT1_SCORE_HIST)
    if n == 0:
        return (0.0, 0.0)
    recent = scores[-n:][::-1]                      # newest first, matching the C++ layout
    plain = sum(recent) / n
    wsum = 0.0
    tot = 0.0
    for i, v in enumerate(recent):
        w = MT1_RECENCY_W[min(i // 4, len(MT1_RECENCY_W) - 1)]
        wsum += w * v
        tot += w
    weighted = wsum / tot if tot > 0 else 0.0
    return (weighted, plain)


def _mt1_slot_mature(meta):
    return meta.get('n_pred', 0) >= MT1_POOL_MIN_AGE


def _mt1_slot_record(meta, score):
    scores = meta.setdefault('scores', [])
    scores.append(float(score))
    if len(scores) > MT1_SCORE_HIST:
        del scores[:-MT1_SCORE_HIST]
    meta['n_pred'] = meta.get('n_pred', 0) + 1


def _mt1_barred_lineages(metas, previously_barred):
    """Lineages over MT1_POOL_LINEAGE_CAP of the pool, with hysteresis back to _RESUME."""
    counts: dict = {}
    for m in metas:
        lin = m.get('lineage', 0)
        counts[lin] = counts.get(lin, 0) + 1
    cap = MT1_POOL_LINEAGE_CAP * len(metas)
    resume = MT1_POOL_LINEAGE_RESUME * len(metas)
    barred = set()
    for lin, n in counts.items():
        limit = resume if lin in previously_barred else cap
        if n > limit:
            barred.add(lin)
    return barred, counts


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


def _mt1_slot_path(industry, model_dir, slot):
    return os.path.join(model_dir, f'mt1_{industry}_model_{slot}.pt')


def _mt1_load_slot(industry, model_dir, slot):
    m = MT1Net()
    path = _mt1_slot_path(industry, model_dir, slot)
    if os.path.exists(path):
        try:
            m.load_state_dict(torch.load(path, weights_only=True))
        except Exception as e:                                   # noqa: BLE001 - reseed on damage
            print(f'Warning: mt1_{industry} slot {slot} unreadable ({e}) — re-initialised')
    m.eval()
    return m


def _mt1_bootstrap(industry, model_dir):
    """Create the pool on first run: seed every slot from mt1_{ind}_best.pt if it exists.

    Seeding all 200 from one model is deliberate. The first cull is MT1_POOL_MIN_AGE runs away,
    and mutation diverges them well before then; starting from the deployed model means
    production is never allocating on a pool of untrained randoms.
    """
    best_path = os.path.join(model_dir, f'mt1_{industry}_best.pt')
    base = MT1Net()
    if os.path.exists(best_path):
        try:
            base.load_state_dict(torch.load(best_path, weights_only=True))
        except Exception as e:                                   # noqa: BLE001
            print(f'Warning: mt1_{industry}_best.pt unreadable ({e}) — bootstrapping from random')
    for slot in range(MT1_POOL_SLOTS):
        if slot == 0:
            torch.save(base.state_dict(), _mt1_slot_path(industry, model_dir, slot))
        else:
            child = _mutate_generic(base, MT1Net, UPKEEP_MT1_SIGMA)
            torch.save(child.state_dict(), _mt1_slot_path(industry, model_dir, slot))


def upkeep_mt1_industry(industry, model_dir, in74_t, actual_d,
                        sigma=UPKEEP_MT1_SIGMA, rolling_state=None):
    """One production evolution step for one industry's MT1 pool.

    Mirrors mt1_step_day in training_v4.cpp, in the same three parts and the same order:

      1. SCORE   the predictions parked on the previous run, against `actual_d` — the P&L this
                 industry's StockNN actually realised this session.
      2. EVOLVE  cull the worst mature individuals, breed replacements from the unbarred elite.
      3. PREDICT the next session with every individual; park the answers in rolling_state.

    `actual_d` grades yesterday's call, not today's. Today's call cannot be graded until the next
    session exists, which is the whole point: a model is never scored on a day it has seen.

    Returns (pred, score0) — the deployed model's prediction for the NEXT session in dollars, and
    the score it earned on this one (0.0 on the first run, when nothing was parked).
    """
    if rolling_state is None:
        rolling_state = {}
    rs = rolling_state.setdefault(industry, {})
    actuals = rs.setdefault('actuals', [])
    metas = rs.setdefault('metas', [{'scores': [], 'n_pred': 0, 'lineage': i}
                                    for i in range(MT1_POOL_SLOTS)])
    pending = rs.get('pending')
    best_slot = rs.get('best_slot', 0)
    next_lineage = rs.get('next_lineage', MT1_POOL_SLOTS)
    barred = set(rs.get('barred', []))
    day = rs.get('day', 0)

    if not os.path.exists(_mt1_slot_path(industry, model_dir, 0)):
        _mt1_bootstrap(industry, model_dir)

    in74_t = in74_t.detach()
    score0 = 0.0

    # ── 1. SCORE ────────────────────────────────────────────────────────────────
    if pending and len(pending) == MT1_POOL_SLOTS and actual_d is not None:
        baseline = mt1_baseline(actuals)
        floor_v = mt1_floor(actuals)
        # `actuals` holds sessions strictly before this one — it is appended to below.
        assert mt1_windows_are_causal(day, day - 1, day - 1)
        for slot in range(MT1_POOL_SLOTS):
            v = mt1_score(actual_d, pending[slot], baseline, floor_v)
            _mt1_slot_record(metas[slot], v)
            if slot == best_slot:
                score0 = v
        rs['pending'] = None

    if actual_d is not None:
        actuals.append(float(actual_d))
        if len(actuals) > MT1_BASELINE_DAYS:
            del actuals[:-MT1_BASELINE_DAYS]

    # ── 2. EVOLVE ───────────────────────────────────────────────────────────────
    order = sorted(range(MT1_POOL_SLOTS),
                   key=lambda i: (_mt1_slot_mature(metas[i]), _mt1_slot_rank_key(metas[i])),
                   reverse=True)
    mature = [i for i in order if _mt1_slot_mature(metas[i])]

    if score0 and mature:
        barred, _counts = _mt1_barred_lineages(metas, barred)
        n_cull = int(MT1_POOL_CULL_PCT * len(mature) + 0.5)
        n_elite = max(1, int(MT1_POOL_ELITE_PCT * len(mature)))
        parents = [i for i in mature[:n_elite] if metas[i].get('lineage') not in barred]
        if not parents:
            parents = [mature[0]]           # every lineage barred: breed anyway rather than freeze
        for c in range(n_cull):
            victim = mature[len(mature) - 1 - c]
            if victim == best_slot:
                continue
            par = parents[c % len(parents)]
            child = _mutate_generic(_mt1_load_slot(industry, model_dir, par), MT1Net, sigma)
            torch.save(child.state_dict(), _mt1_slot_path(industry, model_dir, victim))
            metas[victim] = {'scores': [], 'n_pred': 0, 'lineage': metas[par].get('lineage', 0)}
        best_slot = order[0]

    # ── 3. PREDICT ──────────────────────────────────────────────────────────────
    preds = []
    for slot in range(MT1_POOL_SLOTS):
        m = _mt1_load_slot(industry, model_dir, slot)
        with torch.no_grad():
            preds.append(mt1_pred(m(in74_t)[0, 0]))
        del m

    rs['pending'] = preds
    rs['best_slot'] = best_slot
    rs['next_lineage'] = next_lineage
    rs['barred'] = sorted(barred)
    rs['day'] = day + 1

    # The deployed model is whichever individual ranked first; publish it for inference.
    best_sd = torch.load(_mt1_slot_path(industry, model_dir, best_slot), weights_only=True)
    torch.save(best_sd, os.path.join(model_dir, f'mt1_{industry}_best.pt'))
    return preds[best_slot], score0


# ── MT2 upkeep ─────────────────────────────────────────────────────────────────

# ── living conditional model ────────────────────────────────────────────────────
# Re-evaluated every run against RECENT StockNN performance, which is what makes it "living":
# the counts decay, so the percentages and the streak length N both move as behaviour changes,
# and the memory length itself is re-selected from data rather than fixed.
LIVING_BN_FILE = 'living_bn.json'
LIVING_BN_HISTORY = 750       # days of per-industry P&L retained, for re-selecting the half-life
LIVING_BN_REFIT_EVERY = 20    # runs between half-life re-selection (it is the expensive step)
LIVING_BN_MIN_REFIT = 400     # days needed before a half-life estimate means anything


def upkeep_living_bn(model_dir, actual_by_industry, industry_list):
    """One daily step of the living conditional model. Returns its current report.

    `actual_by_industry` is each industry's realised book change for the session -- StockNN's own
    P&L on the portfolio it chose, not raw stock returns. A missing industry is recorded as NaN,
    not 0.0: zero is the value a hard-floor reset writes, and `classify` treats it as "no state"
    on purpose, so passing 0.0 for "unknown" would silently mix the two.
    """
    import numpy as _np

    from living_bn import LivingBN, fit_half_life

    path = os.path.join(model_dir, LIVING_BN_FILE)
    blob = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                blob = json.load(f)
        except (OSError, ValueError) as e:
            print(f"living_bn: {path} unreadable ({e}) — starting fresh")
            blob = {}

    n_ind = len(industry_list)
    model = (LivingBN.from_dict(blob['model']) if blob.get('model')
             else LivingBN(n_ind=n_ind))
    if model.n_ind != n_ind:                       # industry count changed; state is meaningless
        print(f"living_bn: industry count {model.n_ind} -> {n_ind}, resetting")
        model = LivingBN(n_ind=n_ind)
        blob = {}

    hist = blob.get('history', [])
    row = [float(actual_by_industry.get(ind, _np.nan)) if actual_by_industry.get(ind) is not None
           else float('nan') for ind in industry_list]
    hist.append(row)
    hist = hist[-LIVING_BN_HISTORY:]

    if len(hist) >= 3:
        model.observe(_np.array(hist[-3]), _np.array(hist[-2]), _np.array(hist[-1]))

    runs = int(blob.get('runs_since_refit', 0)) + 1
    if runs >= LIVING_BN_REFIT_EVERY and len(hist) >= LIVING_BN_MIN_REFIT:
        arr = _np.array(hist, dtype=float)
        try:
            best, _ = fit_half_life(arr, warmup=min(250, len(hist) // 2))
            if best != model.half_life:
                print(f"living_bn: half-life {model.half_life} -> {best}")
            # Rebuild from history: the decay is baked into every count, so changing it means
            # re-accumulating rather than carrying forward counts weighted the old way.
            model = LivingBN(n_ind=n_ind, half_life=best)
            for t in range(2, len(arr)):
                model.observe(arr[t - 2], arr[t - 1], arr[t])
        except Exception as e:                                   # noqa: BLE001
            print(f"living_bn: half-life refit failed ({e}); keeping {model.half_life}")
        runs = 0

    with open(path, 'w') as f:
        json.dump({'model': model.to_dict(), 'history': hist,
                   'runs_since_refit': runs}, f)
    return model.report()


def upkeep_mt2(model_dir, mt1_slot0_outputs, actual_perf, industry_list,
               sigma=UPKEEP_MT2_SIGMA):
    """
    One evolution step for the MT2NN cross-industry allocator pool.

    mt1_slot0_outputs: {ind: predicted_next_session_pnl_dollars} — one MT1 output per industry
                       for each industry, produced by upkeep_mt1_industry().
                       No normalization applied — raw activations passed directly to MT2.
    actual_perf:       {ind: float} fractional return per industry for scoring.
    industry_list:     list of industry keys in canonical INDUSTRY_NAMES order.

    File naming: mt2_model_{slot}.pt (prefix 'mt2').

    Returns (best_pts, slot0_pts, injected_flag).
    """
    prefix    = 'mt2'
    best_path = os.path.join(model_dir, 'mt2_best.pt')

    # Build in12: one MT1 dollar prediction per industry, signed, unnormalised. Mirrors
    # training_v4.cpp. The signed prediction replaces the old four channels outright — it carries
    # both the direction call and the size, which is all the four ever encoded between them.
    in12 = [float(mt1_slot0_outputs.get(ind, 0.0)) for ind in industry_list]
    in12_t = torch.tensor(in12, dtype=torch.float32).unsqueeze(0)   # (1, 12)

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
            out = m(in12_t)   # (1, 48)
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
        out0 = m0(in12_t)
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
                            h_out = hm(in12_t)
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
    MT1 input per industry = 74 = [37 market ‖ 37 portfolio] features; its single output is that
    industry's predicted next-session StockNN P&L in dollars. Loads mt1_{ind}_best.pt for each
    industry and mt2_best.pt, runs the chain, returns (allocations, tier_map, mt1_outputs).

    Caller should fall back to MasterNN/equal allocation if mt2_best.pt is absent.

    allocations:  {ind: dollar_amount}
    tier_map:     {ind: 0-3}
    mt1_outputs:  {ind: predicted_next_session_pnl_dollars}
    """
    from training_lib import build_master_features, tiers_to_alloc

    industry_list = list(industries.keys())
    today888 = build_master_features(mkt_val_history, pf_val_history, industry_list)

    mt1_outputs: dict = {}
    for i, ind in enumerate(industry_list):
        in74_t  = today888[:, i * 74:(i + 1) * 74]
        best_pt = os.path.join(model_dir, f"mt1_{ind}_best.pt")
        mt1_m   = MT1Net()
        if os.path.exists(best_pt):
            try:
                mt1_m.load_state_dict(torch.load(best_pt, weights_only=True))
            except Exception as e:                               # noqa: BLE001
                print(f"Warning: could not load mt1_{ind}_best.pt: {e}")
        mt1_m.eval()
        with torch.no_grad():
            mt1_outputs[ind] = mt1_pred(mt1_m(in74_t)[0, 0])
        del mt1_m

    # MT2 input: the 12 predictions in industry order, signed and unnormalised.
    in12_t = torch.tensor([mt1_outputs[ind] for ind in industry_list],
                          dtype=torch.float32).unsqueeze(0)       # (1, 12)

    mt2_path = os.path.join(model_dir, 'mt2_best.pt')
    mt2_m    = MT2NN()
    if os.path.exists(mt2_path):
        try:
            mt2_m.load_state_dict(torch.load(mt2_path, weights_only=True))
        except Exception as e:
            print(f"Warning: could not load mt2_best.pt: {e}")
    mt2_m.eval()
    with torch.no_grad():
        out = mt2_m(in12_t)
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
