"""
upkeep.py — Single-day evolution step for production upkeep.

Imported by production_v2.py after each trading day. Provides three upkeep functions:

    upkeep_industry()      — one evolution step for a StockNN industry pool, with burst
                             refinement now correctly enabled (UPKEEP_SIGMA passed through).
                             Fixes the silent bug in training_v2.train_industry_one_day
                             where daily_sigma was never forwarded to step_industry.

    upkeep_mt1_industry()  — one evolution step for one MT1NN industry pool (12 pools total),
                             followed by 4 burst refinement passes at sigma/2, /4, /8, /16.
                             MT1 is scored directly (direction + range + accuracy) with no
                             portfolio simulation.

    upkeep_mt2()           — one evolution step for the MT2NN cross-industry allocator pool.
                             Uses MT1 slot0 outputs (normalized via running stats) as input.
                             No burst refinement — portfolio simulation already runs all 200
                             slots. Diversity injection fires when best_pts < -1.

Constants:
    UPKEEP_SIGMA = 0.004  — base sigma for burst refinement (half of full-train 0.008)
    MT1_SCALE    = 0.05   — tanh scale for expected delta (±5% range)
    RANGE_SCALE  = 0.02   — half-width at which score_range = 1/e ≈ 0.37
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

UPKEEP_SIGMA  = 0.004
MT1_SCALE     = 0.05
RANGE_SCALE   = 0.02


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


# ── MT1 scoring ────────────────────────────────────────────────────────────────

def _mt1_score(out3, actual):
    """
    Score one MT1 slot against the industry's actual fractional return.

    out3:   raw logit tensor shape (3,)
    actual: float — fractional return (slot0_industry_value / baseline - 1)
    Returns float in [0.0, 1.0].
    """
    conf     = torch.sigmoid(out3[0]).item()
    delta    = torch.tanh(out3[1]).item() * MT1_SCALE
    range_hw = F.softplus(out3[2]).item()

    score_dir   = 1.0 if (conf >= 0.5) == (actual >= 0.0) else 0.0
    in_range    = abs(actual - delta) <= range_hw
    score_range = math.exp(-range_hw / RANGE_SCALE) if in_range else 0.0
    score_acc   = max(0.0, 1.0 - min(abs(actual - delta) / (abs(actual) + 1e-9), 1.0))

    return 0.50 * score_dir + 0.33 * score_range + 0.17 * score_acc


def _mt1_decode(model, in37_t):
    """Run MT1 inference and decode outputs to (conf, delta, range_hw)."""
    model.eval()
    with torch.inference_mode():
        out3 = model(in37_t).squeeze(0)
    conf     = torch.sigmoid(out3[0]).item()
    delta    = torch.tanh(out3[1]).item() * MT1_SCALE
    range_hw = F.softplus(out3[2]).item()
    return conf, delta, range_hw


# ── MT1 burst refinement ───────────────────────────────────────────────────────

def _mt1_burst(prefix, model_dir, in37_t, actual_perf_i, burst_sigma):
    """
    One burst refinement pass for an MT1 pool.

    Generates 200 mutants (10 per ELITE_POOL parent), scores via _mt1_score,
    merges top-ELITE_COUNT with current elites (cap: 2 burst replacements),
    and regenerates wavg slots 17–19. Returns new best score.
    """
    burst_candidates = []
    for elite_slot in range(ELITE_POOL):
        parent = load_slot_model(prefix, model_dir, elite_slot, MT1NN)
        for _ in range(10):
            child = _mutate_generic(parent, MT1NN, burst_sigma)
            child.eval()
            with torch.inference_mode():
                out3 = child(in37_t).squeeze(0)
            burst_candidates.append((child, _mt1_score(out3, actual_perf_i)))
        del parent
    burst_candidates.sort(key=lambda x: x[1], reverse=True)
    top_burst = burst_candidates[:ELITE_COUNT]

    current_elites = []
    for rank in range(ELITE_COUNT):
        m = load_slot_model(prefix, model_dir, rank, MT1NN)
        m.eval()
        with torch.inference_mode():
            out3 = m(in37_t).squeeze(0)
        current_elites.append((m, _mt1_score(out3, actual_perf_i)))

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
    if new_best > prev_best:
        log(f"[mt1/{sn(prefix[4:])}] Burst σ={burst_sigma:.5f}: {burst_count} replacement(s), best={new_best:.4f}")
    else:
        log(f"[mt1/{sn(prefix[4:])}] Burst σ={burst_sigma:.5f}: best={new_best:.4f} — no improvement")

    for rank, (m, _) in enumerate(new_elites):
        save_slot_model(prefix, model_dir, rank, m)

    elite_scores = [s for _, s in new_elites]
    for n_avg, wavg_slot in [(5, ELITE_COUNT), (10, ELITE_COUNT + 1), (15, ELITE_COUNT + 2)]:
        k    = min(n_avg, ELITE_COUNT)
        wavg = compute_weighted_avg_model(prefix, model_dir, list(range(k)), elite_scores[:k], MT1NN)
        save_slot_model(prefix, model_dir, wavg_slot, wavg)
        del wavg

    del burst_candidates, current_elites, top_burst, all_cands, new_elites
    gc.collect()
    return new_best


# ── MT2 norm stats ─────────────────────────────────────────────────────────────

def load_mt2_norm_stats(model_dir):
    """Load MT2 running normalization stats. Returns zero-initialized dict on missing/error."""
    path = os.path.join(model_dir, 'mt2_norm_stats.json')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {'delta_mean': 0.0, 'delta_var': 0.0,
            'range_mean': 0.0, 'range_var': 0.0, 'count': 0}


def save_mt2_norm_stats(model_dir, stats):
    try:
        with open(os.path.join(model_dir, 'mt2_norm_stats.json'), 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        log(f"WARNING: could not save mt2_norm_stats: {e}")


def _update_norm_stats(stats, delta, range_hw):
    """Welford's online algorithm — update running mean/variance for delta and range in-place."""
    n = stats['count'] + 1
    d = delta - stats['delta_mean']
    stats['delta_mean'] += d / n
    stats['delta_var']  += d * (delta - stats['delta_mean'])
    r = range_hw - stats['range_mean']
    stats['range_mean'] += r / n
    stats['range_var']  += r * (range_hw - stats['range_mean'])
    stats['count'] = n


def _normalize_stat(val, mean, var, count):
    std = math.sqrt(var / count) if count > 1 else 1.0
    return (val - mean) / (std + 1e-9)


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

def upkeep_mt1_industry(industry, model_dir, in37_t, actual_perf_i, sigma=UPKEEP_SIGMA):
    """
    One evolution step for one industry's MT1NN pool, with 4 burst passes.

    industry:      industry key (e.g. 'energy')
    model_dir:     directory containing model files
    in37_t:        (1, 37) tensor — this industry's slice of build_master_features output
    actual_perf_i: float — fractional return (slot0_score / baseline - 1) for today
    sigma:         base mutation sigma

    File naming: mt1_{industry}_model_{slot}.pt (uses _model_path with prefix 'mt1_{industry}')

    Returns (best_mt1_score, slot0_mt1_score, slot0_conf, slot0_delta, slot0_range_hw).
    """
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

    in37_t = in37_t.detach()

    # Score all N_SLOTS slots
    scores = []
    for slot in range(N_SLOTS):
        m = load_slot_model(prefix, model_dir, slot, MT1NN)
        m.eval()
        with torch.inference_mode():
            out3 = m(in37_t).squeeze(0)
        scores.append((slot, _mt1_score(out3, actual_perf_i)))
        del m

    # Record slot0 outputs before selection (selection may overwrite slot 0)
    slot0_m = load_slot_model(prefix, model_dir, 0, MT1NN)
    slot0_conf, slot0_delta, slot0_range_hw = _mt1_decode(slot0_m, in37_t)
    del slot0_m

    best_score  = max(s for _, s in scores)
    slot0_score = dict(scores)[0]

    log(f"[mt1/{sn(industry)}] best={best_score:.4f} slot0={slot0_score:.4f} "
        f"actual={actual_perf_i:+.4f}")

    _select_and_mutate(prefix, model_dir, MT1NN, scores, sigma)

    # 4 burst refinement passes
    log(f"[mt1/{sn(industry)}] Running 4 burst passes ...")
    for burst_num in range(4):
        _mt1_burst(prefix, model_dir, in37_t, actual_perf_i, sigma / (2 ** (burst_num + 1)))

    # Save slot 0 (post-selection best) as the best model for production inference
    slot0_final = load_slot_model(prefix, model_dir, 0, MT1NN)
    try:
        torch.save(slot0_final.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save {prefix}_best.pt: {e}")
    del slot0_final

    return best_score, slot0_score, slot0_conf, slot0_delta, slot0_range_hw


# ── MT2 upkeep ─────────────────────────────────────────────────────────────────

def upkeep_mt2(model_dir, mt1_slot0_outputs, norm_stats, actual_perf, industry_list,
               sigma=UPKEEP_SIGMA):
    """
    One evolution step for the MT2NN cross-industry allocator pool.

    mt1_slot0_outputs: {ind: (conf, delta, range_hw)} — MT1 slot0 decoded outputs
                       for each industry, produced by upkeep_mt1_industry().
    norm_stats:        running statistics dict (mutated in place via _update_norm_stats).
                       Persist between calls with load/save_mt2_norm_stats().
    actual_perf:       {ind: float} fractional return per industry for scoring.
    industry_list:     list of industry keys in canonical INDUSTRY_NAMES order.

    File naming: mt2_model_{slot}.pt (prefix 'mt2').

    Returns (best_pts, slot0_pts, injected_flag). Updates norm_stats in place.
    """
    prefix    = 'mt2'
    best_path = os.path.join(model_dir, 'mt2_best.pt')

    # Update running norm stats from today's MT1 slot0 outputs
    for ind in industry_list:
        _, delta, range_hw = mt1_slot0_outputs.get(ind, (0.5, 0.0, 0.02))
        _update_norm_stats(norm_stats, delta, range_hw)

    # Build normalized in36 input
    in36 = []
    for ind in industry_list:
        conf, delta, range_hw = mt1_slot0_outputs.get(ind, (0.5, 0.0, 0.02))
        in36.append(conf)
        in36.append(_normalize_stat(delta,    norm_stats['delta_mean'], norm_stats['delta_var'], norm_stats['count']))
        in36.append(_normalize_stat(range_hw, norm_stats['range_mean'], norm_stats['range_var'], norm_stats['count']))
    in36_t = torch.tensor(in36, dtype=torch.float32).unsqueeze(0)   # (1, 36)

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
            out = m(in36_t)   # (1, 48)
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
        out0 = m0(in36_t)
    del m0
    tier0 = out0.view(12, 4).argmax(dim=1).tolist()
    tier_counts = [sum(1 for t in tier0 if t == c) for c in range(4)]

    log(f"[mt2] best_pts={best_pts:+.2f} slot0_pts={slot0_pts:+.2f} ideal={ideal_pts} "
        f"tiers(0/1/2/3)={tier_counts[0]}/{tier_counts[1]}/{tier_counts[2]}/{tier_counts[3]}")

    injected = False
    if best_pts >= -1.0:
        _select_and_mutate(prefix, model_dir, MT2NN, pred_scores, sigma)
    else:
        injected = True
        half = ELITE_COUNT // 2
        log(f"[mt2] best_pts={best_pts:.2f} < -1 — diversity injection into bottom {ELITE_COUNT - half} elites")
        for inject_rank in range(half, ELITE_COUNT):
            source_rank = inject_rank - half
            elite = load_slot_model(prefix, model_dir, source_rank, MT2NN)
            blend = blend_model_halfway(elite, MT2NN)
            save_slot_model(prefix, model_dir, inject_rank, blend)
            del elite, blend

    slot0_final = load_slot_model(prefix, model_dir, 0, MT2NN)
    try:
        torch.save(slot0_final.state_dict(), best_path)
    except Exception as e:
        log(f"WARNING: could not save mt2_best.pt: {e}")
    del slot0_final

    return best_pts, slot0_pts, injected


# ── Production inference (MT1 → MT2) ──────────────────────────────────────────

def run_mt_inference(model_dir, industries, ind_value_history, norm_stats,
                     zero_counts, total_cash):
    """
    MT1→MT2 inference chain for daily capital allocation in production.

    Loads mt1_{ind}_best.pt for each industry and mt2_best.pt, runs the full
    chain, and returns (allocations, tier_map, mt1_outputs).

    Caller should fall back to MasterNN/equal allocation if mt2_best.pt is absent.

    allocations:  {ind: dollar_amount}
    tier_map:     {ind: 0-3}
    mt1_outputs:  {ind: (conf, delta, range_hw)} — slot0 decoded MT1 outputs
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
            conf, delta, range_hw = _mt1_decode(mt1_m, in37_t)
        mt1_outputs[ind] = (conf, delta, range_hw)
        del mt1_m

    in36 = []
    for ind in industry_list:
        conf, delta, range_hw = mt1_outputs[ind]
        in36.append(conf)
        in36.append(_normalize_stat(delta,    norm_stats['delta_mean'], norm_stats['delta_var'], norm_stats['count']))
        in36.append(_normalize_stat(range_hw, norm_stats['range_mean'], norm_stats['range_var'], norm_stats['count']))
    in36_t = torch.tensor(in36, dtype=torch.float32).unsqueeze(0)

    mt2_path = os.path.join(model_dir, 'mt2_best.pt')
    mt2_m    = MT2NN()
    if os.path.exists(mt2_path):
        try:
            mt2_m.load_state_dict(torch.load(mt2_path, weights_only=True))
        except Exception as e:
            print(f"Warning: could not load mt2_best.pt: {e}")
    mt2_m.eval()
    with torch.no_grad():
        out = mt2_m(in36_t)
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
