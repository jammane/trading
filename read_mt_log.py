#!/usr/bin/env python3
"""
read_mt_log.py — Read and summarize mt_training_log.bin from training_v4_cpp.

Usage:
  python read_mt_log.py /path/to/mt_training_log.bin
  python read_mt_log.py /path/to/mt_training_log.bin --pass 2         # single pass only
  python read_mt_log.py /path/to/mt_training_log.bin --industry tech  # filter industry
  python read_mt_log.py /path/to/mt_training_log.bin --per-day        # every day, not a summary

V12 only. The V1–V11 parsers are gone with the five-pool MT1 they described: every column they
decoded (four component pools × four stats, the OOS activation twin, the per-channel skill scores,
the direction pool's lifecycle) names a thing that no longer exists, and no V1–V11 log survives —
the old run directories were cleared when the universe was re-normalized. A reader that silently
accepts a record it cannot interpret is worse than one that refuses.
"""

import argparse
import math
import os
import struct
import sys
from collections import defaultdict

HEADER_SIZE    = 16
MT_LOG_MAGIC   = 0x4D543132        # "MT12" — must match MT_LOG_MAGIC in training_v4.cpp
MT_LOG_VERSION = 12
N_IND          = 12

# Mirrors MTLogRecord in training_v4.cpp. KEEP IN SYNC — a layout drift here decodes into
# plausible wrong numbers rather than an error, which is why the size is asserted below.
#
#   uint32 pass_num, actual_day
#   float  mt1_pred[12]        — deployed model's call for the NEXT session, in dollars
#   float  mt1_actual_d[12]    — realised P&L scored on this day (yesterday's call)
#   float  mt1_baseline[12]    — trailing mean of actual: the predictor MT1 must beat
#   float  mt1_floor[12]       — score-denominator floor in force this day
#   float  mt1_score0[12]      — deployed model's score
#   float  mt1_score_mean[12]  — pool mean
#   float  mt1_score_best[12]  — pool max  (max-of-200: rises with pool size under a null)
#   float  mt1_score_min[12]   — pool min
#   float  mt1_pool_stats[12][5]  — mature, culled, max_lineage, distinct_lineages, mean_retire_age
#   float  mt1_life[12][5]        — cumulative retirement-age histogram
#   float  mt2_best_pts, mt2_slot0_pts, mt2_ideal_pts
#   uint8  mt2_injected + 3 pad
#   float  mt2_consensus_flat_pts, mt2_consensus_wtd_pts, mt2_slot0_pts_pf, mt2_slot0_pts_mkt
RECORD_FMT  = ('<II' + 'f' * (8 * N_IND) + 'f' * (5 * N_IND) + 'f' * (5 * N_IND)
               + 'fff' + 'B3x' + '4f')
RECORD_SIZE = struct.calcsize(RECORD_FMT)
assert RECORD_SIZE == 904, f'V12 record must be 904 bytes, got {RECORD_SIZE}'

MT1_FIELDS = ('pred', 'actual', 'baseline', 'floor',
              'score0', 'score_mean', 'score_best', 'score_min')
POOL_STATS = ('mature', 'culled', 'lineage_max', 'lineage_n', 'retire_age')
LIFE_BUCKETS = ('8-15', '16-31', '32-63', '64-127', '128+')

# Taken from the universe rather than hardcoded, so a symbol swap or an industry rename cannot
# leave the reader labelling the wrong columns.
from universe import INDUSTRIES as _UNIVERSE_INDUSTRIES        # noqa: E402

INDUSTRY_NAMES = list(_UNIVERSE_INDUSTRIES.keys())


def parse_log(path):
    """Return (industries, records). Refuses anything that is not a V12 log."""
    with open(path, 'rb') as f:
        blob = f.read()
    if len(blob) < HEADER_SIZE:
        sys.exit(f'{path}: too short to hold a header')
    magic, version, n_ind, _ = struct.unpack('<4I', blob[:HEADER_SIZE])
    if magic != MT_LOG_MAGIC:
        sys.exit(f'{path}: bad magic 0x{magic:08X} — not an mt_training_log.bin')
    if version != MT_LOG_VERSION:
        sys.exit(f'{path}: log version {version}, this reader handles {MT_LOG_VERSION} only.\n'
                 'Versions 1-11 described the five-pool MT1 and are no longer decodable.')
    if n_ind != N_IND:
        sys.exit(f'{path}: {n_ind} industries, expected {N_IND}')

    body = blob[HEADER_SIZE:]
    if len(body) % RECORD_SIZE:
        print(f'Warning: {len(body) % RECORD_SIZE} trailing bytes ignored '
              '(run still in flight?)', file=sys.stderr)
    records = []
    for off in range(0, len(body) - RECORD_SIZE + 1, RECORD_SIZE):
        v = struct.unpack(RECORD_FMT, body[off:off + RECORD_SIZE])
        i = 2
        rec = {'pass': v[0], 'day': v[1], 'mt1': {}}
        for name in MT1_FIELDS:
            rec['mt1'][name] = list(v[i:i + N_IND]); i += N_IND
        rec['mt1']['pool_stats'] = [list(v[i + k * 5:i + k * 5 + 5]) for k in range(N_IND)]
        i += 5 * N_IND
        rec['mt1']['life'] = [list(v[i + k * 5:i + k * 5 + 5]) for k in range(N_IND)]
        i += 5 * N_IND
        rec['mt2'] = {'best_pts': v[i], 'slot0_pts': v[i + 1], 'ideal_pts': v[i + 2],
                      'injected': v[i + 3],
                      'consensus_flat': v[i + 4], 'consensus_wtd': v[i + 5],
                      'slot0_pts_pf': v[i + 6], 'slot0_pts_mkt': v[i + 7]}
        records.append(rec)
    return INDUSTRY_NAMES[:n_ind], records


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float('nan')


def _corr(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys, strict=True)
             if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float('nan')
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    dx = math.sqrt(sum((a - mx) ** 2 for a, b in pairs))
    dy = math.sqrt(sum((b - my) ** 2 for a, b in pairs))
    return num / (dx * dy) if dx > 0 and dy > 0 else float('nan')


def _ind_indices(industries, industry_filter):
    if not industry_filter:
        return list(range(len(industries)))
    idx = [i for i, n in enumerate(industries) if industry_filter.lower() in n.lower()]
    if not idx:
        sys.exit(f'No industry matches "{industry_filter}". Known: {", ".join(industries)}')
    return idx


def print_per_day(industries, recs, industry_filter=None):
    idx = _ind_indices(industries, industry_filter)
    print(f"{'day':>5} {'ind':<16} {'actual':>10} {'pred':>10} {'base':>10} "
          f"{'floor':>8} {'s0':>6} {'mean':>6} {'mature':>7} {'cull':>5}")
    for r in recs:
        for i in idx:
            m = r['mt1']
            print(f"{r['day']:>5} {industries[i]:<16} "
                  f"{m['actual'][i]:>10.1f} {m['pred'][i]:>10.1f} {m['baseline'][i]:>10.1f} "
                  f"{m['floor'][i]:>8.1f} {m['score0'][i]:>6.3f} {m['score_mean'][i]:>6.3f} "
                  f"{m['pool_stats'][i][0]:>7.0f} {m['pool_stats'][i][1]:>5.0f}")


def print_mt1_summary(industries, recs, industry_filter=None):
    """The three numbers that decide whether MT1 works, per industry.

    A score of 0.5 means the model tied the trailing mean. The MEAN column is the one to read:
    `best` is max-of-200 and rises with pool size even when nothing is learned, which is the
    artifact that made the old in-sample direction numbers look like 81% skill.

    `corr` is the correlation between the prediction and the outcome it was made for, over the
    whole window. It is the only column here that cannot be produced by selection on noise.
    """
    idx = _ind_indices(industries, industry_filter)
    print('\n── MT1 ──────────────────────────────────────────────────────────────────────')
    print('score 0.5 = tied the trailing mean.  read MEAN, not BEST (best = max-of-200).')
    print(f"\n{'industry':<16} {'n':>5} {'s0':>7} {'mean':>7} {'best':>7} {'min':>7} "
          f"{'corr':>7} {'>0.5':>6} {'mature':>7} {'lin':>5}")
    for i in idx:
        scored = [r for r in recs if r['mt1']['score_mean'][i] > 0]
        if not scored:
            print(f'{industries[i]:<16} {0:>5}   (no scored days)')
            continue
        s0 = [r['mt1']['score0'][i] for r in scored]
        corr = _corr([r['mt1']['pred'][i] for r in scored[:-1]],
                     [r['mt1']['actual'][i] for r in scored[1:]])
        frac = sum(1 for v in s0 if v > 0.5) / len(s0)
        print(f"{industries[i]:<16} {len(scored):>5} {_mean(s0):>7.3f} "
              f"{_mean([r['mt1']['score_mean'][i] for r in scored]):>7.3f} "
              f"{_mean([r['mt1']['score_best'][i] for r in scored]):>7.3f} "
              f"{_mean([r['mt1']['score_min'][i] for r in scored]):>7.3f} "
              f"{corr:>7.3f} {frac:>6.1%} "
              f"{_mean([r['mt1']['pool_stats'][i][0] for r in scored]):>7.1f} "
              f"{_mean([r['mt1']['pool_stats'][i][3] for r in scored]):>5.1f}")

    print('\n  corr = prediction on day t vs the P&L realised on t+1 — the only column here that')
    print('  selection on noise cannot manufacture. A pool can hold score near 0.5 with corr 0.')


def print_target_summary(industries, recs, industry_filter=None):
    """What MT1 is being asked to predict. Worth reading before any score is believed."""
    idx = _ind_indices(industries, industry_filter)
    print('\n── Target ───────────────────────────────────────────────────────────────────')
    print(f"{'industry':<16} {'mean':>10} {'sd':>10} {'mean/sd':>8} {'neg':>6} "
          f"{'|base err|':>11} {'floor':>9}")
    for i in idx:
        acts = [r['mt1']['actual'][i] for r in recs if r['mt1']['score_mean'][i] > 0]
        if len(acts) < 2:
            continue
        mu = _mean(acts)
        sd = math.sqrt(sum((a - mu) ** 2 for a in acts) / (len(acts) - 1))
        base_err = _mean([abs(r['mt1']['actual'][i] - r['mt1']['baseline'][i])
                          for r in recs if r['mt1']['score_mean'][i] > 0])
        print(f'{industries[i]:<16} {mu:>10.1f} {sd:>10.1f} '
              f'{(mu / sd if sd else float("nan")):>8.3f} '
              f'{sum(1 for a in acts if a < 0) / len(acts):>6.1%} {base_err:>11.1f} '
              f"{_mean([r['mt1']['floor'][i] for r in recs if r['mt1']['score_mean'][i] > 0]):>9.1f}")


def print_pool_summary(industries, recs, industry_filter=None):
    """Lifecycle: is the pool turning over, and is one lineage eating it?"""
    idx = _ind_indices(industries, industry_filter)
    if not recs:
        return
    print('\n── Pool lifecycle ───────────────────────────────────────────────────────────')
    print(f"{'industry':<16} {'mature':>7} {'cull/d':>7} {'lin_max':>8} {'lin_n':>6} "
          f"{'retire':>7}   {'  '.join(f'{b:>7}' for b in LIFE_BUCKETS)}")
    last = recs[-1]
    for i in idx:
        st = [r['mt1']['pool_stats'][i] for r in recs if r['mt1']['pool_stats'][i][0] > 0]
        if not st:
            continue
        life = last['mt1']['life'][i]
        print(f'{industries[i]:<16} {_mean([s[0] for s in st]):>7.1f} '
              f'{_mean([s[1] for s in st]):>7.2f} {_mean([s[2] for s in st]):>8.1f} '
              f'{_mean([s[3] for s in st]):>6.1f} {_mean([s[4] for s in st]):>7.1f}   '
              + '  '.join(f'{v:>7.0f}' for v in life))
    print('\n  lin_max is the largest lineage. Above 25 of 200 it is barred from breeding until it')
    print('  falls back under 20 — if it sits at the cap the pool is a monoculture in all but name.')


def print_mt2_summary(recs):
    if not recs:
        return
    print('\n── MT2 ──────────────────────────────────────────────────────────────────────')
    graded = [r for r in recs if r['mt2']['ideal_pts'] != 0]
    if not graded:
        print('  no graded MT2 days')
        return
    ideal = _mean([r['mt2']['ideal_pts'] for r in graded])
    print(f"  days graded     {len(graded)}")
    print(f"  slot0 pts       {_mean([r['mt2']['slot0_pts'] for r in graded]):>8.3f}")
    print(f"  best pts        {_mean([r['mt2']['best_pts'] for r in graded]):>8.3f}")
    print(f"  ideal pts       {ideal:>8.3f}   (ceiling)")
    print(f"  consensus flat  {_mean([r['mt2']['consensus_flat'] for r in graded]):>8.3f}")
    print(f"  consensus wtd   {_mean([r['mt2']['consensus_wtd'] for r in graded]):>8.3f}")
    print(f"  slot0 on pf     {_mean([r['mt2']['slot0_pts_pf'] for r in graded]):>8.3f}   (trained objective)")
    print(f"  slot0 on mkt    {_mean([r['mt2']['slot0_pts_mkt'] for r in graded]):>8.3f}   (diagnostic)")
    print(f"  injections      {sum(r['mt2']['injected'] for r in graded)}")


def main():
    parser = argparse.ArgumentParser(description='Summarize mt_training_log.bin (V12)')
    parser.add_argument('log', help='Path to mt_training_log.bin')
    parser.add_argument('--pass', dest='passnum', type=int, default=None,
                        help='Only this pass (1-based)')
    parser.add_argument('--industry', default=None, help='Substring filter on industry name')
    parser.add_argument('--per-day', action='store_true', help='Print every day, not a summary')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f'{args.log}: not found')
    industries, records = parse_log(args.log)
    if not records:
        sys.exit(f'{args.log}: header is valid but holds no records')

    by_pass = defaultdict(list)
    for r in records:
        by_pass[r['pass']].append(r)

    print(f'{args.log}: {len(records)} records, {len(by_pass)} pass(es), '
          f'days {records[0]["day"]}-{records[-1]["day"]}')

    passes = sorted(by_pass)
    if args.passnum is not None:
        if args.passnum - 1 not in by_pass:
            sys.exit(f'No pass {args.passnum}. Present: {[p + 1 for p in passes]}')
        passes = [args.passnum - 1]

    for p in passes:
        recs = by_pass[p]
        print(f'\n{"=" * 78}\nPASS {p + 1} — {len(recs)} days '
              f'({recs[0]["day"]}-{recs[-1]["day"]})\n{"=" * 78}')
        if args.per_day:
            print_per_day(industries, recs, args.industry)
        print_target_summary(industries, recs, args.industry)
        print_mt1_summary(industries, recs, args.industry)
        print_pool_summary(industries, recs, args.industry)
        print_mt2_summary(recs)


if __name__ == '__main__':
    main()
