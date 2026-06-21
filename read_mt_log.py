#!/usr/bin/env python3
"""
read_mt_log.py — Read and summarize mt_training_log.bin from training_v4_cpp.

Usage:
  python read_mt_log.py /path/to/mt_training_log.bin
  python read_mt_log.py /path/to/mt_training_log.bin --pass 2        # single pass only
  python read_mt_log.py /path/to/mt_training_log.bin --industry tech  # filter industry
"""

import argparse
import os
import struct
import sys
from collections import defaultdict

HEADER_SIZE  = 16
RECORD_SIZE  = 168
MAGIC        = 0x4D543132   # 'MT12'
N_IND        = 12

INDUSTRY_NAMES = [
    'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
    'consumer_services', 'health_care', 'industrials', 'consumer_staples',
    'energy', 'utilities', 'real_estate', 'materials',
]

# Record layout (168 bytes):
#   uint32  pass_num         (0)
#   uint32  actual_day       (4)
#   float32 mt1_best[12]     (8..55)
#   float32 mt1_slot0[12]    (56..103)
#   float32 mt1_mean[12]     (104..151)
#   float32 mt2_best_pts     (152)
#   float32 mt2_slot0_pts    (156)
#   float32 mt2_ideal_pts    (160)
#   uint8   mt2_injected     (164)
#   uint8   padding[3]       (165..167)

RECORD_FMT = '<II' + 'f'*12 + 'f'*12 + 'f'*12 + 'fff' + 'B3x'
assert struct.calcsize(RECORD_FMT) == RECORD_SIZE


def parse_log(path):
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
        if len(hdr) < HEADER_SIZE:
            sys.exit('ERROR: file too short for header')
        magic, version, n_ind, _ = struct.unpack('<IIII', hdr)
        if magic != MAGIC:
            sys.exit(f'ERROR: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})')
        records = []
        while True:
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                break
            vals = struct.unpack(RECORD_FMT, raw)
            rec = {
                'pass':         vals[0],
                'day':          vals[1],
                'mt1_best':     list(vals[2:14]),
                'mt1_slot0':    list(vals[14:26]),
                'mt1_mean':     list(vals[26:38]),
                'mt2_best_pts': vals[38],
                'mt2_slot0_pts':vals[39],
                'mt2_ideal_pts':vals[40],
                'mt2_injected': vals[41],
            }
            records.append(rec)
    return records


def _mean(lst):
    return sum(lst) / len(lst) if lst else float('nan')


def _thirds(recs, key_fn):
    n = len(recs)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    e = n // 3
    early = recs[:e or 1]
    late  = recs[-e or len(recs):]
    mid   = recs[e:n - e] if n >= 6 else recs
    return _mean([key_fn(r) for r in early]), \
           _mean([key_fn(r) for r in mid]),   \
           _mean([key_fn(r) for r in late])


def print_pass_summary(pass_num, recs, industry_filter=None):
    print(f'\n{"="*70}')
    print(f'  Pass {pass_num}  ({len(recs)} days, day {recs[0]["day"]} – {recs[-1]["day"]})')
    print(f'{"="*70}')

    # MT2 trend
    mt2_early, mt2_mid, mt2_late = _thirds(recs, lambda r: r['mt2_best_pts'])
    mt2_inj = sum(r['mt2_injected'] for r in recs)
    avg_ideal = _mean([r['mt2_ideal_pts'] for r in recs])
    print(f'  MT2 best_pts  early={mt2_early:+.2f}  mid={mt2_mid:+.2f}  late={mt2_late:+.2f}'
          f'  (ideal avg={avg_ideal:+.2f}  inj={mt2_inj}/{len(recs)})')

    # MT1 per-industry trend
    print('  MT1 slot0 score (early → late):')
    for i, name in enumerate(INDUSTRY_NAMES):
        if industry_filter and industry_filter.lower() not in name:
            continue
        e2, m2, l2 = _thirds(recs, lambda r, ii=i: r['mt1_slot0'][ii])
        best_avg   = _mean([r['mt1_best'][i] for r in recs])
        mean_avg   = _mean([r['mt1_mean'][i] for r in recs])
        print(f'    {name:<28s}  slot0: {e2:.3f}→{m2:.3f}→{l2:.3f}  '
              f'(best={best_avg:.3f}  pool_mean={mean_avg:.3f})')


def main():
    parser = argparse.ArgumentParser(description='Summarize mt_training_log.bin')
    parser.add_argument('log', help='Path to mt_training_log.bin')
    parser.add_argument('--pass', dest='passnum', type=int, default=None,
                        help='Limit output to a single pass number')
    parser.add_argument('--industry', default=None,
                        help='Substring filter for industry names')
    args = parser.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f'ERROR: file not found: {args.log}')

    records = parse_log(args.log)
    if not records:
        sys.exit('No records found in log file.')

    print(f'Log: {args.log}')
    print(f'Total records: {len(records)}')

    by_pass = defaultdict(list)
    for r in records:
        by_pass[r['pass']].append(r)

    passes = sorted(by_pass.keys())
    if args.passnum is not None:
        if args.passnum not in by_pass:
            sys.exit(f'Pass {args.passnum} not found. Available: {passes}')
        passes = [args.passnum]

    for p in passes:
        print_pass_summary(p, by_pass[p], industry_filter=args.industry)

    print()


if __name__ == '__main__':
    main()
