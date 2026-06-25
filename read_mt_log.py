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

HEADER_SIZE     = 16
RECORD_SIZE_V1  = 168   # original: best + slot0 + mean
RECORD_SIZE_V2  = 216   # added mt1_min[12]
RECORD_SIZE_V3  = 792   # added dir/rng/acc component stats (4 stats × 3 components × 12 ind)
RECORD_SIZE_V4  = 984   # added conf4 component stats (4 stats × 5 components × 12 ind)
RECORD_SIZE_V5  = 1032  # added mt1_dir_correct_dbl[12] — mean n_correct_dbl for direction pool
MAGIC           = 0x4D543132   # 'MT12'
N_IND           = 12

INDUSTRY_NAMES = [
    'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
    'consumer_services', 'health_care', 'industrials', 'consumer_staples',
    'energy', 'utilities', 'real_estate', 'materials',
]

# Record layout v1 (168 bytes):
#   uint32 pass_num, actual_day; float mt1_best/slot0/mean[12]; float mt2×3; uint8 inj; pad[3]
#
# Record layout v2 (216 bytes — adds mt1_min[12]):
#   ... same as v1 + float mt1_min[12] before MT2 fields
#
# Record layout v3 (792 bytes — adds direction/range/accuracy component stats):
#   uint32 pass_num, actual_day
#   float  mt1_{best,slot0,mean,min}[12]      — composite
#   float  mt1_dir_{best,slot0,mean,min}[12]  — direction component
#   float  mt1_rng_{best,slot0,mean,min}[12]  — range component
#   float  mt1_acc_{best,slot0,mean,min}[12]  — accuracy component
#   float  mt2_best_pts, mt2_slot0_pts, mt2_ideal_pts
#   uint8  mt2_injected; uint8 pad[3]
#
# Record layout v4 (984 bytes — adds conf4 component stats):
#   ... same as v3 + float mt1_conf4_{best,slot0,mean,min}[12] before MT2 fields

RECORD_FMT_V1 = '<II' + 'f'*12 + 'f'*12 + 'f'*12 + 'fff' + 'B3x'
RECORD_FMT_V2 = '<II' + 'f'*12 + 'f'*12 + 'f'*12 + 'f'*12 + 'fff' + 'B3x'
RECORD_FMT_V3 = '<II' + 'f'*12 * 16 + 'fff' + 'B3x'  # 4 stats × 4 components × 12 ind
RECORD_FMT_V4 = '<II' + 'f'*12 * 20 + 'fff' + 'B3x'  # 4 stats × 5 components × 12 ind
RECORD_FMT_V5 = '<II' + 'f'*12 * 21 + 'fff' + 'B3x'  # + dir_correct_dbl[12]
assert struct.calcsize(RECORD_FMT_V1) == RECORD_SIZE_V1
assert struct.calcsize(RECORD_FMT_V2) == RECORD_SIZE_V2
assert struct.calcsize(RECORD_FMT_V3) == RECORD_SIZE_V3
assert struct.calcsize(RECORD_FMT_V4) == RECORD_SIZE_V4
assert struct.calcsize(RECORD_FMT_V5) == RECORD_SIZE_V5


def parse_log(path):
    file_size = os.path.getsize(path)
    data_size = file_size - HEADER_SIZE
    fits_v5 = data_size > 0 and data_size % RECORD_SIZE_V5 == 0
    fits_v4 = data_size > 0 and data_size % RECORD_SIZE_V4 == 0
    fits_v3 = data_size > 0 and data_size % RECORD_SIZE_V3 == 0
    fits_v2 = data_size > 0 and data_size % RECORD_SIZE_V2 == 0
    fits_v1 = data_size > 0 and data_size % RECORD_SIZE_V1 == 0
    if not fits_v1 and not fits_v2 and not fits_v3 and not fits_v4 and not fits_v5:
        sys.exit(f'ERROR: data size {data_size} not divisible by any known record size '
                 f'({RECORD_SIZE_V1}, {RECORD_SIZE_V2}, {RECORD_SIZE_V3}, '
                 f'{RECORD_SIZE_V4}, or {RECORD_SIZE_V5})')
    # Prefer newest format that fits
    if fits_v5:
        rec_size, fmt, ver = RECORD_SIZE_V5, RECORD_FMT_V5, 5
    elif fits_v4:
        rec_size, fmt, ver = RECORD_SIZE_V4, RECORD_FMT_V4, 4
    elif fits_v3:
        rec_size, fmt, ver = RECORD_SIZE_V3, RECORD_FMT_V3, 3
    elif fits_v2:
        rec_size, fmt, ver = RECORD_SIZE_V2, RECORD_FMT_V2, 2
    else:
        rec_size, fmt, ver = RECORD_SIZE_V1, RECORD_FMT_V1, 1

    _nan12 = [float('nan')] * N_IND

    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
        if len(hdr) < HEADER_SIZE:
            sys.exit('ERROR: file too short for header')
        magic, version, n_ind, _ = struct.unpack('<IIII', hdr)
        if magic != MAGIC:
            sys.exit(f'ERROR: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})')

        records = []
        while True:
            raw = f.read(rec_size)
            if len(raw) < rec_size:
                break
            vals = struct.unpack(fmt, raw)
            if ver == 5:
                rec = {
                    'pass':                  vals[0],
                    'day':                   vals[1],
                    'mt1_best':              list(vals[2:14]),
                    'mt1_slot0':             list(vals[14:26]),
                    'mt1_mean':              list(vals[26:38]),
                    'mt1_min':               list(vals[38:50]),
                    'mt1_dir_best':          list(vals[50:62]),
                    'mt1_dir_slot0':         list(vals[62:74]),
                    'mt1_dir_mean':          list(vals[74:86]),
                    'mt1_dir_min':           list(vals[86:98]),
                    'mt1_rng_best':          list(vals[98:110]),
                    'mt1_rng_slot0':         list(vals[110:122]),
                    'mt1_rng_mean':          list(vals[122:134]),
                    'mt1_rng_min':           list(vals[134:146]),
                    'mt1_acc_best':          list(vals[146:158]),
                    'mt1_acc_slot0':         list(vals[158:170]),
                    'mt1_acc_mean':          list(vals[170:182]),
                    'mt1_acc_min':           list(vals[182:194]),
                    'mt1_conf4_best':        list(vals[194:206]),
                    'mt1_conf4_slot0':       list(vals[206:218]),
                    'mt1_conf4_mean':        list(vals[218:230]),
                    'mt1_conf4_min':         list(vals[230:242]),
                    'mt1_dir_correct_dbl':   list(vals[242:254]),
                    'mt2_best_pts':          vals[254],
                    'mt2_slot0_pts':         vals[255],
                    'mt2_ideal_pts':         vals[256],
                    'mt2_injected':          vals[257],
                }
            elif ver == 4:
                rec = {
                    'pass':                  vals[0],
                    'day':                   vals[1],
                    'mt1_best':              list(vals[2:14]),
                    'mt1_slot0':             list(vals[14:26]),
                    'mt1_mean':              list(vals[26:38]),
                    'mt1_min':               list(vals[38:50]),
                    'mt1_dir_best':          list(vals[50:62]),
                    'mt1_dir_slot0':         list(vals[62:74]),
                    'mt1_dir_mean':          list(vals[74:86]),
                    'mt1_dir_min':           list(vals[86:98]),
                    'mt1_rng_best':          list(vals[98:110]),
                    'mt1_rng_slot0':         list(vals[110:122]),
                    'mt1_rng_mean':          list(vals[122:134]),
                    'mt1_rng_min':           list(vals[134:146]),
                    'mt1_acc_best':          list(vals[146:158]),
                    'mt1_acc_slot0':         list(vals[158:170]),
                    'mt1_acc_mean':          list(vals[170:182]),
                    'mt1_acc_min':           list(vals[182:194]),
                    'mt1_conf4_best':        list(vals[194:206]),
                    'mt1_conf4_slot0':       list(vals[206:218]),
                    'mt1_conf4_mean':        list(vals[218:230]),
                    'mt1_conf4_min':         list(vals[230:242]),
                    'mt1_dir_correct_dbl':   [0.0] * N_IND,
                    'mt2_best_pts':          vals[242],
                    'mt2_slot0_pts':         vals[243],
                    'mt2_ideal_pts':         vals[244],
                    'mt2_injected':          vals[245],
                }
            elif ver == 3:
                rec = {
                    'pass':              vals[0],
                    'day':               vals[1],
                    'mt1_best':          list(vals[2:14]),
                    'mt1_slot0':         list(vals[14:26]),
                    'mt1_mean':          list(vals[26:38]),
                    'mt1_min':           list(vals[38:50]),
                    'mt1_dir_best':      list(vals[50:62]),
                    'mt1_dir_slot0':     list(vals[62:74]),
                    'mt1_dir_mean':      list(vals[74:86]),
                    'mt1_dir_min':       list(vals[86:98]),
                    'mt1_rng_best':      list(vals[98:110]),
                    'mt1_rng_slot0':     list(vals[110:122]),
                    'mt1_rng_mean':      list(vals[122:134]),
                    'mt1_rng_min':       list(vals[134:146]),
                    'mt1_acc_best':      list(vals[146:158]),
                    'mt1_acc_slot0':     list(vals[158:170]),
                    'mt1_acc_mean':      list(vals[170:182]),
                    'mt1_acc_min':       list(vals[182:194]),
                    'mt1_conf4_best':        _nan12, 'mt1_conf4_slot0': _nan12,
                    'mt1_conf4_mean':        _nan12, 'mt1_conf4_min':   _nan12,
                    'mt1_dir_correct_dbl':   [0.0] * N_IND,
                    'mt2_best_pts':          vals[194],
                    'mt2_slot0_pts':     vals[195],
                    'mt2_ideal_pts':     vals[196],
                    'mt2_injected':      vals[197],
                }
            elif ver == 2:
                rec = {
                    'pass':              vals[0],
                    'day':               vals[1],
                    'mt1_best':          list(vals[2:14]),
                    'mt1_slot0':         list(vals[14:26]),
                    'mt1_mean':          list(vals[26:38]),
                    'mt1_min':           list(vals[38:50]),
                    'mt1_dir_best':      _nan12, 'mt1_dir_slot0': _nan12,
                    'mt1_dir_mean':      _nan12, 'mt1_dir_min':   _nan12,
                    'mt1_rng_best':      _nan12, 'mt1_rng_slot0': _nan12,
                    'mt1_rng_mean':      _nan12, 'mt1_rng_min':   _nan12,
                    'mt1_acc_best':      _nan12, 'mt1_acc_slot0': _nan12,
                    'mt1_acc_mean':      _nan12, 'mt1_acc_min':   _nan12,
                    'mt1_conf4_best':        _nan12, 'mt1_conf4_slot0': _nan12,
                    'mt1_conf4_mean':        _nan12, 'mt1_conf4_min':   _nan12,
                    'mt1_dir_correct_dbl':   [0.0] * N_IND,
                    'mt2_best_pts':          vals[50],
                    'mt2_slot0_pts':     vals[51],
                    'mt2_ideal_pts':     vals[52],
                    'mt2_injected':      vals[53],
                }
            else:  # v1
                rec = {
                    'pass':              vals[0],
                    'day':               vals[1],
                    'mt1_best':          list(vals[2:14]),
                    'mt1_slot0':         list(vals[14:26]),
                    'mt1_mean':          list(vals[26:38]),
                    'mt1_min':           _nan12,
                    'mt1_dir_best':      _nan12, 'mt1_dir_slot0': _nan12,
                    'mt1_dir_mean':      _nan12, 'mt1_dir_min':   _nan12,
                    'mt1_rng_best':      _nan12, 'mt1_rng_slot0': _nan12,
                    'mt1_rng_mean':      _nan12, 'mt1_rng_min':   _nan12,
                    'mt1_acc_best':      _nan12, 'mt1_acc_slot0': _nan12,
                    'mt1_acc_mean':      _nan12, 'mt1_acc_min':   _nan12,
                    'mt1_conf4_best':        _nan12, 'mt1_conf4_slot0': _nan12,
                    'mt1_conf4_mean':        _nan12, 'mt1_conf4_min':   _nan12,
                    'mt1_dir_correct_dbl':   [0.0] * N_IND,
                    'mt2_best_pts':          vals[38],
                    'mt2_slot0_pts':     vals[39],
                    'mt2_ideal_pts':     vals[40],
                    'mt2_injected':      vals[41],
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


def print_per_day(pass_num, recs, industry_filter=None):
    import math
    has_min = not math.isnan(recs[0]['mt1_min'][0])
    print(f'\n{"="*70}')
    print(f'  Pass {pass_num}  per-day  ({len(recs)} days, day {recs[0]["day"]} – {recs[-1]["day"]})')
    print(f'{"="*70}')
    header = f'  {"day":>4}  {"mt2_best":>9} {"mt2_s0":>9} {"mt2_ideal":>9} {"inj":>3}'
    for i, name in enumerate(INDUSTRY_NAMES):
        if industry_filter and industry_filter.lower() not in name:
            continue
        short = name[:10]
        header += f'  {short+":s0":>14} {short+":max":>14} {short+":mn":>14}'
        if has_min:
            header += f' {short+":min":>14}'
    print(header)
    for r in recs:
        line = (f'  {r["day"]:>4}  {r["mt2_best_pts"]:>+9.2f} {r["mt2_slot0_pts"]:>+9.2f}'
                f' {r["mt2_ideal_pts"]:>+9.2f} {"Y" if r["mt2_injected"] else ".":>3}')
        for i, name in enumerate(INDUSTRY_NAMES):
            if industry_filter and industry_filter.lower() not in name:
                continue
            s0  = r['mt1_slot0'][i]
            mx  = r['mt1_best'][i]
            mn  = r['mt1_mean'][i]
            mi  = r['mt1_min'][i]
            line += f'  {s0:>14.3f} {mx:>14.3f} {mn:>14.3f}'
            if has_min:
                line += f' {mi:>14.3f}'
        print(line)


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

    # MT1 per-industry trends
    import math
    has_comp = not math.isnan(recs[0]['mt1_min'][0])
    has_comp_v3 = not math.isnan(recs[0]['mt1_dir_best'][0])

    print(f'  MT1 composite (early→mid→late) — slot0 | max | mean | min')
    for i, name in enumerate(INDUSTRY_NAMES):
        if industry_filter and industry_filter.lower() not in name:
            continue
        s0_e, s0_m, s0_l = _thirds(recs, lambda r, ii=i: r['mt1_slot0'][ii])
        mx_e, mx_m, mx_l = _thirds(recs, lambda r, ii=i: r['mt1_best'][ii])
        mn_e, mn_m, mn_l = _thirds(recs, lambda r, ii=i: r['mt1_mean'][ii])
        mi_e, mi_m, mi_l = _thirds(recs, lambda r, ii=i: r['mt1_min'][ii])
        min_str = f'  min: {mi_e:.3f}→{mi_m:.3f}→{mi_l:.3f}' if has_comp else ''
        print(f'    {name:<28s}'
              f'  slot0: {s0_e:.3f}→{s0_m:.3f}→{s0_l:.3f}'
              f'  max: {mx_e:.3f}→{mx_m:.3f}→{mx_l:.3f}'
              f'  mean: {mn_e:.3f}→{mn_m:.3f}→{mn_l:.3f}'
              f'{min_str}')

    has_conf4 = not math.isnan(recs[0]['mt1_conf4_best'][0])

    if has_comp_v3:
        components = [
            ('direction',   'mt1_dir_best',   'mt1_dir_slot0',   'mt1_dir_mean',   'mt1_dir_min'),
            ('range',       'mt1_rng_best',   'mt1_rng_slot0',   'mt1_rng_mean',   'mt1_rng_min'),
            ('accuracy',    'mt1_acc_best',   'mt1_acc_slot0',   'mt1_acc_mean',   'mt1_acc_min'),
        ]
        if has_conf4:
            components.append(
                ('confidence', 'mt1_conf4_best', 'mt1_conf4_slot0', 'mt1_conf4_mean', 'mt1_conf4_min')
            )
        for comp_label, best_key, s0_key, mean_key, min_key in components:
            print(f'  MT1 {comp_label} component (early→mid→late) — slot0 | max | mean | min')
            for i, name in enumerate(INDUSTRY_NAMES):
                if industry_filter and industry_filter.lower() not in name:
                    continue
                s0_e, s0_m, s0_l = _thirds(recs, lambda r, ii=i, k=s0_key:   r[k][ii])
                mx_e, mx_m, mx_l = _thirds(recs, lambda r, ii=i, k=best_key:  r[k][ii])
                mn_e, mn_m, mn_l = _thirds(recs, lambda r, ii=i, k=mean_key:  r[k][ii])
                mi_e, mi_m, mi_l = _thirds(recs, lambda r, ii=i, k=min_key:   r[k][ii])
                print(f'    {name:<28s}'
                      f'  slot0: {s0_e:.3f}→{s0_m:.3f}→{s0_l:.3f}'
                      f'  max: {mx_e:.3f}→{mx_m:.3f}→{mx_l:.3f}'
                      f'  mean: {mn_e:.3f}→{mn_m:.3f}→{mn_l:.3f}'
                      f'  min: {mi_e:.3f}→{mi_m:.3f}→{mi_l:.3f}')


def main():
    parser = argparse.ArgumentParser(description='Summarize mt_training_log.bin')
    parser.add_argument('log', help='Path to mt_training_log.bin')
    parser.add_argument('--pass', dest='passnum', type=int, default=None,
                        help='Limit output to a single pass number')
    parser.add_argument('--industry', default=None,
                        help='Substring filter for industry names')
    parser.add_argument('--per-day', action='store_true',
                        help='Print per-day numbers instead of early/mid/late summary')
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
        if args.per_day:
            print_per_day(p, by_pass[p], industry_filter=args.industry)
        else:
            print_pass_summary(p, by_pass[p], industry_filter=args.industry)

    print()


if __name__ == '__main__':
    main()
