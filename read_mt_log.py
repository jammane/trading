#!/usr/bin/env python3
"""
read_mt_log.py — Read and summarize mt_training_log.bin from training_v4_cpp.

Usage:
  python read_mt_log.py /path/to/mt_training_log.bin
  python read_mt_log.py /path/to/mt_training_log.bin --pass 2        # single pass only
  python read_mt_log.py /path/to/mt_training_log.bin --industry tech  # filter industry
"""

import argparse
import math
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
RECORD_SIZE_V6  = 1044  # added mt1_dir_injected[12] — direction collapse injection flags
RECORD_SIZE_V7  = 1244  # added mt1_slot0_act[12][4] + mt2_consensus flat/wtd
RECORD_SIZE_V8  = 1252  # added mt2_slot0_pts_pf + mt2_slot0_pts_mkt (dual-graded deployed slot0)
RECORD_SIZE_V9  = 1300  # added mt1_actual_d[12] — the MT1 target, so runs can be re-graded offline.
                        # V9 records are also written PER BLOCK-DAY, not once per 25-day block.
RECORD_SIZE_V10 = 1684  # added mt1_oos_act[12][4] + mt1_skill[12][4]. The OOS activations come from
                        # the head0/tail0 snapshot taken at BLOCK START, so the model that produced
                        # them never saw the day it is graded on — mt1_slot0_act is in-sample by
                        # construction and cannot be used to measure skill. mt1_dir_injected also
                        # carries real values again from V10 (it was hardcoded to 0 through V9).
RECORD_SIZE_V11 = 2212  # added mt1_dir_stats[12][6] + mt1_dir_life[12][5] — the direction pool's
                        # forward-accumulation lifecycle (v0.5.0.0): mature fraction, models culled,
                        # max/count of lineage shares, mean secondary score, mean retirement age,
                        # and the cumulative retirement-age histogram.
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
RECORD_FMT_V5 = '<II' + 'f'*12 * 21 + 'fff' + 'B3x'       # + dir_correct_dbl[12]
RECORD_FMT_V6 = '<II' + 'f'*12 * 21 + 'fff' + 'B3x' + '12B'  # + mt1_dir_injected[12]
RECORD_FMT_V7 = '<II' + 'f'*12 * 21 + 'fff' + 'B3x' + '12B' + '50f'  # + slot0_act[48] + consensus[2]
RECORD_FMT_V8 = '<II' + 'f'*12 * 21 + 'fff' + 'B3x' + '12B' + '52f'  # + slot0_pts_pf/mkt
RECORD_FMT_V9 = '<II' + 'f'*12 * 21 + 'fff' + 'B3x' + '12B' + '52f' + '12f'  # + mt1_actual_d[12]
RECORD_FMT_V10 = RECORD_FMT_V9 + '96f'   # + mt1_oos_act[12][4] + mt1_skill[12][4]
RECORD_FMT_V11 = RECORD_FMT_V10 + '132f'  # + mt1_dir_stats[12][6] + mt1_dir_life[12][5]
assert struct.calcsize(RECORD_FMT_V11) == RECORD_SIZE_V11
assert struct.calcsize(RECORD_FMT_V10) == RECORD_SIZE_V10
assert struct.calcsize(RECORD_FMT_V9) == RECORD_SIZE_V9
assert struct.calcsize(RECORD_FMT_V8) == RECORD_SIZE_V8
assert struct.calcsize(RECORD_FMT_V7) == RECORD_SIZE_V7
assert struct.calcsize(RECORD_FMT_V1) == RECORD_SIZE_V1
assert struct.calcsize(RECORD_FMT_V2) == RECORD_SIZE_V2
assert struct.calcsize(RECORD_FMT_V3) == RECORD_SIZE_V3
assert struct.calcsize(RECORD_FMT_V4) == RECORD_SIZE_V4
assert struct.calcsize(RECORD_FMT_V5) == RECORD_SIZE_V5
assert struct.calcsize(RECORD_FMT_V6) == RECORD_SIZE_V6


# record layout version -> (record size, struct format)
_LAYOUTS = {
    11: (RECORD_SIZE_V11, RECORD_FMT_V11),
    10: (RECORD_SIZE_V10, RECORD_FMT_V10), 9: (RECORD_SIZE_V9, RECORD_FMT_V9),
    8:  (RECORD_SIZE_V8,  RECORD_FMT_V8),  7: (RECORD_SIZE_V7, RECORD_FMT_V7),
    6:  (RECORD_SIZE_V6,  RECORD_FMT_V6),  5: (RECORD_SIZE_V5, RECORD_FMT_V5),
    4:  (RECORD_SIZE_V4,  RECORD_FMT_V4),  3: (RECORD_SIZE_V3, RECORD_FMT_V3),
    2:  (RECORD_SIZE_V2,  RECORD_FMT_V2),  1: (RECORD_SIZE_V1, RECORD_FMT_V1),
}


def parse_log(path):
    file_size = os.path.getsize(path)
    data_size = file_size - HEADER_SIZE

    _nan12 = [float('nan')] * N_IND

    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
        if len(hdr) < HEADER_SIZE:
            sys.exit('ERROR: file too short for header')
        magic, version, n_ind, _ = struct.unpack('<IIII', hdr)
        if magic != MAGIC:
            sys.exit(f'ERROR: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})')

        # Trust the header version when the data actually divides by that layout. Sniffing on size
        # alone is ambiguous — 421 V9 records divide evenly by the V10 record size — and the writer
        # has always stamped the true version; it simply was never consulted. Falls back to
        # newest-that-fits for files whose header version predates this table.
        rec_size = fmt = ver = None
        hdr_layout = version + 1           # MT_LOG_VERSION n writes record layout V(n+1)
        if hdr_layout in _LAYOUTS and data_size > 0 and data_size % _LAYOUTS[hdr_layout][0] == 0:
            rec_size, fmt = _LAYOUTS[hdr_layout]
            ver = hdr_layout
        else:
            for v in sorted(_LAYOUTS, reverse=True):
                size, vfmt = _LAYOUTS[v]
                if data_size > 0 and data_size % size == 0:
                    rec_size, fmt, ver = size, vfmt, v
                    break
        if ver is None:
            sys.exit(f'ERROR: data size {data_size} not divisible by any known record size '
                     f'({sorted(s for s, _ in _LAYOUTS.values())})')

        records = []
        while True:
            raw = f.read(rec_size)
            if len(raw) < rec_size:
                break
            vals = struct.unpack(fmt, raw)
            if ver >= 9:
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
                    'mt1_dir_injected':      list(vals[258:270]),
                    'mt1_slot0_act':         [list(vals[270+i*4:274+i*4]) for i in range(12)],
                    'mt2_consensus_flat_pts': vals[318],
                    'mt2_consensus_wtd_pts':  vals[319],
                    'mt2_slot0_pts_pf':       vals[320],
                    'mt2_slot0_pts_mkt':      vals[321],
                    'mt1_actual_d':           list(vals[322:334]),
                }
                if ver >= 10:
                    # Out-of-sample twin of mt1_slot0_act (block-start snapshot) + the per-channel
                    # skill scores it backs. Use THESE, not mt1_slot0_act, to judge skill.
                    rec['mt1_oos_act'] = [list(vals[334 + i*4:338 + i*4]) for i in range(12)]
                    rec['mt1_skill']   = [list(vals[382 + i*4:386 + i*4]) for i in range(12)]
                else:
                    rec['mt1_oos_act'] = [[float('nan')] * 4 for _ in range(12)]
                    rec['mt1_skill']   = [[float('nan')] * 4 for _ in range(12)]
                if ver >= 11:
                    # Direction pool lifecycle. stats = {mature_frac, culled, lineage_max,
                    # lineage_n, mean_secondary, mean_retire_age}; life = cumulative age histogram.
                    rec['mt1_dir_stats'] = [list(vals[430 + i*6:436 + i*6]) for i in range(12)]
                    rec['mt1_dir_life']  = [list(vals[502 + i*5:507 + i*5]) for i in range(12)]
                else:
                    rec['mt1_dir_stats'] = [[float('nan')] * 6 for _ in range(12)]
                    rec['mt1_dir_life']  = [[float('nan')] * 5 for _ in range(12)]
            elif ver == 8:
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
                    'mt1_dir_injected':      list(vals[258:270]),
                    'mt1_slot0_act':         [list(vals[270+i*4:274+i*4]) for i in range(12)],
                    'mt2_consensus_flat_pts': vals[318],
                    'mt2_consensus_wtd_pts':  vals[319],
                    'mt2_slot0_pts_pf':       vals[320],
                    'mt2_slot0_pts_mkt':      vals[321],
                }
            elif ver == 7:
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
                    'mt1_dir_injected':      list(vals[258:270]),
                    'mt1_slot0_act':         [list(vals[270+i*4:274+i*4]) for i in range(12)],
                    'mt2_consensus_flat_pts': vals[318],
                    'mt2_consensus_wtd_pts':  vals[319],
                }
            elif ver == 6:
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
                    'mt1_dir_injected':      list(vals[258:270]),
                }
            elif ver == 5:
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
                    'mt1_dir_injected':      [0] * N_IND,
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
                    'mt1_dir_injected':      [0] * N_IND,
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
                    'mt1_dir_injected':  [0] * N_IND,
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
                    'mt1_dir_injected':  [0] * N_IND,
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
                    'mt1_dir_injected':  [0] * N_IND,
                }
            # Defaults for fields older formats lack, so consumers can read them unconditionally.
            rec.setdefault('mt1_actual_d', _nan12)          # V9+
            rec.setdefault('mt1_slot0_act', [[float('nan')] * 4 for _ in range(N_IND)])  # V7+
            records.append(rec)
    return records


def _mean(lst):
    return sum(lst) / len(lst) if lst else float('nan')


def _corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = _mean(xs), _mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (sx * sy) if sx > 0 and sy > 0 else float('nan')


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
    header = f'  {"day":>4}  {"mt2_best":>9} {"mt2_s0":>9} {"mt2_ideal":>9} {"m2i":>3} {"dir_inj":>7}'
    for i, name in enumerate(INDUSTRY_NAMES):
        if industry_filter and industry_filter.lower() not in name:
            continue
        short = name[:10]
        header += f'  {short+":s0":>14} {short+":max":>14} {short+":mn":>14}'
        if has_min:
            header += f' {short+":min":>14}'
    print(header)
    for r in recs:
        dir_inj_inds = [INDUSTRY_NAMES[i][:4] for i in range(N_IND) if r['mt1_dir_injected'][i]]
        dir_inj_str = ','.join(dir_inj_inds) if dir_inj_inds else '.'
        line = (f'  {r["day"]:>4}  {r["mt2_best_pts"]:>+9.2f} {r["mt2_slot0_pts"]:>+9.2f}'
                f' {r["mt2_ideal_pts"]:>+9.2f} {"Y" if r["mt2_injected"] else ".":>3}'
                f' {dir_inj_str:>7}')
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


def _print_oos_summary(recs, industry_filter=None):
    """Leak-free (V10+) section: grade the block-start snapshot, which never saw the day it is
    scored on, against the same target. Everything else in this report is in-sample — the model
    behind mt1_slot0_act was selected on a window CONTAINING that day — so this block is the only
    part that measures skill rather than fit. The in-sample twin is printed alongside; a large gap
    between the two IS the overfit."""
    import math
    if math.isnan(recs[0]['mt1_oos_act'][0][0]):
        return   # pre-V10 log

    SCALE = 10000.0
    print('  MT1 OUT-OF-SAMPLE (block-start snapshot) vs in-sample — the gap is the overfit')
    print(f'    {"industry":<24s} {"dir OOS":>8s} {"dir IS":>7s} | {"|d| OOS":>8s} {"|d| IS":>7s} |'
          f' {"skill: dir":>10s} {"acc":>6s} {"rng":>6s} {"cfd":>6s}  {"inj":>4s}')
    tot_o = tot_i = tot_n = 0
    for i, name in enumerate(INDUSTRY_NAMES):
        if industry_filter and industry_filter.lower() not in name:
            continue
        oh = ih = n = 0
        ox, oy, ix = [], [], []
        for r in recs:
            a = r['mt1_actual_d'][i]
            oh += (r['mt1_oos_act'][i][0] >= 0.5) == (a >= 0)
            ih += (r['mt1_slot0_act'][i][0] >= 0.5) == (a >= 0)
            ox.append(abs(r['mt1_oos_act'][i][1]) * SCALE)
            ix.append(abs(r['mt1_slot0_act'][i][1]) * SCALE)
            oy.append(abs(a))
            n += 1
        tot_o += oh; tot_i += ih; tot_n += n
        sk = [_mean([r['mt1_skill'][i][c] for r in recs]) for c in range(4)]
        inj = sum(r['mt1_dir_injected'][i] for r in recs)
        print(f'    {name:<24s} {100*oh/n:7.2f}% {100*ih/n:6.2f}% | {_corr(ox, oy):8.3f}'
              f' {_corr(ix, oy):7.3f} | {sk[0]:10.3f} {sk[1]:6.3f} {sk[2]:6.3f} {sk[3]:6.3f}'
              f'  {inj:4d}')
    if tot_n:
        print(f'    {"OVERALL":<24s} {100*tot_o/tot_n:7.2f}% {100*tot_i/tot_n:6.2f}%'
              f'   (50.00% = no skill; skill 0 = no better than a constant baseline)')


def _print_dir_pool_summary(recs, industry_filter=None):
    """Direction pool lifecycle (V11+). Under forward accumulation the pool is 200 persistent
    individuals rather than 183-of-200 regenerated daily, so these are the numbers that say whether
    the lifecycle is behaving: mature fraction should settle near 60%, culls near 10/day, and
    lineage share is the direct read on monoculture that we previously had to infer from spread."""
    if math.isnan(recs[0]['mt1_dir_stats'][0][0]):
        return   # pre-V11 log
    print('  MT1 DIRECTION POOL lifecycle (forward accumulation)')
    print(f'    {"industry":<24s} {"mature":>7s} {"cull/d":>7s} {"lin max":>8s} {"lin n":>6s}'
          f' {"mean sec":>9s} {"retire age":>11s}  {"age histogram 8/16/32/64/128+":>32s}')
    for i, name in enumerate(INDUSTRY_NAMES):
        if industry_filter and industry_filter.lower() not in name:
            continue
        st = [_mean([r['mt1_dir_stats'][i][k] for r in recs]) for k in range(6)]
        life = recs[-1]['mt1_dir_life'][i]          # cumulative — last record holds the totals
        tot = sum(life) or 1.0
        hist = '/'.join(f'{100*v/tot:.0f}' for v in life)
        print(f'    {name:<24s} {100*st[0]:6.1f}% {st[1]:7.2f} {100*st[2]:7.1f}% {st[3]:6.1f}'
              f' {st[4]:9.3f} {st[5]:11.1f}  {hist:>32s}')
    print('    (mature target ~60%, cull ~10/day; lin max is the largest lineage share of the pool)')


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

    _print_oos_summary(recs, industry_filter)
    _print_dir_pool_summary(recs, industry_filter)

    print('  MT1 composite (early→mid→late) — slot0 | max | mean | min')
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
                inj_str = ''
                if comp_label == 'direction':
                    n_inj = sum(r['mt1_dir_injected'][i] for r in recs)
                    if n_inj:
                        inj_str = f'  [dir-inj={n_inj}]'
                print(f'    {name:<28s}'
                      f'  slot0: {s0_e:.3f}→{s0_m:.3f}→{s0_l:.3f}'
                      f'  max: {mx_e:.3f}→{mx_m:.3f}→{mx_l:.3f}'
                      f'  mean: {mn_e:.3f}→{mn_m:.3f}→{mn_l:.3f}'
                      f'  min: {mi_e:.3f}→{mi_m:.3f}→{mi_l:.3f}'
                      f'{inj_str}')


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
