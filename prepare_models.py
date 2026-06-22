#!/usr/bin/env python3
"""
prepare_models.py — Convert PyTorch .pt elite models to flat float32 .bin files
for the C++ trainer (training_v4_cpp).

Run this once before the first C++ training run, or whenever you want to seed
the C++ trainer from a Python-trained checkpoint.

Usage:
  python prepare_models.py --account acct0
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch

# Shared layer definitions — single source of truth for binary layout.
# Each entry: (state_dict_key_prefix, out_size, in_size)
# For inject layers, key_prefix uses the ModuleList index notation.
STOCK_LAYER_DEFS = [
    ('fc_seed',  120,  60),
    # inject layers 0..13: fc_inject.{i}
    *[(f'fc_inject.{i}', 125 + 5 * i, 180 + 5 * i) for i in range(14)],
    ('fc_today', 300, 398),
    ('fc_flat1', 300, 300),
    ('fc_flat2', 300, 300),
    ('fc_fc1',   237, 300),
    ('fc_fc2',   174, 237),
    ('fc_fc3',   111, 174),
    ('fc_out',    48, 111),
]

MASTER_LAYER_DEFS = [
    ('fc1',    444, 444),
    ('fc2',    444, 444),
    ('fc3',    312, 444),
    ('fc4',    180, 312),
    ('fc_out',  48, 180),
]

MT1_LAYER_DEFS = [
    ('fc1',    37, 37),
    ('fc2',    29, 37),
    ('fc3',    20, 29),
    ('fc4',    12, 20),
    ('fc_out',  3, 12),
]

# MT2 layout mirrors C++ binary offsets (FC1, FC2, LSTM L1, LSTM L2, taper1-3, fc_out).
# Stored as (key, shape) rather than (prefix, out, in) because LSTM has bias-only entries.
MT2_LAYOUT = [
    ('fc1.weight',         (36, 36)),
    ('fc1.bias',           (36,)),
    ('fc2.weight',         (36, 36)),
    ('fc2.bias',           (36,)),
    ('lstm.weight_ih_l0',  (144, 3)),
    ('lstm.weight_hh_l0',  (144, 36)),
    ('lstm.bias_ih_l0',    (144,)),
    ('lstm.bias_hh_l0',    (144,)),
    ('lstm.weight_ih_l1',  (144, 36)),
    ('lstm.weight_hh_l1',  (144, 36)),
    ('lstm.bias_ih_l1',    (144,)),
    ('lstm.bias_hh_l1',    (144,)),
    ('taper1.weight',      (66, 72)),
    ('taper1.bias',        (66,)),
    ('taper2.weight',      (60, 66)),
    ('taper2.bias',        (60,)),
    ('taper3.weight',      (54, 60)),
    ('taper3.bias',        (54,)),
    ('fc_out.weight',      (48, 54)),
    ('fc_out.bias',        (48,)),
]

ELITE_POOL = 20


def state_dict_to_arr(state_dict, layer_defs):
    """Flatten a PyTorch state_dict to a contiguous float32 numpy array."""
    parts = []
    for prefix, out_size, in_size in layer_defs:
        w = state_dict[f'{prefix}.weight'].float().numpy()
        b = state_dict[f'{prefix}.bias'].float().numpy()
        assert w.shape == (out_size, in_size), \
            f'{prefix}.weight shape {w.shape} != ({out_size}, {in_size})'
        assert b.shape == (out_size,), \
            f'{prefix}.bias shape {b.shape} != ({out_size},)'
        parts.append(w.ravel())
        parts.append(b.ravel())
    return np.concatenate(parts).astype(np.float32)


def mt2_state_dict_to_arr(state_dict):
    """Flatten MT2NN state_dict to float32 array matching C++ binary layout."""
    parts = []
    for key, shape in MT2_LAYOUT:
        t = state_dict[key].float().numpy()
        assert t.shape == shape, f'{key}: expected shape {shape}, got {t.shape}'
        parts.append(t.ravel())
    return np.concatenate(parts).astype(np.float32)


def convert_industry(prefix, load_dir, output_dir, layer_defs, label):
    """Convert all ELITE_POOL PyTorch .pt files for a prefix to flat float32 .bin for the C++ trainer."""
    converted = 0
    for slot in range(ELITE_POOL):
        src = os.path.join(load_dir, f'{prefix}_model_{slot}.pt')
        if not os.path.exists(src):
            print(f'  [{label}] slot {slot:2d}: {src} not found — skipping')
            continue
        try:
            sd = torch.load(src, map_location='cpu', weights_only=True)
            arr = state_dict_to_arr(sd, layer_defs)
            dst = os.path.join(output_dir, f'{prefix}_elite_{slot}.bin')
            arr.tofile(dst)
            converted += 1
        except Exception as e:
            print(f'  [{label}] slot {slot:2d}: ERROR — {e}')
    print(f'  [{label}] {converted}/{ELITE_POOL} elite slots converted')


def convert_mt2(load_dir, output_dir):
    """Convert MT2NN .pt elite slots to C++ .bin files."""
    converted = 0
    for slot in range(ELITE_POOL):
        src = os.path.join(load_dir, f'mt2_model_{slot}.pt')
        if not os.path.exists(src):
            print(f'  [mt2] slot {slot:2d}: {src} not found — skipping')
            continue
        try:
            sd  = torch.load(src, map_location='cpu', weights_only=True)
            arr = mt2_state_dict_to_arr(sd)
            dst = os.path.join(output_dir, f'mt2_elite_{slot}.bin')
            arr.tofile(dst)
            converted += 1
        except Exception as e:
            print(f'  [mt2] slot {slot:2d}: ERROR — {e}')
    print(f'  [mt2] {converted}/{ELITE_POOL} elite slots converted')


def convert_mt2_norm_stats(load_dir, output_dir):
    """Convert Python JSON norm stats to C++ binary format (4 doubles + 1 int = 36 bytes)."""
    import json, math
    src = os.path.join(load_dir, 'mt2_norm_stats.json')
    if not os.path.exists(src):
        print('  [mt2_norm_stats] JSON file not found — skipping')
        return
    try:
        with open(src) as f:
            s = json.load(f)
        count = s.get('count', 0)
        if count == 0:
            print('  [mt2_norm_stats] count=0 — skipping (no data yet)')
            return
        dm = s['delta_mean']
        dv = s['delta_var']    # Welford M2
        rm = s['range_mean']
        rv = s['range_var']
        # Reconstruct C++ sum / sum_sq from Welford mean/M2:
        #   sum = mean * count
        #   sum_sq = M2 + mean^2 * count   (because M2 = sum_sq - sum^2/count)
        delta_sum  = dm * count
        delta_sum2 = dv + dm * dm * count
        range_sum  = rm * count
        range_sum2 = rv + rm * rm * count
        dst = os.path.join(output_dir, 'mt2_norm_stats.bin')
        with open(dst, 'wb') as f:
            f.write(struct.pack('<ddddi', delta_sum, delta_sum2, range_sum, range_sum2, count))
        print(f'  [mt2_norm_stats] written to {dst} (count={count})')
    except Exception as e:
        print(f'  [mt2_norm_stats] ERROR — {e}')


def main():
    parser = argparse.ArgumentParser(description='Convert .pt elite models to .bin for C++ trainer')
    parser.add_argument('--account', default='acct0', help='Account identifier (e.g. acct0); derives models/ACCOUNT/training as load and output dir')
    args = parser.parse_args()

    load_dir   = os.path.join('models', args.account, 'training')
    output_dir = load_dir
    os.makedirs(output_dir, exist_ok=True)

    industries = [
        'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
        'consumer_services', 'health_care', 'industrials', 'consumer_staples',
        'energy', 'utilities', 'real_estate', 'materials',
    ]

    print(f'Converting industry elite models from {load_dir} → {output_dir}')
    for ind in industries:
        convert_industry(ind, load_dir, output_dir, STOCK_LAYER_DEFS, ind)

    print(f'Converting master elite models from {load_dir} → {output_dir}')
    convert_industry('master', load_dir, output_dir, MASTER_LAYER_DEFS, 'master')

    print(f'Converting MT1 elite models from {load_dir} → {output_dir}')
    for ind in industries:
        convert_industry(f'mt1_{ind}', load_dir, output_dir, MT1_LAYER_DEFS, f'mt1_{ind}')

    print(f'Converting MT2 elite models from {load_dir} → {output_dir}')
    convert_mt2(load_dir, output_dir)

    print(f'Converting MT2 norm stats from {load_dir} → {output_dir}')
    convert_mt2_norm_stats(load_dir, output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
