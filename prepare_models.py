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

# Branched MT1NN — layer order defines the flat .bin layout; must match models.py MT1NN
# and the C++ MT1_* offsets. (prefix, out_size, in_size). Total 2,218 params.
MT1_LAYER_DEFS = [
    ('a1', 20, 20), ('a2', 20, 20),   # block A: vol + poly
    ('b1',  6, 10), ('b2',  4,  6),   # block B: daily returns
    ('c1',  5,  7), ('c2',  4,  5),   # block C: decade returns
    ('d1', 22, 28), ('d2', 16, 22),   # block D: fusion taper
    ('d3', 10, 16), ('d4',  4, 10),
]

# MT2 layout mirrors C++ binary offsets (FC1, FC2, LSTM L1, LSTM L2, taper1-3, fc_out).
# Stored as (key, shape) rather than (prefix, out, in) because LSTM has bias-only entries.
MT2_LAYOUT = [
    ('fc1.weight',         (36, 48)),
    ('fc1.bias',           (36,)),
    ('fc2.weight',         (36, 36)),
    ('fc2.bias',           (36,)),
    ('lstm.weight_ih_l0',  (144, 4)),
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

ELITE_POOL       = 20
MT1_COMP_PARENTS = 25  # 17 direct + 3 wavg + 5 injection per component pool
MT1_POOL_NAMES   = ('dir', 'acc', 'rng', 'cfd')


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


def convert_industry(prefix, load_dir, output_dir, layer_defs, label, n_elites=None):
    """Convert elite PyTorch .pt files for a prefix to flat float32 .bin for the C++ trainer."""
    if n_elites is None:
        n_elites = ELITE_POOL
    converted = 0
    for slot in range(n_elites):
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
    print(f'  [{label}] {converted}/{n_elites} elite slots converted')


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

    print(f'Converting MT1 component pool models from {load_dir} → {output_dir}')
    for ind in industries:
        for pool in MT1_POOL_NAMES:
            convert_industry(f'mt1_{ind}_{pool}', load_dir, output_dir, MT1_LAYER_DEFS,
                             f'mt1_{ind}_{pool}', n_elites=MT1_COMP_PARENTS)

    print(f'Converting MT2 elite models from {load_dir} → {output_dir}')
    convert_mt2(load_dir, output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
