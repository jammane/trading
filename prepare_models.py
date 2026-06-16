#!/usr/bin/env python3
"""
prepare_models.py — Convert PyTorch .pt elite models to flat float32 .bin files
for the C++ trainer (training_v4_cpp).

Run this once before the first C++ training run, or whenever you want to seed
the C++ trainer from a Python-trained checkpoint.

Usage:
  python prepare_models.py --load-dir models --output models
  python prepare_models.py --load-dir /tmp/py_out --output /tmp/cpp_out
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


def convert_industry(prefix, load_dir, output_dir, layer_defs, label):
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


def main():
    parser = argparse.ArgumentParser(description='Convert .pt elite models to .bin for C++ trainer')
    parser.add_argument('--load-dir', required=True, help='Directory containing .pt files')
    parser.add_argument('--output',   required=True, help='Directory to write .bin files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    industries = [
        'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
        'consumer_services', 'health_care', 'industrials', 'consumer_staples',
        'energy', 'utilities', 'real_estate', 'materials',
    ]

    print(f'Converting industry elite models from {args.load_dir} → {args.output}')
    for ind in industries:
        convert_industry(ind, args.load_dir, args.output, STOCK_LAYER_DEFS, ind)

    print(f'Converting master elite models from {args.load_dir} → {args.output}')
    convert_industry('master', args.load_dir, args.output, MASTER_LAYER_DEFS, 'master')

    print('Done.')


if __name__ == '__main__':
    main()
