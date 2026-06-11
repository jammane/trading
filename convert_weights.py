#!/usr/bin/env python3
"""
convert_weights.py — Convert C++ flat float32 .bin elite models back to PyTorch .pt files
for use with production_v2.py and inspect_trades.py.

Run this after a C++ training run to make the trained models available to the Python stack.

Usage:
  python convert_weights.py --models-dir models --output models
  python convert_weights.py --models-dir /tmp/cpp_out --output /tmp/cpp_pt
"""

import argparse
import os

import numpy as np
import torch

from models import StockNN, MasterNN
from prepare_models import STOCK_LAYER_DEFS, MASTER_LAYER_DEFS, ELITE_POOL


def arr_to_state_dict(arr, layer_defs, model_class):
    """Reconstruct a PyTorch state_dict from a flat float32 numpy array."""
    offset = 0
    state_dict = {}
    for prefix, out_size, in_size in layer_defs:
        n_w = out_size * in_size
        n_b = out_size
        w = arr[offset:offset + n_w].reshape(out_size, in_size)
        b = arr[offset + n_w:offset + n_w + n_b]
        state_dict[f'{prefix}.weight'] = torch.from_numpy(w.copy())
        state_dict[f'{prefix}.bias']   = torch.from_numpy(b.copy())
        offset += n_w + n_b
    assert offset == len(arr), f'Consumed {offset} floats but array has {len(arr)}'
    return state_dict


def convert_industry(prefix, models_dir, output_dir, layer_defs, model_class, label):
    converted = 0
    for slot in range(ELITE_POOL):
        src = os.path.join(models_dir, f'{prefix}_elite_{slot}.bin')
        if not os.path.exists(src):
            print(f'  [{label}] slot {slot:2d}: {src} not found — skipping')
            continue
        try:
            arr = np.fromfile(src, dtype=np.float32)
            sd  = arr_to_state_dict(arr, layer_defs, model_class)
            m   = model_class()
            m.load_state_dict(sd)
            dst = os.path.join(output_dir, f'{prefix}_model_{slot}.pt')
            torch.save(m.state_dict(), dst)
            converted += 1
        except Exception as e:
            print(f'  [{label}] slot {slot:2d}: ERROR — {e}')
    print(f'  [{label}] {converted}/{ELITE_POOL} elite slots converted')


def main():
    parser = argparse.ArgumentParser(description='Convert .bin C++ elite models to .pt for Python stack')
    parser.add_argument('--models-dir', required=True, help='Directory containing .bin files')
    parser.add_argument('--output',     required=True, help='Directory to write .pt files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    industries = [
        'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
        'consumer_services', 'health_care', 'industrials', 'consumer_staples',
        'energy', 'utilities', 'real_estate', 'materials',
    ]

    print(f'Converting industry elite models from {args.models_dir} → {args.output}')
    for ind in industries:
        convert_industry(ind, args.models_dir, args.output, STOCK_LAYER_DEFS, StockNN, ind)

    print(f'Converting master elite models from {args.models_dir} → {args.output}')
    convert_industry('master', args.models_dir, args.output, MASTER_LAYER_DEFS, MasterNN, 'master')

    print('Done.')


if __name__ == '__main__':
    main()
