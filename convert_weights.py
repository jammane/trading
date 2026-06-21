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
import json
import os
import struct

import numpy as np
import torch

from models import StockNN, MasterNN, MT1NN, MT2NN
from prepare_models import (
    STOCK_LAYER_DEFS, MASTER_LAYER_DEFS, MT1_LAYER_DEFS, MT2_LAYOUT, ELITE_POOL
)


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


def arr_to_mt2_state_dict(arr):
    """Reconstruct MT2NN state_dict from flat float32 array using MT2_LAYOUT."""
    offset = 0
    state_dict = {}
    for key, shape in MT2_LAYOUT:
        n = 1
        for d in shape:
            n *= d
        t = torch.from_numpy(arr[offset:offset + n].reshape(shape).copy())
        state_dict[key] = t
        offset += n
    assert offset == len(arr), f'MT2: consumed {offset} floats but array has {len(arr)}'
    return state_dict


def convert_mt2_norm_stats(models_dir, output_dir):
    """Convert C++ binary norm stats (36 bytes) to Python JSON format."""
    src = os.path.join(models_dir, 'mt2_norm_stats.bin')
    if not os.path.exists(src):
        print('  [mt2_norm_stats] .bin file not found — skipping')
        return
    try:
        with open(src, 'rb') as f:
            raw = f.read(36)
        delta_sum, delta_sum2, range_sum, range_sum2, count = struct.unpack('<ddddi', raw)
        if count == 0:
            print('  [mt2_norm_stats] count=0 in binary — skipping')
            return
        dm = delta_sum / count
        dv = delta_sum2 - delta_sum * delta_sum / count   # Welford M2
        rm = range_sum / count
        rv = range_sum2 - range_sum * range_sum / count
        stats = {
            'delta_mean': dm,
            'delta_var':  dv,
            'range_mean': rm,
            'range_var':  rv,
            'count':      count,
        }
        dst = os.path.join(output_dir, 'mt2_norm_stats.json')
        with open(dst, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'  [mt2_norm_stats] written to {dst} (count={count})')
    except Exception as e:
        print(f'  [mt2_norm_stats] ERROR — {e}')


def convert_industry(prefix, models_dir, output_dir, layer_defs, model_class, label):
    import shutil
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

    # Slot 0 is the production model — copy to _best.pt for production_v2.py
    slot0 = os.path.join(output_dir, f'{prefix}_model_0.pt')
    if os.path.exists(slot0):
        shutil.copy2(slot0, os.path.join(output_dir, f'{prefix}_best.pt'))
        print(f'  [{label}] _best.pt written (copy of slot 0)')


def convert_mt2(models_dir, output_dir):
    """Convert MT2NN C++ .bin elite slots to PyTorch .pt files."""
    import shutil
    converted = 0
    for slot in range(ELITE_POOL):
        src = os.path.join(models_dir, f'mt2_elite_{slot}.bin')
        if not os.path.exists(src):
            print(f'  [mt2] slot {slot:2d}: {src} not found — skipping')
            continue
        try:
            arr = np.fromfile(src, dtype=np.float32)
            sd  = arr_to_mt2_state_dict(arr)
            m   = MT2NN()
            m.load_state_dict(sd)
            dst = os.path.join(output_dir, f'mt2_model_{slot}.pt')
            torch.save(m.state_dict(), dst)
            converted += 1
        except Exception as e:
            print(f'  [mt2] slot {slot:2d}: ERROR — {e}')
    print(f'  [mt2] {converted}/{ELITE_POOL} elite slots converted')

    slot0 = os.path.join(output_dir, 'mt2_model_0.pt')
    if os.path.exists(slot0):
        shutil.copy2(slot0, os.path.join(output_dir, 'mt2_best.pt'))
        print('  [mt2] mt2_best.pt written (copy of slot 0)')


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

    print(f'Converting MT1 elite models from {args.models_dir} → {args.output}')
    for ind in industries:
        convert_industry(f'mt1_{ind}', args.models_dir, args.output, MT1_LAYER_DEFS, MT1NN, f'mt1_{ind}')

    print(f'Converting MT2 elite models from {args.models_dir} → {args.output}')
    convert_mt2(args.models_dir, args.output)

    print(f'Converting MT2 norm stats from {args.models_dir} → {args.output}')
    convert_mt2_norm_stats(args.models_dir, args.output)

    print('Done.')


if __name__ == '__main__':
    main()
