#!/usr/bin/env python3
"""
convert_weights.py — Convert C++ flat float32 .bin elite models back to PyTorch .pt files
for use with production_v2.py and inspect_trades.py.

Run this after a C++ training run to make the trained models available to the Python stack.

Usage:
  python convert_weights.py --account acct0
"""

import argparse
import os

import numpy as np
import torch

from models import StockNN, MasterNN, MT1NN, MT2NN
from prepare_models import (
    STOCK_LAYER_DEFS, MASTER_LAYER_DEFS, MT1_LAYER_DEFS, MT2_LAYOUT,
    ELITE_POOL,
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



def convert_industry(prefix, models_dir, output_dir, layer_defs, model_class, label, n_elites=None):
    """Convert C++ .bin elite files for a prefix to PyTorch .pt, copying slot 0 to _best.pt."""
    import shutil
    if n_elites is None:
        n_elites = ELITE_POOL
    converted = 0
    for slot in range(n_elites):
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
    print(f'  [{label}] {converted}/{n_elites} elite slots converted')

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


def _convert_mt1_best(ind, models_dir, output_dir):
    """Convert mt1_{ind}_comp_0.bin (best composite model) to mt1_{ind}_best.pt."""
    import shutil
    src = os.path.join(models_dir, f'mt1_{ind}_comp_0.bin')
    if not os.path.exists(src):
        print(f'  [mt1/{ind}] mt1_{ind}_comp_0.bin not found — skipping')
        return
    try:
        arr = np.fromfile(src, dtype=np.float32)
        sd  = arr_to_state_dict(arr, MT1_LAYER_DEFS, MT1NN)
        m   = MT1NN()
        m.load_state_dict(sd)
        dst = os.path.join(output_dir, f'mt1_{ind}_best.pt')
        torch.save(m.state_dict(), dst)
        print(f'  [mt1/{ind}] mt1_{ind}_best.pt written from comp_0.bin')
    except Exception as e:
        print(f'  [mt1/{ind}] ERROR converting comp_0.bin: {e}')


def main():
    parser = argparse.ArgumentParser(description='Convert .bin C++ elite models to .pt for Python stack')
    parser.add_argument('--account', default='acct0', help='Account identifier (e.g. acct0); derives models/ACCOUNT/training as source and output dir')
    args = parser.parse_args()

    models_dir = os.path.join('models', args.account, 'training')
    output_dir = models_dir
    os.makedirs(output_dir, exist_ok=True)

    industries = [
        'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
        'consumer_services', 'health_care', 'industrials', 'consumer_staples',
        'energy', 'utilities', 'real_estate', 'materials',
    ]

    print(f'Converting industry elite models from {models_dir} → {output_dir}')
    for ind in industries:
        convert_industry(ind, models_dir, output_dir, STOCK_LAYER_DEFS, StockNN, ind)

    print(f'Converting master elite models from {models_dir} → {output_dir}')
    convert_industry('master', models_dir, output_dir, MASTER_LAYER_DEFS, MasterNN, 'master')

    print(f'Converting MT1 composite best models from {models_dir} → {output_dir}')
    for ind in industries:
        _convert_mt1_best(ind, models_dir, output_dir)

    print(f'Converting MT2 elite models from {models_dir} → {output_dir}')
    convert_mt2(models_dir, output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
