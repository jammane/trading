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

import numpy as np
import torch

from models import MT1NET_LAYER_DEFS, MT1NET_PARAMS

# Shared layer definitions — single source of truth for binary layout.
# Each entry: (state_dict_key_prefix, out_size, in_size)
# For inject layers, key_prefix uses the ModuleList index notation.
STOCK_LAYER_DEFS = [
    ('fc_seed',  120,  60),
    # inject layers 0..13: fc_inject.{i}
    *[(f'fc_inject.{i}', 125 + 5 * i, 180 + 5 * i) for i in range(14)],
    ('fc_today', 300, 422),
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

# MT2 layout mirrors C++ binary offsets (FC1, FC2, LSTM L1, LSTM L2, taper1-3, fc_out).
# Stored as (key, shape) rather than (prefix, out, in) because LSTM has bias-only entries.
MT2_LAYOUT = [
    ('fc1.weight',         (36, 12)),
    ('fc1.bias',           (36,)),
    ('fc2.weight',         (36, 36)),
    ('fc2.bias',           (36,)),
    ('lstm.weight_ih_l0',  (144, 1)),
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

# MT1: one pool of persistent individuals per industry. Every slot carries its own rolling score
# register, so every slot has a weight file — there are no seed-regenerated mutation slots.
MT1_POOL_SLOTS      = 200
# Metadata sidecar written by save_mt1_pool in training_v4.cpp. KEEP IN SYNC with MT1_META_MAGIC /
# MT1_META_VERSION there and with MT1SlotMeta in mt1_pool.h: the sidecar is a separate file
# precisely because .bin weight files are headerless and validated by element count alone.
MT1_META_MAGIC      = 0x4D543150        # "MT1P"
MT1_META_VERSION    = 1
MT1_SCORE_HIST      = 16
MT1_SLOT_META_BYTES = MT1_SCORE_HIST * 4 + 4 + 4   # float score[16] + uint16 n_pred(+pad) + uint32


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


def convert_mt1_pool(ind, load_dir, output_dir):
    """Convert one industry's MT1 pool .pt → the .bin names load_or_init_mt1_pool reads.

    Deliberately not convert_industry: that writes `{prefix}_elite_{slot}.bin`, and the MT1 pool
    has no elite slots — 200 persistent individuals, read as `mt1_{ind}_slot_{n}.bin`. A name
    mismatch here is silent, because load_bin treats a missing file as "random-init this slot".
    """
    converted = 0
    for slot in range(MT1_POOL_SLOTS):
        src = os.path.join(load_dir, f'mt1_{ind}_model_{slot}.pt')
        if not os.path.exists(src):
            continue
        try:
            sd = torch.load(src, map_location='cpu', weights_only=True)
            arr = state_dict_to_arr(sd, MT1NET_LAYER_DEFS)
            if len(arr) != MT1NET_PARAMS:
                print(f'  [mt1/{ind}] slot {slot}: {len(arr)} floats, expected {MT1NET_PARAMS}'
                      ' — skipped')
                continue
            arr.tofile(os.path.join(output_dir, f'mt1_{ind}_slot_{slot}.bin'))
            converted += 1
        except Exception as e:                                   # noqa: BLE001 - report and continue
            print(f'  [mt1/{ind}] slot {slot}: ERROR — {e}')
    print(f'  [mt1/{ind}] {converted}/{MT1_POOL_SLOTS} slots converted')
    # No metadata sidecar is written: seeding from .pt gives weights but no score history, so the
    # pool starts every register empty. load_or_init_mt1_pool handles a missing sidecar by design.


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

    print(f'Converting MT1 pools from {load_dir} → {output_dir}')
    for ind in industries:
        convert_mt1_pool(ind, load_dir, output_dir)

    print(f'Converting MT2 elite models from {load_dir} → {output_dir}')
    convert_mt2(load_dir, output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
