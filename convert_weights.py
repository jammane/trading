#!/usr/bin/env python3
"""
convert_weights.py — Convert C++ flat float32 .bin elite models back to PyTorch .pt files
for use with production_v2.py and inspect_trades.py.

Run this after a C++ training run to make the trained models available to the Python stack.

Usage:
  python convert_weights.py --account acct0
  python convert_weights.py --account acct0 --industry-dir models/acct0/training/champion
  python convert_weights.py --source-dir /root/some_run --output-dir /tmp/pt

Directory selection:
  --source-dir    where the .bin files are read from (default: models/ACCOUNT/training)
  --industry-dir  overrides the source for the StockNN industry elites ONLY
  --output-dir    where the .pt files are written    (default: --source-dir)

--industry-dir exists for the champion store written by the v0.6.3.0 pass-boundary seeding
(PASS_SEEDING.md). `champion/` holds the per-industry best StockNN elites and NOTHING ELSE —
master, MT1 and MT2 are saved to the run root — so pointing --source-dir at it would convert the
industries and silently skip everything else. Splitting the two is what makes the deliverable
correct: StockNN from champion/, master/MT1/MT2 from the run root.
"""

import argparse
import os

import struct

import numpy as np
import torch

from models import MT1NET_LAYER_DEFS, MT1NET_PARAMS, MT1Net, MT2NN, MasterNN, StockNN
from prepare_models import (
    ELITE_POOL,
    MASTER_LAYER_DEFS,
    MT1_META_MAGIC,
    MT1_META_VERSION,
    MT1_POOL_SLOTS,
    MT1_SLOT_META_BYTES,
    MT2_LAYOUT,
    STOCK_LAYER_DEFS,
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


def _mt1_meta_path(models_dir, ind):
    return os.path.join(models_dir, f'mt1_{ind}_meta.bin')


def _read_mt1_meta(models_dir, ind):
    """Read the pool metadata sidecar written by save_mt1_pool.

    Returns (best_slot, n_slots) or (0, MT1_POOL_SLOTS) when the sidecar is missing or rejected.
    The sidecar is what says WHICH of the 200 individuals is deployed — without it slot 0 is a
    guess, and the deployed model is the one whose rolling register ranked first, not a fixed slot.
    """
    path = _mt1_meta_path(models_dir, ind)
    if not os.path.exists(path):
        return 0, MT1_POOL_SLOTS
    try:
        with open(path, 'rb') as f:
            magic, version, n_slots, hist = struct.unpack('<4I', f.read(16))
            if magic != MT1_META_MAGIC or version != MT1_META_VERSION:
                print(f'  [mt1/{ind}] metadata sidecar rejected (magic/version) — assuming slot 0')
                return 0, MT1_POOL_SLOTS
            f.seek(16 + n_slots * MT1_SLOT_META_BYTES)
            best_slot = struct.unpack('<i', f.read(4))[0]
        if not 0 <= best_slot < n_slots:
            return 0, n_slots
        return best_slot, n_slots
    except (OSError, struct.error) as e:
        print(f'  [mt1/{ind}] metadata sidecar unreadable ({e}) — assuming slot 0')
        return 0, MT1_POOL_SLOTS


def _convert_mt1_pool(ind, models_dir, output_dir):
    """Convert one industry's 200-individual MT1 pool .bin → .pt, and write the deployed best.

    Replaces the old head + 4-tail compose. There is no composition step any more: a slot file IS
    a whole model.
    """
    best_slot, n_slots = _read_mt1_meta(models_dir, ind)
    converted = 0
    for slot in range(n_slots):
        src = os.path.join(models_dir, f'mt1_{ind}_slot_{slot}.bin')
        if not os.path.exists(src):
            continue
        try:
            arr = np.fromfile(src, dtype=np.float32)
            if len(arr) != MT1NET_PARAMS:
                print(f'  [mt1/{ind}] slot {slot}: {len(arr)} floats, expected {MT1NET_PARAMS}'
                      ' — skipped')
                continue
            m = MT1Net()
            m.load_state_dict(arr_to_state_dict(arr, MT1NET_LAYER_DEFS, None))
            torch.save(m.state_dict(), os.path.join(output_dir, f'mt1_{ind}_model_{slot}.pt'))
            converted += 1
            if slot == best_slot:
                torch.save(m.state_dict(), os.path.join(output_dir, f'mt1_{ind}_best.pt'))
        except Exception as e:                                   # noqa: BLE001 - report and continue
            print(f'  [mt1/{ind}] slot {slot}: ERROR — {e}')
    if converted == 0:
        print(f'  [mt1/{ind}] no slot .bin found — nothing converted')
    else:
        print(f'  [mt1/{ind}] {converted}/{n_slots} slots converted; '
              f'best = slot {best_slot} → mt1_{ind}_best.pt')


def main():
    parser = argparse.ArgumentParser(
        description='Convert .bin C++ elite models to .pt for the Python stack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='With pass-boundary seeding active the last pass is NOT necessarily the best;\n'
               'convert the champion store with --industry-dir <run>/champion.')
    parser.add_argument('--account', default='acct0',
                        help='Account identifier (e.g. acct0); derives models/ACCOUNT/training')
    parser.add_argument('--source-dir', default=None,
                        help='Read .bin from here instead of the --account path')
    parser.add_argument('--industry-dir', default=None,
                        help='Override the source for StockNN industry elites only '
                             '(e.g. .../training/champion, which holds nothing else)')
    parser.add_argument('--output-dir', default=None,
                        help='Write .pt here (default: the source directory)')
    args = parser.parse_args()

    models_dir   = args.source_dir   or os.path.join('models', args.account, 'training')
    industry_dir = args.industry_dir or models_dir
    output_dir   = args.output_dir   or models_dir

    for label, d in (('source', models_dir), ('industry source', industry_dir)):
        if not os.path.isdir(d):
            parser.error(f'{label} directory does not exist: {d}')
    os.makedirs(output_dir, exist_ok=True)

    industries = [
        'tech_hardware', 'tech_software_ai', 'financials', 'consumer_discretionary',
        'consumer_services', 'health_care', 'industrials', 'consumer_staples',
        'energy', 'utilities', 'real_estate', 'materials',
    ]

    # A mistyped path would otherwise convert nothing and still exit 0, which is exactly how the
    # wrong models get deployed. Fail loudly instead.
    present = [i for i in industries
               if os.path.exists(os.path.join(industry_dir, f'{i}_elite_0.bin'))]
    if not present:
        parser.error(f'no industry *_elite_0.bin files in {industry_dir} — wrong directory?')
    if len(present) < len(industries):
        missing = [i for i in industries if i not in present]
        print(f'\n  *** WARNING: only {len(present)}/{len(industries)} industries found in '
              f'{industry_dir}')
        print(f'  *** missing: {", ".join(missing)}')
        print('  *** those industries will have NO .pt written and production would fall back to '
              'whatever is already there.\n')

    if industry_dir != models_dir:
        print(f'StockNN industries  <- {industry_dir}')
        print(f'master / MT1 / MT2  <- {models_dir}')

    print(f'Converting industry elite models from {industry_dir} → {output_dir}')
    for ind in industries:
        convert_industry(ind, industry_dir, output_dir, STOCK_LAYER_DEFS, StockNN, ind)

    print(f'Converting master elite models from {models_dir} → {output_dir}')
    convert_industry('master', models_dir, output_dir, MASTER_LAYER_DEFS, MasterNN, 'master')

    print(f'Converting MT1 pools ({MT1_POOL_SLOTS} individuals/industry) from {models_dir} → {output_dir}')
    for ind in industries:
        _convert_mt1_pool(ind, models_dir, output_dir)

    print(f'Converting MT2 elite models from {models_dir} → {output_dir}')
    convert_mt2(models_dir, output_dir)

    print('Done.')


if __name__ == '__main__':
    main()
