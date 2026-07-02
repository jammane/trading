"""
Functional harness for upkeep_mt1_industry's heads/tails block cycle (Increment 4B).

Runs the production daily MT1 upkeep over several synthetic days in a temp model dir and asserts
the head/tail block-alternating contract:
  - each call returns a finite 10-tuple (composite + direction slot0 activations),
  - a valid composed MT1NN is written to mt1_{ind}_best.pt,
  - the 4 tail pools evolve every run while the head pool evolves only on block boundaries
    (every MT1_BLOCK_DAYS runs) — i.e. tail slot0 changes more often than head slot0.

SKIPPED until 4B lands (current upkeep still uses the old component-pool architecture). Unskip
in the 4B commit; it is the validation gate for that rewrite.
"""

import hashlib
import os

import pytest
import torch

import upkeep as upkeep_mod
from upkeep import upkeep_mt1_industry
from models import MT1NN, MT1Head, MT1Tail

TAILS = ("dir", "acc", "rng", "cfd")


def _md5(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def test_upkeep_mt1_head_tail_block_cycle(tmp_path, monkeypatch):
    # Small block so a few days exercise >1 block boundary; small pool for test speed.
    monkeypatch.setattr(upkeep_mod, "MT1_BLOCK_DAYS", 3, raising=False)
    monkeypatch.setattr(upkeep_mod, "MT1_COMP_SLOTS", 40, raising=False)
    ind = "energy"
    md = str(tmp_path)
    rolling: dict = {}
    torch.manual_seed(0)

    n_days = 7  # > 2 * block(3)
    head_hashes, best_hashes = [], []
    for _ in range(n_days):
        in37 = torch.randn(1, 37)
        actual_d = float(torch.randn(1).item()) * 500.0
        ret = upkeep_mt1_industry(ind, md, in37, actual_d, rolling_state=rolling)
        assert len(ret) == 10, "upkeep_mt1_industry must return a 10-tuple"
        for v in ret:
            fv = float(v)
            assert fv == fv and abs(fv) < 1e9, f"non-finite return value {v}"
        head_hashes.append(_md5(os.path.join(md, f"mt1_{ind}_head_model_0.pt")))
        best_hashes.append(_md5(os.path.join(md, f"mt1_{ind}_best.pt")))

    # Head/tail pool files exist with the right classes.
    MT1Head().load_state_dict(
        torch.load(os.path.join(md, f"mt1_{ind}_head_model_0.pt"), weights_only=True))
    for t in TAILS:
        MT1Tail().load_state_dict(
            torch.load(os.path.join(md, f"mt1_{ind}_tail_{t}_model_0.pt"), weights_only=True))

    # best.pt is a valid composed MT1NN with finite (1,4) output.
    best = os.path.join(md, f"mt1_{ind}_best.pt")
    assert os.path.exists(best), "composed mt1_{ind}_best.pt not written"
    m = MT1NN()
    m.load_state_dict(torch.load(best, weights_only=True))
    m.eval()
    with torch.inference_mode():
        out = m(torch.randn(1, 37))
    assert out.shape == (1, 4) and torch.isfinite(out).all()

    # Block cadence: the production model (composed head0+tail0) evolves every run via the daily
    # tail phase, while the head pool only changes on block boundaries (every MT1_BLOCK_DAYS runs).
    head_changes = sum(1 for a, b in zip(head_hashes, head_hashes[1:]) if a != b)
    best_changes = sum(1 for a, b in zip(best_hashes, best_hashes[1:]) if a != b)
    assert best_changes > head_changes, (
        f"production model should evolve more often than the head "
        f"(best={best_changes}, head={head_changes})")
    assert head_changes >= 1, "head pool should evolve at least once across 2 block boundaries"
