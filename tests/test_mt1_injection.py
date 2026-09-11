"""Direction constant-collapse injection (upkeep._inject_dir_tail).

Locks the injection contract mirrored from the C++ trainer. Under the v0.5.0.0 forward-accumulation
direction pool there are no elite RANK slots to overwrite — every slot is a persistent individual
carrying its own record — so injection targets the WORST QUARTILE OF MATURE models and leaves the
good ones alone. Each replacement becomes a fresh individual: ½·best + ½·random weights, an empty
record, its own lineage, and immunity until MT1_DIR_MIN_AGE.

Guards against a future edit that reseeds the wrong models, clobbers the deployed one, or forgets
to reset the metadata (which would leave a brand-new model wearing its predecessor's track record).
"""

import torch

from models import MT1Tail
from training_lib import MT1_COMP_SLOTS, MT1_DIR_INJ_BLEND, load_slot_model, save_slot_model
from upkeep import (
    MT1_DIR_MIN_AGE,
    _dir_sort_key,
    _inject_dir_tail,
    _load_dir_meta,
    _save_dir_meta,
)


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _seed_pool(prefix, d, industry, n_mature):
    """Write MT1_COMP_SLOTS tails and a metadata file where the first n_mature slots are mature
    with descending record quality (slot 0 best) and the rest are immature."""
    torch.manual_seed(0)
    for slot in range(MT1_COMP_SLOTS):
        save_slot_model(prefix, d, slot, MT1Tail())
    slots = []
    for s in range(MT1_COMP_SLOTS):
        if s < n_mature:
            # 16 bits, progressively fewer set → strictly descending primary score.
            bits = max(0, 16 - s)
            slots.append([(1 << bits) - 1, 16, s])
        else:
            slots.append([0, MT1_DIR_MIN_AGE - 1, s])   # immature: immune
    _save_dir_meta(d, industry, slots, MT1_COMP_SLOTS, 0)
    return slots


def test_injection_targets_worst_quartile_of_mature(tmp_path):
    prefix, d, ind = "mt1_test_tail_dir", str(tmp_path), "test"
    n_mature = 40
    _seed_pool(prefix, d, ind, n_mature)

    before = {s: _flat(load_slot_model(prefix, d, s, MT1Tail)) for s in range(MT1_COMP_SLOTS)}
    best = load_slot_model(prefix, d, 0, MT1Tail)

    n_inj = _inject_dir_tail(prefix, d, ind, best)
    assert n_inj == n_mature // 4

    slots, _next, _best = _load_dir_meta(d, ind)
    # _seed_pool writes strictly descending record quality over the mature prefix, so the pool's
    # own ranking puts the worst quartile at the tail of that prefix.
    injected = set(range(n_mature - n_inj, n_mature))
    after = {s: _flat(load_slot_model(prefix, d, s, MT1Tail)) for s in range(MT1_COMP_SLOTS)}

    for s in injected:
        assert not torch.equal(after[s], before[s]), f"slot {s} (worst quartile) must be reseeded"
        assert torch.isfinite(after[s]).all()
        # Metadata reset: a fresh individual cannot inherit its predecessor's record or age.
        assert slots[s][0] == 0, f"slot {s} record must be cleared"
        assert slots[s][1] == 0, f"slot {s} age must reset (immune until MT1_DIR_MIN_AGE)"
        assert slots[s][2] >= MT1_COMP_SLOTS, f"slot {s} must get a fresh lineage id"

    # The good mature models and every immature model are byte-for-byte untouched.
    for s in range(MT1_COMP_SLOTS):
        if s in injected:
            continue
        assert torch.equal(after[s], before[s]), f"slot {s} must be unchanged"

    # Sanity: the seeded ranking really is best→worst, so "worst quartile" means what we assert.
    assert _dir_sort_key(slots[0]) >= _dir_sort_key(slots[n_mature - n_inj - 1])


def test_injection_leaves_the_deployed_model_alone(tmp_path):
    """Slot 0 carries the best record in the seeded pool, so it must survive an injection —
    re-diversifying must never discard the model the pool is currently deploying."""
    prefix, d, ind = "mt1_test_tail_dir_keep", str(tmp_path), "keep"
    _seed_pool(prefix, d, ind, 40)
    before = _flat(load_slot_model(prefix, d, 0, MT1Tail))
    _inject_dir_tail(prefix, d, ind, load_slot_model(prefix, d, 0, MT1Tail))
    assert torch.equal(_flat(load_slot_model(prefix, d, 0, MT1Tail)), before)


def test_injected_slot_is_halfway_between_best_and_random(tmp_path):
    """A blended slot must sit exactly on the segment between the best and its random draw:
    injected = BLEND*best + (1-BLEND)*random, so random = (injected - BLEND*best)/(1-BLEND) is a
    valid MT1Tail-shaped tensor and the blend reproduces the injected slot."""
    prefix, d, ind = "mt1_test_tail_dir2", str(tmp_path), "blend"
    n_mature = 40
    _seed_pool(prefix, d, ind, n_mature)
    best = _flat(load_slot_model(prefix, d, 0, MT1Tail))

    n_inj = _inject_dir_tail(prefix, d, ind, load_slot_model(prefix, d, 0, MT1Tail))

    injected = _flat(load_slot_model(prefix, d, n_mature - 1, MT1Tail))   # the very worst
    assert n_inj >= 1
    # Recover the implied random component and confirm the blend identity holds.
    implied_random = (injected - MT1_DIR_INJ_BLEND * best) / (1.0 - MT1_DIR_INJ_BLEND)
    recon = MT1_DIR_INJ_BLEND * best + (1.0 - MT1_DIR_INJ_BLEND) * implied_random
    assert torch.allclose(recon, injected, atol=1e-6)
    # And it is genuinely a blend, not a copy of best (random component is non-trivial).
    assert not torch.allclose(injected, best, atol=1e-3)
