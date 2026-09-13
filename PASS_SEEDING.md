# Staged: per-industry proportional pass-boundary seeding (StockNN)

**Status: STAGED — do not implement yet.**

Gated on the five-pass v0.6.2.1 run finishing. This changes pass-boundary logic in the trainer;
rebuilding `build/training_v4_cpp` mid-run would leave passes 3–5 inconsistent with 1–2, and the
running process would not pick it up regardless.

Scope is **StockNN only**. MT1 is deferred (see the end).

---

## What exists today

There is **no pass-boundary decision at all**. Every industry's elites carry forward
unconditionally. At the top of each pass: portfolios reset to `IND_STARTING_CASH` with zero
holdings, all 200 slots copy slot 0, `ind_val_hist` / `mkt_val_hist` / `zero_counts` are zeroed,
OHLCV histories cleared, the elite-history ring resets (`hist_head = hist_count = 0` at
`day_num == 0`). Only weights persist.

## What to build

At each pass boundary, **per industry** — the pools are already per-industry, so no restructuring
is needed:

**1. Metric.** Slot 0's **percent** portfolio change over the **last 15 days** of the pass, for
`pass_old` and `pass_current`.

Percent rather than dollars so books of different size compete fairly — at day 805 pass 1 held
$56,708 in `tech_hardware` against pass 2's $34,695, and a dollar metric rewards the larger book
mechanically. Slot 0 rather than pool max, because `E[max of 200]` grows with pool *spread*, which
would reward a merged pool for its diversity rather than its quality.

**2. Share of the seed.** Direction is `sign(r_new − r_old)` in every branch:

| case | share_new |
|---|---|
| both positive | `r_new / (r_old + r_new)` |
| exactly one negative | `1.0` if `r_new > 0` else `0.0` |
| both negative | `abs(r_old) / (abs(r_old) + abs(r_new))` — smaller loss takes the larger share |

A single logistic `1/(1 + exp(-(r_new - r_old)/T))` reproduces all three without the discontinuity
at zero. The one-negative branch is the weakest measured (51.6%) and is the only one that can still
discard a whole set on a near-tie, so the logistic is the preferred form if `T` can be calibrated.

**3. Seed by proportional interleave** of the two sets' **20 persisted elites**, preserving rank
order within each set:

```
apct, bpct   = 100*share_a, 100*share_b
apos = bpos  = 0
aspd = bspd  = 0
while len(out) < ELITE_POOL:            # 20, NOT 200 — see Mechanics
    while aspd < 100 and bspd < 100:
        aspd += apct;  bspd += bpct
    if aspd > bspd: out.push(A[apos]); apos += 1; aspd -= 100
    else:           out.push(B[bpos]); bpos += 1; bspd -= 100
```

Verified: exact counts at every split (34.7% → 7 of 20, expected 6.9), rank order preserved within
each set, degenerate 0%/100% terminate cleanly, no infinite loop.

**4. Day 1 of the new pass.** The interleaved models occupy slots 0–19 in interleave order.
Mutations generate normally. After day 1 is scored, `selection_and_mutation` imposes the standard
layout (0–16 elites, 17–19 wavg, 20–199 mutations) as usual. No special ranking pass is needed.

**5. Champion.** The better of the two passes becomes the new `pass_old`; the other is not
retained. Two snapshots live at any time.

**6. Reserve.** The best ending model set is held separately for the final deliverable,
independent of seeding, so a wrong gate call can never cost the best models.

## Mechanics that constrain the design (verified in `training_v4.cpp`)

- **Only `ELITE_POOL = 20` models persist per industry** (`save_industry_elites`, ~line 2945),
  plus a 50-model history ring that resets at pass boundaries. Slots 20–199 are computed into a
  single scratch buffer (`mut_buf`) during the forward pass and discarded — they have **no stored
  existence**. A literal 200-model carryover would need 200 × 12 × 3.54 MB = **8.5 GB per
  snapshot** and would only import untested Gaussian perturbations. The cheap alternative, if ever
  wanted, is persisting the 180 `mut_seeds` (8 bytes each); mutations are a deterministic function
  of `(parent, seed, sigma)`.
- **StockNN uses uniform `parent = mut_i / MUTATIONS_PER_PARENT`** — 9 children per parent — not
  MT1's weighted `kChildren` table. The 180 children therefore inherit the seed proportion
  automatically under *any* ordering (7 parents → 63 children, 13 → 117). Interleave order only
  determines which model occupies slot 0 (production model and baseline source) and slots 17–19.
- **No slot-0 dependency at pass start.** All 200 portfolios reset identical, so `baseline` is
  `IND_STARTING_CASH` regardless of ordering.

Storage: 20 × 12 × 3.54 MB = **850 MB per snapshot**, ~1.7 GB for champion plus current.

## Measured expectations

744 non-overlapping 15-day industry-windows, passes 1–2 of the v0.6.2.1 run:

```
regime     windows   mean share   direction accuracy
both+        38.6%       0.497          60.6%
one-         29.4%       0.543          51.6%
both-        32.0%       0.501          48.7%
overall                                 54.2%
```

Only the both-positive regime carries signal. Allocation averages ~10 of 20 slots, sd ~6.6.
Resolution is 5% per slot, so a share below 2.5% gives the minority set zero models. Exact ties
fall to B, so assign which pass is A and which is B deliberately.

**Why 15 days rather than the full pass.** The pool at a pass end is not the pool that existed
mid-pass — models from the middle are long since culled. A full-pass percentage measures the
trajectory, not the surviving models that actually seed the next pass. This is a near-term
predictor by design, and it accepts a weaker sample to measure the right object.

**Why a weak signal is acceptable here.** The proportional form degrades gracefully: under noise
it lands near 50/50 rather than flipping categorically, and the champion and reserve rules mean a
wrong call cannot cost the best models. A hard gate at the same accuracy would make a full wrong
decision ~46% of the time.

**What the current run will add.** Everything above rests on a single pass-to-pass transition
(1→2). Passes 3, 4 and 5 supply three more. If the effect decays across four transitions, that is
itself the plateau signal and the gate becomes unnecessary; if it holds, the gate has something to
act on.

## Deferred

- **MT1.** Needs to work first — direction sits at 52–53% out-of-sample with negative skill in all
  12 industries. Its tails already carry per-channel scores and a baseline→0 / ideal→1 secondary
  normalization (used to equalise the four channels in the composite), which is the normalization
  to reuse rather than inventing one.
- **MT2.** Pulls from both passes regardless of per-industry outcomes: its 48 inputs are MT1
  slot-0 activations spanning all 12 industries, so per-industry seeding presents an input mixture
  it never trained on. Academic while paper runs `--flat-allocation`.

## Considered and rejected

- **Freezing slot 0 during the judging window** — measures a single model, not the pool-with-
  selection that actually carries forward.
- **Judging on the full pass** — wrong object; see above.
- **Judging on pool max** — rewards pool variance rather than quality.
- **Raw dollar amounts** — biased toward whichever book is already larger.
- **Interleaving 200 models** — 180 of them do not exist as stored models.
