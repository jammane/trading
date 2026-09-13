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

**5. Champion.** The better of the two passes becomes the new reference for the next boundary.
Because each boundary compares the champion against the finishing pass, the champion is already a
running maximum under the metric — so it doubles as the **reserve** for the final deliverable and
only one store is needed. A wrong gate call can therefore never cost the best models.

## Persistence

The design above is not implementable without somewhere to record *what the champion scored* and
*where its weights live*. Two facts force this:

- The metric is otherwise recoverable only by regex-scraping `prod=$` out of a 39 MB `train.log`.
  The trainer would be reading back its own console output as load-bearing state.
- **After the first boundary the champion is a per-industry composite.** `energy`'s champion may
  come from pass 2 while `financials`' comes from pass 1, so no single pass directory or log
  section describes "the champion". Per-industry scores must be recorded when crowned and carried
  forward.

### `pass_reference.csv`

Lives in the run's model directory (`models/acct#/training/`) beside the weights — it is
load-bearing state that must travel with them, not a log.

```
pass,industry,day_start,day_end,champ_pass,champ_pct,chal_pass,chal_pct,regime,share_new,slots_new,new_champ_pass,new_champ_pct
```

Append-only, one row per industry per boundary. The reader takes the **last row per industry** and
reads `new_champ_pass` / `new_champ_pct` — the champion entering the next comparison. Everything
earlier is audit trail: every decision the gate made, the inputs it made it on, and what it did.

CSV rather than the `mt1_{ind}_tail_dir_meta.bin` binary-sidecar pattern because this is a decision
record meant to be read by eye and by pandas. Writing is `fprintf`, reading is `getline` +
`sscanf`; `training_log.csv` is the precedent. Twelve rows per boundary.

### `champion/`

A sibling directory holding `{ind}_elite_{0..19}.bin` per industry, each copied from whichever pass
won *that industry*. 20 x 12 x 3.54 MB = **850 MB**, one store serving both champion and reserve.

`convert_weights.py` should target this directory rather than the live training directory when
producing the deliverable — the last pass is not necessarily the best. On the v0.6.2.1 run's first
boundary, pass 2 finished **6.2% behind** pass 1.

### Boundary sequence — and its ordering hazard

`load_or_init_industry` reads `{dir}/{ind}_elite_{slot}.bin` from the live output directory and
`save_industry_elites` writes there, so at the end of pass N **that directory is pass N's elites**.
Writing the blend into it destroys exactly the models the crowning step then needs. Order matters:

1. Capture the metric: slot-0 baseline at `stop_day - 15` and at `stop_day`, both already in memory
   during the pass. `pct = end/start - 1`.
2. Read the last row for this industry from `pass_reference.csv` -> `champ_pass`, `champ_pct`.
3. **Stage** pass N's 20 elites to a scratch directory (file copies, 71 MB per industry).
4. Compute `share_new` from `(champ_pct, chal_pct)`, then interleave `champion/` with the staged
   copies into the live output directory — that is pass N+1's seed.
5. Crown: if pass N won, replace `champion/{ind}_elite_*` from the staged copies.
6. Append the row to `pass_reference.csv`.

Stage to disk, not memory: two full sets is 142 MB per industry on a box already mlocking ~720 MB
with ~500 MB free. Disk is the cheaper resource.

### Edge cases

- **No reference file** (first boundary, after pass 1): pass 1 is champion by default, seed 100%
  from it, write the row with an empty `chal_pct`. Matches today's behaviour.
- **Pass shorter than 15 days** (diagnostic runs): skip the gate, seed 100% from the finishing
  pass, write no row.
- **`--no-save`**: write neither the reference file nor the champion store.
- **`--master-only`**: StockNN is not trained, so the gate is meaningless — skip.
- **Missing or corrupt champion file**: fall back to 100% from the finishing pass and log it
  loudly. Silent fallback is precisely `load_bin`'s failure mode, and it has cost a run before.

### Acceptance criteria for the implementation

After a 2-pass run: `pass_reference.csv` holds 12 rows, `champion/` holds 240 files, and replaying
the recorded `champ_pct` / `chal_pct` through the branch rule reproduces the recorded `slots_new`
exactly.

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

Storage: 20 × 12 × 3.54 MB = **850 MB** for the champion store. The finishing pass's elites
already sit in the run directory, and the staging copies add another 850 MB transiently
during a boundary only — so peak additional cost is ~1.7 GB against 44 GB free.

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
