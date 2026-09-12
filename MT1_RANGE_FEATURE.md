# Staged: add `(H−L)/A` as an MT1 input

**Status: STAGED — do not implement yet.**
Two gates must clear first:

1. Paper trading is running (the v0.6.2.1 five-pass run finishes, models deploy, cron enabled).
2. The portfolio-vol validation below passes. It has **not** been run.

This is a BREAKING change to every MT1 `.bin`/`.pt`, so it must not land mid-run or mid-rollout.

---

## What to add

One feature: **`(H−L)/A`** where `A = (2O+3C+H+L)/7` — the day's range as a fraction of price,
averaged equal-weight across the industry's 12 names.

**Not** `close_pos`, and **not** `close_vs_wap`. Add the ratio itself. The two v0.6.x StockNN
inputs jointly encode it (`close_pos / close_vs_wap = A/(H−L)`), but a linear layer cannot
divide — the same reasoning that made `close_pos` additive to StockNN. Feeding both and hoping
the net forms the quotient reproduces the exact gap the feature is meant to close.

## Why — measured 2026-09-12 on 144 symbols × 1255 sessions (2021-09-13 → 2026-09-11)

All 37 existing MT1 features derive from **one cumulative close-value series** (daily returns,
10-day bucket returns, mean |return|, polynomial fits of the level path). There is no open, high
or low anywhere in them, so any O/H/L-derived quantity is outside their span by construction.

Against forward `MT1_VOL_DAYS = 20` realized vol, 732 non-overlapping industry-windows:

```
corr(trailing mean|ret|  = MT1 feature[17],  fwd 20d vol) = +0.6650
corr((H−L)/A,                                fwd 20d vol) = +0.5282
corr(|close_pos|,                            fwd 20d vol) = +0.2122

R²  trailing vol alone       = 0.4422
R²  trailing vol + (H−L)/A   = 0.4760     delta +0.0338   (+7.6% relative)
```

`close_pos` alone is weak (0.212). The ratio is what carries the signal.

## Why NOT the direction/delta channel

Against MT1's direction target — 10-day forward industry return vs the cross-sectional median,
sampled every 10th day so windows never overlap (n=123):

```
close_pos              rank IC -0.0055   t -0.17
close_vs_wap                   -0.0088   t -0.28
(H−L)/A                        -0.0151   t -0.52
CONTROL trailing 10d return    -0.0030   t -0.10
```

Nothing — **including the control**, which MT1 already receives as features [10..16]. At the
cross-industry level over 10 days, nothing here predicts relative returns. This is consistent
with direction scoring negative OOS skill in all 12 industries on every run to date, and it
means adding inputs will not fix that channel. Scope this change to the vol channel only.

## BLOCKING prerequisite — validate against *portfolio* vol

The measurement above used the **industry index's** forward vol. MT1's actual target is the
forward vol of the **deployed slot-0 portfolio**.

This distinction has already bitten once. The v0.6.0.0-era study found range estimators
(Parkinson / Garman–Klass / Rogers–Satchell) tie close-to-close on raw correlation (0.561–0.568
vs 0.568) but go **negative OOS R² (−0.01 to −0.03 vs +0.17)** specifically against the
portfolio target — "biased for a portfolio target, scale doesn't transfer."

That study used them as direct *estimators*, where scale must transfer; this proposal uses
`(H−L)/A` as an incremental *input* where the model learns the mapping. Genuinely different
question — but close enough to the prior negative result that it must be re-measured before
committing.

**The validation:** recompute the R² table above with the target replaced by the forward 20-day
realized vol of slot-0's deployed portfolio. Needs per-day slot-0 holdings, which requires a run
with saved weights. `ht_train_v060a` ran `--no-save` and cannot supply them; the v0.6.2.1 run
in flight **does** save weights, so `inspect_trades.py` can reconstruct them.

Proceed only if the incremental R² survives against portfolio vol. If it does not, close this
out — the index-vol result is then explained by the same bias the earlier study found.

## Implementation checklist

Feature count goes 37 → 38, dual input 74 → 76, master vector 888 → 912.

- `training_lib._build_ind_features37` (~line 634) — append the new feature; rename to match.
- `build_ind_features37` in `training_v4.cpp` (~line 1920) — **C++ twin, keep bit-identical**.
  Called twice at ~1974-1975 with offsets 0 and 37; both offsets shift.
- `models.MT1Head` — slices `x[:, 0:10], x[:, 10:17], x[:, 17:37]`; the third slice widens.
- `models.MT1DualHead` — input 74 → 76; `self.mkt` reads `x[:, 0:37]`, that boundary moves.
- Param constants **disagree today and must be reconciled during implementation**: CLAUDE.md says
  MT1NN is "37→4, ~3,412 params" (the 37 is already stale — it is 74), `models.py` docstring says
  9,208, `training_v4.cpp` says `MT1NN_PARAMS = 2218`. `load_bin` validates by exact element count
  and **falls back to random init silently** on mismatch, so a wrong constant produces a run that
  looks fine and has learned nothing.
- Tests: mirror the `TestTodayLayout` pattern — pin the offsets on both sides so a Python/C++
  divergence fails loudly instead of misaligning every feature past the first block.
- Regenerate MT1 models from scratch. Every `mt1_*.bin` / `.pt` from v0.6.x and earlier becomes
  unloadable. Start from a clean `--output` directory.
- Version: BREAKING → `0.7.0.0`.

## Deliberately out of scope

- Changing StockNN. `close_pos` and `close_vs_wap` stay exactly as they are in slots 15 and 16.
- The direction, accuracy and confidence channels.
