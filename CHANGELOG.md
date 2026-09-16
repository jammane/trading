# Changelog

All notable changes to this project are documented here.
Entries are prepended automatically on each commit via the Claude commit hook.

---

## [7ef9dc1] — 2026-09-16
v0.6.8.0: --no-orders for catch-up runs

## [f4e4a7f] — 2026-09-16
v0.6.7.0: log the mutation success rate (Rechenberg's 1/5 statistic)

## [0eae448] — 2026-09-16
v0.6.6.0: --no-save no longer disables training; --load-dir is a seed again

## [4800edf] — 2026-09-16
v0.6.5.0: training pauses while a production cycle is running

## [a24e124] — 2026-09-15
convert_weights: select the source directory by parameter

## [b130eee] — 2026-09-14
v0.6.4.0: clock-seed the RNG, re-seeded at the start of every pass

## [373a77e] — 2026-09-14
v0.6.3.0: per-industry proportional pass-boundary seeding

## [04cfd44] — 2026-09-13
PASS_SEEDING: specify the pass-reference file and champion store

## [f44c685] — 2026-09-13
Stage per-industry proportional pass-boundary seeding (StockNN)

## [104c1cd] — 2026-09-12
Reconcile the MT1 param constants; delete the dead single-trunk block

## [efd7a97] — 2026-09-12
Stage (H-L)/A as a future MT1 vol-channel input

## [d2d7a43] — 2026-09-12
v0.6.2.0: --flat-allocation for paper; fix the production_v2 import break

## [3884674] — 2026-09-12
Swap TMHC -> CCS: Taylor Morrison delisted

## [a79f007] — 2026-09-12
download_daily: detect dead tickers instead of reporting them as healthy

## [2570e6b] — 2026-09-12
v0.6.1.0: StockNN close_pos + close_vs_wap per-symbol features (BREAKING)

## [8b27044] — 2026-06-01
Correct development status: paper trading untested

## [5804d35] — 2026-06-01
Add docstrings, linting config, and update README for public release

## [415d8c7] — 2026-06-01
Merge pull request #1 from jammane/add-claude-github-actions-1780331371873

## [d690f87] — 2026-06-01
"Claude Code Review workflow"

## [4dd3875] — 2026-06-01
"Claude PR Assistant workflow"

## [859d836] — 2026-05-12
Cleaned up repo: removing models and stock data from history

## [5ceaa2a] — 2026-04-10
Initial commit
