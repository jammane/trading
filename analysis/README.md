# Side-test analysis scripts

One-shot studies run against a finished pass (`mt1_dataset.bin`, `holdings_log.csv`,
`bayes_online.csv`). Kept because each one encodes a control or a correction that was expensive to
find, and re-deriving them is how the same mistake gets made twice.

| script | question | headline result |
|---|---|---|
| `tiered.py` | tier-0/1/2 allocation overlay vs flat | +14.1pp at W=60, but t < 1.2 |
| `interval.py` | the same across look-backs 5..120, daily and every-d | monotone 5 > 10 > 20 > 30 |
| `short_d.py` | sub-5-day look-backs, turnover and cost sensitivity | 43.5%/day turnover; negative net of 10bp |
| `quarterly.py` | is per-industry efficiency persistent, is there a trend | split-half **-0.223**; no persistence |
| `qtrly_live.py` | quarterly scorecard of the ONLINE trees in a live pass | k=1 converges to exactly always-up |
| `qtrly_offline.py` | same metric, offline, across the expanded interval set | best cell -0.36pp |
| `qtrly_start.py` | does feeding the trees the first 400 days hurt | no: 0.03pp difference |

## Controls these scripts carry, and why

**Permutation control on every allocation result.** `tiered.py` / `interval.py` / `short_d.py`
re-run with the *same weight vector* assigned to a random permutation of industries. The rolling
tier rule's apparent +14 to +88pp sits inside that null in every cell -- the shape of the weights
does the work, not the choice of industry.

**Transaction costs.** `short_d.py` charges turnover. Without it a 43.5%/day strategy looks
tradeable; with 10bp per side the best cell goes from +5.71 to -2.99 bp/day.

**Log differences, not arithmetic.** `quarterly.py` uses `log(1+r) - log(1+bh)` so quarterly
figures sum to the whole-period gap. The arithmetic version put hardware at +108pp where the
decomposable measure says +2.8%.

**Identical scoring days across variants.** `qtrly_start.py` scores both learn-from-d17 and
learn-from-d400 on the same days, or the comparison measures the day set rather than the change.
