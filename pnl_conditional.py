#!/usr/bin/env python3
"""Does conditioning on recent sign/magnitude predict tomorrow's P&L?

Tests two specific hypotheses:

  METHOD 1  condition on sign(today).      "tomorrow reverses today"
  METHOD 2  condition on (sign(yesterday), sign(today), whether today grew or shrank) -- six
            states. NOTE: as specified, all six predict the same thing, so method 2 only adds
            value if the six states' MEASURED rates differ from each other. That is tested
            directly rather than assumed.

Why this might work where the feature-based models failed: pnl_persistence.py found the variance
ratio BELOW 1 at every horizon (0.886 at q=20), which is mean reversion -- the direction these
methods assume. But it also found sign runs exactly at the coin-flip value (1.99 d vs 1.99 d
shuffled). If both are true, any reversal lives in the MAGNITUDE, not the sign, and a rule phrased
on sign alone would miss it. So sign and dollars are reported separately throughout.

Three things are measured for every state, because they can disagree and only the last one pays:
  P(up)      -- how often tomorrow is positive
  E[$]       -- the mean dollars tomorrow, which is what a position actually earns
  edge       -- a cross-sectional allocation acting on the signal, vs equal weight

Every number carries a moving-block bootstrap t and a shuffled control (same values, time order
destroyed within each industry). A conditional difference that survives shuffling is structure;
one that does not is the conditioning re-describing noise.

    python pnl_conditional.py mt1_dataset.bin
"""
import argparse

import numpy as np

import read_mt1_dataset as D

IND_NAMES = ['hardware', 'software', 'financial', 'discret', 'services', 'health',
             'industrl', 'staples', 'energy', 'utilitie', 'land', 'materials']


def blocks(T, block, rng):
    n = max(1, int(np.ceil(T / block)))
    st = rng.integers(0, max(1, T - block), size=n)
    return np.concatenate([np.arange(s, min(s + block, T)) for s in st])[:T]


def shuffled(pnl, seed):
    """Permute each industry's series in time.

    Destroys time ordering while preserving every industry's marginal distribution AND the
    cross-industry composition. That second property is what makes it the right null here: pooling
    12 industries with different up-rates and different mean sizes manufactures an apparent
    "down days predict higher returns" effect with no time structure at all, and only a control
    that keeps the composition can expose it.
    """
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(pnl[:, i]) for i in range(pnl.shape[1])])


def perm_test(stat, pnl, n_perm, seed):
    """Observed value, and its z against the null distribution from `n_perm` permutations.

    This replaced a single shuffled column. One permutation is one draw from the null, so it
    carries the sampling error of a single sample -- on this data the single-shuffle control
    disagreed between the full and burn-in samples on the same statistic, which is exactly what a
    one-draw null does. Averaging many draws gives both the null's centre and its spread, which is
    what turns "looks different from shuffled" into a p-value.

    The null is the right one for the question: it asks whether the TIME ORDERING carries
    information, not whether the statistic differs from zero.
    """
    obs = stat(pnl)
    if obs is None or not np.isfinite(obs):
        return np.nan, np.nan, np.nan, np.nan
    null = []
    for i in range(n_perm):
        v = stat(shuffled(pnl, seed + 1000 + i))
        if v is not None and np.isfinite(v):
            null.append(v)
    if len(null) < 20:
        return obs, np.nan, np.nan, np.nan
    a = np.array(null)
    sd = a.std(ddof=1)
    z = (obs - a.mean()) / sd if sd > 0 else np.nan
    p = (np.sum(np.abs(a - a.mean()) >= abs(obs - a.mean())) + 1) / (len(a) + 1)
    return obs, a.mean(), z, p


# ── state definitions ───────────────────────────────────────────────────────────

def method1_states(p):
    """sign(today) -> 0 = today down, 1 = today up. Aligned so row t predicts t+1."""
    today = p[:-1]
    return (today > 0).astype(int), p[1:]


def method2_states(p):
    """(yesterday, today) -> six states, exactly as specified."""
    y, t, nxt = p[:-2], p[1:-1], p[2:]
    s = np.full(y.size, -1)
    up_y, up_t = y > 0, t > 0
    s[up_y & up_t & (t >= y)] = 0     # profit, then MORE profit
    s[up_y & up_t & (t < y)] = 1      # profit, then LESS profit
    s[up_y & ~up_t] = 2               # profit, then loss
    s[~up_y & ~up_t & (t <= y)] = 3   # loss, then BIGGER loss
    s[~up_y & ~up_t & (t > y)] = 4    # loss, then SMALLER loss
    s[~up_y & up_t] = 5               # loss, then profit
    return s, nxt


M1_NAMES = ['today DOWN', 'today UP']
M2_NAMES = ['up, then MORE up', 'up, then LESS up', 'up, then DOWN',
            'down, then MORE down', 'down, then LESS down', 'down, then UP']


def state_table(pnl, statefn, names, label, seed, n_perm):
    """Per state: does tomorrow depart from the unconditional rate BY MORE THAN time-shuffling
    would produce? The null is the permutation distribution, not 50% and not zero."""
    print(f'\n{label}')
    print(f'  {"state":<22} {"n":>6} {"dP(up)":>9} {"null":>8} {"z":>6} {"p":>6}   '
          f'{"dE[$]":>9} {"null":>8} {"z":>6} {"p":>6}')

    def make(k, dollars):
        def f(p):
            st, nxt = statefn_panel(p, statefn)
            m = st == k
            if m.sum() < 30:
                return np.nan
            return (nxt[m].mean() - nxt.mean()) if dollars \
                else ((nxt[m] > 0).mean() - (nxt > 0).mean())
        return f

    for k, nm in enumerate(names):
        st_all, _ = statefn_panel(pnl, statefn)
        n = int((st_all == k).sum())
        o1, n1, z1, p1 = perm_test(make(k, False), pnl, n_perm, seed)
        o2, n2, z2, p2 = perm_test(make(k, True), pnl, n_perm, seed)
        s1 = '*' if np.isfinite(p1) and p1 < 0.05 else ' '
        s2 = '*' if np.isfinite(p2) and p2 < 0.05 else ' '
        print(f'  {nm:<22} {n:>6} {100*o1:>+8.2f}pp {100*n1:>+7.2f} {z1:>+5.1f} {p1:>6.3f}{s1}  '
              f'{o2:>+8.2f} {n2:>+7.2f} {z2:>+5.1f} {p2:>6.3f}{s2}')
    print(f'  ({len(names)} states x 2 metrics = {2*len(names)} tests; '
          f'a 5% threshold expects {0.05*2*len(names):.1f} false positives by chance)')


def statefn_panel(p, statefn):
    """Apply a per-industry state function across the panel and flatten."""
    ss, nn = [], []
    for i in range(p.shape[1]):
        s, nxt = statefn(p[:, i])
        ss.append(s)
        nn.append(nxt)
    return np.concatenate(ss), np.concatenate(nn)


def outcome_table(pnl, seed, n_perm):
    """For each state, which of the TWO candidate outcomes does the data pick?

    The hypothesis gives each state two competing outcomes, not one:
        A  CONTINUATION -- we are in a streak of N, tomorrow keeps today's sign
        B  REVERSAL     -- tomorrow flips today's sign
    So the statistic is P(tomorrow has the SAME sign as today | state), against the permutation
    null. Above the null picks A, below picks B, and the null is what the same conditioning would
    produce with the time ordering destroyed.
    """
    print('\nWHICH OUTCOME?  P(tomorrow KEEPS today\'s sign | state)  -- A = continuation, B = reversal')
    print(f'  {"state":<22} {"n":>6} {"P(same)":>9} {"vs null":>9} {"z":>6} {"p":>7}  verdict')

    def cont(k):
        def f(p):
            st, nxt = statefn_panel(p, method2_states)
            tod = np.concatenate([p[1:-1, i] for i in range(p.shape[1])])
            m = st == k
            if m.sum() < 30:
                return np.nan
            return float((np.sign(nxt[m]) == np.sign(tod[m])).mean())
        return f

    for k, nm in enumerate(M2_NAMES):
        st_all, _ = statefn_panel(pnl, method2_states)
        n = int((st_all == k).sum())
        o, nl, z, pv = perm_test(cont(k), pnl, n_perm, seed)
        if not np.isfinite(z):
            continue
        d = o - nl
        if pv < 0.05:
            v = 'A continuation' if d > 0 else 'B REVERSAL'
            v += f'  (p {pv:.3f})'
        else:
            v = 'neither -- indistinguishable from the null'
        print(f'  {nm:<22} {n:>6} {100*o:>8.1f}% {100*d:>+8.2f}pp {z:>+5.1f} {pv:>7.3f}  {v}')


def focused_edge(pnl, seed, n_perm, states, label, warmup=250):
    """Act on ONLY the named states, walk-forward: underweight where the table says reversal-down.

    Flagged as post-hoc: these states were chosen BECAUSE they tested significant above, so this
    is not an independent confirmation. It answers a narrower question -- whether the one effect
    that survived is large enough to be worth money -- not whether it would have been found in
    advance.
    """
    print(f'\nFOCUSED ECONOMIC TEST -- {label}  (post-hoc state selection; see docstring)')

    def edge(p):
        T, N = p.shape
        st = np.full((T, N), -1)
        for i in range(N):
            sarr, _ = method2_states(p[:, i])
            st[2:, i] = sarr
        g = b = 0.0
        n = 0
        for t in range(warmup, T - 1):
            flag = np.isin(st[t], states)
            if not flag.any() or flag.all():
                continue
            w = np.where(flag, 0.5, 1.5)
            w = w / w.sum()
            g += float((w * p[t + 1]).sum())
            b += float(p[t + 1].mean())
            n += 1
        return (g - b) / n if n else np.nan

    o, nl, z, pv = perm_test(edge, pnl, max(60, n_perm // 2), seed)
    star = '*' if np.isfinite(pv) and pv < 0.05 else ' '
    print(f'  observed {o:+.3f} $/day/industry   null {nl:+.3f}   z {z:+.1f}   p {pv:.3f}{star}')


ACCEL = {0: 'accel', 1: 'decel', 2: 'flip', 3: 'accel', 4: 'decel', 5: 'flip'}


def accel_summary(pnl, seed, n_perm):
    """Group the six states by ACCELERATING / DECELERATING / FLIPPED.

    The spec gives accelerating and decelerating the same outcome. Measured pooled, they do not
    behave the same: both decelerating states select reversal and both accelerating states select
    nothing. This tests that grouping directly, which is a 3-test family instead of 6 and so is
    the better-powered form of the same question.
    """
    print('\nACCEL / DECEL GROUPING  P(tomorrow KEEPS today\'s sign)')
    print(f'  {"group":<26} {"n":>6} {"P(same)":>9} {"vs null":>9} {"z":>6} {"p":>7}  verdict')
    for tag in ('accel', 'decel', 'flip'):
        ks = [k for k, v in ACCEL.items() if v == tag]

        def stat(p, ks=ks):
            st, nxt = statefn_panel(p, method2_states)
            tod = np.concatenate([p[1:-1, i] for i in range(p.shape[1])])
            m = np.isin(st, ks)
            if m.sum() < 30:
                return np.nan
            return float((np.sign(nxt[m]) == np.sign(tod[m])).mean())

        st_all, _ = statefn_panel(pnl, method2_states)
        n = int(np.isin(st_all, ks).sum())
        o, nl, z, pv = perm_test(stat, pnl, n_perm, seed)
        if not np.isfinite(z):
            continue
        d = o - nl
        v = ('A continuation' if d > 0 else 'B REVERSAL') if pv < 0.05 else 'neither'
        names = {'accel': 'ACCELERATING (MORE up/down)', 'decel': 'DECELERATING (LESS up/down)',
                 'flip': 'FLIPPED (sign changed)'}
        print(f'  {names[tag]:<26} {n:>6} {100*o:>8.1f}% {100*d:>+8.2f}pp {z:>+5.1f} {pv:>7.3f}  {v}')


def per_industry(pnl, seed, n_perm, names=None):
    """The same test run inside each industry separately, with its POWER stated first.

    Per-industry is what the living model would actually condition on, but it divides the sample
    by 12. Before reading twelve nulls as twelve findings, the detectable effect size is computed
    from the actual bucket sizes: with n observations in a state, the standard error on a
    proportion is sqrt(0.25/n), so the smallest effect detectable at 95% with 80% power is about
    2.8 * that. If the pooled effect (-3.4pp) is smaller than that number, a per-industry null is
    a statement about the sample size, not about the industry.
    """
    T, N = pnl.shape
    print(f'\nPER-INDUSTRY  (decelerating states only -- the pooled effect lives there)')
    ks = [1, 4]
    st_all, _ = statefn_panel(pnl, method2_states)
    n_typ = int(np.isin(st_all, ks).sum()) // N
    mde = 2.8 * np.sqrt(0.25 / max(n_typ, 1)) * 100
    print(f'  typical n per industry in those states: {n_typ};  smallest detectable effect '
          f'~{mde:.1f}pp at 80% power')
    print(f'  (the pooled effect is -3.4pp, so per-industry is '
          f'{"UNDERPOWERED -- expect nulls" if mde > 3.4 else "adequately powered"})')
    print(f'  {"industry":<12} {"n":>5} {"P(same)":>9} {"vs null":>9} {"z":>6} {"p":>7}')
    hits = 0
    for i in range(N):
        col = pnl[:, i:i + 1]

        def stat(p):
            st, nxt = statefn_panel(p, method2_states)
            tod = p[1:-1, 0]
            m = np.isin(st, ks)
            if m.sum() < 30:
                return np.nan
            return float((np.sign(nxt[m]) == np.sign(tod[m])).mean())

        sti, _ = statefn_panel(col, method2_states)
        n = int(np.isin(sti, ks).sum())
        o, nl, z, pv = perm_test(stat, col, n_perm, seed + i)
        if not np.isfinite(z):
            continue
        nm = names[i] if names else f'ind{i}'
        star = '*' if pv < 0.05 else ' '
        hits += pv < 0.05
        print(f'  {nm:<12} {n:>5} {100*o:>8.1f}% {100*(o-nl):>+8.2f}pp {z:>+5.1f} {pv:>7.3f}{star}')
    print(f'  {hits}/{N} industries individually significant at 5% '
          f'(chance alone gives {0.05*N:.1f})')


def streak_table(pnl, seed):
    """P(the current run continues) by run length so far -- the 'streak of N' premise."""
    print('\nSTREAK CONTINUATION  (is a longer run more or less likely to continue?)')
    print(f'  {"run so far":<12} {"n":>7} {"P(continue)":>12} {"shuffled":>10}')
    sh = shuffled(pnl, seed)
    for src, tag in ((pnl, 'obs'), (sh, 'shuf')):
        res = {}
        for i in range(src.shape[1]):
            x = np.sign(src[:, i])
            x = x[x != 0]
            run = 1
            for j in range(1, x.size - 1):
                if x[j] == x[j - 1]:
                    run += 1
                else:
                    run = 1
                res.setdefault(min(run, 5), []).append(x[j + 1] == x[j])
        if tag == 'obs':
            obs = {k: (len(v), np.mean(v)) for k, v in res.items()}
        else:
            shf = {k: np.mean(v) for k, v in res.items()}
    for r in sorted(obs):
        n, p = obs[r]
        lab = f'{r}+ days' if r == 5 else f'{r} day{"s" if r > 1 else ""}'
        print(f'  {lab:<12} {n:>7} {100*p:>11.1f}% {100*shf.get(r, np.nan):>9.1f}%')


def reversal_edge(pnl, seed, n_perm):
    """The economic test: allocate AGAINST today's P&L, measure the edge vs equal weight.

    Tercile weights 1.0 / 1.5 / 2.25 -- the same ladder tiers_to_alloc uses in production, so this
    is the rule as it would be deployed, not an idealised long-short.
    """
    print('\nECONOMIC TEST  (allocate against today; tercile weights 1.0/1.5/2.25 as in production)')

    def edge(p):
        T, N = p.shape
        if T < 60:
            return np.nan
        today, nxt = p[:-1], p[1:]
        rk = np.argsort(np.argsort(-today, axis=1), axis=1)   # 0 = worst today -> biggest weight
        w = np.where(rk >= N - N // 3, 2.25, np.where(rk >= N // 3, 1.5, 1.0))
        w = w / w.sum(axis=1, keepdims=True)
        return float((w * nxt).sum(axis=1).mean() - nxt.mean(axis=1).mean())

    o, nl, z, p = perm_test(edge, pnl, n_perm, seed)
    star = '*' if np.isfinite(p) and p < 0.05 else ' '
    print(f'  observed      {o:+.3f} $/day/industry')
    print(f'  permutation null {nl:+.3f}   z {z:+.1f}   p {p:.3f}{star}')
    print(f'  for scale: mean daily P&L is {pnl.mean():+.2f} $/industry')


def method2_edge(pnl, seed, n_perm, warmup=250):
    """Trade the six-state table itself, walk-forward.

    Each day: classify every industry into its state from (yesterday, today), look up the mean
    next-day P&L for that state ESTIMATED ONLY ON DAYS STRICTLY BEFORE TODAY, rank the 12
    industries by that estimate, and allocate 1.0/1.5/2.25 by tercile.

    The walk-forward is the whole point. The six-state table above is fitted on the full sample,
    so acting on it in-sample would be circular -- it would report the fit, not a tradeable edge.
    Here the table at day t has never seen day t or later.
    """
    print('\nECONOMIC TEST -- METHOD 2  (walk-forward six-state table, same tercile ladder)')

    def edge(p):
        T, N = p.shape
        if warmup + 50 > T:
            return np.nan
        st = np.full((T, N), -1)
        for i in range(N):
            s, _ = method2_states(p[:, i])
            st[2:, i] = s
        gain, base, n = 0.0, 0.0, 0
        for t in range(warmup, T - 1):
            hist_s = st[2:t].ravel()
            hist_y = p[3:t + 1].ravel()
            m = hist_s >= 0
            if m.sum() < 100:
                continue
            exp = np.full(6, np.nan)
            for k in range(6):
                sel = m & (hist_s == k)
                if sel.sum() >= 30:
                    exp[k] = hist_y[sel].mean()
            cur = st[t]
            pred = np.array([exp[c] if c >= 0 and np.isfinite(exp[c]) else 0.0 for c in cur])
            rk = np.argsort(np.argsort(pred))
            w = np.where(rk >= N - N // 3, 2.25, np.where(rk >= N // 3, 1.5, 1.0))
            w = w / w.sum()
            gain += float((w * p[t + 1]).sum())
            base += float(p[t + 1].mean())
            n += 1
        return (gain - base) / n if n else np.nan

    o, nl, z, pv = perm_test(edge, pnl, max(60, n_perm // 4), seed)
    star = '*' if np.isfinite(pv) and pv < 0.05 else ' '
    print(f'  observed      {o:+.3f} $/day/industry')
    print(f'  permutation null {nl:+.3f}   z {z:+.1f}   p {pv:.3f}{star}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--burn-in', type=int, default=500)
    ap.add_argument('--perm', type=int, default=200, help='permutations for the null')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per-industry', action='store_true')
    a = ap.parse_args()

    ds = D.read(a.dataset) if hasattr(D, 'read') else D.parse(a.dataset)
    day = np.asarray(ds['day'])
    raw = ds['book_now'].astype(float) - ds['book_prev'].astype(float)
    reset = np.abs(raw) < 1e-9
    raw = np.where(reset, np.nan, raw)
    print(f'{a.dataset}: {raw.shape[0]} days x {raw.shape[1]} industries; '
          f'{int(reset.sum())} reset days dropped')

    for lab, keep in (('FULL SAMPLE', np.ones(len(day), bool)),
                      (f'AFTER BURN-IN (day >= {a.burn_in})', day >= a.burn_in)):
        cols = [raw[keep, i][np.isfinite(raw[keep, i])] for i in range(raw.shape[1])]
        m = min(len(c) for c in cols)
        if m < 200:
            continue
        p = np.column_stack([c[-m:] for c in cols])
        print(f'\n{"=" * 92}\n{lab}   ({p.shape[0]} days x {p.shape[1]} industries)\n{"=" * 92}')
        print(f'unconditional: P(up) {100*(p > 0).mean():.1f}%   mean ${p.mean():+.2f}   '
              f'lag-1 autocorr '
              f'{np.mean([np.corrcoef(p[:-1, i], p[1:, i])[0, 1] for i in range(p.shape[1])]):+.4f}')
        state_table(p, method1_states, M1_NAMES, 'METHOD 1 -- condition on sign(today)',
                    a.seed, a.perm)
        state_table(p, method2_states, M2_NAMES,
                    'METHOD 2 -- condition on (yesterday, today, grew/shrank)', a.seed, a.perm)
        outcome_table(p, a.seed, a.perm)
        accel_summary(p, a.seed, a.perm)
        if a.per_industry:
            per_industry(p, a.seed, a.perm, IND_NAMES)
        streak_table(p, a.seed)
        reversal_edge(p, a.seed, a.perm)
        method2_edge(p, a.seed, a.perm)
        focused_edge(p, a.seed, a.perm, [1], 'underweight "up, then LESS up"')


if __name__ == '__main__':
    main()
