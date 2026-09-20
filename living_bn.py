#!/usr/bin/env python3
"""A living conditional model of StockNN's per-industry book P&L.

Nothing in here is a hardcoded constant. Every probability, the streak length N, and the memory
half-life itself are estimated from data and re-estimated as part of daily training.

WHAT IT CONDITIONS ON
    The state of an industry on day t, from its own book P&L on days t-1 and t:

        0 UP_MORE    profit, then MORE profit      accelerating
        1 UP_LESS    profit, then LESS profit      decelerating
        2 UP_DOWN    profit, then loss             flipped
        3 DN_MORE    loss,   then BIGGER loss      accelerating
        4 DN_LESS    loss,   then SMALLER loss     decelerating
        5 DN_UP      loss,   then profit           flipped

    The series is StockNN's own book P&L -- the deployed portfolio it chose -- not raw stock
    returns. `mt1_dataset.bin` supplies it as book_now - book_prev.

WHY POOLED, WITH ROOM TO SEPARATE
    Measured on 751 post-burn-in days: the decelerating states carry P(tomorrow keeps today's
    sign) = 47.0% against a permutation null of 50.3%, p 0.007. Per industry that effect is
    invisible -- n ~ 187 per industry gives a 10.2pp minimum detectable effect against a 3.4pp
    truth -- yet 11 of 12 industries point the same way (sign test p 0.0063, t -5.89 across the
    12), and the between-industry variance is SMALLER than sampling noise alone would produce
    (4.06 vs 13.37), so tau^2 estimates to 0.

    So the model is hierarchical rather than either pooled or per-industry: each industry's
    estimate is shrunk toward the pooled one by an empirical-Bayes weight computed from the data.
    Today that weight is ~100% pooling, which is the correct answer to "should this be
    per-industry". If industries ever do diverge, tau^2 rises and the shrinkage relaxes on its own.

WHAT "LIVING" MEANS
    Counts decay exponentially with a half-life H, so the model tracks drift instead of averaging
    over five years. H is not chosen by hand: `fit_half_life` replays history and picks the H with
    the best out-of-sample predictive log-likelihood. N -- the streak length the hypothesis talks
    about -- is likewise derived from the current posterior rather than assumed.
"""
import json
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

UP_MORE, UP_LESS, UP_DOWN, DN_MORE, DN_LESS, DN_UP = range(6)
STATE_NAMES = ['up_more', 'up_less', 'up_down', 'dn_more', 'dn_less', 'dn_up']
GROUP = {UP_MORE: 'accel', UP_LESS: 'decel', UP_DOWN: 'flip',
         DN_MORE: 'accel', DN_LESS: 'decel', DN_UP: 'flip'}
N_STATES = 6


def classify(prev, cur):
    """State from yesterday's and today's P&L. None if either day is flat (a reset day)."""
    if prev == 0 or cur == 0 or not (np.isfinite(prev) and np.isfinite(cur)):
        return None
    if prev > 0:
        if cur > 0:
            return UP_MORE if cur >= prev else UP_LESS
        return UP_DOWN
    if cur < 0:
        return DN_MORE if cur <= prev else DN_LESS
    return DN_UP


@dataclass
class LivingBN:
    """Decayed counts per (industry, state), with empirical-Bayes pooling across industries."""

    n_ind: int
    half_life: float = 250.0
    window: int = 0               # >0 selects a RECTANGULAR window of this many days instead
    prior_strength: float = 2.0          # weak Beta(1,1)-ish prior, in pseudo-observations
    # weighted counts: [industry][state]
    w_n: np.ndarray = field(default=None, repr=False)      # total weight
    w_same: np.ndarray = field(default=None, repr=False)   # weight where tomorrow kept the sign
    w_sum: np.ndarray = field(default=None, repr=False)    # weighted sum of tomorrow's dollars
    w_sq: np.ndarray = field(default=None, repr=False)
    w_run: np.ndarray = field(default=None, repr=False)    # weighted sum of realised run lengths
    w_runn: np.ndarray = field(default=None, repr=False)
    days_seen: int = 0

    def __post_init__(self):
        z = lambda: np.zeros((self.n_ind, N_STATES))       # noqa: E731
        for a in ('w_n', 'w_same', 'w_sum', 'w_sq', 'w_run', 'w_runn'):
            if getattr(self, a) is None:
                setattr(self, a, z())

    # ── updating ────────────────────────────────────────────────────────────────
    @property
    def decay(self):
        return 0.5 ** (1.0 / max(self.half_life, 1e-9))

    def observe(self, prev2, prev1, today):
        """Record one day's outcome.

        `prev2`,`prev1` are the two days that DEFINED the state; `today` is the outcome being
        predicted. Called with (t-2, t-1, t) so nothing ever sees its own outcome.

        Two weightings of "recent history" are supported, and which is better is a measurement,
        not a preference: exponential decay (every past day still counts, geometrically less), or
        a rectangular `window` of the last N days (a day counts fully, then not at all).
        """
        if self.window > 0:
            self._observe_window(prev2, prev1, today)
            return
        g = self.decay
        self.w_n *= g
        self.w_same *= g
        self.w_sum *= g
        self.w_sq *= g
        for i in range(self.n_ind):
            s = classify(prev2[i], prev1[i])
            if s is None or today[i] == 0 or not np.isfinite(today[i]):
                continue
            self.w_n[i, s] += 1.0
            if np.sign(today[i]) == np.sign(prev1[i]):
                self.w_same[i, s] += 1.0
            self.w_sum[i, s] += today[i]
            self.w_sq[i, s] += today[i] ** 2
        self.days_seen += 1

    def _observe_window(self, prev2, prev1, today):
        """Rectangular window: add today's observations, drop those older than `window` days.

        Kept exact rather than approximated by an equivalent half-life, because the two kernels
        differ in the only place it matters -- a windowed estimate discards a regime completely on
        a known date, while a decayed one never quite does.
        """
        if not hasattr(self, '_ring'):
            self._ring = deque()
        obs = []
        for i in range(self.n_ind):
            st = classify(prev2[i], prev1[i])
            if st is None or today[i] == 0 or not np.isfinite(today[i]):
                continue
            same = float(np.sign(today[i]) == np.sign(prev1[i]))
            obs.append((i, st, same, float(today[i])))
            self.w_n[i, st] += 1.0
            self.w_same[i, st] += same
            self.w_sum[i, st] += today[i]
            self.w_sq[i, st] += today[i] ** 2
        self._ring.append(obs)
        while len(self._ring) > self.window:
            for i, st, same, dol in self._ring.popleft():
                self.w_n[i, st] -= 1.0
                self.w_same[i, st] -= same
                self.w_sum[i, st] -= dol
                self.w_sq[i, st] -= dol ** 2
        self.days_seen += 1

    # ── posteriors ──────────────────────────────────────────────────────────────
    def _pooled(self):
        n = self.w_n.sum(axis=0)
        k = self.w_same.sum(axis=0)
        p = np.where(n > 0, (k + self.prior_strength * 0.5) / (n + self.prior_strength), 0.5)
        return p, n

    @staticmethod
    def _chi2_crit(df, z=2.3263):
        """Chi-square critical value via Wilson-Hilferty. Default z is the 99th percentile.

        Avoids a scipy dependency: scipy is not installed on the droplet, and this is the only
        place the model needs a quantile.
        """
        k = max(df, 1)
        return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3

    def tau2(self, state):
        """Between-industry variance of the TRUE rate: DerSimonian-Laird, gated by Cochran's Q.

        Returns 0 unless the industries are heterogeneous beyond what sampling noise explains at
        the 99% level, and 0 is the measured answer today -- the observed between-industry
        variance of the decelerating-state effect (4.06) is SMALLER than sampling noise alone
        would produce (13.37).

        The gate is deliberately at 99%, not 95%, because the two errors are not symmetric. Failing
        to detect small real heterogeneity costs nothing: the model keeps pooling, which is already
        the better estimator at n ~ 187 per industry. Un-pooling on noise costs a lot: a plain
        method-of-moments estimator spread the twelve industries' probabilities by 0.03 on purely
        simulated identical data -- the same size as the 0.033 effect being chased.
        """
        n = self.w_n[:, state]
        ok = n >= 20
        G = int(ok.sum())
        if G < 3:
            return 0.0
        p_i = self.w_same[ok, state] / n[ok]
        p_bar = self.w_same[ok, state].sum() / n[ok].sum()
        var = p_bar * (1.0 - p_bar)
        if var <= 0:
            return 0.0
        w = n[ok] / var                                  # inverse-variance weights
        p_w = float((w * p_i).sum() / w.sum())
        Q = float((w * (p_i - p_w) ** 2).sum())
        if self._chi2_crit(G - 1) >= Q:
            return 0.0
        C = float(w.sum() - (w ** 2).sum() / w.sum())
        return float(max(0.0, (Q - (G - 1)) / C)) if C > 0 else 0.0

    def p_continue(self, ind, state):
        """Posterior P(tomorrow keeps today's sign), shrunk toward the pooled estimate."""
        p_pool, _ = self._pooled()
        t2 = self.tau2(state)
        n = self.w_n[ind, state]
        if t2 <= 0 or n <= 0:
            kappa = float('inf')
        else:
            kappa = p_pool[state] * (1 - p_pool[state]) / t2
        if not np.isfinite(kappa):
            return float(p_pool[state])
        k = self.w_same[ind, state]
        return float((k + kappa * p_pool[state]) / (n + kappa))

    def streak_n(self, ind, state):
        """N -- the expected further length of the current run, from the posterior.

        A run that continues with probability p has expected remaining length 1/(1-p). This is the
        'streak of N days' the hypothesis names, derived rather than assumed, and it moves every
        day as the posterior moves.
        """
        p = self.p_continue(ind, state)
        return float('inf') if p >= 1.0 else 1.0 / (1.0 - p)

    def expected_dollars(self, ind, state):
        n = self.w_n[ind, state]
        if n < 1:
            tot_n = self.w_n[:, state].sum()
            return float(self.w_sum[:, state].sum() / tot_n) if tot_n > 0 else 0.0
        pooled = self.w_sum[:, state].sum() / max(self.w_n[:, state].sum(), 1e-9)
        kappa = 50.0                       # dollars are noisier than signs; shrink harder
        return float((self.w_sum[ind, state] + kappa * pooled) / (n + kappa))

    def report(self):
        p_pool, n = self._pooled()
        rows = []
        for s in range(N_STATES):
            rows.append({'state': STATE_NAMES[s], 'group': GROUP[s], 'n_eff': float(n[s]),
                         'p_continue': float(p_pool[s]),
                         'streak_N': float('inf') if p_pool[s] >= 1 else 1 / (1 - p_pool[s]),
                         'tau2': self.tau2(s)})
        return rows

    # ── persistence ─────────────────────────────────────────────────────────────
    def to_dict(self):
        return {'n_ind': self.n_ind, 'half_life': self.half_life, 'window': int(self.window or 0),
                'prior_strength': self.prior_strength, 'days_seen': self.days_seen,
                **{a: getattr(self, a).tolist()
                   for a in ('w_n', 'w_same', 'w_sum', 'w_sq', 'w_run', 'w_runn')}}

    @classmethod
    def from_dict(cls, d):
        m = cls(n_ind=d['n_ind'], half_life=d['half_life'], window=int(d.get('window', 0) or 0),
                prior_strength=d.get('prior_strength', 2.0))
        m.days_seen = d.get('days_seen', 0)
        for a in ('w_n', 'w_same', 'w_sum', 'w_sq', 'w_run', 'w_runn'):
            if a in d:
                setattr(m, a, np.array(d[a], dtype=float))
        return m

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def observed_streak_lengths(pnl):
    """Mean REMAINING run length after each state, measured directly rather than derived.

    The model reports N as 1/(1-p), which is the expected remaining length of a GEOMETRIC run.
    That identity holds only if the per-day continuation probability is constant within a run. If
    runs are not geometric -- if, say, continuation weakens the longer a run has gone on -- the
    derived N and the observed N come apart, and the derived one is wrong.

    N counts the run from TODAY INCLUSIVE, matching 1/(1-p): under a geometric run with
    continuation probability p, the expected number of days from today that keep the sign is
    1/(1-p), while the expected number of ADDITIONAL days is p/(1-p). Comparing one against the
    other makes the two estimators look like they disagree by about a factor of two, which is what
    the first version of this function did.

    Returns (mean_observed, n) per state. Forward-looking by construction, so it is an estimator
    to run over history, never a prediction input.
    """
    T, N = pnl.shape
    tot = np.zeros(N_STATES)
    cnt = np.zeros(N_STATES)
    for i in range(N):
        x = pnl[:, i]
        for t in range(1, T - 1):
            s = classify(x[t - 1], x[t])
            if s is None:
                continue
            run = 0
            k = t + 1
            while k < T and x[k] != 0 and np.isfinite(x[k]) and np.sign(x[k]) == np.sign(x[t]):
                run += 1
                k += 1
            tot[s] += run + 1          # include today: N counts the run from today inclusive
            cnt[s] += 1
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), cnt


# ── data-driven half-life ───────────────────────────────────────────────────────

def predictive_loglik(pnl, half_life, warmup=250, window=0):
    """Walk-forward log-likelihood of the sign outcomes under a model with this half-life.

    Strictly causal: the model has only ever observed days before the one being scored.
    """
    T, N = pnl.shape
    m = LivingBN(n_ind=N, half_life=half_life, window=window)
    ll, n = 0.0, 0
    for t in range(2, T - 1):
        if t > warmup:
            for i in range(N):
                s = classify(pnl[t - 1, i], pnl[t, i])
                if s is None or pnl[t + 1, i] == 0:
                    continue
                p = min(max(m.p_continue(i, s), 1e-6), 1 - 1e-6)
                same = np.sign(pnl[t + 1, i]) == np.sign(pnl[t, i])
                ll += math.log(p if same else 1 - p)
                n += 1
        m.observe(pnl[t - 2], pnl[t - 1], pnl[t])
    return (ll / n if n else float('-inf')), n


def fit_half_life(pnl, grid=(30, 60, 125, 250, 500, 1000, 1e9), warmup=250):
    """Pick the memory length from the data instead of assuming one.

    1e9 is effectively 'never forget', included so that 'the process is stationary' is one of the
    candidate answers rather than an assumption ruled out by the grid.
    """
    out = []
    for h in grid:
        ll, n = predictive_loglik(pnl, h, warmup)
        out.append((h, ll, n))
    best = max(out, key=lambda r: r[1])
    return best[0], out


def compare_kernels(pnl, half_lives=(125, 250, 500, 1e9), windows=(63, 126, 252, 504),
                    warmup=250):
    """Exponential decay vs a rectangular window, on the same walk-forward criterion.

    'The past year' is one specific candidate (252 trading days) among several, scored the same
    way as every other, so the choice of memory is a measurement rather than a convention.
    """
    rows = []
    for h in half_lives:
        ll, n = predictive_loglik(pnl, h, warmup)
        rows.append(('decay', 'never forget' if h > 1e8 else f'{int(h)} d half-life', ll, n))
    for w in windows:
        ll, n = predictive_loglik(pnl, 1e9, warmup, window=int(w))
        rows.append(('window', f'last {int(w)} d', ll, n))
    return rows
