"""The committee allocation production trades: a loser committee puts 4 industries in sell-only,
and a winner committee tilts the other 8.

Both committees vote with `votes_member` (`committee_race.py`), chosen on the v5 champion
composite:

  W1  TANH32 grad, MT1 grdC, MT1 evo, MT1S-rep evo, MT1S-stk grad      picks the 4 to tilt toward
  L1  TANH32 grad, ALLOC bayes mean, MT1S-stk evo                      flags the 4 to sell only

Members are listed strongest first by their composite mean gap (the winner race for W1, the
mirrored loser race for L1), because `votes_member` weights each member 1.1x the one below it.

The allocation splits today's cash. The 4 industries L1 flags get none, so they only sell, and
they are never liquidated (a position the model does not sell stays held). The other 8 share the
cash at (1 - t)/8 each, plus t/4 to each of the 4 W1 picks among them, with t = 0.5. W1 votes
over all 12 but funds only among the 8. This is the rule `/root/sell_only.py` replayed on v5:
composite +10.53 bp/day against flat, 8 of 10 passes.

Predictions come from the race models the daily training step maintains, in each entry's own
units. Only their order across the 12 industries matters to a vote.
"""
import os

import numpy as np

import committee_race as cr
import race_models as rm

METHOD = "votes_member"
WINNERS = ("TANH32 grad", "MT1 grdC", "MT1 evo", "MT1S-rep evo", "MT1S-stk grad")
LOSERS = ("TANH32 grad", "ALLOC bayes mean", "MT1S-stk evo")
TILT = 0.5
BAYES = "ALLOC bayes mean"
ENTRIES = tuple(dict.fromkeys(WINNERS + LOSERS))


def load_model(entry, industry, state_dir, seed_dir):
    """The entry's model for one industry: the daily-trained copy in `state_dir` if the trainer
    has written one, otherwise the champion's."""
    name = f"race_{rm.slug(entry)}_{industry}.bin"
    for d in (state_dir, seed_dir):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return rm.RaceModel(entry, np.fromfile(p, np.float32))
    raise FileNotFoundError(f"{name} in neither {state_dir} nor {seed_dir}")


def industry_inputs(bars, hold, cash, intent, book_base):
    """One industry's race inputs at today's close. bars: [12] (o, h, l, c) or None for a symbol
    without a usable bar; hold: [12] shares; intent: [12, 4] StockNN output.
    -> (in146, raw slices18, ratio slices18)"""
    valid = [b is not None for b in bars]
    o, h, lo, c = (np.array([b[k] if b else 0.0 for b in bars], np.float64) for k in range(4))
    in146 = rm.build_cfeat146(o, h, lo, c, valid, hold, cash, intent, book_base=book_base)
    it = np.asarray(intent, np.float64)
    sl = [rm.mt1p_slices(in146, c, valid, cash, hold, it[:, 0], it[:, 1], it[:, 2], it[:, 3], raw)
          for raw in (True, False)]
    return in146, sl[0], sl[1]


def predict(industries, inputs, in74, bayes_mu, state_dir, seed_dir):
    """-> {entry: [12] predictions}, NaN where an industry has no inputs.

    inputs: {ind: industry_inputs(...)}; in74: {ind: 74 MT1 features}; bayes_mu: {ind: posterior
    mean daily return}."""
    out = {e: np.full(len(industries), np.nan) for e in ENTRIES}
    for i, ind in enumerate(industries):
        if ind in bayes_mu:
            out[BAYES][i] = bayes_mu[ind]
        if ind not in inputs:
            continue
        in146, raw18, ratio18 = inputs[ind]
        for e in ENTRIES:
            if e == BAYES:
                continue
            m = load_model(e, ind, state_dir, seed_dir)
            if m.arch == "MT1":
                out[e][i] = m.predict(in74=in74[ind])
            elif m.arch == "MT1P":
                out[e][i] = m.predict(slices18=ratio18 if m.evo else raw18)
            else:
                out[e][i] = m.predict(in146=in146)
    return out


def vote(members, preds, ok, losers=False, mask=None):
    """[12] bool: the committee's 4 under METHOD, exactly as committee_race scores one day.

    losers=True votes on negated predictions, so each member's top 4 are the 4 it expects to do
    worst. mask restricts which industries can be funded; each member still votes over all ok."""
    n = len(ok)
    ok = np.asarray(ok, bool)[None, :]
    P = dict(real=np.zeros((1, n)), flat=np.zeros(1), ok=ok,
             pred={e: (-1.0 if losers else 1.0) * np.asarray(preds[e], float)[None, :]
                   for e in members})
    F = {e: (*cr.per_entry(P["pred"][e], ok), np.zeros(1)) for e in members}
    k = len(members)
    scores, votes, zmean = cr.committee_scores(list(members), [cr.STEP ** (k - 1 - j)
                                                                for j in range(k)], P, F)
    fundable = ok if mask is None else ok & np.asarray(mask, bool)[None, :]
    return cr.fund(scores[METHOD], fundable, votes, zmean)[0]


def decide(industries, preds, cash, tilt=TILT):
    """-> dict(sell_only=[inds], winners=[inds], allocations={ind: $}).

    An industry with a NaN from any member cannot be voted on: it is left out of both
    committees and gets the base share."""
    n = len(industries)
    ok = np.all([np.isfinite(preds[e]) for e in ENTRIES], axis=0)
    clean = {e: np.nan_to_num(preds[e], nan=0.0) for e in ENTRIES}
    flagged = vote(LOSERS, clean, ok, losers=True) if ok.sum() > 4 else np.zeros(n, bool)
    keep = ~flagged
    winners = vote(WINNERS, clean, ok, mask=keep) if (ok & keep).sum() >= 4 else np.zeros(n, bool)
    share = np.where(keep, (1.0 - tilt) / keep.sum(), 0.0) + winners * tilt / max(winners.sum(), 1)
    if not winners.any():
        share = keep / keep.sum()
    return dict(sell_only=[industries[i] for i in range(n) if flagged[i]],
                winners=[industries[i] for i in range(n) if winners[i]],
                allocations={ind: float(cash * share[i]) for i, ind in enumerate(industries)})


# ── The daily record ─────────────────────────────────────────────────────────────────────────────
#
# Each run records its decision (pending_decision.json) and the next run settles it against what
# the paper account actually did, writing rows in race_daily.csv's format so paper and training
# can be read with the same tools. The trainer's own race_daily.csv for the same day (archived by
# upkeep_cpp.archive_logs) is the simulated twin: the same real book, filled by the trainer's fill
# model instead of the market.

N_PICK = cr.N_PICK


def header(industries):
    return (["pass", "day", "date", "entry", "ok_n", "flat_bp", "top4_bp", "gap_bp"]
            + [f"pick{k}" for k in range(1, N_PICK + 1)] + list(industries))


def industry_returns(pending, closes, positions, fills):
    """Each industry's real book return over the session after the decision, bp; NaN where the
    industry had no book.

    Book at the decision = shares held x that day's close + the cash it was given. Book a session
    later = shares now x today's close + that cash - what its buys cost + what its sells raised.
    fills: [(symbol, side 'buy'|'sell', qty, avg price)]. A symbol with no close today
    (suspended) is held at its decision close."""
    flow = {}
    for sym, side, qty, px in fills:
        flow[sym] = flow.get(sym, 0.0) + (qty * px if side == "sell" else -qty * px)
    out = []
    for ind in pending["industries"]:
        syms = pending["symbols"][ind]
        cash = pending["allocations"][ind] if ind not in pending["sell_only"] else 0.0
        v0 = cash + sum(pending["holdings"].get(s, 0.0) * pending["closes"].get(s, 0.0)
                        for s in syms)
        # A suspended symbol has no close today: it is held at its decision close, not at 0.
        v1 = cash + sum(positions.get(s, 0.0) * closes.get(s, pending["closes"].get(s, 0.0))
                        + flow.get(s, 0.0) for s in syms)
        out.append((v1 / v0 - 1.0) * 1e4 if v0 > 1.0 else np.nan)
    return np.array(out)


def settle_rows(pending, real_bp, account_bp, day):
    """race_daily rows for one settled decision: REALIZED, each member, the two committees, and
    PORTFOLIO (the account's actual return; its 12 columns are the dollars each industry was
    given). The L1 row's picks are the 4 it flagged, so its gap is negative when they lagged."""
    inds = pending["industries"]
    ok = np.isfinite(real_bp)
    flat = float(real_bp[ok].mean()) if ok.any() else np.nan
    tag = "paper-dry" if pending.get("no_orders") else "paper"
    base = [tag, day, pending["date"]]

    def fmt(x):
        return "" if not np.isfinite(x) else f"{x:.4f}"

    def row(entry, picks, cols, top=None):
        idx = [inds.index(p) for p in picks]
        r = [real_bp[i] for i in idx if ok[i]]
        top = (float(np.mean(r)) if r else np.nan) if top is None else top
        return base + [entry, int(ok.sum()), fmt(flat), fmt(top), fmt(top - flat)] \
            + (list(picks) + [""] * N_PICK)[:N_PICK] + [f"{c:.10g}" if np.isfinite(c) else ""
                                                        for c in cols]

    rows = [base + ["REALIZED", int(ok.sum()), fmt(flat), "", ""] + [""] * N_PICK
            + [fmt(x) for x in real_bp]]
    for e in ENTRIES:
        p = np.array(pending["preds"][e], float)
        order = [i for i in np.argsort(-np.nan_to_num(p, nan=-np.inf), kind="stable")
                 if np.isfinite(p[i])][:N_PICK]
        rows.append(row(e, [inds[i] for i in order], p))
    nan12 = [np.nan] * len(inds)
    rows.append(row("W1 committee", pending["winners"], nan12))
    rows.append(row("L1 committee", pending["sell_only"], nan12))
    rows.append(row("PORTFOLIO", [], [pending["allocations"][i] for i in inds], top=account_bp))
    return rows


def append_rows(path, industries, rows):
    import csv
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header(industries))
        w.writerows(rows)
