"""Production's daily training step: drive `training_v4_cpp --upkeep` and read what it leaves.

Each session production trains ONE day with the same C++ code that trained the race -- the day
it last placed orders on, filled against the bars that have arrived since -- then converts the
StockNN elites to .pt and runs the updated models. There is deliberately no Python copy of the
race training: a second implementation is how production's buy loop once drifted 8x over budget.

  write_book()   the real books as of the decision day, for the next run to train on
  run_upkeep()   one training day; the trainer refuses a day it has already trained (rc 3)
  read_state()   upkeep_state.txt: the bayes posterior and MT1's value curves, among others
  seed_state()   the first state: the bayes posterior replayed from the race's champion composite

State lives in one directory (the trainer's --output): StockNN elites and history ring, the
evolutionary pools, race_<entry>_<ind>.bin (+ .adam.bin) for every entry in use, and
upkeep_state.txt. The champion store is the --load-dir seed, read only where the state is empty.
"""
import json
import math
import os
import subprocess

STATE_FILE = "upkeep_state.txt"
BOOK_VERSION = 1
BALLOC_GAMMA = 0.98            # = BALLOC_GAMMA in training_v4.cpp
# The race's StockNN settings (v5): production's daily step trains as the race did.
SIGMA, MASTER_SIGMA = 0.008, 0.006


def universe_order(path="universe.json"):
    """{industry: [symbols]} in the order the trainer reads them (q0..q11)."""
    with open(path) as f:
        return json.load(f)


def write_book(path, date, books, base, universe=None):
    """books: {industry: {'cash': float, 'holdings': {sym: qty}, 'stop_prices': {sym: price}}}
    base:  {industry: starting capital}, CN_BOOK's denominator (IND_STARTING_CASH in training).
    Every industry in the universe must be present; symbols are written in universe.json order,
    the order the trainer indexes them by."""
    universe = universe or universe_order()
    missing = set(universe) - set(books)
    if missing:
        raise ValueError(f"book has no entry for {sorted(missing)}")
    lines = [f"upkeep_book {BOOK_VERSION}", f"date {date}"]
    for ind, syms in universe.items():
        b = books[ind]
        if not (base[ind] > 0):
            raise ValueError(f"{ind}: base must be > 0")
        hold = [float(b["holdings"].get(s, 0.0)) for s in syms]
        stop = [float(b.get("stop_prices", {}).get(s, 0.0)) for s in syms]
        if not all(math.isfinite(x) for x in hold + stop + [b["cash"]]):
            raise ValueError(f"{ind}: non-finite value in book")
        lines.append(f"ind {ind} base {base[ind]:.6f} cash {float(b['cash']):.6f} hold "
                     + " ".join(f"{x:.6f}" for x in hold) + " stop "
                     + " ".join(f"{x:.6f}" for x in stop))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def read_state(state_dir):
    """-> dict(date, hist_count, mkt{ind: [..]}, pf{ind: [..]}, bayes{ind: (mu, m2, w)}, kv{key: v}),
    or None if the trainer has not written one yet."""
    p = os.path.join(state_dir, STATE_FILE)
    if not os.path.exists(p):
        return None
    st = dict(date="", hist_count=0, mkt={}, pf={}, bayes={}, kv={})
    with open(p) as f:
        head = f.readline().split()
        if head != ["upkeep_state", "1"]:
            raise ValueError(f"{p}: not an upkeep_state v1")
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "date":
                st["date"] = t[1] if len(t) > 1 else ""
            elif t[0] == "hist":
                st["hist_count"] = int(t[1])
            elif t[0] in ("mkt", "pf"):
                st[t[0]][t[1]] = [float(x) for x in t[2:]]
            elif t[0] == "bayes":
                st["bayes"][t[1]] = tuple(float(x) for x in t[2:5])
            else:
                st["kv"][" ".join(t[:-1])] = float(t[-1])
    return st


def write_state(state_dir, st, industries):
    """The trainer's text format (write_upkeep_state). Used only to seed a new state."""
    p = os.path.join(state_dir, STATE_FILE)
    n = st.get("hist_count", 0)
    lines = ["upkeep_state 1", f"date {st.get('date', '')}".rstrip(), f"hist {n}"]
    for ind in industries:
        lines.append(" ".join(["mkt", ind] + [repr(float(x)) for x in st.get("mkt", {}).get(ind, [])[:n]]))
        lines.append(" ".join(["pf", ind] + [repr(float(x)) for x in st.get("pf", {}).get(ind, [])[:n]]))
    for ind in industries:
        if ind in st.get("bayes", {}):
            mu, m2, w = st["bayes"][ind]
            lines.append(f"bayes {ind} {mu!r} {m2!r} {w!r}")
    for k, v in sorted(st.get("kv", {}).items()):
        lines.append(f"{k} {v!r}")
    os.makedirs(state_dir, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, p)


def bayes_replay(returns, ok, gamma=BALLOC_GAMMA):
    """The trainer's posterior update, day by day: returns/ok are [T, 12], returns as fractions.
    -> (mu, m2, w), each a list of 12."""
    n = len(returns[0]) if len(returns) else 0
    mu, m2, w = [0.0] * n, [0.0] * n, [0.0] * n
    for t in range(len(returns)):
        for i in range(n):
            if not ok[t][i]:
                continue
            x = float(returns[t][i])
            w[i] = gamma * w[i] + 1.0
            d = x - mu[i]
            mu[i] += d / w[i]
            m2[i] = gamma * m2[i] + d * (x - mu[i])
    return mu, m2, w


def seed_state(state_dir, race_daily, pass_reference):
    """A first state whose bayes posterior has seen the champion composite's 252 days -- each
    industry's slot-0 return from its champion pass -- so the allocator starts where the race
    left it rather than abstaining for its warm-up. The value curves stay empty; the trainer
    rebuilds them from the bars on its first day. Refuses to overwrite an existing state."""
    import committee_race as cr
    if read_state(state_dir) is not None:
        raise FileExistsError(f"{state_dir}/{STATE_FILE} exists; not overwriting a live state")
    entries, passes = cr.load(race_daily)
    comp = cr.champion_composite(entries, passes, cr.read_champions(pass_reference))
    mu, m2, w = bayes_replay((comp["real"] / 1e4).tolist(), comp["ok"].tolist())
    inds = comp["inds"]
    write_state(state_dir, dict(bayes={ind: (mu[i], m2[i], w[i]) for i, ind in enumerate(inds)}),
                inds)
    return {ind: mu[i] for i, ind in enumerate(inds)}


def run_upkeep(binary, book_path, state_dir, seed_dir, race_only, log_path,
               sigma=SIGMA, master_sigma=MASTER_SIGMA, workers=2, timeout=3600):
    """One training day. -> the trainer's return code: 0 trained, 2 the next session is not in
    stock_data yet, 3 this day was already trained, anything else a failure (see log_path)."""
    cmd = [binary, "--upkeep", book_path, "--output", state_dir, "--load-dir", seed_dir,
           "--race-only", ",".join(race_only), "--sigma", str(sigma),
           "--master-sigma", str(master_sigma), "--workers", str(workers)]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
