"""Production's daily training step: drive `training_v4_cpp --upkeep` and read what it leaves.

Each session production trains ONE day with the same C++ code that trained the race -- the day
it last placed orders on, filled against the bars that have arrived since -- then converts the
StockNN elites to .pt and runs the updated models. There is deliberately no Python copy of the
race training: a second implementation is how production's buy loop once drifted 8x over budget.

  write_book()   the real books as of the decision day, for the next run to train on
  run_upkeep()   one training day; the trainer refuses a day it has already trained (rc 3)
  read_state()   upkeep_state.txt: the bayes posterior and MT1's value curves, among others
  seed_state()   the first state: the bayes posterior replayed from the race's champion composite
  race_book()    the champions' simulated books where the race stopped, to start a catch-up from
  catch_up()     train every session from a book to the newest bars, chaining the simulated books

State lives in one directory (the trainer's --output): StockNN elites and history ring, the
evolutionary pools, race_<entry>_<ind>.bin (+ .adam.bin) for every entry in use, and
upkeep_state.txt. The champion store is the --load-dir seed, read only where the state is empty.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys

STATE_FILE = "upkeep_state.txt"
SIM_BOOK = "book_sim_next.txt"   # written by every --upkeep: slot 0's book after the day
IND_STARTING_CASH = 25_000.0     # the race's book base
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


def race_book(run_dir, path, day=None, universe=None):
    """Write the champions' simulated books entering race day `day` (0-based, the trainer's
    actual_day; default the last day race_daily.csv logged) as a BOOK, and return its date.

    holdings_log.csv rows are written 1-based after the day's trades, so row `day` is the book the
    race carried INTO day `day`: what each industry's champion pass held at that close, before
    trading on it. Stops are not logged and start at 0."""
    import committee_race as cr
    champ = cr.read_champions(os.path.join(run_dir, "pass_reference.csv"))
    dates = {}
    with open(os.path.join(run_dir, "race_daily.csv")) as f:
        for r in csv.DictReader(f):
            if r["entry"] == "REALIZED":
                dates[(int(r["pass"]), int(r["day"]))] = r["date"]
    if day is None:
        day = max(d for (_, d) in dates)
    universe = universe or universe_order()
    rows = {}
    with open(os.path.join(run_dir, "holdings_log.csv")) as f:
        for r in csv.DictReader(f):
            ind = r["industry"]
            if int(r["day"]) == day and int(r["pass"]) == champ.get(ind):
                rows[ind] = r
    missing = set(universe) - set(rows)
    if missing:
        raise ValueError(f"holdings_log.csv has no day-{day} row for {sorted(missing)}")
    date = dates[(champ[next(iter(universe))], day)]
    books = {ind: {"cash": float(rows[ind]["cash"]),
                   "holdings": {s: float(rows[ind][f"q{j}"]) for j, s in enumerate(syms)},
                   "stop_prices": {}}
             for ind, syms in universe.items()}
    write_book(path, date, books, {ind: IND_STARTING_CASH for ind in universe}, universe)
    return date


def catch_up(binary, state_dir, seed_dir, race_only, book, log_path, max_days=250):
    """Train from `book` through the newest session in stock_data, one --upkeep per day, each
    day's book being the simulated book the day before ended with. -> days trained.
    Stops cleanly at rc 2 (no later session yet); a day already in the state (rc 3) is skipped by
    moving on to the book that day left, so an interrupted catch-up can simply be re-run."""
    nxt = os.path.join(state_dir, SIM_BOOK)
    cur = os.path.join(state_dir, "book_catchup.txt")
    if os.path.abspath(book) != os.path.abspath(cur):
        shutil.copy2(book, cur)
    trained = 0
    for _ in range(max_days):
        rc = run_upkeep(binary, cur, state_dir, seed_dir, race_only, log_path)
        if rc == 2:
            break
        if rc not in (0, 3):
            raise RuntimeError(f"training_v4_cpp --upkeep failed (rc {rc}); see {log_path}")
        if rc == 3 and not os.path.exists(nxt):
            break
        trained += rc == 0
        shutil.copy2(nxt, cur)
    return trained


def main(argv=None):
    ap = argparse.ArgumentParser(description="daily C++ training: seed a state, catch up a gap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("seed", help="new state with the bayes posterior from the race")
    a.add_argument("--state", required=True)
    a.add_argument("--run-dir", required=True, help="the race run (race_daily.csv, pass_reference.csv)")
    b = sub.add_parser("catch-up", help="train every session since the race stopped")
    b.add_argument("--state", required=True)
    b.add_argument("--seed-dir", required=True, help="the champion store")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--binary", default="build/training_v4_cpp")
    b.add_argument("--race-only", required=True, help="comma-separated race entries")
    b.add_argument("--log", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "seed":
        mu = seed_state(a.state, os.path.join(a.run_dir, "race_daily.csv"),
                        os.path.join(a.run_dir, "pass_reference.csv"))
        print("seeded bayes posterior (mean, bp/day): "
              + ", ".join(f"{k} {v * 1e4:+.1f}" for k, v in mu.items()))
        return 0
    book = os.path.join(a.state, "book_race_end.txt")
    date = race_book(a.run_dir, book)
    print(f"catch-up from {date}, the race's last decision day, on its champions' books")
    n = catch_up(a.binary, a.state, a.seed_dir, a.race_only.split(","), book, a.log)
    st = read_state(a.state)
    print(f"trained {n} day(s); state now through {st['date'] if st else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
