"""The Friday training pass: swap what must be swapped, train one new pass, promote its champions.

Every Friday, after paper and prod have placed their orders, the dev worktree adds one full pass
to the standing race run (`training_v4_cpp --continue-run`), judged per industry against the
champions paper trades:

  1. wait     until production's trading lock is gone
  2. swap     a symbol whose data stopped arriving, or the outlier of an industry whose prices
              drifted past the swap thresholds, is replaced by the first of its industry's ready
              replacements (replacements.py) that still passes its checks. The universe change
              is committed to dev and to paper, and paper's recorded book forgets the old symbol.
  3. train    one pass on the whole race roster; a swapped industry's pass is crowned outright
              (--crown), since its champion was raised on a symbol it no longer trades. When the
              order rules (training_lib.ORDER_RULES) differ from those the run's champions were
              trained under (<run>/order_rules.txt), EVERY industry is crowned: the old models
              learned on trades the broker does not make. When every industry is crowned, three
              passes are trained instead of one: the first is crowned and the other two compete
              with it on the champion rule, so the replacements are the best of three
  4. promote  every industry whose champion changed is promoted to paper: its daily-training
              state is cleared, so the next session re-seeds it from the new champion, and its
              StockNN is converted for inference now. Prod is promoted only for crowned industries
              (a swap or new order rules), whose old models no longer match what is traded.
  5. report   the champions, the swaps, and the committee races re-run on the new champion
              composite beside the committees paper trades. Committees are reported, never
              changed automatically.

The report goes to <run>/weekly/<date>.md and stdout.

    python weekly_pass.py [--dry-run]     # from the dev worktree
    python weekly_pass.py --swap-only     # step 2 alone: before a fresh run, or a mid-week delisting
    python weekly_pass.py --promote-run RUN   # a fresh run's champions replace all 12 on paper
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RUN_DIR = os.environ.get("WEEKLY_RUN", "/root/ht_race_v6")
PAPER_REPO = os.environ.get("PAPER_REPO", "/root/trading")
ACCOUNT = "acct0"
LOCK = os.environ.get("TRADING_LOCK_PATH", "/run/trading/trading_active.lock")
WATCHLIST = "universe_watchlist.json"
TRAIN_ARGS = ["--sigma", "0.008", "--master-sigma", "0.006",
              "--sigma-decay", "1.0", "--start-day", "17", "--stop-day", "1255"]
WEEKLY_PASSES = 1
REPLACE_ALL_PASSES = 3      # every industry crowned: the trainer crowns the first, two compete


def passes_for(industries, crown):
    """One pass a week; three when every industry's model is being replaced outright, so the
    replacement is the best of three passes rather than whatever one pass produced."""
    return REPLACE_ALL_PASSES if crown and set(industries) <= set(crown) else WEEKLY_PASSES


RULES_FILE = "order_rules.txt"


def recorded_order_rules(run_dir):
    """The order-rules version the run's champions were trained under; 1 if never recorded (every
    run before training_lib.plan_orders)."""
    try:
        return int(Path(run_dir, RULES_FILE).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 1


def rules_crown(run_dir, industries, crown, current):
    """-> (crown, report lines). If the champions were trained under other order rules, every
    industry's pass is crowned outright, whatever it scores against them."""
    was = recorded_order_rules(run_dir)
    if was == current:
        return list(crown), []
    return sorted(set(crown) | set(industries)), [
        "", "## Order rules",
        f"- the champions were trained under order rules v{was}; training now uses v{current}, "
        "so every industry's pass is crowned outright and replaces its model"]


def record_order_rules(run_dir, current):
    Path(run_dir, RULES_FILE).write_text(f"{current}\n")


# ── 2. Which symbols must go ─────────────────────────────────────────────────────────────────────

def drift_ratio(prices):
    p = [x for x in prices if x and x > 0]
    return float(np.mean(p) / np.median(p)) if p else 1.0


def swaps_needed(universe, last_dates, closes, thresholds):
    """-> [(industry, symbol, reason)], at most one per industry. A symbol whose newest bar is
    behind the universe's newest session has stopped arriving (the first sign it is no longer
    tradable). Otherwise an industry whose mean/median price ratio is past a swap threshold loses
    the member furthest out on the side that drives the ratio: the dearest if it is high, the
    cheapest if it is low."""
    session = max(last_dates.values(), default="")
    out = []
    for ind, syms in universe.items():
        stale = [s for s in syms if last_dates.get(s, "") < session]
        if stale:
            s = stale[0]
            out.append((ind, s, f"data stopped (last bar {last_dates.get(s) or 'never'}, "
                                f"universe at {session})"))
            continue
        r = drift_ratio([closes.get(s) for s in syms])
        if r > thresholds["mean_median_swap_above"]:
            s = max(syms, key=lambda x: closes.get(x, 0.0))
            out.append((ind, s, f"price drift: mean/median {r:.3f} > "
                                f"{thresholds['mean_median_swap_above']}"))
        elif r < thresholds["mean_median_swap_below"]:
            s = min(syms, key=lambda x: closes.get(x, float("inf")))
            out.append((ind, s, f"price drift: mean/median {r:.3f} < "
                                f"{thresholds['mean_median_swap_below']}"))
    return out


def choose_replacement(industry, old, universe, closes, ready, recheck):
    """The first ready replacement that still passes its checks (recheck(c) -> (price, reason))
    and leaves the industry's mean/median ratio inside the WATCH band. -> (candidate, notes)."""
    notes = []
    for c in ready:
        price, reason = recheck(c)
        if reason:
            notes.append(f"{c['symbol']} skipped: {reason}")
            continue
        prices = [closes.get(s) for s in universe[industry] if s != old] + [price]
        r = drift_ratio(prices)
        if not 0.96 <= r <= 1.20:
            notes.append(f"{c['symbol']} skipped: would leave mean/median at {r:.3f}")
            continue
        return dict(c, price=price), notes
    return None, notes


def clear_book_slot(book_path, industry, slot):
    """Paper's recorded book after a swap: the old symbol's shares and stop at `slot` read as 0,
    since the book is positional and that slot now belongs to the new symbol. (Production sells
    the old position as an orphan.)"""
    if not os.path.exists(book_path):
        return False
    lines = Path(book_path).read_text().splitlines()
    for k, line in enumerate(lines):
        t = line.split()
        if len(t) > 6 and t[0] == "ind" and t[1] == industry:
            h, st = t.index("hold"), t.index("stop")
            t[h + 1 + slot] = "0.000000"
            t[st + 1 + slot] = "0.000000"
            lines[k] = " ".join(t)
            with open(book_path + ".tmp", "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(book_path + ".tmp", book_path)
            return True
    return False


# ── 4. Promotion ────────────────────────────────────────────────────────────────────────────────

def industry_state_files(state_dir, industry):
    """Every file of one industry's daily-training state: StockNN elites and history ring, race
    models and their Adam moments, and the evolutionary pools with their metadata."""
    pats = [f"{industry}_elite_*.bin", f"{industry}_hist.bin", f"race_*_{industry}.bin",
            f"race_*_{industry}.adam.bin", f"r_*_{industry}_meta.bin",
            f"r_*_{industry}_slot_*.bin"]
    out = set()
    for p in pats:
        out |= set(glob.glob(os.path.join(state_dir, p)))
    return sorted(out)


def drop_state_keys(state_path, industries):
    """Remove the per-industry learning rates and step counts from upkeep_state.txt, so the
    re-seeded models start from the trainer's defaults."""
    if not os.path.exists(state_path):
        return 0
    keep, n = [], 0
    for line in Path(state_path).read_text().splitlines():
        t = line.split()
        if len(t) == 4 and t[0] in ("lr", "tanh32_step") and t[2] in industries:
            n += 1
            continue
        keep.append(line)
    with open(state_path + ".tmp", "w") as f:
        f.write("\n".join(keep) + "\n")
    os.replace(state_path + ".tmp", state_path)
    return n


def promote(model_dir, champion_dir, industries, reset_state=True):
    """Promote the champion of each industry to one production model directory: clear its
    daily-training state (paper; the next session re-seeds from champion/) and convert its StockNN
    elites to .pt for inference now."""
    from convert_weights import STOCK_LAYER_DEFS, convert_industry
    from models import StockNN
    state_dir = os.path.join(model_dir, "cpp")
    removed = 0
    if reset_state and os.path.isdir(state_dir):
        for ind in industries:
            for p in industry_state_files(state_dir, ind):
                os.remove(p)
                removed += 1
        drop_state_keys(os.path.join(state_dir, "upkeep_state.txt"), set(industries))
    for ind in industries:
        convert_industry(ind, champion_dir, model_dir, STOCK_LAYER_DEFS, StockNN, ind)
    return removed


def last_pass(ref_path):
    """The run's newest pass number; 0 if it has none."""
    try:
        with open(ref_path) as f:
            return max((int(t.split(",")[0]) for t in f if t[:1].isdigit()), default=0)
    except OSError:
        return 0


def crowned(ref_path, first, last=None):
    """{industry: (regime, score bp/day, previous champion pass, champion pass)} for industries
    whose champion after pass `last` is one of the passes first..last. The regime is the one its
    champion pass was judged under; the previous champion is the one standing before `first`."""
    last = first if last is None else last
    rows = {}
    with open(ref_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("pass,"):
                continue
            t = line.strip().split(",")
            if len(t) == 13:
                rows[int(t[0]), t[1]] = t
    out = {}
    for (p, ind), t in rows.items():
        champ = int(t[11])
        if p == last and first <= champ <= last:
            out[ind] = (rows.get((champ, ind), t)[8], float(t[12]) * 1e4,
                        int(rows.get((first, ind), t)[4]), champ)
    return out


# ── 5. Committees ───────────────────────────────────────────────────────────────────────────────

def race_daily_files(run_dir):
    return [os.path.join(run_dir, "race_daily.csv")] + sorted(
        glob.glob(os.path.join(run_dir, "continue_p*", "race_daily.csv")),
        key=lambda p: int(re.search(r"continue_p(\d+)", p).group(1)))


def committee_report(run_dir):
    """The traded committees on the new champion composite, beside the best of each race."""
    import committee_alloc as ca
    import committee_race as cr
    files = race_daily_files(run_dir)
    champ = cr.read_champions(os.path.join(run_dir, "pass_reference.csv"))
    lines = []
    for losers, name, members in ((False, "W1 (winners)", ca.WINNERS),
                                  (True, "L1 (losers)", ca.LOSERS)):
        entries, passes = cr.load(files)
        P = cr.champion_composite(entries, passes, champ)
        T = len(P["days"])
        pick = np.array([ca.vote(members, {e: P["pred"][e][t] for e in members}, P["ok"][t],
                                 losers=losers) for t in range(T)])
        sign = -1.0 if losers else 1.0
        gap = sign * ((pick * P["real"]).sum(1) / cr.N_PICK - P["flat"])
        c = cr.champion_check(files, os.path.join(run_dir, "pass_reference.csv"), losers=losers)
        cg = np.array([r["gaps"][0] for r in c["cres"]])
        better = int((cg > gap.mean()).sum())
        lines.append(f"- **{name}** {', '.join(members)}: composite {gap.mean():+.2f} bp/day over "
                     f"{T} days (se {gap.std() / np.sqrt(T):.2f}); {better} of {len(cg)} raced "
                     f"configurations score higher")
        for i in np.argsort(-cg, kind="stable")[:5]:
            r = c["res"][i]
            lines.append(f"  - {cg[i]:+.2f}  {r['method']}  {r['committee']}")
    return lines


# ── Orchestration ───────────────────────────────────────────────────────────────────────────────

def sh(cmd, cwd=None, check=True, **kw):
    print("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str), text=True, **kw)


def wait_for_trading(lock=LOCK, poll=60, limit=4 * 3600):
    t0 = time.time()
    while os.path.exists(lock):
        if time.time() - t0 > limit:
            raise RuntimeError(f"{lock} still held after {limit // 3600} h")
        time.sleep(poll)


def in_trading_window(now=None):
    """Weekdays 16:45-17:50 ET, around the 17:05 paper run: a promotion started here could still
    be clearing state when production reads it, before production has taken its lock."""
    now = now or dt.datetime.now()
    return now.weekday() < 5 and dt.time(16, 45) <= now.time() < dt.time(17, 50)


def wait_for_quiet(poll=60):
    while in_trading_window():
        time.sleep(poll)
    wait_for_trading()


def trainer_running():
    """Is any training_v4_cpp (other than a production --upkeep day) running?"""
    r = subprocess.run(["pgrep", "-af", "training_v4_cpp"], capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if "--upkeep" not in ln and "pgrep" not in ln]


LIVE_STATE_FILES = ("book_next.txt", "book_base.json", "pending_decision.json",
                    "pending_decision.json.settled")     # paper's own, not the models'


def promote_run(run_dir, paper_dir, prod_dir, binary="build/training_v4_cpp"):
    """A fresh run's champions replace every industry on paper, caught up to the present first.

    The run's data ends the day it started, so its champions are days behind. A new daily-training
    state is built beside the live one (cpp.<run>): seeded from the run (bayes posterior replayed
    from its champion composite), then trained one --upkeep per session from the run's last
    decision day on its champions' simulated books, through the newest session in stock_data --
    as paper's first state was built from v5. Only then, outside the paper run, is it swapped in
    with paper's own files (real book, book base, pending decision) carried over, paper's
    champion/ re-pointed at the run's, and the caught-up StockNN converted. The old state is kept
    as cpp.prev-<date>. -> report lines."""
    import production_v2
    import upkeep_cpp
    champ = os.path.join(run_dir, "champion")
    ref = os.path.join(run_dir, "pass_reference.csv")
    inds = list(json.loads(Path("universe.json").read_text()))
    missing = [i for i in inds if not os.path.exists(os.path.join(champ, f"{i}_elite_0.bin"))]
    if missing:
        raise RuntimeError(f"{champ} has no champion for {', '.join(missing)}; nothing promoted")
    if glob.glob(os.path.join(run_dir, "continue_p*")):
        # the catch-up starts from one book, every champion's on the same day; passes added by
        # --continue-run end later than the run's own, and their books are in continue_p<N>/
        raise RuntimeError(f"{run_dir} has continued passes; promote_run is for a fresh run "
                           "(a weekly pass promotes its own champions)")
    name = os.path.basename(os.path.normpath(run_dir))
    live = os.path.join(paper_dir, "cpp")
    stage = os.path.join(paper_dir, f"cpp.{name}")
    logs = os.path.join(PAPER_REPO, "logs", ACCOUNT, "paper")
    log = os.path.join(logs, f"upkeep_cpp_catchup_{name}.log")
    archive = os.path.join(logs, f"training_catchup_{name}")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    upkeep_cpp.seed_state(stage, race_daily_files(run_dir), ref)
    book = os.path.join(stage, "book_race_end.txt")
    start = upkeep_cpp.race_book(run_dir, book)
    race_only = list(production_v2.UPKEEP_RACE_ONLY)
    n = upkeep_cpp.catch_up(binary, stage, champ, race_only, book, log, archive=archive)
    wait_for_quiet()
    # a session may have arrived while waiting; catch_up skips the days it already trained
    n += upkeep_cpp.catch_up(binary, stage, champ, race_only,
                             os.path.join(stage, "book_catchup.txt"), log, archive=archive)
    through = upkeep_cpp.read_state(stage)["date"]
    for f in LIVE_STATE_FILES:
        if os.path.exists(os.path.join(live, f)):
            shutil.copy2(os.path.join(live, f), os.path.join(stage, f))
    if os.path.exists(live):
        os.rename(live, os.path.join(paper_dir, f"cpp.prev-{dt.date.today().isoformat()}"))
    os.rename(stage, live)
    link = os.path.join(paper_dir, "champion")
    if os.path.islink(link):
        os.remove(link)
    elif os.path.exists(link):
        os.rename(link, link + ".old")
    os.symlink(champ, link)
    production_v2.convert_stocknn(live, paper_dir)
    rep = [f"- paper: champion/ -> {champ}; all {len(inds)} industries on a new daily state",
           f"- caught up from {start} (the run's last decision day) through {through}: {n} "
           f"session(s) trained on simulated books (log {log})",
           "- bayes posterior replayed from the run's champion composite"]
    if os.path.isdir(prod_dir):
        promote(prod_dir, champ, inds, reset_state=False)
        rep.append(f"- prod: promoted all {len(inds)}")
    else:
        rep.append(f"- prod: {prod_dir} does not exist (prod not set up); nothing promoted")
    won = crowned(ref, 1, last_pass(ref))
    rep.append("- champions: " + ", ".join(f"{i} p{c} ({s:+.2f} bp/day)"
                                            for i, (_, s, _, c) in sorted(won.items())))
    return rep


def committee_races(run_dir):
    """The full committee races, winners and losers, on a run's champion composite, written beside
    it. Report only: the committees paper trades are never changed automatically."""
    rep = []
    files = race_daily_files(run_dir)
    ref = os.path.join(run_dir, "pass_reference.csv")
    for flag, out in (([], "committee_race_champions.txt"),
                      (["--losers"], "committee_race_losers.txt")):
        dest = os.path.join(run_dir, out)
        with open(dest, "w") as f:
            r = sh([sys.executable, "committee_race.py", *files, "--champions", ref, *flag,
                    "--top", "40"], check=False, stdout=f, stderr=subprocess.STDOUT)
        rep.append(f"- {'loser' if flag else 'winner'} race: rc {r.returncode}, {dest}")
    return rep


def last_bar_dates_and_closes(symbols, data_dir="stock_data"):
    dates, closes = {}, {}
    for s in symbols:
        try:
            days = json.loads(Path(data_dir, f"{s}.json").read_text()).get("days", [])
            if days:
                dates[s], closes[s] = days[-1]["date"], float(days[-1]["close"])
        except (OSError, ValueError):
            pass
    return dates, closes


def git_commit_swap(message, files, paper_repo=PAPER_REPO):
    """Commit the swap on dev and carry the same commit to paper (cherry-pick: the rest of dev
    is not released by a swap). -> paper updated?"""
    sh(["git", "add"] + files)
    sh(["git", "commit", "-q", "-m", message])
    sh("git pull -q --rebase origin dev && git push -q origin dev")
    sha = sh(["git", "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    r = sh(f"git pull -q origin paper && git cherry-pick {sha} && git push -q origin paper",
           cwd=paper_repo, check=False)
    if r.returncode != 0:
        sh("git cherry-pick --abort", cwd=paper_repo, check=False)
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be swapped and promoted; train, commit and change nothing")
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--swap-only", action="store_true",
                    help="make the swaps due now (commit to dev and paper) and stop: no pass")
    ap.add_argument("--promote-run", metavar="RUN",
                    help="promote a finished fresh run's champions to every industry and stop")
    a = ap.parse_args(argv)
    today = dt.date.today().isoformat()
    rep = [f"# Weekly pass {today}", ""]
    paper_dir = os.path.join(PAPER_REPO, "models", ACCOUNT, "paper")
    prod_dir = os.path.join(PAPER_REPO, "models", ACCOUNT, "prod")

    if a.promote_run:
        rep = [f"# Promotion of {a.promote_run} {today}", ""] + promote_run(
            a.promote_run, paper_dir, prod_dir)
        rep += ["", "## Committees (report only)"]
        for f in (committee_races, committee_report):
            try:
                rep += f(a.promote_run)
            except Exception as e:
                rep.append(f"- {f.__name__} failed: {e}")
        os.makedirs(os.path.join(a.promote_run, "weekly"), exist_ok=True)
        Path(a.promote_run, "weekly", f"promote_{today}.md").write_text("\n".join(rep) + "\n")
        print("\n".join(rep))
        return 0
    busy = trainer_running()
    if busy and not a.dry_run:
        print("a trainer is running; the weekly pass waits for it to finish:\n  " + "\n  ".join(busy))
        return 1

    wait_for_trading()
    if not a.dry_run:
        sh("git pull -q --rebase origin dev")
        sh("cmake --build build --target training_v4_cpp -j2 > /dev/null")

    # 2. swaps
    universe = json.loads(Path("universe.json").read_text())
    wl = json.loads(Path(WATCHLIST).read_text())
    syms = [s for v in universe.values() for s in v]
    dates, closes = last_bar_dates_and_closes(syms)
    need = swaps_needed(universe, dates, closes, wl["thresholds"])
    if need and not a.dry_run:
        sh([sys.executable, "download_daily.py"], check=False)   # a one-day feed gap retries here
        dates, closes = last_bar_dates_and_closes(syms)
        need = swaps_needed(universe, dates, closes, wl["thresholds"])
    import replacements as rp
    tradable = rp._alpaca_tradable()
    swaps, crown = {}, []
    rep.append("## Swaps")
    for ind, old, why in need:
        peers = rp.peer_profile([s for s in universe[ind] if s != old])

        def recheck(c, peers=peers):
            st, reason = rp.vet(c["symbol"], c["exchange"], rp._yf_history(c["symbol"]), peers,
                                tradable(c["symbol"]))
            return (st or {}).get("price"), reason
        cand, notes = choose_replacement(ind, old, universe, closes, rp.ready(wl, ind), recheck)
        rep += [f"- {ind}: {old} -- {why}"] + [f"  - {n}" for n in notes]
        if cand is None:
            rep.append("  - **CONCERN: no ready replacement passes; not swapped, it stays "
                       "suspended**")
            continue
        rep.append(f"  - -> **{cand['symbol']}** ${cand['price']:.2f}")
        swaps[old] = (ind, cand)
    if not need:
        rep.append("- none")
    if swaps and not a.dry_run:
        import swap_symbols
        swap_symbols.swap_in_file(f"universe_{ACCOUNT}.py", {o: c["symbol"] for o, (_, c) in swaps.items()})
        swap_symbols.export_universe_json(os.getcwd(), ACCOUNT)
        for old, (ind, c) in swaps.items():
            wl.setdefault("completed", {})[old] = dict(industry=ind, replaced_with=c["symbol"],
                                                       swap_date=today, reason="weekly pass",
                                                       price_at_swap=c["price"])
            wl["replacements"][ind] = [r for r in wl["replacements"][ind]
                                       if r["symbol"] != c["symbol"]]
            slot = universe[ind].index(old)
            clear_book_slot(os.path.join(paper_dir, "cpp", "book_next.txt"), ind, slot)
            crown.append(ind)
        rp.save_watchlist(wl)
        sh([sys.executable, "download_daily.py"], check=False)      # 5-year history for the new symbols
        ok = git_commit_swap("Weekly swap: " + ", ".join(f"{o} -> {c['symbol']} ({i})"
                                                        for o, (i, c) in swaps.items()),
                             [f"universe_{ACCOUNT}.py", "universe.json", WATCHLIST])
        if not ok:
            rep.append("- **CONCERN: the swap commit did not apply to paper; paper keeps the old "
                       "universe until it is merged by hand**")

    if a.swap_only:
        print("\n".join(rep + ["", "- swap only: no pass trained"]))
        return 0

    # 3. train
    from training_lib import ORDER_RULES
    swapped = set(crown)
    crown, note = rules_crown(a.run_dir, universe, crown, ORDER_RULES)
    rep += note
    n_passes = passes_for(universe, crown)
    rep += ["", "## Pass"]
    if n_passes > 1:
        rep.append(f"- every industry is replaced: {n_passes} passes, the first crowned and the "
                   "rest competing with it")
    if a.dry_run:
        rep.append("- dry run: no pass trained")
        print("\n".join(rep))
        return 0
    skip = Path(a.run_dir, "race_skip.txt").read_text().strip()
    log = os.path.join(a.run_dir, f"weekly_{today}.log")
    cmd = (["./build/training_v4_cpp", "--continue-run", "--output", a.run_dir,
            "--passes", str(n_passes)] + TRAIN_ARGS)
    if skip:
        cmd += ["--race-skip", skip]
    if crown:
        cmd += ["--crown", ",".join(crown)]
    ref = os.path.join(a.run_dir, "pass_reference.csv")
    first = last_pass(ref) + 1
    with open(log, "w") as f:
        rc = sh(cmd, check=False, stdout=f, stderr=subprocess.STDOUT).returncode
    pass_num = last_pass(ref)
    span = f"pass {pass_num}" if pass_num == first else f"passes {first}-{pass_num}"
    rep.append(f"- {span}, rc {rc}, log {log}")
    if rc != 0:
        rep.append("- **CONCERN: the pass failed; nothing promoted**")
    else:
        if note:
            record_order_rules(a.run_dir, ORDER_RULES)
        won = crowned(ref, first, pass_num)

        def why(i):
            return "crowned after its swap" if i in swapped else "crowned: new order rules"

        rep.append("- champions taken: " + (", ".join(
            f"{i} ({s:+.2f} bp/day, "
            + (f"p{c}, " if pass_num > first else "")
            + (why(i) if r == "crown" or i in crown else f"replaces p{p}") + ")"
            for i, (r, s, p, c) in sorted(won.items())) or "none"))
        # 4. promote
        champ = os.path.join(a.run_dir, "champion")
        if won:
            wait_for_trading()
            n = promote(paper_dir, champ, sorted(won))
            rep.append(f"- paper: promoted {', '.join(sorted(won))} ({n} state files reset)")
        if crown and os.path.isdir(prod_dir):
            promote(prod_dir, champ, crown, reset_state=False)
            rep.append(f"- prod: promoted the crowned {', '.join(crown)}")
        elif crown:
            rep.append(f"- prod: {prod_dir} does not exist (prod not set up); nothing promoted")
        # 5. committees
        rep += ["", "## Committees (report only)"]
        try:
            rep += committee_report(a.run_dir)
        except Exception as e:
            rep.append(f"- committee race failed: {e}")
    os.makedirs(os.path.join(a.run_dir, "weekly"), exist_ok=True)
    out = os.path.join(a.run_dir, "weekly", f"{today}.md")
    with open(out, "w") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\nreport -> {out}")
    shutil.copy2(out, os.path.join(a.run_dir, "weekly", "latest.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
