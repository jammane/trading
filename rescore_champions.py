"""Re-score a JN run's champions under the current fill model, and ask what the old scores cost.

A champion's score in pass_reference.csv is only comparable with a challenger trained under the same
fill model (fees.FILL_MODEL). When the fill model changes, the champions are re-scored at once and
nothing is replaced (user rule, 2026-11-23); the trainer refuses a JN run whose fill_model.txt is
stale until this has run.

Re-scoring runs the trainer with --jn-frozen --frozen-slots 1 over the current data window: each
industry's champion elite 0 trades its own book for the whole window with no selection, and the
champion judge (PassJudge, pass_reference v3) scores it. Note what that is: the champion MODEL,
frozen. The champion's old score came from the PASS that produced it, a pool evolving daily; the
two measure different things, and a challenger is still a pass.

    python rescore_champions.py rescore RUN
        Re-scores RUN/champion, appends one `rescore-f<N>` row per industry to
        RUN/pass_reference.csv (same pass number, so the next boundary reads the new score) and
        writes RUN/fill_model.txt.

    python rescore_champions.py unfair RUN --seed-dir DIR --from-pass P
        DIR holds the champions as they stood before pass P (for JN9: /root/ht_jn9_seed). Re-scores
        them, then replays passes P.. against the re-scored scores and lists the boundaries that
        would have crowned but did not. First-order only: a pass crowned earlier would have
        seeded the later passes differently.

Run from a worktree with universe.json and stock_data/ (the trainer reads both).
"""
import argparse
import os
import re
import subprocess
import sys

from fees import FILL_MODEL

FROZEN_RE = re.compile(r"FROZEN SCORE (\S+) ([-+0-9.]+) bp/day days (\d+)-(\d+) resets (\d+) "
                       r"recent (\d+)")


def frozen_scores(trainer, champion_dir, out_dir, profile="jn_profiles/training.json"):
    """{industry: (score bp/day, day_start, day_end, resets, recent)} for champion_dir's elite 0s."""
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists -- remove it or choose another (a frozen pass needs a fresh dir)")
    os.makedirs(out_dir)
    cmd = [trainer, "--jn-stocknn", "--jn-frozen", "--frozen-slots", "1",
           "--load-dir", champion_dir, "--output", out_dir, "--passes", "1", "--start-day", "0",
           "--workers", "1", "--race-only", "MT1C grad",
           "--jn-profile", profile, "--jn-no-bulk"]
    log = os.path.join(out_dir, "train.log")
    print("running:", " ".join(cmd), f"> {log}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit(f"trainer exited {rc}; see {log}")
    out = parse_frozen(open(log).read())
    if not out:
        sys.exit(f"no FROZEN SCORE lines in {log}")
    return out


def parse_frozen(text):
    return {m.group(1): (float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)),
                         int(m.group(6))) for m in FROZEN_RE.finditer(text)}


def read_rows(ref_path):
    """pass_reference.csv data rows, in file order, as dicts (all 13 columns as strings)."""
    cols = None
    rows = []
    with open(ref_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.strip().split(",")
            if cols is None:
                cols = t
                continue
            if len(t) == len(cols):
                rows.append(dict(zip(cols, t)))
    return rows


def standing(rows):
    """{industry: (champion pass, champion score as stored, last pass number)} from the last row."""
    out = {}
    for r in rows:
        out[r["industry"]] = (int(r["new_champ_pass"]), float(r["new_champ_pct"]), int(r["pass"]))
    return out


def rescore_rows(rows, scores, fill_model=FILL_MODEL):
    """One row per industry recording the re-score: the champion stays, its score is replaced."""
    out = []
    for ind, (cp, old, last) in standing(rows).items():
        if ind not in scores:
            sys.exit(f"no frozen score for {ind}")
        new = scores[ind][0] / 1e4
        ds, de = scores[ind][1], scores[ind][2]
        out.append(f"{last},{ind},{ds},{de},{cp},{old:+.8f},{cp},{new:+.8f},rescore-f{fill_model},"
                   f"0.0000,0,{cp},{new:+.8f}")
    return out


def replay(rows, before, from_pass):
    """Counterfactual champion chain from pass `from_pass` on, per industry.

    before: {industry: score (fractional)} of the champion standing before from_pass, re-scored.
    Returns {industry: [(pass, challenger, champion it faced then, crowned actually, would crown)]}.
    """
    by = {}
    for r in rows:
        if r["regime"].startswith("rescore"):
            continue
        p = int(r["pass"])
        if p >= from_pass:
            by.setdefault(r["industry"], {})[p] = r
    out = {}
    for ind, passes in by.items():
        champ = before[ind]
        seq = []
        for p in sorted(passes):
            r = passes[p]
            chal = float(r["chal_pct"])
            actual = int(r["new_champ_pass"]) == p
            would = chal > champ
            seq.append((p, chal, champ, actual, would))
            if would:
                champ = chal
        out[ind] = seq
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rescore", "unfair"])
    ap.add_argument("run")
    ap.add_argument("--trainer", default="./build/training_v4_cpp")
    ap.add_argument("--seed-dir", help="unfair: the champions standing before --from-pass")
    ap.add_argument("--from-pass", type=int, help="unfair: the first pass trained under the current fill model")
    a = ap.parse_args()
    ref = os.path.join(a.run, "pass_reference.csv")
    rows = read_rows(ref)
    if not rows:
        sys.exit(f"{ref}: no rows")

    if a.mode == "rescore":
        scores = frozen_scores(a.trainer, os.path.join(a.run, "champion"),
                               os.path.join(a.run, f"rescore_f{FILL_MODEL}"))
        new_rows = rescore_rows(rows, scores)
        with open(ref, "a") as f:
            for line in new_rows:
                f.write(line + "\n")
        with open(os.path.join(a.run, "fill_model.txt"), "w") as f:
            f.write(f"{FILL_MODEL}\n")
        print(f"\nchampions re-scored under fill model {FILL_MODEL} (frozen elite 0, bp/day):")
        print(f"{'industry':24s} {'pass':>4s} {'old':>8s} {'new':>8s} {'change':>8s}  resets (last 2y)")
        for ind, (cp, old, _) in standing(rows).items():
            s = scores[ind]
            print(f"{ind:24s} {cp:4d} {old * 1e4:+8.2f} {s[0]:+8.2f} {s[0] - old * 1e4:+8.2f}  {s[3]} ({s[4]})")
        print(f"\nappended {len(new_rows)} rows to {ref}; wrote fill_model.txt = {FILL_MODEL}")
        return

    if not a.seed_dir or not a.from_pass:
        sys.exit("unfair needs --seed-dir and --from-pass")
    scores = frozen_scores(a.trainer, a.seed_dir,
                           os.path.join(a.run, f"rescore_seed_p{a.from_pass - 1}_f{FILL_MODEL}"))
    before = {ind: s[0] / 1e4 for ind, s in scores.items()}
    res = replay(rows, before, a.from_pass)
    missed = gained = lost = 0
    print(f"\npasses {a.from_pass}+ against the champions re-scored under fill model {FILL_MODEL} "
          f"(bp/day; * = crowned):")
    for ind, seq in res.items():
        cells = []
        for p, chal, champ, actual, would in seq:
            tag = ("MISSED" if would and not actual else "kept" if actual and would
                   else "LOST" if actual else "")
            missed += would and not actual
            gained += would
            lost += actual and not would
            cells.append(f"p{p} {chal * 1e4:+7.2f} vs {champ * 1e4:+7.2f}{' ' + tag if tag else ''}")
        print(f"{ind:24s} " + " | ".join(cells))
    n = sum(len(s) for s in res.values())
    print(f"\n{n} boundaries: {gained} would have crowned against the re-scored champions, "
          f"{missed} of them did not crown (lost to the stale score); "
          f"{lost} crowned that would not have.")


if __name__ == "__main__":
    main()
