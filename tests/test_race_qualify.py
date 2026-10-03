"""race_qualify.py: the qualifier rule for the next MT1 race.

> 40% of counted passes must beat flat. Unchanged entries count v3 + v4 (5/10 in, 4/10 out);
entries changed in v4 count v4 only (3/5 in, 2/5 out). The boundary is the whole rule, so it is
pinned on both sides. Also: v3's renamed rows are read under their v4 names, infrastructure
entries are never skipped, the Bayesian allocator qualifies on either line, and a partial log
refuses to decide.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "race_qualify.py"


def _log(path, entries, bayes=None, passes=5):
    """entries: {raw name as the trainer prints it: [gap per pass]}; bayes: {"mean": [...]}"""
    lines = []
    for p in range(1, passes + 1):
        lines.append(f"[00:00:00]    ===== MT1 race, pass {p} : x =====")
        for kind, gaps in (bayes or {}).items():
            g = gaps[p - 1]
            lines.append(f"[00:00:00]    ALLOC bayes {kind:<8} (gamma 0.98) bp/day over 9 days: "
                         f"flat +8.00 | top4 {8 + g:+.2f} ({g:+.2f} vs flat) | long-short +0.00")
        for name, gaps in entries.items():
            g = gaps[p - 1]
            lines.append(f"[00:00:00]    {name:<14} loss 0.1 | within-day rank +0.0010 over 9 ind-days")
            lines.append(f"        alloc bp/day: flat +8.00 | top4 {8 + g:+.2f} ({g:+.2f} vs flat) | "
                         f"top6 +8.00 (+0.00) | long-short +0.00")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _run(v3, v4):
    r = subprocess.run([sys.executable, str(SCRIPT), v3, v4], capture_output=True, text=True)
    skip_line = [ln for ln in r.stdout.splitlines() if ln.startswith("RACE_SKIP=")]
    skip = set(filter(None, skip_line[0][len("RACE_SKIP="):].split(","))) if skip_line else None
    return r.returncode, skip, r.stdout


W = [+1, +1, +1, +1, +1]
L = [-1, -1, -1, -1, -1]
BAYES_OK = {"mean": W, "thompson": W}


def test_unchanged_entry_boundary_is_strictly_above_40pct(tmp_path):
    # 2/5 in v3 + 3/5 in v4 = 5/10 -> in;  2/5 + 2/5 = 4/10 -> out
    v3 = _log(tmp_path / "v3", {"MT1C     evo ": [1, 1, -1, -1, -1],
                                "MT1C     gdmn": [1, 1, -1, -1, -1]}, BAYES_OK)
    v4 = _log(tmp_path / "v4", {"MT1C     evo ": [1, 1, 1, -1, -1],
                                "MT1C     gdmn": [1, 1, -1, -1, -1]}, BAYES_OK)
    rc, skip, _ = _run(v3, v4)
    assert rc == 0
    assert "MT1C evo" not in skip
    assert "MT1C gdmn" in skip


def test_changed_entry_counts_v4_only(tmp_path):
    # TANH32 grad changed in v4: its v3 losses must not count. 3/5 in v4 -> in, 2/5 -> out.
    v3 = _log(tmp_path / "v3", {"TANH32   grad": L, "TANH32   gdmn": W}, BAYES_OK)
    v4 = _log(tmp_path / "v4", {"TANH32   grad": [1, 1, 1, -1, -1],
                                "TANH32   gdmn": [1, 1, -1, -1, -1]}, BAYES_OK)
    rc, skip, _ = _run(v3, v4)
    assert rc == 0
    assert "TANH32 grad" not in skip      # 3/5, v3's 0/5 ignored
    assert "TANH32 gdmn" in skip          # 2/5, v3's 5/5 ignored


def test_v3_renamed_rows_count_under_v4_names(tmp_path):
    # v3 "MT1S-stk evo" IS v4 "MT1S-rep evo"; v4 "MT1S-stk evo" is a new per-stock entry.
    v3 = _log(tmp_path / "v3", {"MT1S-stk evo ": W}, BAYES_OK)
    v4 = _log(tmp_path / "v4", {"MT1S-rep evo ": L, "MT1S-stk evo ": [1, 1, -1, -1, -1]}, BAYES_OK)
    rc, skip, _ = _run(v3, v4)
    assert rc == 0
    assert "MT1S-rep evo" not in skip     # 5/10 via its v3 history
    assert "MT1S-stk evo" in skip         # 2/5, and v3's 5/5 belongs to MT1S-rep


def test_infrastructure_entries_are_never_skipped(tmp_path):
    v3 = _log(tmp_path / "v3", {"MT1C     grad": L, "MT1      evo ": L}, BAYES_OK)
    v4 = _log(tmp_path / "v4", {"MT1C     grad": L, "MT1      evo ": L}, BAYES_OK)
    rc, skip, out = _run(v3, v4)
    assert rc == 0
    assert not ({"MT1C grad", "MT1 evo"} & skip)
    assert "deployed entry" in out and "logged pool" in out


def test_bayes_qualifies_on_either_line_and_is_skipped_as_one(tmp_path):
    v3 = _log(tmp_path / "v3", {"MT1C     evo ": W}, {"mean": L, "thompson": W})
    v4 = _log(tmp_path / "v4", {"MT1C     evo ": W}, {"mean": L, "thompson": W})
    rc, skip, _ = _run(v3, v4)
    assert rc == 0 and "ALLOC bayes" not in skip
    v3 = _log(tmp_path / "v3", {"MT1C     evo ": W}, {"mean": L, "thompson": L})
    v4 = _log(tmp_path / "v4", {"MT1C     evo ": W}, {"mean": L, "thompson": L})
    rc, skip, _ = _run(v3, v4)
    assert rc == 0 and "ALLOC bayes" in skip
    assert not any(s.startswith("ALLOC bayes ") for s in skip)   # one switch, not two lines


def test_partial_race_refuses_to_decide(tmp_path):
    v3 = _log(tmp_path / "v3", {"MT1C     evo ": W}, BAYES_OK)
    v4 = _log(tmp_path / "v4", {"MT1C     evo ": W}, BAYES_OK, passes=4)
    rc, skip, _ = _run(v3, v4)
    assert rc == 2
    assert skip is None                   # no RACE_SKIP line for a caller to act on
