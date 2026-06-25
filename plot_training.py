#!/usr/bin/env python3
"""
plot_training.py — Download training logs from the droplet and generate SVG progress charts.

Usage:
    python plot_training.py                       # use existing logs_local/ files, latest pass
    python plot_training.py --download            # pull fresh logs first
    python plot_training.py --pass 2              # display pass 2 (default: latest)
    python plot_training.py --download --account acct0 --host root@165.22.6.112

Outputs (git-ignored):
    plots/industry_performance.svg   — StockNN elite portfolio value per industry
                                       (○ = floor-reset event at $22.5K, ● = reset-to $25K)
    plots/mt1_composite.svg          — MT1 composite score per industry
    plots/mt1_direction.svg          — MT1 direction component
    plots/mt1_range.svg              — MT1 range component
    plots/mt1_accuracy.svg           — MT1 accuracy component
    plots/mt1_confidence.svg         — MT1 confidence component
    plots/mt2_performance.svg        — MT2 allocation score (elite pool stats)

Lines: solid = mean, dashed = max & min  (slot0 excluded per design)
Each chart shows one pass only; x-axis = day within that pass.
When more than 100 days are present, values are averaged into ~100 display points.
"""

import argparse
import csv
import math
import struct
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.lines as mlines
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

# ── Configuration ─────────────────────────────────────────────────────────────
DROPLET_HOST    = "root@165.22.6.112"
REMOTE_LOG      = "/root/trading/logs/{account}/training"
LOCAL_LOG_DIR   = Path("logs_local")
PLOT_DIR        = Path("plots")

DAYS_PER_PASS     = 1255
FIG_W             = 16      # fixed width; height varies per chart
TARGET_POINTS     = 100     # max display points per chart (avg-downsample when exceeded)
IND_STARTING_CASH = 25_000.0
FLOOR_VALUE       = IND_STARTING_CASH * 0.9   # $22,500 — hard-floor reset threshold

INDUSTRIES = [
    "tech_hardware", "tech_software_ai", "financials", "consumer_discretionary",
    "consumer_services", "health_care", "industrials", "consumer_staples",
    "energy", "utilities", "real_estate", "materials",
]
IND_LABEL = {
    "tech_hardware":          "Hardware",
    "tech_software_ai":       "Software/AI",
    "financials":             "Financials",
    "consumer_discretionary": "Cons. Discr.",
    "consumer_services":      "Cons. Svc",
    "health_care":            "Health Care",
    "industrials":            "Industrials",
    "consumer_staples":       "Cons. Staples",
    "energy":                 "Energy",
    "utilities":              "Utilities",
    "real_estate":            "Real Estate",
    "materials":              "Materials",
}
# 12 perceptually distinct colors
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#f0a500", "#00897b",
]
IND_COLOR = dict(zip(INDUSTRIES, COLORS))

_COMP_NAMES = ["composite", "direction", "range", "accuracy", "confidence"]
_STAT_NAMES = ["best", "slot0", "mean", "min"]

MT1_COMP_LABEL = {
    "composite":  "Composite  (0.50×dir + 0.33×range + 0.17×acc)",
    "direction":  "Direction  — 5-day sum  (conf if up, 1−conf if down)",
    "range":      "Range  — calibration tightness",
    "accuracy":   "Accuracy  — dollar error vs floor",
    "confidence": "Confidence  — out[3] vs range-geometry ideal",
}

# ── Binary log constants (v4, 984 bytes/record) ───────────────────────────────
MT_LOG_MAGIC   = 0x4D543132  # 'MT12'
RECORD_SIZE_V4 = 984
HEADER_SIZE    = 16
_RECORD_STRUCT = struct.Struct("<II" + "f" * 243 + "Bxxx")
assert _RECORD_STRUCT.size == RECORD_SIZE_V4, f"struct size mismatch: {_RECORD_STRUCT.size}"

# ── Download ──────────────────────────────────────────────────────────────────
def download_logs(host: str, account: str) -> None:
    remote = f"{host}:{REMOTE_LOG.format(account=account)}"
    LOCAL_LOG_DIR.mkdir(exist_ok=True)
    for fname in ("training_log.csv", "mt_training_log.bin"):
        src = f"{remote}/{fname}"
        dst = LOCAL_LOG_DIR / fname
        print(f"  scp {src} → {dst}")
        result = subprocess.run(["scp", src, str(dst)], capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"scp failed: {result.stderr.strip()}")

# ── Parse training_log.csv ────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v.strip()) for k, v in row.items()})
    return rows

# ── Parse mt_training_log.bin ─────────────────────────────────────────────────
def load_binary_log(path: Path) -> list[dict]:
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        sys.exit(f"{path}: too small to be a valid log ({len(data)} bytes)")

    magic, version, n_ind, _ = struct.unpack_from("<IIII", data, 0)
    if magic != MT_LOG_MAGIC:
        sys.exit(f"{path}: bad magic {magic:#010x} (expected {MT_LOG_MAGIC:#010x})")
    if version != 3:
        sys.exit(f"{path}: log version {version} — this tool requires v3 (984-byte records)")

    records = []
    offset = HEADER_SIZE
    while offset + RECORD_SIZE_V4 <= len(data):
        raw = _RECORD_STRUCT.unpack_from(data, offset)
        pass_num   = raw[0] + 1   # C++ writes 0-indexed; normalise to match CSV's 1-indexed
        actual_day = raw[1]

        f = raw[2:242]  # 240 MT1 floats
        mt1 = {}
        for ci, comp in enumerate(_COMP_NAMES):
            mt1[comp] = {}
            for si, stat in enumerate(_STAT_NAMES):
                base = ci * 48 + si * 12
                mt1[comp][stat] = list(f[base : base + 12])

        records.append({
            "pass":      pass_num,
            "day":       actual_day,
            "mt1":       mt1,
            "mt2_best":  raw[242],
            "mt2_slot0": raw[243],
            "mt2_ideal": raw[244],
            "mt2_inj":   raw[245],
        })
        offset += RECORD_SIZE_V4

    return records

# ── Downsampling ──────────────────────────────────────────────────────────────
def _smooth_seg(xs: list, ys: list, target: int) -> tuple[list, list]:
    """Average (xs, ys) into `target` evenly-spaced bins when len > target."""
    n = len(xs)
    if n <= target:
        return list(xs), list(ys)
    out_xs, out_ys = [], []
    for k in range(target):
        lo = int(k * n / target)
        hi = min(n, int((k + 1) * n / target) + 1)
        if lo >= hi:
            hi = lo + 1
        out_xs.append(sum(xs[lo:hi]) / (hi - lo))
        out_ys.append(sum(ys[lo:hi]) / (hi - lo))
    return out_xs, out_ys

def _smooth(xs: list, ys: list) -> tuple[list, list]:
    """Smooth a simple series (no NaN breaks) to ~TARGET_POINTS."""
    return _smooth_seg(xs, ys, TARGET_POINTS)

# ── Floor-reset detection ─────────────────────────────────────────────────────
def _is_reset(mean_v: float, max_v: float, min_v: float) -> bool:
    # Pre-fix binary: 0.0 is the sentinel returned on a floor-reset day
    if mean_v == 0.0:
        return True
    # Post-fix binary: floor reset returns IND_STARTING_CASH for all three
    if (mean_v == IND_STARTING_CASH
            and max_v == IND_STARTING_CASH
            and min_v == IND_STARTING_CASH):
        return True
    return False

# ── Industry display series (floor-reset aware) ───────────────────────────────
def _build_industry_series(
    xs: list[int],
    raw_means: list[float],
    raw_maxes: list[float],
    raw_mins:  list[float],
) -> tuple[tuple[list, list], tuple[list, list], tuple[list, list], list[int]]:
    """
    Return (mean_series, max_series, min_series, reset_xs):
      - On reset days: line extends DOWN to FLOOR_VALUE (open-circle position),
        then a NaN break, then IND_STARTING_CASH as the first point of the
        next segment (filled-dot position connecting to the next day).
      - Each continuous segment is proportionally smoothed to ~TARGET_POINTS.
    """
    total_n = len(xs)

    def _split_and_smooth(raw: list[float]) -> tuple[list, list]:
        reset_xs_set = {
            xs[j] for j in range(total_n)
            if _is_reset(raw_means[j], raw_maxes[j], raw_mins[j])
        }
        segs: list[tuple[list, list]] = []
        seg_x: list[float] = []
        seg_y: list[float] = []
        for x, v in zip(xs, raw):
            if x in reset_xs_set:
                seg_x.append(float(x))
                seg_y.append(FLOOR_VALUE)   # line drops to open-circle level
                segs.append((seg_x, seg_y))
                seg_x = [float(x)]
                seg_y = [IND_STARTING_CASH] # filled-dot level starts next segment
            else:
                seg_x.append(float(x))
                seg_y.append(v)
        if seg_x:
            segs.append((seg_x, seg_y))

        out_xs: list[float] = []
        out_ys: list[float] = []
        for i, (sx, sy) in enumerate(segs):
            seg_target = max(2, round(TARGET_POINTS * len(sx) / total_n))
            smx, smy = _smooth_seg(sx, sy, seg_target)
            if i:
                out_xs.append(math.nan)
                out_ys.append(math.nan)
            out_xs.extend(smx)
            out_ys.extend(smy)
        return out_xs, out_ys

    reset_xs = [
        xs[j] for j in range(total_n)
        if _is_reset(raw_means[j], raw_maxes[j], raw_mins[j])
    ]
    return (
        _split_and_smooth(raw_means),
        _split_and_smooth(raw_maxes),
        _split_and_smooth(raw_mins),
        reset_xs,
    )

# ── Shared plot helpers ───────────────────────────────────────────────────────
def _style_ax(ax, title: str, xlabel: str, ylabel: str, title_size: int = 13) -> None:
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8)

def _industry_legend(ax, extra_handles: list | None = None) -> None:
    handles = [
        mlines.Line2D([], [], color=IND_COLOR[ind], linewidth=1.8, label=IND_LABEL[ind])
        for ind in INDUSTRIES
    ]
    handles += [
        mlines.Line2D([], [], color="#444444", linewidth=1.8, linestyle="-",  label="─── mean"),
        mlines.Line2D([], [], color="#444444", linewidth=0.9, linestyle="--", label="- - max / min"),
    ]
    if extra_handles:
        handles += extra_handles
    ax.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.14), ncol=7,
        fontsize=8, framealpha=0.95, edgecolor="#cccccc",
    )

def _plot_band(ax, xs, means, maxes, mins, color: str) -> None:
    ax.plot(xs, means, color=color, linewidth=1.5, alpha=0.90, zorder=3)
    ax.plot(xs, maxes, color=color, linewidth=0.75, linestyle="--", alpha=0.50, zorder=2)
    ax.plot(xs, mins,  color=color, linewidth=0.75, linestyle="--", alpha=0.50, zorder=2)

def _save(fig, out_path: Path) -> None:
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

# ── Industry performance SVG ──────────────────────────────────────────────────
def plot_industry(rows: list[dict], pass_num: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    fig.subplots_adjust(bottom=0.23)

    xs = [int(r["day"]) for r in rows]
    has_resets = False

    for ind in INDUSTRIES:
        raw_means = [r[f"{ind}_elite_mean"] for r in rows]
        raw_maxes = [r[f"{ind}_elite_max"]  for r in rows]
        raw_mins  = [r[f"{ind}_elite_min"]  for r in rows]

        (mean_xs, mean_ys), (max_xs, max_ys), (min_xs, min_ys), reset_xs = \
            _build_industry_series(xs, raw_means, raw_maxes, raw_mins)

        color = IND_COLOR[ind]
        _plot_band(ax, mean_xs, mean_ys, max_ys, min_ys, color)

        if reset_xs:
            has_resets = True
            # Open circle at FLOOR_VALUE (where the reset was triggered)
            ax.scatter(
                reset_xs, [FLOOR_VALUE] * len(reset_xs),
                s=55, facecolors="none", edgecolors=color,
                linewidths=1.5, zorder=5,
            )
            # Filled dot at IND_STARTING_CASH (where portfolios reset to)
            ax.scatter(
                reset_xs, [IND_STARTING_CASH] * len(reset_xs),
                s=30, facecolors=color, edgecolors=color,
                linewidths=1.0, zorder=5,
            )

    ax.axhline(IND_STARTING_CASH, color="#aaaaaa", linewidth=0.6, linestyle=":", zorder=1)
    ax.axhline(FLOOR_VALUE,        color="#ffaaaa", linewidth=0.6, linestyle=":", zorder=1)
    _style_ax(ax,
              f"Industry Elite Performance (StockNN) — Pass {pass_num}",
              f"Day (Pass {pass_num})", "Portfolio Value ($)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    extra = []
    if has_resets:
        extra = [
            mlines.Line2D([], [], color="#888888", marker="o", linestyle="none",
                          markersize=6, markerfacecolor="none", markeredgewidth=1.5,
                          label="○  floor reset ($22.5K)"),
            mlines.Line2D([], [], color="#888888", marker="o", linestyle="none",
                          markersize=5, markerfacecolor="#888888",
                          label="●  reset-to ($25K)"),
        ]
    _industry_legend(ax, extra_handles=extra)
    _save(fig, out_path)

# ── MT1 single-component SVG ──────────────────────────────────────────────────
def plot_mt1_component(records: list[dict], comp: str, pass_num: int, out_path: Path,
                       csv_rows: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    fig.subplots_adjust(bottom=0.23)

    xs_raw = [r["day"] for r in records]
    n_ind = len(INDUSTRIES)
    for i, ind in enumerate(INDUSTRIES):
        means_raw = [r["mt1"][comp]["mean"][i] for r in records]
        bests_raw = [r["mt1"][comp]["best"][i] for r in records]
        mins_raw  = [r["mt1"][comp]["min"][i]  for r in records]

        xs_m, means = _smooth(xs_raw, means_raw)
        xs_b, bests = _smooth(xs_raw, bests_raw)
        xs_n, mins_s = _smooth(xs_raw, mins_raw)

        _plot_band(ax, xs_m, means, bests, mins_s, IND_COLOR[ind])

    # All-industry mean/max/min: average across industries of each per-day stat
    all_means_raw = [sum(r["mt1"][comp]["mean"]) / n_ind for r in records]
    all_bests_raw = [sum(r["mt1"][comp]["best"]) / n_ind for r in records]
    all_mins_raw  = [sum(r["mt1"][comp]["min"])  / n_ind for r in records]
    xs_am, all_means = _smooth(xs_raw, all_means_raw)
    xs_ab, all_bests = _smooth(xs_raw, all_bests_raw)
    xs_an, all_mins  = _smooth(xs_raw, all_mins_raw)
    ax.plot(xs_am, all_means, color="black", linewidth=2.5, linestyle=(0, (4, 2)),
            zorder=5, label="All-ind mean (mean)")
    ax.plot(xs_ab, all_bests, color="black", linewidth=1.8, linestyle=(0, (2, 2)),
            zorder=5, label="All-ind mean (max)")
    ax.plot(xs_an, all_mins,  color="black", linewidth=1.8, linestyle=(0, (1, 2)),
            zorder=5, label="All-ind mean (min)")

    ax.set_ylim(0, 6)
    ylabel = "5-day sum score (0 – 6)"
    if comp == "direction":
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle="--",
                   label="Random baseline (3.0)")

        # Mean n_correct_dbl across all industries (primary sort key denominator)
        # Plotted on left axis (same 0-6 scale as scores). Triangle markers.
        cdb_raw = [
            sum(r["mt1_dir_correct_dbl"]) / n_ind
            for r in records
        ]
        xs_cdb, cdb_vals = _smooth(xs_raw, cdb_raw)
        # Choose markevery so ~12-15 triangles appear regardless of series length
        markevery = max(1, len(xs_cdb) // 14)
        ax.plot(xs_cdb, cdb_vals, color="#e06000", linewidth=2.2,
                marker="^", markersize=6, markevery=markevery,
                zorder=6, label="Mean correct predictions (today×2)")

        # % industries with rising portfolio value day-over-day (right y-axis)
        if csv_rows:
            csv_by_day = {int(r["day"]): r for r in csv_rows}
            net_pos_raw = []
            for r in records:
                day = r["day"]
                row = csv_by_day.get(day)
                prev_row = csv_by_day.get(day - 1)
                if row and prev_row:
                    pct = 100.0 * sum(
                        1 for ind in INDUSTRIES
                        if float(row.get(f"{ind}_elite_mean", 0)) > float(prev_row.get(f"{ind}_elite_mean", 0))
                    ) / n_ind
                else:
                    pct = 0.0
                net_pos_raw.append(pct)
        else:
            net_pos_raw = [
                100.0 * sum(1 for i in range(n_ind) if r["mt1"]["direction"]["mean"][i] > 3.0) / n_ind
                for r in records
            ]
        xs_np, net_pos = _smooth(xs_raw, net_pos_raw)
        ax2 = ax.twinx()
        ax2.plot(xs_np, net_pos, color="#808080", linewidth=2.5, zorder=4,
                 label="% portfolio up day-over-day")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("% industries up day-over-day", fontsize=10, color="#606060")
        ax2.tick_params(labelsize=8, colors="#606060")
        ax2.spines["right"].set_color("#808080")
    else:
        ax2 = None

    _style_ax(ax,
              f"MT1 — {MT1_COMP_LABEL[comp]} — Pass {pass_num}",
              f"Day (Pass {pass_num})", ylabel)
    extra = [
        mlines.Line2D([], [], color="black", linewidth=2.5, linestyle=(0, (4, 2)), label="All-ind mean (mean)"),
        mlines.Line2D([], [], color="black", linewidth=1.8, linestyle=(0, (2, 2)), label="All-ind mean (max)"),
        mlines.Line2D([], [], color="black", linewidth=1.8, linestyle=(0, (1, 2)), label="All-ind mean (min)"),
    ]
    if comp == "direction":
        extra.append(mlines.Line2D([], [], color="#e06000", linewidth=2.2,
                                   marker="^", markersize=6,
                                   label="Mean correct predictions (today×2)"))
    if ax2 is not None:
        extra.append(mlines.Line2D([], [], color="#808080", linewidth=2.5,
                                   label="% industries up day-over-day (right axis)"))
    _industry_legend(ax, extra_handles=extra)
    _save(fig, out_path)

# ── MT2 SVG ───────────────────────────────────────────────────────────────────
_MT2_INJ_THRESHOLD = -7.0   # injection fires when ≥75% of pool scores below this
_MT2_BASELINE_WINDOW = 30   # days to look back when computing the random baseline


def _ideal_to_tier_counts(ideal_pts: float) -> tuple[int, int, int, int]:
    """Map mt2_ideal_pts → (n0, n1, n2, n3) optimal tier counts for that day."""
    best_n, best_diff = 0, float("inf")
    for n in range(13):
        if n == 0:   exp = 0.0
        elif n == 1: exp = 3.0
        elif n == 2: exp = 5.0
        else:
            base = n // 3; rem = n % 3
            n1 = base + (1 if rem >= 1 else 0)
            n2 = base + (1 if rem >= 2 else 0)
            n3 = n - n1 - n2
            exp = float(n1 + 2 * n2 + 3 * n3)
        diff = abs(exp - ideal_pts)
        if diff < best_diff:
            best_diff, best_n = diff, n
    n_pos = best_n
    if n_pos == 0: return (12, 0, 0, 0)
    if n_pos == 1: return (11, 0, 0, 1)
    if n_pos == 2: return (10, 0, 1, 1)
    base = n_pos // 3; rem = n_pos % 3
    n1 = base + (1 if rem >= 1 else 0)
    n2 = base + (1 if rem >= 2 else 0)
    n3 = n_pos - n1 - n2
    return (12 - n_pos, n1, n2, n3)


def _master_points(pred: int, opt: int) -> float:
    if opt == 0:
        return 0.0 if pred == 0 else -2.0 - 0.25 * pred
    if pred == 0: return -float(opt)
    if pred <= opt: return float(pred)
    return float(opt) - 0.25 * (pred - opt)


def _mt2_realistic_random_baseline(rows: list[dict]) -> tuple[float, int, tuple]:
    """
    Expected MT2 score for a random-tier guess that uses the same tier-count
    distribution observed in recent data:

      t0m = mean(# tier-0 industries over last 30 MT2 days), rounded to int
      Remaining 12-t0m split into tiers 1/2/3 by the optimal-tiers rule
        (lower tiers absorb the remainder → n1 ≥ n2 ≥ n3)
      Random guess: randomly assign industries to tiers with these fixed counts
        (each industry independently gets tier j with prob assign_j/12)

    Returns (expected_score, t0m_int, (n0, n1, n2, n3)).
    """
    mt2_rows = [r for r in rows if r.get("mt2_ideal_pts") is not None]
    window = mt2_rows[-_MT2_BASELINE_WINDOW:]
    if not window:
        return (-7.0, 6, (6, 2, 2, 2))

    # Mean optimal tier counts over window
    sum_opt = [0.0, 0.0, 0.0, 0.0]
    for r in window:
        n0, n1, n2, n3 = _ideal_to_tier_counts(r["mt2_ideal_pts"])
        sum_opt[0] += n0; sum_opt[1] += n1
        sum_opt[2] += n2; sum_opt[3] += n3
    n = len(window)
    mean_opt = [s / n for s in sum_opt]

    # Assignment counts: round t0m, distribute remainder by optimal-tiers rule
    t0m = round(mean_opt[0])
    n_nonzero = 12 - t0m
    if n_nonzero == 0:   an1, an2, an3 = 0, 0, 0
    elif n_nonzero == 1: an1, an2, an3 = 0, 0, 1
    elif n_nonzero == 2: an1, an2, an3 = 0, 1, 1
    else:
        base = n_nonzero // 3; rem = n_nonzero % 3
        an1 = base + (1 if rem >= 1 else 0)
        an2 = base + (1 if rem >= 2 else 0)
        an3 = n_nonzero - an1 - an2
    assign = [t0m, an1, an2, an3]

    # E[score] = sum_k mean_opt[k] * E[_master_points(rand_j, k)]
    # where rand_j ~ assign distribution (assign[j]/12 per industry)
    e_score = 0.0
    for k in range(4):
        e_opt_k = sum(assign[j] / 12.0 * _master_points(j, k) for j in range(4))
        e_score += mean_opt[k] * e_opt_k

    return (e_score, t0m, (t0m, an1, an2, an3))


def plot_mt2(rows: list[dict], pass_num: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    fig.subplots_adjust(bottom=0.13)

    xs_raw    = [int(r["day"]) for r in rows]
    means_raw = [r["mt2_elite_mean_pts"] for r in rows]
    maxes_raw = [r["mt2_elite_max_pts"]  for r in rows]
    mins_raw  = [r["mt2_elite_min_pts"]  for r in rows]
    ideals_raw= [r["mt2_ideal_pts"]      for r in rows]

    xs_m, means  = _smooth(xs_raw, means_raw)
    xs_b, maxes  = _smooth(xs_raw, maxes_raw)
    xs_n, mins_s = _smooth(xs_raw, mins_raw)
    xs_i, ideals = _smooth(xs_raw, ideals_raw)

    ax.plot(xs_m, means,  color="#1f77b4", linewidth=1.5,  alpha=0.90, label="Elite mean")
    ax.plot(xs_b, maxes,  color="#1f77b4", linewidth=0.75, alpha=0.55, linestyle="--", label="Elite max")
    ax.plot(xs_n, mins_s, color="#1f77b4", linewidth=0.75, alpha=0.55, linestyle="--", label="Elite min")
    ax.plot(xs_i, ideals, color="#2ca02c", linewidth=1.8,  alpha=0.90, label="Ideal (slot0 basis)")

    rand_score, t0m, assign = _mt2_realistic_random_baseline(rows)
    n0, n1, n2, n3 = assign
    ax.axhline(rand_score, color="#d62728", linewidth=1.0, alpha=0.90, linestyle="-",
               label=f"Random baseline (t0m={t0m}: {n0}×T0 {n1}×T1 {n2}×T2 {n3}×T3 → E={rand_score:.2f})")

    ax.axhline(_MT2_INJ_THRESHOLD, color="#ff7f0e", linewidth=0.8, alpha=0.85, linestyle="-",
               label=f"75% injection threshold ({_MT2_INJ_THRESHOLD:.0f} pts)")
    ax.axhline(0, color="#dddddd", linewidth=0.5, zorder=1)
    _style_ax(ax,
              f"MT2 Allocation Score (Elite Pool) — Pass {pass_num}",
              f"Day (Pass {pass_num})", "Score (pts)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=9, framealpha=0.95, edgecolor="#cccccc")
    _save(fig, out_path)

# ── Pass helpers ──────────────────────────────────────────────────────────────
def _pass_max_day(all_rows: list[dict], pass_num: int) -> int:
    return max((int(r["day"]) for r in all_rows if int(r["pass"]) == pass_num), default=0)

def _is_complete(all_rows: list[dict], pass_num: int) -> bool:
    return _pass_max_day(all_rows, pass_num) >= DAYS_PER_PASS

def _generate_all(rows: list[dict], records: list[dict], pass_num: int, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    plot_industry(rows, pass_num, out_dir / "industry_performance.svg")
    if records:
        for comp in _COMP_NAMES:
            plot_mt1_component(records, comp, pass_num, out_dir / f"mt1_{comp}.svg",
                               csv_rows=rows)
    else:
        print(f"  (no MT1 records for pass {pass_num} — MT1 activates at day 25)")
    plot_mt2(rows, pass_num, out_dir / "mt2_performance.svg")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate training progress SVGs")
    ap.add_argument("--download", action="store_true", help="Pull fresh logs from droplet")
    ap.add_argument("--host",    default=DROPLET_HOST, help="Droplet SSH host")
    ap.add_argument("--account", default="acct0",      help="Account (e.g. acct0)")
    ap.add_argument("--pass",    dest="pass_num", type=int, default=None,
                    help="Pass to display in plots/ (default: last completed, or pass 1)")
    args = ap.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)
    LOCAL_LOG_DIR.mkdir(exist_ok=True)

    if args.download:
        print("Downloading logs from droplet...")
        download_logs(args.host, args.account)

    csv_path = LOCAL_LOG_DIR / "training_log.csv"
    bin_path = LOCAL_LOG_DIR / "mt_training_log.bin"

    for p in (csv_path, bin_path):
        if not p.exists():
            sys.exit(f"Missing {p} — run with --download or copy manually")

    print("Parsing logs...")
    all_rows    = load_csv(csv_path)
    all_records = load_binary_log(bin_path)
    if not all_rows:
        sys.exit("training_log.csv is empty")
    if not all_records:
        sys.exit("mt_training_log.bin has no records (MT1 activates at day 25)")

    all_passes = sorted({int(r["pass"]) for r in all_rows}
                        | {r["pass"] for r in all_records})
    completed  = [p for p in all_passes if _is_complete(all_rows, p)]

    # Default: last completed pass, or pass 1 if none are complete
    if args.pass_num is not None:
        display_pass = args.pass_num
        if display_pass not in all_passes:
            sys.exit(f"Pass {display_pass} not in logs (available: {all_passes})")
    else:
        display_pass = completed[-1] if completed else (all_passes[0] if all_passes else 1)

    print(f"  Available passes: {all_passes}  |  completed: {completed or 'none'}  |"
          f"  display: pass {display_pass}")

    # Archive charts for every completed pass in plots/passN/
    for p in completed:
        pdir = PLOT_DIR / f"pass{p}"
        print(f"Archiving pass {p} → {pdir}/")
        rows_p = [r for r in all_rows    if int(r["pass"]) == p]
        recs_p = [r for r in all_records if     r["pass"]  == p]
        _generate_all(rows_p, recs_p, p, pdir)

    # Generate display charts in plots/
    print(f"Generating display charts (pass {display_pass}) → {PLOT_DIR}/")
    rows_d = [r for r in all_rows    if int(r["pass"]) == display_pass]
    recs_d = [r for r in all_records if     r["pass"]  == display_pass]
    if not rows_d:
        sys.exit(f"No CSV rows for pass {display_pass}")
    max_day = max(int(r["day"]) for r in rows_d)
    print(f"  {len(rows_d)} CSV rows (day 1–{max_day})  |  {len(recs_d)} MT1 records")
    _generate_all(rows_d, recs_d, display_pass, PLOT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
