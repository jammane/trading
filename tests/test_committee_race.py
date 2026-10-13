"""committee_race.py: the post-race race of voting committees over race_daily.csv.

Graded on synthetic files with a KNOWN answer: the entrant rule (most wins, at least 6 with
ties, united with the 6 best means), the 1.1-step weightings on a hand-worked day, committees of
independent noisy-but-informative models beating their best member (the reason to vote), pure
noise not looking consistent, and the Bayesian tree being causal and learning which member to
trust.
"""
import numpy as np

import committee_race as cr

INDS = [f"i{k}" for k in range(12)]


def _write(path, models, passes=4, days=150, seed=0):
    """models: {name: noise sd, or None for pure noise}; returns are N(0, 100) bp per industry."""
    rng = np.random.default_rng(seed)
    hdr = "pass,day,date,entry,ok_n,flat_bp,top4_bp,gap_bp,pick1,pick2,pick3,pick4," + ",".join(INDS)
    lines = [hdr]
    for p in range(1, passes + 1):
        for d in range(400, 400 + days):
            r = rng.normal(0, 100, 12)
            flat = r.mean()
            lines.append(f"{p},{d},2025-01-01,REALIZED,12,{flat:.6f},,,,,,,"
                         + ",".join(f"{x:.10g}" for x in r))
            for name, sd in models.items():
                pred = rng.normal(0, 100, 12) if sd is None else r + rng.normal(0, sd, 12)
                top = np.argsort(-pred, kind="stable")[:4]
                t4 = r[top].mean()
                lines.append(f"{p},{d},2025-01-01,{name},12,{flat:.6f},{t4:.6f},{t4 - flat:.6f},"
                             + ",".join(INDS[k] for k in top) + ","
                             + ",".join(f"{x:.10g}" for x in pred))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _day(real, preds):
    """one-day pass dict + features for hand-worked cases"""
    ok = np.ones((1, 12), bool)
    real = np.asarray(real, float)[None]
    P = dict(real=real, ok=ok, flat=real.mean(1))
    F = {}
    for name, p in preds.items():
        rk, z, pos = cr.per_entry(np.asarray(p, float)[None], ok)
        F[name] = (rk, z, pos, np.zeros(1))
    return P, F


def test_entrant_pool_most_wins_with_ties_plus_best_means():
    s = {"A": (9, 1.0), "B": (8, 1.0), "C": (8, 0.5), "D": (7, 0.2), "E": (7, 0.1),
         "F": (6, 0.3), "G": (6, 0.0), "H": (5, 0.9), "I": (4, 3.0), "J": (3, -1.0)}
    pool = cr.entrant_pool(s)
    # by wins: A B C D E F, and G is tied with F at 6 -> in; by mean: I A B H C F
    assert pool == ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


def test_committee_of_copies_reproduces_the_single_model():
    real = np.arange(12.0)
    pred = [5, 9, 1, 7, 3, 11, 0, 2, 4, 6, 8, 10]
    P, F = _day(real, {"A": pred})
    F.update({n: F["A"] for n in "BCD"})
    out = cr.run_committee(list("ABCD"), [1.331, 1.21, 1.1, 1.0], P, F)
    single = np.sort(real[np.argsort(pred)[-4:]]).mean() - real.mean()
    for meth in ("votes", "votes_pick", "votes_member", "votes_both", "borda", "zrecord"):
        assert abs(out[meth][0] - single) < 1e-9, meth


def test_one_point_one_steps_on_a_hand_worked_day():
    # A (strongest) prefers i0..i3, B prefers i4..i7; both in the same order of strength
    real = [10, 20, 30, 40, 1, 2, 3, 4, 0, 0, 0, 0]
    a = [12, 11, 10, 9, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 12, 11, 10, 9, 0, 0, 0, 0]
    P, F = _day(real, {"A": a, "B": b})
    flat = np.mean(real)
    out = cr.run_committee(["A", "B"], [1.1, 1.0], P, F)
    # member weights: A's four picks outrank B's
    assert abs(out["votes_member"][0] - (np.mean([10, 20, 30, 40]) - flat)) < 1e-9
    # pick weights: each member's #1 and #2 (1.331, 1.21) beat any #3 (1.1)
    assert abs(out["votes_pick"][0] - (np.mean([10, 20, 1, 2]) - flat)) < 1e-9
    # both: A#1 1.4641; A#2 and B#1 1.331; then A#3 and B#2 tie at 1.21, and the tie goes to
    # the higher mean z -- B's #2 (z 1.50) over A's #3 (z 1.30)
    assert abs(out["votes_both"][0] - (np.mean([10, 20, 1, 2]) - flat)) < 1e-9


def test_independent_informative_models_vote_better_than_their_best_member(tmp_path):
    path = _write(tmp_path / "rd.csv", {f"M{k}": 400.0 for k in range(6)}, passes=4, days=150)
    _, pool, ps, res = cr.run(path, sizes=(6,))
    r = next(x for x in res if x["method"] == "borda")
    assert r["w_best"] >= 3                                           # of 4 passes
    assert r["w_flat"] == 4


def test_pure_noise_committees_are_not_consistent(tmp_path):
    path = _write(tmp_path / "rd.csv", {f"N{k}": None for k in range(7)}, passes=6, days=120,
                  seed=3)
    _, _, ps, res = cr.run(path, sizes=(4, 5))
    assert sum(r["w_flat"] == len(ps) for r in res) / len(res) < 0.15
    assert abs(np.mean([r["gaps"].mean() for r in res])) < 5.0        # bp/day; signal case ~40


def test_record_is_causal():
    g = np.array([1.0, -2.0, 3.0, 4.0, -5.0])
    base = cr.record(g)
    for t in range(len(g)):
        h = g.copy()
        h[t] = 1e6
        assert np.allclose(cr.record(h)[:t + 1], base[:t + 1])


def test_bayes_tree_is_causal():
    rng = np.random.default_rng(1)
    T = 40
    pos = np.where(rng.random((3, T, 12)) < 0.33, 0, -1)
    real = rng.normal(0, 100, (T, 12))
    ok = np.ones((T, 12), bool)
    base = cr.bayes_tree(pos, real, real.mean(1), ok)
    for t in (0, 7, 25):
        r2 = real.copy()
        r2[t] += 1e5                                                  # day t's outcome only
        s2 = cr.bayes_tree(pos, r2, r2.mean(1), ok)
        assert np.allclose(s2[:t + 1], base[:t + 1])                  # decisions <= t unchanged


def test_bayes_tree_learns_which_member_to_trust(tmp_path):
    # one informative member among noise: a plain vote is diluted, the tree finds the pattern
    models = {"good": 150.0, **{f"N{k}": None for k in range(3)}}
    path = _write(tmp_path / "rd.csv", models, passes=4, days=250, seed=5)
    _, _, _, res = cr.run(path, sizes=(4,), pool=list(models))
    g = {r["method"]: r["gaps"].mean() for r in res}
    assert g["bayes_tree"] > g["votes"] + 5.0


def test_pass_pool_is_winners_plus_the_best_six():
    s = {e: (int(g > 0), g, np.array([g])) for e, g in
         zip("ABCDEFGHI", [5, 4, 3, 2, 1, 0.5, 0.2, -1, -2], strict=True)}
    assert cr.pass_pool(s) == list("ABCDEFG")             # seven winners, no losers needed
    s = {e: (int(g > 0), g, np.array([g])) for e, g in
         zip("ABCDEFG", [5, 4, -1, -2, -3, -4, -5], strict=True)}
    assert cr.pass_pool(s) == list("ABCDEF")              # two winners, filled to six by gap


def test_watch_credits_the_informative_member_and_counts_eligibility(tmp_path):
    models = {"good": 40, "m1": 150, "m2": 150, "m3": 150, "m4": 150, "noise": None,
              "noise2": None}
    w = cr.watch(_write(tmp_path / "rd.csv", models, passes=3, days=150, seed=3))
    assert w["ps"] == [1, 2, 3]
    assert len(w["entrants"]["good"]) == 3
    eff = {e: np.mean(list(v.values())) for e, v in w["effect"].items()}
    assert eff["good"] > 0 and eff["good"] == max(eff.values())
    for e in ("noise", "noise2"):
        if e in eff:
            assert eff[e] < 0, e
    for c, p in w["top_c"].items():
        assert len(p) <= w["eligible"](c)
        assert c == " + ".join(sorted(c.split(" + ")))      # one key per set, across passes
    assert sum(s["won"] for s in w["methods"].values()) == 3


def test_champion_composite_takes_each_industry_from_its_champion_pass(tmp_path):
    path = _write(tmp_path / "rd.csv", {"good": 30.0, "noise": None}, passes=3, days=40)
    ref = tmp_path / "pass_reference.csv"
    champ = {i: (k % 3) + 1 for k, i in enumerate(INDS)}
    ref.write_text("# pass_reference v2\npass,industry,new_champ_pass\n"
                   + "".join(f"2,{i},1\n" for i in INDS)                 # superseded pass
                   + "".join(f"3,{i},{p}\n" for i, p in champ.items()))
    assert cr.read_champions(str(ref)) == champ
    entries, passes = cr.load(path)
    C = cr.champion_composite(entries, passes, champ)
    for j, i in enumerate(INDS):
        assert np.array_equal(C["real"][:, j], passes[champ[i]]["real"][:, j])
        assert np.array_equal(C["pred"]["good"][:, j], passes[champ[i]]["pred"]["good"][:, j])
    # the gap is recomputed from the composite's own top 4, not copied from any pass
    top = np.argsort(-C["pred"]["good"], axis=1, kind="stable")[:, :4]
    want = np.take_along_axis(C["real"], top, 1).mean(1) - C["real"].mean(1)
    assert np.allclose(C["gap"]["good"], want)
    assert C["gap"]["good"].mean() > C["gap"]["noise"].mean()


def test_loser_race_flags_the_worst_four():
    """Mirrored, a model's "top 4" is its bottom 4: gap = flat - their return, hit = they lost."""
    real = np.array([-50, -40, -30, -20, 5, 10, 15, 20, 25, 30, 35, 40], float)
    P = cr.mirror(dict(real=real[None], ok=np.ones((1, 12), bool), flat=real.mean(keepdims=True),
                       pred={"A": real[None].copy()}, gap={"A": np.zeros(1)}))
    flat = real.mean()
    assert np.isclose(P["gap"]["A"][0], flat - np.mean([-50, -40, -30, -20]))
    F = {"A": (*cr.per_entry(P["pred"]["A"], P["ok"]), np.zeros(1))}
    gap, hit = cr.run_committee(["A"], [1.0], P, F)["votes"]
    assert np.isclose(gap, flat - np.mean([-50, -40, -30, -20])) and hit == 1.0


def test_loser_race_rewards_a_model_that_knows_only_the_losers(tmp_path):
    """Informative on the bottom of the ranking only: wins the loser race outright."""
    path = _write(tmp_path / "rd.csv", {"N1": None, "N2": None})
    rows = (tmp_path / "rd.csv").read_text().splitlines()
    rng = np.random.default_rng(3)
    out = [rows[0]]
    for line in rows[1:]:
        f = line.split(",")
        out.append(line)
        if f[3] == "REALIZED":
            r = np.array(f[12:], float)
            pred = rng.normal(0, 1, 12)
            pred[np.argsort(r)[:4]] -= 10           # its 4 lowest are today's 4 worst
            out.append(",".join(f[:3] + ["L", "12", f[5], "", "0", "", "", "", ""]
                                + [f"{x:.6g}" for x in pred]))
    (tmp_path / "rd.csv").write_text("\n".join(out) + "\n")
    lose = cr.standings(*cr.load(path, losers=True))
    assert lose["L"][0] == 4 and lose["L"][1] > 50
    assert lose["L"][1] > max(lose["N1"][1], lose["N2"][1])
    _, passes = cr.load(path)
    top = np.mean([cr.entry_gap(P["pred"]["L"], P["real"], P["ok"], P["flat"]).mean()
                   for P in passes.values()])
    # its top 4 are random among the 8 non-losers: an edge at the top, but only about half
    assert 0.3 * lose["L"][1] < top < 0.7 * lose["L"][1]
