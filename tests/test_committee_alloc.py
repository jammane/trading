"""committee_alloc: the committee allocation production trades, and its daily record.

The vote must be committee_race's own `votes_member` on one day, or paper trades a committee the
race never scored. The allocation must put nothing in a flagged industry (sell-only, never
liquidated) and tilt the rest exactly as the replay did. The record must compute each industry's
real return from what the account actually did.
"""
import csv

import numpy as np
import pytest

import committee_alloc as ca
import committee_race as cr

INDS = [f"i{k}" for k in range(12)]


def _preds(seed=0):
    rng = np.random.default_rng(seed)
    return {e: rng.normal(size=12) for e in ca.ENTRIES}


def test_vote_is_committee_race_on_one_day():
    """Same picks as committee_race's scoring path on a 1-day pass, for both committees."""
    for seed in range(20):
        p = _preds(seed)
        ok = np.ones(12, bool)
        for members, losers in ((ca.WINNERS, False), (ca.LOSERS, True)):
            P = dict(real=np.zeros((1, 12)), flat=np.zeros(1), ok=ok[None],
                     pred={e: (-1 if losers else 1) * p[e][None] for e in members})
            F = {e: (*cr.per_entry(P["pred"][e], P["ok"]), np.zeros(1)) for e in members}
            k = len(members)
            s, v, z = cr.committee_scores(list(members), [1.1 ** (k - 1 - j) for j in range(k)],
                                          P, F)
            want = cr.fund(s["votes_member"], P["ok"], v, z)[0]
            assert (ca.vote(members, p, ok, losers=losers) == want).all()


def test_member_weight_breaks_a_split_vote_toward_the_strongest_member():
    """Two members disagree on the 4th pick; the stronger (listed first) member's pick wins."""
    a = np.array([9, 8, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0], float)      # top 4: 0 1 2 3
    b = np.array([9, 8, 7, 0, 6, 0, 0, 0, 0, 0, 0, 0], float)      # top 4: 0 1 2 4
    pick = ca.vote(("A", "B"), {"A": a, "B": b}, np.ones(12, bool))
    assert pick.tolist()[:5] == [True, True, True, True, False]
    pick = ca.vote(("B", "A"), {"A": a, "B": b}, np.ones(12, bool))
    assert pick.tolist()[:5] == [True, True, True, False, True]


def test_loser_vote_flags_the_lowest():
    x = np.arange(12, dtype=float)
    pick = ca.vote(("A",), {"A": x}, np.ones(12, bool), losers=True)
    assert np.flatnonzero(pick).tolist() == [0, 1, 2, 3]


def test_mask_keeps_the_winners_out_of_the_flagged():
    x = np.arange(12, dtype=float)[::-1]                             # best are 0..3
    mask = np.ones(12, bool)
    mask[:2] = False
    assert np.flatnonzero(ca.vote(("A",), {"A": x}, np.ones(12, bool), mask=mask)).tolist() \
        == [2, 3, 4, 5]


def test_allocation_sell_only_four_and_tilt_the_rest():
    d = ca.decide(INDS, _preds(3), 3000.0)
    a = d["allocations"]
    assert len(d["sell_only"]) == 4 and len(d["winners"]) == 4
    assert not set(d["sell_only"]) & set(d["winners"])
    assert sum(a.values()) == pytest.approx(3000.0)
    for ind in INDS:
        want = 0.0 if ind in d["sell_only"] else 3000.0 * (0.5 / 8 + (0.5 / 4 if ind in
                                                                          d["winners"] else 0))
        assert a[ind] == pytest.approx(want)


def test_an_industry_without_predictions_sits_out_and_takes_the_base_share():
    p = _preds(4)
    p["MT1 evo"][5] = np.nan
    d = ca.decide(INDS, p, 1000.0)
    assert "i5" not in d["sell_only"] and "i5" not in d["winners"]
    assert d["allocations"]["i5"] == pytest.approx(1000.0 * 0.5 / 8)
    assert sum(d["allocations"].values()) == pytest.approx(1000.0)


def test_models_load_from_the_daily_state_before_the_champion(tmp_path):
    import race_models as rm
    st, seed = tmp_path / "cpp", tmp_path / "champion"
    st.mkdir()
    seed.mkdir()
    name = f"race_{rm.slug('TANH32 grad')}_i0.bin"
    np.full(rm.TANH32_PARAMS, 1, np.float32).tofile(seed / name)
    assert ca.load_model("TANH32 grad", "i0", str(st), str(seed)).W[0] == 1
    np.full(rm.TANH32_PARAMS, 2, np.float32).tofile(st / name)
    assert ca.load_model("TANH32 grad", "i0", str(st), str(seed)).W[0] == 2
    with pytest.raises(FileNotFoundError):
        ca.load_model("TANH32 grad", "i1", str(st), str(seed))


def _pending():
    return dict(date="2026-10-13", industries=["a", "b", "c"],
                symbols={"a": ["A1", "A2"], "b": ["B1"], "c": ["C1"]},
                allocations={"a": 100.0, "b": 200.0, "c": 0.0}, sell_only=["c"], winners=["b"],
                holdings={"A1": 2, "C1": 10}, closes={"A1": 50.0, "A2": 20.0, "B1": 10.0,
                                                      "C1": 5.0},
                preds={e: [3.0, 2.0, 1.0] for e in ca.ENTRIES}, equity=1000.0)


def test_industry_return_counts_holdings_fills_and_the_cash_it_was_given():
    closes = {"A1": 55.0, "A2": 21.0, "B1": 11.0, "C1": 4.0}
    positions = {"A1": 2, "A2": 3, "B1": 0, "C1": 6}
    fills = [("A2", "buy", 3, 20.5), ("C1", "sell", 4, 4.5)]
    r = ca.industry_returns(_pending(), closes, positions, fills)
    a0, a1 = 100 + 2 * 50, 100 + 2 * 55 + 3 * 21 - 3 * 20.5
    c0, c1 = 10 * 5, 6 * 4 + 4 * 4.5                     # sell-only: no cash in its book
    assert r[0] == pytest.approx((a1 / a0 - 1) * 1e4)
    assert r[1] == pytest.approx(0.0)                     # cash only, nothing bought
    assert r[2] == pytest.approx((c1 / c0 - 1) * 1e4)


def test_settled_rows_are_race_daily_rows(tmp_path):
    real = np.array([100.0, -50.0, -200.0])
    rows = ca.settle_rows(_pending(), real, 12.5, 0)
    log = tmp_path / "committee_daily.csv"
    ca.append_rows(str(log), ["a", "b", "c"], rows)
    ca.append_rows(str(log), ["a", "b", "c"], rows)
    got = list(csv.DictReader(log.open()))
    assert len(got) == 2 * len(rows)                       # header written once
    by = {r["entry"]: r for r in got[:len(rows)]}
    flat = (100 - 50 - 200) / 3
    assert float(by["REALIZED"]["flat_bp"]) == pytest.approx(flat)
    assert float(by["REALIZED"]["c"]) == pytest.approx(-200)
    assert by["L1 committee"]["pick1"] == "c"
    assert float(by["L1 committee"]["gap_bp"]) == pytest.approx(-200 - flat)
    assert float(by["W1 committee"]["top4_bp"]) == pytest.approx(-50)
    assert float(by["PORTFOLIO"]["gap_bp"]) == pytest.approx(12.5 - flat)
    assert float(by["PORTFOLIO"]["b"]) == pytest.approx(200)
    assert by["TANH32 grad"]["pick1"] == "a" and float(by["TANH32 grad"]["top4_bp"]) == \
        pytest.approx(flat)
    assert set(ca.ENTRIES) <= set(by)
