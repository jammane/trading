"""The weekly pass and its replacement list (weekly_pass.py, replacements.py).

What must hold: a symbol is swapped only for the two documented reasons, and only for a vetted
replacement that keeps its industry in band; a candidate failing any check is never offered; and
promoting a champion clears exactly its own industry's daily-training state -- no other
industry's, whose names can share words with it.
"""
import math

import numpy as np
import pytest

import replacements as rp
import weekly_pass as wp

TH = {"mean_median_swap_above": 1.35, "mean_median_swap_below": 0.93}


def _bars(n, price, vol, dollar=5e6, seed=0):
    rng = np.random.default_rng(seed)
    lp = np.cumsum(rng.normal(0, vol / math.sqrt(252), n))
    c = price * np.exp(lp - lp[-1])                    # ends exactly at `price`
    return [dict(open=x, high=x * 1.01, low=x * 0.99, close=x, volume=dollar / x) for x in c]


PEERS = dict(vol_min=0.30, vol_max=0.50, vol_med=0.40, price_med=60.0, liq_min=4e6)


def test_vet_passes_a_candidate_that_resembles_the_industry():
    st, why = rp.vet("X", "NMS", _bars(1300, 60, 0.4), PEERS, True)
    assert why is None and st["price"] == pytest.approx(60)


@pytest.mark.parametrize("kw, frag", [
    (dict(exchange="PNK"), "exchange"),
    (dict(n=1000), "history"),
    (dict(price=95), "price"),
    (dict(price=25), "price"),
    (dict(vol=0.9), "volatility"),
    (dict(vol=0.1), "volatility"),
    (dict(dollar=1e6), "traded"),
    (dict(tradable=False), "Alpaca"),
])
def test_vet_rejects_each_failed_check(kw, frag):
    a = dict(exchange="NMS", n=1300, price=60, vol=0.4, dollar=5e6, tradable=True) | kw
    _, why = rp.vet("X", a["exchange"], _bars(a["n"], a["price"], a["vol"], a["dollar"]), PEERS,
                    a["tradable"])
    assert why and frag in why


def test_refresh_keeps_three_vetted_candidates_and_never_offers_excluded_ones():
    uni = {"ind_a": ["A1", "A2", "A3"], "ind_b": ["B1", "B2", "B3"]}
    member = {s: _bars(300, 60 + k, 0.4, seed=k) for k, s in enumerate(sum(uni.values(), []))}
    cands = {"C1": (60, 0.40), "C2": (62, 0.42), "C3": (70, 0.45), "C4": (58, 0.38),
             "C1B": (60.5, 0.40), "C5": (95, 0.4), "OLD": (60, 0.4), "BAD": (60, 0.4),
             "A2": (60, 0.4), "SIB": (60, 0.4)}
    names = {s: f"{s} Inc" for s in cands} | {"C1B": "C1 Inc", "SIB": "A1 Inc"}
    wl = {"completed": {"OLD": {}}, "rejected": {"BAD": "accounting concern"},
          "replacements": {"ind_a": [{"symbol": "C3"}]}}
    profile = {s: ("Semis", f"{s} Inc") for s in sum(uni.values(), [])} | {"A3": ("Banks", "A3 Inc")}
    out, report = rp.refresh(
        uni, wl, profile=lambda s: profile[s],
        screen=lambda y: [(s, "NMS", names[s]) for s in cands] if y == "Semis" else [("ZZ", "NMS", "Z")],
        history=lambda s: _bars(1300, *cands[s]),
        tradable=lambda s: True, bars_fn=lambda s: member[s], today="2026-10-14")
    a = [r["symbol"] for r in out["ind_a"]]
    b = [r["symbol"] for r in out["ind_b"]]
    assert len(a) == 3 and not {"C5", "OLD", "BAD", "A2", "SIB"} & set(a)
    assert not ({"C1", "C1B"} <= set(a + b))            # one share class per company
    assert "ZZ" not in a                                 # one member's Yahoo industry is not screened
    assert not set(a) & set(b)                           # each symbol offered to one industry
    assert b and "CONCERN" in report[1]                 # ind_b is left with fewer than 3
    assert [r["score"] for r in out["ind_a"]] == sorted(r["score"] for r in out["ind_a"])


def test_swaps_are_for_stopped_data_or_drift_only():
    uni = {"a": ["A1", "A2", "A3"], "b": ["B1", "B2", "B3"], "c": ["C1", "C2", "C3"]}
    dates = {s: "2026-10-16" for s in sum(uni.values(), [])} | {"A2": "2026-10-15"}
    closes = {"A1": 60, "A2": 60, "A3": 60, "B1": 40, "B2": 45, "B3": 140,
              "C1": 60, "C2": 61, "C3": 59}
    got = wp.swaps_needed(uni, dates, closes, TH)
    assert [(i, s) for i, s, _ in got] == [("a", "A2"), ("b", "B3")]
    assert "data stopped" in got[0][2] and "drift" in got[1][2]


def test_replacement_is_the_first_that_rechecks_and_keeps_the_band():
    uni = {"a": ["A1", "A2", "A3", "A4"]}
    closes = {"A1": 35, "A2": 36, "A3": 37, "A4": 200}
    ready = [{"symbol": "R1", "exchange": "NMS"}, {"symbol": "R2", "exchange": "NMS"},
             {"symbol": "R3", "exchange": "NMS"}]
    prices = {"R1": (61, "not an active, tradable asset on Alpaca"), "R2": (89, None),
              "R3": (57, None)}
    c, notes = wp.choose_replacement("a", "A4", uni, closes, ready, lambda r: prices[r["symbol"]])
    assert c["symbol"] == "R3" and c["price"] == 57
    assert "R1 skipped" in notes[0] and "R2 skipped" in notes[1]


def test_clear_book_slot_zeroes_only_that_industry_and_slot(tmp_path):
    p = tmp_path / "book.txt"
    p.write_text("upkeep_book 1\ndate 2026-10-16\n"
                 "ind a base 100.0 cash 5.0 hold 1 2 3 stop 9 8 7\n"
                 "ind b base 100.0 cash 5.0 hold 1 2 3 stop 9 8 7\n")
    assert wp.clear_book_slot(str(p), "a", 1)
    a, b = p.read_text().splitlines()[2:]
    assert a.split()[7:10] == ["1", "0.000000", "3"] and a.split()[11:14] == ["9", "0.000000", "7"]
    assert b.endswith("hold 1 2 3 stop 9 8 7")


def test_promotion_clears_exactly_one_industrys_state(tmp_path):
    names = ["consumer_services_elite_0.bin", "consumer_services_hist.bin",
             "race_TANH32___grad_consumer_services.bin",
             "race_TANH32___grad_consumer_services.adam.bin",
             "r_mt1_consumer_services_meta.bin", "r_mt1_consumer_services_slot_7.bin",
             "consumer_staples_elite_0.bin", "race_TANH32___grad_consumer_staples.bin",
             "r_mt1_consumer_staples_slot_7.bin", "mt2_elite_0.bin", "upkeep_state.txt"]
    for n in names:
        (tmp_path / n).write_text("x")
    got = {p.split("/")[-1] for p in wp.industry_state_files(str(tmp_path), "consumer_services")}
    assert got == set(names[:6])


def test_state_keys_dropped_for_promoted_industries_only(tmp_path):
    p = tmp_path / "upkeep_state.txt"
    p.write_text("upkeep_state 1\ndate 2026-10-15\nmkt energy 1 2\nbayes energy 1 2 3\n"
                 "lr TANH32___grad energy 0.001\nlr TANH32___grad utilities 0.002\n"
                 "tanh32_step TANH32___grad energy 7\nadam_step TANH32___grad 9\n")
    assert wp.drop_state_keys(str(p), {"energy"}) == 2
    t = p.read_text()
    assert "lr TANH32___grad energy" not in t and "tanh32_step" not in t
    assert "lr TANH32___grad utilities" in t and "bayes energy" in t and "adam_step" in t


def test_crowned_reads_the_pass_rows(tmp_path):
    p = tmp_path / "pass_reference.csv"
    p.write_text("# pass_reference v2\npass,industry,day_start,day_end,champ_pass,champ_pct,"
                 "chal_pass,chal_pct,regime,share_new,slots_new,new_champ_pass,new_champ_pct\n"
                 "11,energy,18,1255,4,+0.001,11,+0.002,x,0.6,12,11,+0.00200000\n"
                 "11,utilities,18,1255,9,+0.003,11,+0.002,x,0.4,8,9,+0.00300000\n"
                 "11,materials,18,1255,11,+0.001,0,+0.0,crown,1.0,20,11,+0.00100000\n")
    got = wp.crowned(str(p), 11)
    assert set(got) == {"energy", "materials"} and got["materials"][0] == "crown"
    assert got["energy"][1] == pytest.approx(20.0)


def test_crowned_over_three_passes_reads_the_final_champion(tmp_path):
    # pass 11 crowns both; 12 beats energy's p11; materials keeps p11 through 13
    p = tmp_path / "pass_reference.csv"
    p.write_text("# pass_reference v2\npass,industry,day_start,day_end,champ_pass,champ_pct,"
                 "chal_pass,chal_pct,regime,share_new,slots_new,new_champ_pass,new_champ_pct\n"
                 "10,energy,18,1255,4,+0.001,10,+0.0005,x,0.4,8,4,+0.00100000\n"
                 "11,energy,18,1255,11,+0.001,0,+0.0,crown,1.0,20,11,+0.00100000\n"
                 "11,materials,18,1255,11,+0.002,0,+0.0,crown,1.0,20,11,+0.00200000\n"
                 "12,energy,18,1255,11,+0.001,12,+0.003,both+,0.7,14,12,+0.00300000\n"
                 "12,materials,18,1255,11,+0.002,12,+0.001,both+,0.4,8,11,+0.00200000\n"
                 "13,energy,18,1255,12,+0.003,13,+0.002,both+,0.4,8,12,+0.00300000\n"
                 "13,materials,18,1255,11,+0.002,13,+0.001,both+,0.4,8,11,+0.00200000\n")
    got = wp.crowned(str(p), 11, 13)
    assert got["energy"][3] == 12 and got["energy"][1] == pytest.approx(30.0)
    assert got["materials"][3] == 11 and got["materials"][0] == "crown"
    assert wp.last_pass(str(p)) == 13 and wp.last_pass(str(tmp_path / "none.csv")) == 0


def test_three_passes_only_when_every_industry_is_replaced():
    inds = {"energy": [], "materials": [], "utilities": []}
    assert wp.passes_for(inds, []) == 1
    assert wp.passes_for(inds, ["energy"]) == 1                  # a swap: one pass
    assert wp.passes_for(inds, ["energy", "materials"]) == 1
    assert wp.passes_for(inds, sorted(inds)) == 3                # new order rules: all of them


def test_screener_names_take_the_em_dash():
    vocab = {"Software\u2014Application", "Semiconductors"}
    assert rp.screener_name("Software - Application", vocab) == "Software\u2014Application"
    assert rp.screener_name("Semiconductors", vocab) == "Semiconductors"
    assert rp.screener_name("Shell Companies", vocab) is None


def test_new_order_rules_crown_every_industry(tmp_path):
    inds = ["energy", "materials", "utilities"]
    # never recorded: the run predates the rules, so everything is crowned
    crown, note = wp.rules_crown(tmp_path, inds, [], 2)
    assert crown == sorted(inds) and any("v1" in n for n in note)
    # recorded at the current version: only the swaps are crowned
    wp.record_order_rules(tmp_path, 2)
    assert wp.rules_crown(tmp_path, inds, ["energy"], 2) == (["energy"], [])
    # the rules move on again
    crown, _ = wp.rules_crown(tmp_path, inds, ["energy"], 3)
    assert crown == sorted(inds)
