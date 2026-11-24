"""rescore_champions.py: the frozen-score parse, the re-score rows the trainer reads back, and the
counterfactual replay that counts crowns lost to a stale champion score."""
import rescore_champions as rc

HEADER = ("# pass_reference v3 -- test\n"
          "pass,industry,day_start,day_end,champ_pass,champ_pct,chal_pass,chal_pct,"
          "regime,share_new,slots_new,new_champ_pass,new_champ_pct\n")


def _ref(tmp_path, lines):
    p = tmp_path / "pass_reference.csv"
    p.write_text(HEADER + "".join(line + "\n" for line in lines))
    return str(p)


def _row(p, ind, cp, cpct, chal, npass, npct, regime="one-"):
    return f"{p},{ind},1,1257,{cp},{cpct:+.8f},{p},{chal:+.8f},{regime},0.0000,0,{npass},{npct:+.8f}"


def test_parse_frozen_reads_every_field():
    text = ("x\nFROZEN SCORE energy -3.2500 bp/day days 0-1257 resets 9 recent 4 topup 1 eff_days 390\n"
            "FROZEN SCORE utilities +1.0000 bp/day days 0-1257 resets 0 recent 0 topup 0 eff_days 390\n")
    out = rc.parse_frozen(text)
    assert out == {"energy": (-3.25, 0, 1257, 9, 4), "utilities": (1.0, 0, 1257, 0, 0)}


def test_rescore_rows_keep_the_champion_and_replace_its_score(tmp_path):
    rows = rc.read_rows(_ref(tmp_path, [
        _row(1, "energy", 0, 0.0009, -0.002, 0, 0.0009),
        _row(2, "energy", 0, 0.0009, 0.001, 2, 0.001, "both+"),
    ]))
    out = rc.rescore_rows(rows, {"energy": (-5.0, 0, 1257, 3, 1)}, fill_model=3)
    assert len(out) == 1
    t = out[0].split(",")
    assert len(t) == 13                     # the trainer's read_pass_ref needs all 13 fields
    assert t[0] == "2" and t[4] == "2" and t[11] == "2"    # same pass number, champion p2 stays
    assert float(t[5]) == 0.001             # the old score, for the record
    assert float(t[12]) == -0.0005          # the new score is what the next boundary reads
    assert t[8] == "rescore-f3"


def test_replay_counts_a_crown_lost_to_the_stale_score(tmp_path):
    rows = rc.read_rows(_ref(tmp_path, [
        _row(1, "energy", 0, 0.0010, -0.0004, 0, 0.0010),   # loses to +10 old, beats -6 re-scored
        _row(2, "energy", 0, 0.0010, -0.0005, 0, 0.0010),   # loses to p1's -4 either way
        _row(3, "energy", 0, 0.0010, 0.0012, 3, 0.0012),    # crowned actually and counterfactually
    ]))
    res = rc.replay(rows, {"energy": -0.0006}, from_pass=1)["energy"]
    assert [(p, actual, would) for p, _, _, actual, would in res] == \
        [(1, False, True), (2, False, False), (3, True, True)]
    assert res[1][2] == -0.0004             # p2 faced the counterfactual champion p1, not p0


def test_replay_ignores_rescore_rows_and_earlier_passes(tmp_path):
    rows = rc.read_rows(_ref(tmp_path, [
        _row(1, "energy", 0, 0.0010, 0.0020, 1, 0.0020),
        _row(2, "energy", 1, 0.0020, 0.0001, 1, 0.0020),
        _row(2, "energy", 1, 0.0020, -0.0003, 1, -0.0003, "rescore-f3"),
    ]))
    res = rc.replay(rows, {"energy": 0.0}, from_pass=2)["energy"]
    assert [p for p, *_ in res] == [2]
    assert res[0][4] is True                # 0.0001 beats the re-scored 0.0
