"""Pin the MT binary-log record layout across the C++ writer and the two Python readers.

`training_v4.cpp` writes the record, `read_mt_log.py` and `plot_training.py` each parse it with
their own hand-maintained struct format. Nothing links them, so a field added on one side and
mistyped on another would produce plausible-looking numbers off by a few floats rather than an
error — the same class of silent drift that let a deleted `tests/test_mt1.cpp` carry a stale
replica of the scoring function while reporting green.

These assert the three definitions agree on record size and version, that the layout matches the
field list the C++ struct declares, and that both readers refuse a log they cannot decode.
"""
import re
import struct
from pathlib import Path

import plot_training
import pytest
import read_mt_log

REPO = Path(__file__).resolve().parent.parent
CPP = (REPO / 'training_v4.cpp').read_text()


def _cpp_record_size():
    m = re.search(r'static_assert\(sizeof\(MTLogRecord\) == (\d+)', CPP)
    assert m, 'MTLogRecord static_assert not found in training_v4.cpp'
    return int(m.group(1))


def _cpp_log_version():
    m = re.search(r'MT_LOG_VERSION\s*=\s*(\d+)u', CPP)
    assert m, 'MT_LOG_VERSION not found in training_v4.cpp'
    return int(m.group(1))


def test_cpp_and_readers_agree_on_record_size():
    size = _cpp_record_size()
    assert size == read_mt_log.RECORD_SIZE
    assert size == plot_training.RECORD_SIZE_V12


def test_reader_formats_match_their_declared_sizes():
    assert struct.calcsize(read_mt_log.RECORD_FMT) == read_mt_log.RECORD_SIZE
    assert struct.calcsize(plot_training.RECORD_FMT_V12) == plot_training.RECORD_SIZE_V12


def test_the_two_readers_use_the_same_format_string():
    """Byte-identical, not merely same-sized: two formats can agree on total width while
    disagreeing on where the uint8 sits, which silently shifts every float after it."""
    assert read_mt_log.RECORD_FMT == plot_training.RECORD_FMT_V12


def test_the_two_readers_use_the_same_magic():
    """They disagreed once — read_mt_log looked for "MTLG" while the trainer writes "MT12", so it
    refused every real log. A shared format string is not enough; the header must match too."""
    assert read_mt_log.MT_LOG_MAGIC == plot_training.MT_LOG_MAGIC


def test_magic_matches_the_cpp_writer():
    m = re.search(r'MT_LOG_MAGIC\s*=\s*(0x[0-9A-Fa-f]+)u', CPP)
    assert m, 'MT_LOG_MAGIC not found in training_v4.cpp'
    assert int(m.group(1), 16) == read_mt_log.MT_LOG_MAGIC


def test_cpp_version_matches_both_readers():
    v = _cpp_log_version()
    assert v == read_mt_log.MT_LOG_VERSION
    assert f'version != {v}' in (REPO / 'plot_training.py').read_text(), \
        'plot_training must gate on the same version the trainer writes'


def test_record_size_matches_the_declared_field_list():
    """Derive the size from the struct's own fields, so adding one without updating the readers
    fails here rather than shifting every column to the right of it."""
    n_ind = read_mt_log.N_IND
    expected = (
        4 * 2                       # pass_num, actual_day
        + 4 * len(read_mt_log.MT1_FIELDS) * n_ind
        + 4 * 5 * n_ind             # mt1_pool_stats[12][5]
        + 4 * 5 * n_ind             # mt1_life[12][5]
        + 4 * 3                     # mt2 best/slot0/ideal
        + 1 + 3                     # mt2_injected + pad
        + 4 * 4                     # consensus flat/wtd + slot0 pf/mkt
    )
    assert expected == read_mt_log.RECORD_SIZE


def test_mt1_field_names_appear_in_the_cpp_struct():
    for name in read_mt_log.MT1_FIELDS:
        cpp_name = {'actual': 'mt1_actual_d', 'floor': 'mt1_floor', 'pred': 'mt1_pred',
                    'baseline': 'mt1_baseline'}.get(name, f'mt1_{name}')
        assert cpp_name in CPP, f'{cpp_name} is parsed by read_mt_log but absent from MTLogRecord'


def test_readers_refuse_an_older_log(tmp_path):
    """A V11 log must be rejected, not decoded. Its records are a different architecture, and the
    sizes are close enough that size-sniffing would happily read one as the other."""
    p = tmp_path / 'old.bin'
    p.write_bytes(struct.pack('<4I', read_mt_log.MT_LOG_MAGIC, 11, 12, 0) + b'\x00' * 2212)
    with pytest.raises(SystemExit):
        read_mt_log.parse_log(p)
    with pytest.raises(SystemExit):
        plot_training.load_binary_log(p)


def test_readers_refuse_a_foreign_file(tmp_path):
    p = tmp_path / 'junk.bin'
    p.write_bytes(b'not a log at all, really' + b'\x00' * 900)
    with pytest.raises(SystemExit):
        read_mt_log.parse_log(p)


def test_a_valid_empty_log_is_parsed_not_rejected(tmp_path):
    p = tmp_path / 'empty.bin'
    p.write_bytes(struct.pack('<4I', read_mt_log.MT_LOG_MAGIC, read_mt_log.MT_LOG_VERSION, 12, 0))
    industries, records = read_mt_log.parse_log(p)
    assert records == []
    assert len(industries) == 12


def test_a_single_record_round_trips(tmp_path):
    """Pack a record with a distinct value per field and read it back in the right places."""
    n = read_mt_log.N_IND
    vals = [3, 41]
    for f_i, _name in enumerate(read_mt_log.MT1_FIELDS):
        vals += [float(f_i * 100 + i) for i in range(n)]
    vals += [float(1000 + i) for i in range(5 * n)]      # pool_stats
    vals += [float(2000 + i) for i in range(5 * n)]      # life
    vals += [1.0, 2.0, 3.0, 1]                           # mt2 pts + injected
    vals += [4.0, 5.0, 6.0, 7.0]
    p = tmp_path / 'one.bin'
    p.write_bytes(struct.pack('<4I', read_mt_log.MT_LOG_MAGIC, read_mt_log.MT_LOG_VERSION, n, 0)
                  + struct.pack(read_mt_log.RECORD_FMT, *vals))
    _industries, records = read_mt_log.parse_log(p)
    assert len(records) == 1
    r = records[0]
    assert r['pass'] == 3 and r['day'] == 41
    for f_i, name in enumerate(read_mt_log.MT1_FIELDS):
        assert r['mt1'][name] == [f_i * 100 + i for i in range(n)], f'{name} landed wrong'
    assert r['mt1']['pool_stats'][0] == [1000, 1001, 1002, 1003, 1004]
    assert r['mt1']['pool_stats'][1] == [1005, 1006, 1007, 1008, 1009]
    assert r['mt1']['life'][11] == [2055, 2056, 2057, 2058, 2059]
    assert r['mt2']['ideal_pts'] == 3.0 and r['mt2']['injected'] == 1
    assert r['mt2']['slot0_pts_mkt'] == 7.0


def test_both_readers_decode_the_same_record_identically(tmp_path):
    n = read_mt_log.N_IND
    vals = [0, 30] + [float(i) for i in range(8 * n + 5 * n + 5 * n)] + [1.0, 2.0, 3.0, 0] \
        + [4.0, 5.0, 6.0, 7.0]
    p = tmp_path / 'both.bin'
    p.write_bytes(struct.pack('<4I', read_mt_log.MT_LOG_MAGIC, read_mt_log.MT_LOG_VERSION, n, 0)
                  + struct.pack(read_mt_log.RECORD_FMT, *vals))
    _i, a = read_mt_log.parse_log(p)
    b = plot_training.load_binary_log(p)
    assert a[0]['day'] == b[0]['day']
    assert a[0]['pass'] + 1 == b[0]['pass']      # plot_training normalises to 1-indexed
    for name in read_mt_log.MT1_FIELDS:
        assert a[0]['mt1'][name] == b[0][f'mt1_{name}'], f'{name} differs between readers'
    assert a[0]['mt1']['pool_stats'] == b[0]['mt1_pool_stats']
    assert a[0]['mt2']['slot0_pts_pf'] == b[0]['mt2_slot0_pts_pf']
