"""Pin the MT binary-log record layout across the C++ writer and the two Python readers.

`training_v4.cpp` writes the record, `read_mt_log.py` and `plot_training.py` each parse it with
their own hand-maintained struct format. Nothing linked them, so a field added on one side and
mistyped on another would produce plausible-looking numbers off by a few floats rather than an
error — the same class of silent drift that let `tests/test_mt1.cpp` carry a stale replica of
`compute_mt1_scores` while reporting green.

These assert the three definitions agree on record size, and that the reader's version table is
self-consistent.
"""
import re
import struct
from pathlib import Path

import plot_training
import read_mt_log

REPO = Path(__file__).resolve().parent.parent


def _cpp_record_size():
    src = (REPO / 'training_v4.cpp').read_text()
    m = re.search(r'static_assert\(sizeof\(MTLogRecord\) == (\d+)', src)
    assert m, 'MTLogRecord static_assert not found in training_v4.cpp'
    return int(m.group(1))


def _cpp_log_version():
    src = (REPO / 'training_v4.cpp').read_text()
    m = re.search(r'MT_LOG_VERSION\s*=\s*(\d+)u', src)
    assert m, 'MT_LOG_VERSION not found in training_v4.cpp'
    return int(m.group(1))


def test_cpp_and_readers_agree_on_current_record_size():
    size = _cpp_record_size()
    assert size == read_mt_log.RECORD_SIZE_V11
    assert size == plot_training.RECORD_SIZE_V11


def test_reader_formats_match_their_declared_sizes():
    for ver, (size, fmt) in read_mt_log._LAYOUTS.items():
        assert struct.calcsize(fmt) == size, f'read_mt_log V{ver} format/size mismatch'


def test_plot_training_struct_matches_read_mt_log():
    """Both readers must unpack the current record identically."""
    assert plot_training._RECORD_STRUCT_V11.size == read_mt_log.RECORD_SIZE_V11


def test_header_version_maps_to_current_layout():
    """MT_LOG_VERSION n writes record layout V(n+1) — the mapping read_mt_log relies on to prefer
    the header stamp over size-sniffing."""
    layout = _cpp_log_version() + 1
    assert layout in read_mt_log._LAYOUTS
    assert read_mt_log._LAYOUTS[layout][0] == _cpp_record_size()


def test_v10_field_offsets_are_inside_the_record():
    """mt1_oos_act starts at float index 334 and mt1_skill at 382 (12x4 each); both must land
    within the record, or the reader is silently reading past the end of the struct."""
    n = len(struct.unpack(read_mt_log.RECORD_FMT_V10,
                          b'\x00' * read_mt_log.RECORD_SIZE_V10))
    assert n >= 430, f'V10 unpack yields {n} values, need at least 430'


def test_v11_field_offsets_are_inside_the_record():
    """mt1_dir_stats starts at float index 430 (12x6) and mt1_dir_life at 502 (12x5)."""
    n = len(struct.unpack(read_mt_log.RECORD_FMT_V11,
                          b'\x00' * read_mt_log.RECORD_SIZE_V11))
    assert n >= 562, f'V11 unpack yields {n} values, need at least 562'
