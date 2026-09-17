"""Non-finite OHLC bars must never reach a calculation.

yfinance returns NaN for a session it has no data for, and `json.dump` writes that as a bare
`NaN` literal. The C++ trainer's `atof()` parses it happily and the bar was marked valid
unconditionally, so NaN entered the portfolio valuation. `(int)NaN` is undefined behaviour and
lands on INT32_MIN, which is how `prod=$-2147483648` appeared for SCCO 2026-08-11.

It was not cosmetic. Any statistic over that industry-pass was destroyed -- a quarter return came
out as -4,047,374% -- and it would have been reported as a real number had it been less absurd.

Four layers now: download_daily drops bad bars on fetch, purges ones already on disk, the C++
loader rejects them (and reports a count), and production's loader skips them.
"""
import json
import math

import pytest

BAD = {'date': '2026-08-11', 'open': float('nan'), 'high': float('nan'),
       'low': float('nan'), 'close': float('nan'), 'volume': 0}
GOOD = {'date': '2026-08-12', 'open': 10.0, 'high': 11.0,
        'low': 9.5, 'close': 10.5, 'volume': 1000}


def usable(d):
    return all(isinstance(d.get(k), (int, float)) and math.isfinite(d[k]) and d[k] > 0
               for k in ('open', 'high', 'low', 'close'))


class TestTheBarItself:
    def test_nan_bar_is_not_usable(self):
        assert not usable(BAD)

    def test_good_bar_is_usable(self):
        assert usable(GOOD)

    @pytest.mark.parametrize('field', ['open', 'high', 'low', 'close'])
    def test_any_non_finite_field_rejects_the_bar(self, field):
        for bad_value in (float('nan'), float('inf'), float('-inf'), 0.0, -1.0):
            d = dict(GOOD, **{field: bad_value})
            assert not usable(d), f'{field}={bad_value} should reject'

    def test_nan_survives_a_json_round_trip(self):
        """This is why it reached the trainer: json.dump writes NaN and json.load reads it back."""
        s = json.dumps({'days': [BAD]})
        assert 'NaN' in s
        back = json.loads(s)['days'][0]
        assert math.isnan(back['open'])
        assert not usable(back)


class TestDownloadDailyPurgesOnLoad:
    """_merge keeps an existing bar when the fetch no longer returns it, so dropping bad bars at
    fetch time alone would let one already on disk survive every future run."""

    def test_loader_drops_bad_bars(self, tmp_path, monkeypatch):
        import download_daily
        monkeypatch.setattr(download_daily, 'STOCK_DATA_DIR', str(tmp_path))
        (tmp_path / 'SCCO.json').write_text(json.dumps({'days': [BAD, GOOD]}))
        days = download_daily._load_existing('SCCO')
        assert len(days) == 1 and days[0]['date'] == GOOD['date']

    def test_clean_file_is_untouched(self, tmp_path, monkeypatch):
        import download_daily
        monkeypatch.setattr(download_daily, 'STOCK_DATA_DIR', str(tmp_path))
        (tmp_path / 'AAA.json').write_text(json.dumps({'days': [GOOD]}))
        assert len(download_daily._load_existing('AAA')) == 1

    def test_merge_would_otherwise_preserve_the_bad_bar(self):
        """Pin the mechanism: _merge keeps existing entries the fetch does not replace."""
        import download_daily
        merged = download_daily._merge([BAD, GOOD], [GOOD])   # fetch dropped the bad one
        assert any(not usable(d) for d in merged), \
            'merge preserves unreplaced bars — hence the purge in _load_existing'


class TestProductionLoaderSkipsThem:
    def test_load_stock_data_drops_bad_bars(self, tmp_path, monkeypatch):
        import production_v2
        monkeypatch.setattr(production_v2, 'STOCK_DATA_DIR', str(tmp_path))
        (tmp_path / 'SCCO.json').write_text(json.dumps({'days': [BAD, GOOD]}))
        hist = production_v2.load_stock_data(['SCCO'])
        assert len(hist['SCCO']) == 1
        assert all(math.isfinite(v) for row in hist['SCCO'] for v in row)


class TestTrainerSourceGuards:
    """The C++ twin cannot be imported; guard it at the source."""

    def test_bar_validity_is_checked_not_assumed(self):
        src = open('training_v4.cpp').read()
        assert 'o.valid  = true;' not in src, 'bars are marked valid unconditionally again'
        assert 'std::isfinite(o.open)' in src and 'o.high >= o.low' in src

    def test_int_casts_of_floats_are_nan_safe(self):
        src = open('training_v4.cpp').read()
        assert 'static inline int safe_int(float v)' in src
        for expr in ('(int)baseline', '(int)best_delta', '(int)worst_delta'):
            assert expr not in src, f'{expr} can produce INT32_MIN on NaN'

    def test_rejected_bars_are_reported(self):
        """Silence is what made this expensive — a count must reach the log."""
        src = open('training_v4.cpp').read()
        assert 'g_bad_bars' in src and 'rejected " + std::to_string(g_bad_bars)' in src
