"""Dead-ticker detection in download_daily.

A delisted symbol does not raise — yfinance returns an empty frame, the merge is a no-op, and the
symbol reports `+ 0 day(s)` exactly like a healthy symbol on a weekend. That is how TMHC sat frozen
at 2026-07-23 for seven weeks while reporting a full 1255 days every run. These pin the
cohort-relative check that separates the two cases.
"""
import download_daily as dd


class TestFindStaleSymbols:
    def test_healthy_cohort_has_no_stale(self):
        last = {s: '2026-09-11' for s in ('DHI', 'LEN', 'PHM')}
        cohort, stale = dd.find_stale_symbols(last)
        assert cohort == '2026-09-11'
        assert stale == {}

    def test_weekend_lag_is_not_stale(self):
        # Every symbol lags *today* by several days, but none lags the cohort. A calendar-based
        # check would fire here every Saturday; the cohort-relative one must not.
        last = {s: '2026-09-11' for s in ('DHI', 'LEN', 'PHM', 'TOL')}
        _, stale = dd.find_stale_symbols(last, max_stale_days=0)
        assert stale == {}

    def test_frozen_ticker_is_flagged(self):
        # The real TMHC case: 143 symbols current, one frozen at the delisting date.
        last = {s: '2026-09-11' for s in ('DHI', 'LEN', 'PHM')}
        last['TMHC'] = '2026-07-23'
        cohort, stale = dd.find_stale_symbols(last)
        assert cohort == '2026-09-11'
        assert list(stale) == ['TMHC']
        assert stale['TMHC'] == ('2026-07-23', 50)

    def test_threshold_is_exclusive(self):
        base = {'A': '2026-09-11', 'B': '2026-09-11'}
        at_limit = dict(base, C='2026-09-04')       # exactly 7 days behind
        _, stale = dd.find_stale_symbols(at_limit, max_stale_days=7)
        assert stale == {}, 'a symbol exactly at the limit must not be called dead'

        over = dict(base, C='2026-09-03')           # 8 days behind
        _, stale = dd.find_stale_symbols(over, max_stale_days=7)
        assert list(stale) == ['C']
        assert stale['C'][1] == 8

    def test_ordered_most_stale_first(self):
        last = {'OK': '2026-09-11', 'MILD': '2026-08-11', 'DEAD': '2026-02-11'}
        _, stale = dd.find_stale_symbols(last)
        assert list(stale) == ['DEAD', 'MILD']

    def test_empty_input(self):
        assert dd.find_stale_symbols({}) == (None, {})

    def test_single_symbol_is_its_own_cohort(self):
        # Nothing to compare against — must not flag itself, however old it is.
        cohort, stale = dd.find_stale_symbols({'ONLY': '2020-01-02'})
        assert cohort == '2020-01-02'
        assert stale == {}

    def test_default_threshold_is_a_week(self):
        assert dd.MAX_STALE_DAYS == 7
