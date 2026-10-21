"""production_v2 trades the universe, not the copy of it in state.json.

state.json's 'industries' is written on the first run. Reading it back froze the universe there:
the 2026-10-18 swap ZG -> JOYY reached the trainer but not production, so consumer_services fed
ZG's bars where the trainer fed JOYY's (7 of 107 intents mismatched, all in that industry), and
the new-symbol hold never fired because the new symbol never appeared.
"""
import production_v2
from universe import INDUSTRIES


def stale_state():
    inds = {ind: list(syms) for ind, syms in INDUSTRIES.items()}
    ind = next(iter(inds))
    inds[ind][0] = 'GONE'
    return {'industries': inds, 'known_symbols': [s for v in inds.values() for s in v]}


def test_universe_wins_over_state():
    state = stale_state()
    assert production_v2.current_industries(state) == INDUSTRIES


def test_state_copy_is_refreshed():
    state = stale_state()
    production_v2.current_industries(state)
    assert 'GONE' not in [s for v in state['industries'].values() for s in v]


def test_swapped_in_symbol_is_new_against_known():
    # The new-symbol hold compares the trading universe with state['known_symbols']; with the
    # stale universe the swapped-in symbol never showed up as new.
    state = stale_state()
    inds = production_v2.current_industries(state)
    new = {s for v in inds.values() for s in v} - set(state['known_symbols'])
    assert new == {next(iter(INDUSTRIES.values()))[0]}
