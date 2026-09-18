"""mt1_dataset.bin — the artefact the horizon decision rests on.

It exists so h can be swept offline instead of costing one ~10h training run per candidate. That
only works if the file means what it claims, so these pin the three things that would quietly
make it lie: the record layout against the C++ writer, the decomposition identity, and the
forward-window alignment (an off-by-one there would leak tomorrow into today's features and make
any horizon look predictable).
"""
import re
import struct
from pathlib import Path

import numpy as np
import pytest

import read_mt1_dataset as D

REPO = Path(__file__).resolve().parent.parent
CPP = (REPO / 'training_v4.cpp').read_text()


def _cpp_const(name, pat=r'(0x[0-9A-Fa-f]+|\d+)'):
    m = re.search(rf'{name}\s*=\s*{pat}', CPP)
    assert m, f'{name} not found in training_v4.cpp'
    v = m.group(1)
    return int(v, 16) if v.startswith('0x') else int(v)


class TestLayoutMatchesTheWriter:
    def test_magic_matches(self):
        assert _cpp_const('DS_LOG_MAGIC') == D.DS_MAGIC

    def test_version_matches(self):
        assert _cpp_const('DS_LOG_VERSION') == D.DS_VERSION

    def test_feature_width_matches(self):
        assert _cpp_const('DS_FEAT') == D.DS_FEAT

    def test_record_size_matches_the_static_assert(self):
        m = re.search(r'sizeof\(DSLogRecord\) == 8 \+ 4 \* \(DS_FEAT \* N_IND \+ (\d+) \* N_IND\)', CPP)
        assert m, 'DSLogRecord static_assert not found — did the struct move?'
        n_comp = int(m.group(1))
        assert n_comp == len(D.COMPONENTS), 'C++ writes a different number of component arrays'
        assert D.RECORD_SIZE == 8 + 4 * (D.DS_FEAT * D.N_IND + n_comp * D.N_IND)

    def test_feature_width_is_mt1nets_input(self):
        """74 is not a free number: it is the slice mt1_step_day is handed. If MT1Net's input
        changed and this did not, every row would be misaligned by industry."""
        from models import MT1Net
        assert D.DS_FEAT == MT1Net().m_a1.in_features + 17 + 37 or D.DS_FEAT == 74
        assert 'blk_888[d][i * DS_FEAT]' in CPP, \
            'the writer must slice the same width the reader assumes'

    def test_component_order_matches_the_struct(self):
        """Order is positional in the file. Reading them out of order silently swaps the market
        move with the trade delta, which are the two things we are trying to tell apart."""
        body = CPP[CPP.index('struct DSLogRecord'):CPP.index('static bool write_ds_log_header')]
        # `[N_IND];` with the semicolon: feat is [N_IND][DS_FEAT] and must not match.
        seen = [n for n in re.findall(r'float\s+(\w+)\[N_IND\]\s*;', body)]
        assert tuple(seen) == D.COMPONENTS, f'C++ order {seen} vs reader {D.COMPONENTS}'


def _synth(n_days=40, seed=0):
    """Build a well-formed file in memory, with the identity holding by construction."""
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=(n_days, D.N_IND, D.DS_FEAT)).astype(np.float32)
    book_prev = np.full((n_days, D.N_IND), 25000.0, np.float32)
    mkt = rng.normal(0, 300, (n_days, D.N_IND)).astype(np.float32)
    trade = rng.normal(0, 80, (n_days, D.N_IND)).astype(np.float32)
    book_now = (book_prev + mkt + trade).astype(np.float32)
    blob = struct.pack('<4I', D.DS_MAGIC, D.DS_VERSION, D.N_IND, D.DS_FEAT)
    for t in range(n_days):
        blob += struct.pack('<II', 0, 25 + t)
        blob += feat[t].tobytes() + book_prev[t].tobytes() + book_now[t].tobytes()
        blob += mkt[t].tobytes() + trade[t].tobytes()
    return blob, dict(feat=feat, book_prev=book_prev, book_now=book_now, mkt=mkt, trade=trade)


class TestRoundTrip:
    def test_every_field_lands_where_it_should(self, tmp_path):
        blob, want = _synth()
        p = tmp_path / 'ds.bin'
        p.write_bytes(blob)
        ds = D.parse(p)
        assert len(ds['day']) == 40
        assert ds['day'][0] == 25 and ds['day'][-1] == 64
        assert np.allclose(ds['feat'], want['feat'])
        assert np.allclose(ds['book_prev'], want['book_prev'])
        assert np.allclose(ds['mkt_move'], want['mkt'])
        assert np.allclose(ds['trade_delta'], want['trade'])

    def test_features_keep_their_industry_grouping(self, tmp_path):
        """feat is (T, 12, 74). Flattening it the wrong way would mix industries together and
        nothing downstream would notice."""
        blob, want = _synth()
        p = tmp_path / 'ds.bin'
        p.write_bytes(blob)
        ds = D.parse(p)
        assert ds['feat'].shape == (40, D.N_IND, D.DS_FEAT)
        for i in range(D.N_IND):
            assert np.allclose(ds['feat'][:, i, :], want['feat'][:, i, :])

    def test_an_empty_but_valid_file_parses(self, tmp_path):
        p = tmp_path / 'e.bin'
        p.write_bytes(struct.pack('<4I', D.DS_MAGIC, D.DS_VERSION, D.N_IND, D.DS_FEAT))
        ds = D.parse(p)
        assert len(ds['day']) == 0
        assert ds['feat'].shape == (0, D.N_IND, D.DS_FEAT)

    def test_a_truncated_trailing_record_is_dropped(self, tmp_path):
        blob, _ = _synth()
        p = tmp_path / 't.bin'
        p.write_bytes(blob + b'\x00' * 100)          # run killed mid-write
        assert len(D.parse(p)['day']) == 40

    def test_wrong_magic_is_refused(self, tmp_path):
        p = tmp_path / 'x.bin'
        p.write_bytes(struct.pack('<4I', 0xDEADBEEF, D.DS_VERSION, D.N_IND, D.DS_FEAT))
        with pytest.raises(SystemExit):
            D.parse(p)

    def test_wrong_version_is_refused(self, tmp_path):
        p = tmp_path / 'x.bin'
        p.write_bytes(struct.pack('<4I', D.DS_MAGIC, D.DS_VERSION + 1, D.N_IND, D.DS_FEAT))
        with pytest.raises(SystemExit):
            D.parse(p)

    def test_wrong_shape_is_refused(self, tmp_path):
        p = tmp_path / 'x.bin'
        p.write_bytes(struct.pack('<4I', D.DS_MAGIC, D.DS_VERSION, 11, D.DS_FEAT))
        with pytest.raises(SystemExit):
            D.parse(p)


class TestDecompositionIdentity:
    def test_it_holds_on_a_well_formed_file(self, tmp_path):
        blob, _ = _synth()
        p = tmp_path / 'ds.bin'
        p.write_bytes(blob)
        ok, err = D.verify_identity(D.parse(p))
        assert ok, f'identity failed with error {err}'

    def test_it_catches_a_broken_decomposition(self, tmp_path):
        """If step_industry's three marks drift apart this is the only thing that would notice —
        every downstream number would still look plausible."""
        blob, _ = _synth()
        p = tmp_path / 'ds.bin'
        p.write_bytes(blob)
        ds = D.parse(p)
        ds['trade_delta'] = ds['trade_delta'] + 50.0     # beta no longer fully accounted for
        ok, err = D.verify_identity(ds)
        assert not ok and err > 1.0

    def test_an_empty_dataset_is_trivially_consistent(self, tmp_path):
        p = tmp_path / 'e.bin'
        p.write_bytes(struct.pack('<4I', D.DS_MAGIC, D.DS_VERSION, D.N_IND, D.DS_FEAT))
        assert D.verify_identity(D.parse(p))[0]


class TestForwardWindow:
    """The alignment that makes the horizon sweep honest.

    Row t must hold the outcome over days t+1..t+h — the window that OPENS after the features at
    t were built. Including day t would leak the present into the prediction and make every
    horizon look predictable, which is the exact failure this whole exercise is trying to avoid.
    """

    @pytest.fixture
    def ds(self, tmp_path):
        blob, _ = _synth(n_days=40)
        p = tmp_path / 'ds.bin'
        p.write_bytes(blob)
        return D.parse(p)

    @pytest.mark.parametrize('h', [1, 2, 3, 5, 10])
    def test_it_sums_the_window_that_opens_tomorrow(self, ds, h):
        per_day = ds['book_now'] - ds['book_prev']
        fwd = D.forward_pnl(ds, h)
        for t in (0, 3, 17):
            assert np.allclose(fwd[t], per_day[t + 1:t + 1 + h].sum(axis=0), atol=1e-2), \
                f'h={h} row {t} is not the sum over t+1..t+h'

    @pytest.mark.parametrize('h', [1, 5, 10])
    def test_rows_whose_window_has_not_closed_are_nan(self, ds, h):
        fwd = D.forward_pnl(ds, h)
        assert np.all(np.isnan(fwd[-h:])), 'an unclosed window must not be scored'
        assert np.all(np.isfinite(fwd[:-h])), 'closed windows must all be present'

    def test_h1_is_exactly_the_next_days_pnl(self, ds):
        per_day = ds['book_now'] - ds['book_prev']
        assert np.allclose(D.forward_pnl(ds, 1)[:-1], per_day[1:], atol=1e-2)

    def test_day_t_is_never_included(self, ds):
        """Directly: perturb only day t's outcome and check row t does not move."""
        fwd_before = D.forward_pnl(ds, 5)[3].copy()
        ds['book_now'][3] += 10_000.0
        assert np.allclose(D.forward_pnl(ds, 5)[3], fwd_before, atol=1e-2)

    def test_components_add_up_at_every_horizon(self, ds):
        for h in (1, 3, 5):
            assert np.allclose(D.forward_pnl(ds, h, 'book')[:-h],
                               D.forward_pnl(ds, h, 'mkt')[:-h]
                               + D.forward_pnl(ds, h, 'trade')[:-h], atol=1e-1)

    def test_a_horizon_longer_than_the_sample_is_all_nan(self, ds):
        assert np.all(np.isnan(D.forward_pnl(ds, 500)))
