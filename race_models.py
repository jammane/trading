"""Race entries in production: load a champion race model and run its forward pass.

The trainer saves every race entry's model as a raw float32 array, `race_<slug>_<industry>.bin`
in `champion/`, in the layout of its C++ forward (`mt1_pool.h`, `mt1_grad.h`, `mt1_backprop.h`).
This module mirrors those forwards in numpy, so a committee member predicts in paper exactly as it
did in the race. `tests/test_race_models.py` compiles the C++ forwards and holds every architecture
to them.

Each prediction is in the entry's own units, as in `race_daily.csv`. Only their order across the
12 industries matters to a vote, and the race ranked the raw outputs, so they are not rescaled.

  MT1      MT1Net, 74 trailing features -> 1 (ReLU; raw logit)
  MT1C     MT1CNet, cfeat146 -> 64 -> 24 -> 8 -> 1 (leaky ReLU; raw)
  MT1S     one 14 -> 16 -> 8 -> 1 encoder applied to each of the 12 symbols, summed (raw)
  MT1P     MT1S-stk: one 18 -> 16 -> 8 -> 1 net per stock, summed. The evo entry sums
           tanh(out) x MT1_PRED_SCALE per stock on ratio inputs; the grad entry sums raw outputs
           on raw share counts -- the units each was trained in.
  TANH32   MT1Backprop, cfeat146 -> 32 -> 8 -> 1 (tanh hidden; linear output)
"""
import os

import numpy as np

LEAK = np.float32(0.1)                 # MT1_LEAK
PRED_SCALE = np.float32(10000.0)       # MT1_PRED_SCALE
N_SYMS = 12
PER_SYM = 12                           # MT1C_PER_SYM
C_IN = N_SYMS * PER_SYM + 2            # 146
CN_CASH, CN_BOOK = 144, 145
S_IN = PER_SYM + 2                     # MT1S_PER_SYM, 14
P_IN = PER_SYM + 2 + 4                 # MT1P_IN, 18

# entry name (as in race_daily.csv) -> architecture. The ALLOC bayes lines are not networks.
ARCH = {
    "MT1 evo": "MT1", "MT1 grad": "MT1", "MT1 grdC": "MT1",
    "MT1C evo": "MT1C", "MT1C grad": "MT1C", "MT1C gdmn": "MT1C", "MT1C grnk": "MT1C",
    "MT1S-sum evo": "MT1S", "MT1S-sum grdC": "MT1S", "MT1S-rep evo": "MT1S",
    "MT1S-shr grad": "MT1S", "MT1S-shr grdC": "MT1S",
    "MT1S-stk evo": "MT1P", "MT1S-stk grad": "MT1P", "MT1S-stk grdC": "MT1P",
    "TANH32 grad": "TANH32", "TANH32 gdmn": "TANH32",
}


def _layers(defs):
    """[(out, in)] -> [(w_off, b_off, out, in)] packed as weight then bias, layer by layer."""
    out, off = [], 0
    for o, i in defs:
        out.append((off, off + o * i, o, i))
        off += o * i + o
    return out, off


# MT1Net: the offsets of mt1_pool.h's NT_* constants, built from the same shapes.
_TRUNK = [(20, 20), (20, 20), (6, 10), (4, 6), (5, 7), (4, 5)]      # a1 a2 b1 b2 c1 c2
_MT1, MT1_PARAMS = _layers(_TRUNK + _TRUNK + [(22, 56), (10, 23), (1, 10)])
_MT1C, MT1C_PARAMS = _layers([(64, C_IN), (24, 64), (8, 24), (1, 8)])
_MT1S, MT1S_PARAMS = _layers([(16, S_IN), (8, 16), (1, 8)])
_MT1P, MT1P_PARAMS = _layers([(16, P_IN), (8, 16), (1, 8)])
_T32, TANH32_PARAMS = _layers([(32, C_IN), (8, 32), (1, 8)])
PARAMS = {"MT1": MT1_PARAMS, "MT1C": MT1C_PARAMS, "MT1S": MT1S_PARAMS,
          "MT1P": N_SYMS * MT1P_PARAMS, "TANH32": TANH32_PARAMS}
assert (MT1_PARAMS, MT1C_PARAMS, MT1S_PARAMS, MT1P_PARAMS, TANH32_PARAMS) == \
    (3501, 11177, 385, 449, 4977)


def _dense(W, layer, x):
    w, b, o, i = layer
    return W[w:b].reshape(o, i) @ x + W[b:b + o]


def _relu(x):
    return np.maximum(x, np.float32(0))


def _lrelu(x):
    return np.where(x > 0, x, LEAK * x)


def _trunk(W, L, x37):
    xb, xc, xa = x37[:10], x37[10:17], x37[17:]
    a = _relu(_dense(W, L[1], _relu(_dense(W, L[0], xa))))
    b = _relu(_dense(W, L[3], _relu(_dense(W, L[2], xb))))
    c = _relu(_dense(W, L[5], _relu(_dense(W, L[4], xc))))
    return np.concatenate([a, b, c])


def mt1_forward(W, in74):
    """mt1net_forward(W, in74, 0): the raw logit."""
    h = np.concatenate([_trunk(W, _MT1[0:6], in74[:37]), _trunk(W, _MT1[6:12], in74[37:])])
    d1 = _relu(_dense(W, _MT1[12], h))
    d2 = _relu(_dense(W, _MT1[13], np.append(d1, np.float32(0))))   # reserved d2 input
    return float(_dense(W, _MT1[14], d2)[0])


def mt1c_forward(W, in146):
    x = in146
    for layer in _MT1C[:3]:
        x = _lrelu(_dense(W, layer, x))
    return float(_dense(W, _MT1C[3], x)[0])


def mt1s_slices(in146):
    """[12, 14]: each symbol's 12 features plus the industry cash and book (mt1s_slice)."""
    x = in146[:N_SYMS * PER_SYM].reshape(N_SYMS, PER_SYM)
    ctx = np.broadcast_to(in146[[CN_CASH, CN_BOOK]], (N_SYMS, 2))
    return np.concatenate([x, ctx], 1).astype(np.float32)


def mt1s_forward(W, in146):
    total = np.float32(0)
    for s in mt1s_slices(in146):
        h = _lrelu(_dense(W, _MT1S[1], _lrelu(_dense(W, _MT1S[0], s))))
        total += _dense(W, _MT1S[2], h)[0]
    return float(total)


def mt1p_forward(W, slices18, evo):
    """slices18 [12, 18] (mt1p_slices). W holds the twelve stock nets back to back."""
    total = np.float32(0)
    for j, s in enumerate(slices18):
        Wj = W[j * MT1P_PARAMS:(j + 1) * MT1P_PARAMS]
        h = _lrelu(_dense(Wj, _MT1P[1], _lrelu(_dense(Wj, _MT1P[0], s))))
        out = _dense(Wj, _MT1P[2], h)[0]
        total += np.tanh(out) * PRED_SCALE if evo else out
    return float(total)


def tanh32_forward(W, in146):
    h = np.tanh(_dense(W, _T32[1], np.tanh(_dense(W, _T32[0], in146))))
    return float(_dense(W, _T32[2], h)[0])


IND_STARTING_CASH = 25000.0


def whole_shares(q):
    """training_v4.cpp whole_shares: what an order can actually execute."""
    w = np.floor(np.float32(q) + np.float32(1e-4))
    return float(w) if w > 0 else 0.0


def build_cfeat146(o, h, lo, c, valid, hold, cash, intent, book_base=IND_STARTING_CASH):
    """MT1C's 146 inputs for one industry at today's close (build_mt1c_input_for).

    o, h, l, c, valid: today's bars for the 12 symbols. hold: shares held (the book BEFORE
    today's orders); cash: the industry's cash. intent: [12, 4] StockNN output for the deployed
    model, columns buy_qty, buy_price_frac, sell_all_price_frac, sell_qty.

    Every feature is scale-free except CN_BOOK, which the trainer reads as book / $25,000 -- the
    simulated industry's starting cash. `book_base` is the paper industry's own starting capital,
    so the feature keeps its meaning (the book's growth since the start) at another account size.
    """
    from models import stock_close_pos, stock_close_vs_wap

    def fin(v, fb=0.0):
        return float(v) if np.isfinite(v) else fb

    def clamp(v, lo, hi):
        v = fin(v)
        return lo if v < lo else hi if v > hi else v

    out = np.zeros(C_IN, np.float32)
    book = fin(cash) + sum(fin(hold[j]) * c[j] for j in range(N_SYMS) if valid[j])
    closes = [c[j] for j in range(N_SYMS) if valid[j] and c[j] > 1e-6]
    mean_close = float(np.mean(closes)) if closes else 1.0
    bk = book if book > 1.0 else 1.0
    for j in range(N_SYMS):
        f = j * PER_SYM
        ok = bool(valid[j]) and h[j] > lo[j] and c[j] > 1e-6
        bq, bf, sf, sq = (intent[j][k] for k in range(4))
        if ok:
            A = (2 * o[j] + 3 * c[j] + h[j] + lo[j]) / 7
            out[f:f + 7] = (o[j] / c[j], h[j] / c[j], lo[j] / c[j], c[j] / mean_close,
                            stock_close_pos(o[j], h[j], lo[j], c[j]),
                            stock_close_vs_wap(o[j], h[j], lo[j], c[j]),
                            (h[j] - lo[j]) / A if A > 1e-6 else 0.0)
            out[f + 7] = clamp(fin(hold[j]) * c[j] / bk, 0.0, 1.0)
            afford = whole_shares(fin(cash) / c[j])
            out[f + 8] = clamp(whole_shares(min(fin(bq), afford)) / afford, 0.0, 1.0) \
                if afford > 1e-9 else 0.0
        else:
            out[f:f + 4] = 1.0
        pos = whole_shares(fin(hold[j]))
        out[f + 11] = clamp(whole_shares(min(fin(sq), pos)) / pos, 0.0, 1.0) if pos > 1e-9 else 0.0
        out[f + 9] = clamp(fin(bf, 0.5), 0.0, 1.0)
        out[f + 10] = clamp(fin(sf, 0.5), 0.0, 1.0)
    out[CN_CASH] = clamp(fin(cash) / bk, 0.0, 1.0)
    out[CN_BOOK] = fin(book) / book_base
    return out


def mt1p_slices(in146, close, valid, cash, hold, bqty, bfrac, sfrac, sqty, raw):
    """[12, 18] MT1S-stk inputs for one order set (build_mt1p_slices in training_v4.cpp): the
    MT1S slice, then that stock's four StockNN outputs. `raw` keeps share counts as StockNN
    emitted them (the grad entry); otherwise quantities are ratios to what could be bought or is
    held (the evo entry). A symbol without a valid bar reads as no order."""
    out = np.zeros((N_SYMS, P_IN), np.float32)
    out[:, :S_IN] = mt1s_slices(in146)
    for j in range(N_SYMS):
        ok = bool(valid[j]) and close[j] > 1e-6

        def fin(v, fb=0.0):
            return float(v) if np.isfinite(v) else fb
        bq = max(0.0, fin(bqty[j])) if ok else 0.0
        sq = max(0.0, fin(sqty[j])) if ok else 0.0
        bf = fin(bfrac[j], 0.5) if ok else 0.5
        sf = fin(sfrac[j], 0.5) if ok else 0.5
        if not raw:
            afford = fin(cash) / close[j] if ok else 0.0
            held = fin(hold[j])
            bq = bq / afford if afford > 1e-9 else 0.0
            sq = sq / held if held > 1e-9 else 0.0
        out[j, S_IN:] = (bq, bf, sf, sq)
    return out


def slug(entry):
    """'MT1S-stk evo' -> 'MT1S-stk_evo', 'MT1 evo' -> 'MT1______evo': the trainer pads each
    name's architecture to 9 characters (race_slug)."""
    a, s = entry.rsplit(" ", 1)
    return f"{a:<9}{s}".replace(" ", "_")


class RaceModel:
    """One entry's champion model for one industry."""

    def __init__(self, entry, W):
        self.entry, self.arch = entry, ARCH[entry]
        self.W = np.asarray(W, np.float32)
        if self.W.size != PARAMS[self.arch]:
            raise ValueError(f"{entry}: {self.W.size} weights, expected {PARAMS[self.arch]}")
        self.evo = entry.endswith(" evo")

    @classmethod
    def load(cls, champion_dir, entry, industry):
        path = os.path.join(champion_dir, f"race_{slug(entry)}_{industry}.bin")
        return cls(entry, np.fromfile(path, np.float32))

    def predict(self, in74=None, in146=None, slices18=None):
        """The entry's input: in74 for MT1, in146 for MT1C/MT1S/TANH32, slices18 for MT1P
        (built with mt1p_slices, raw for the grad entry and ratios for evo)."""
        a = self.arch
        if a == "MT1":
            return mt1_forward(self.W, np.asarray(in74, np.float32))
        if a == "MT1P":
            return mt1p_forward(self.W, np.asarray(slices18, np.float32), self.evo)
        x = np.asarray(in146, np.float32)
        return {"MT1C": mt1c_forward, "MT1S": mt1s_forward, "TANH32": tanh32_forward}[a](self.W, x)
