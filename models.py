"""
models.py — Shared neural network definitions.

Single source of truth for StockNN, MasterNN, MT1Net, MT1CNet, MT2INet, and MT2NN.
All training scripts, production_v2.py, and inspect_trades.py import from here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── StockNN `today` vector layout ──────────────────────────────────────────────
# Mirrors TODAY_PER_SYM / TODAY_AGG_OFF / TODAY_STATE_OFF / STOCK_RESERVED in
# training_v4.cpp. The three Python builders (training_lib, production_v2,
# inspect_trades) append sections in this order, so these are the contract those
# builders must satisfy — not indices they read.
TODAY_PER_SYM   = 17                        # per-symbol block width
TODAY_CLOSE_POS = 15                        # close_pos slot within a block
TODAY_CLOSE_WAP = 16                        # close vs weighted average price (v0.6.1.0)
TODAY_RESERVED  = TODAY_CLOSE_WAP           # back-compat alias; slot is no longer reserved
TODAY_N_SYMS    = 12
TODAY_AGG_OFF   = TODAY_N_SYMS * TODAY_PER_SYM    # 204 — cross-symbol aggregates start
TODAY_AGG_LEN   = 15
TODAY_STATE_OFF = TODAY_AGG_OFF + TODAY_AGG_LEN   # 219 — cash, then 12 holdings
TODAY_WIDTH     = TODAY_STATE_OFF + 1 + TODAY_N_SYMS   # 232


def stock_close_vs_wap(open_, high, low, close):
    """Close relative to the day's weighted average price (v0.6.1.0, today feature 16).

        A = (2O + 3C + H + L) / 7                 the weighted average price
        out = (C - A) / A = (4C - 2O - H - L) / (2O + 3C + H + L)

    Shares its NUMERATOR with stock_close_pos; only the denominator differs — 7A here versus
    7(H-L) there. A sits close to C, so this is effectively the /C normalization, which measured
    weaker on the tradeable next-intraday leg (within-industry t +0.93 vs +2.54). It is carried
    anyway because the two together encode (H-L)/A, the range as a fraction of price, which
    neither gives alone and which the net cannot form itself (no division in a linear layer).

    Mirrors stock_close_vs_wap() in training_v4.cpp. Returns 0.0 on a degenerate bar.
    """
    denom = 2.0 * open_ + 3.0 * close + high + low
    if not denom > 1e-9:
        return 0.0
    return (4.0 * close - 2.0 * open_ - high - low) / denom


def stock_close_pos(open_, high, low, close):
    """Intraday position of the close within today's bar (v0.6.0.0, StockNN today feature 15).

        close_pos = (4C - 2O - H - L) / (7(H - L))  ==  2*(C-O)/(H-L) + CLV

    Mirrors stock_close_pos() in training_v4.cpp — keep the two in sync. Returns 0.0 for a
    zero-range or invalid bar. The denominator is (H-L) rather than C: measured within-industry,
    normalizing by C leaves the tradeable O->C leg insignificant (t +0.93 vs t +2.54).
    """
    rng = high - low
    if not rng > 1e-9:
        return 0.0
    return (4.0 * close - 2.0 * open_ - high - low) / (7.0 * rng)


class StockNN(nn.Module):
    """
    FC injection architecture — no LSTM.

    history: (1, 15, 60)  — 15 days oldest→newest, OHLCV × 12 stocks
    today:   (1, 232)     — current day full features:
               OHLCV×12 (60) + ΔOHLCV×12 (60) + price_pos×12 (12)
               + momentum×12 (12) + volatility×12 (12) + vol_ratio×12 (12)
               + dvol_ratio×12 (12) + close_pos×12 (12) + close_vs_wap×12 (12)
               + ind_agg (15) + state (13) = 232

    close_pos = (4C − 2O − H − L) / (7(H − L)), the intraday position of the close
    (v0.6.0.0). Equivalently 2·(C−O)/(H−L) + CLV. The raw O/H/L/C above are dollar
    amounts, so the net can already form this numerator in fc_today but cannot divide;
    the scale-free normalization is what is outside its span.

    Slot 16 carries close_vs_wap as of v0.6.1.0 = (C - A)/A with A = (2O+3C+H+L)/7. It was
    held reserved (constant 0.0) in v0.6.0.0 specifically so it could be filled without another
    breaking retrain: dimensions, offsets and STOCKNN_PARAMS are unchanged between the two.

    Seed   (day 15):  60              → FC → 120
    Inject (×14):     (180+5i)+60     → FC → 125+5i  (grows 120→190)
    Today:            190+232=422     → FC → 300
    Flat:             300             → FC → 300  (×2)
    Funnel:           300→237→174→111→48

    Output (1,48) → reshape (12,4):
      col 0: buy_qty              (ReLU)
      col 1: buy_price_frac       (Sigmoid, 0–1 in low-high range)
      col 2: sell_all_price_frac  (Sigmoid, 0–1 in low-high range)
      col 3: sell_qty             (ReLU)
    """

    def __init__(self):
        super().__init__()
        self.fc_seed   = nn.Linear(60,  120)
        self.fc_inject = nn.ModuleList([
            nn.Linear(180 + 5 * i, 125 + 5 * i) for i in range(14)
        ])
        self.fc_today  = nn.Linear(422, 300)
        self.fc_flat1  = nn.Linear(300, 300)
        self.fc_flat2  = nn.Linear(300, 300)
        self.fc_fc1    = nn.Linear(300, 237)
        self.fc_fc2    = nn.Linear(237, 174)
        self.fc_fc3    = nn.Linear(174, 111)
        self.fc_out    = nn.Linear(111,  48)

    def forward(self, history, today):
        x = F.relu(self.fc_seed(history[:, 0, :]))
        for i, layer in enumerate(self.fc_inject):
            x = F.relu(layer(torch.cat([x, history[:, i + 1, :]], dim=1)))
        x   = F.relu(self.fc_today(torch.cat([x, today], dim=1)))
        x   = F.relu(self.fc_flat1(x))
        x   = F.relu(self.fc_flat2(x))
        x   = F.relu(self.fc_fc1(x))
        x   = F.relu(self.fc_fc2(x))
        x   = F.relu(self.fc_fc3(x))
        out = self.fc_out(x).view(12, 4)
        qty_buy   = F.relu(out[:, 0:1])
        price_buy = torch.sigmoid(out[:, 1:2])
        price_sal = torch.sigmoid(out[:, 2:3])
        qty_sell  = F.relu(out[:, 3:4])
        return torch.cat([qty_buy, price_buy, price_sal, qty_sell], dim=1).view(1, 48)


class MasterNN(nn.Module):
    """
    Flat FC architecture — no injection chain, no history tensor.

    today: (1, 444) — explicit temporal features:
             delta at 18 lookback days × 12 industries    (216)
             2nd-order poly coefs, 5-day window × 12      ( 36)
             3rd-order poly coefs × 4 windows × 12        (192)
             Total: 444

    FC1:    444 → 444  ReLU  (wide)
    FC2:    444 → 444  ReLU  (wide)
    FC3:    444 → 312  ReLU  (taper, step=132)
    FC4:    312 → 180  ReLU  (taper)
    fc_out: 180 →  48        (taper = output)

    Output (1, 48): raw logits, reshape to (12, 4).
      Per-industry softmax over dim=1 → argmax → tier ∈ {0,1,2,3}
      0 = expected net loss; 1/2/3 = positive-return terciles (low→high)
    """

    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(444, 444)
        self.fc2    = nn.Linear(444, 444)
        self.fc3    = nn.Linear(444, 312)
        self.fc4    = nn.Linear(312, 180)
        self.fc_out = nn.Linear(180,  48)

    def forward(self, today):
        x = F.relu(self.fc1(today))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc_out(x)   # (1, 48) raw logits


class MT1Net(nn.Module):
    """Single-output MT1 — the rebuild. ONE network per industry, ONE number out.

    Predicts that industry's NEXT-DAY StockNN P&L in dollars. Replaces the deleted dual head +
    four specialized tails (9,208 params across five 200-slot pools) with 3,501 in one pool.

    Why the head/tail split went: it existed so four components could share a trunk. conf4 was
    ungraded and fed MT2 a constant; direction scored 52.25% OOS against 83.66% in-sample with
    negative skill in all 12 industries; delta aimed at the same 10-day forward relative-return
    target that nothing predicted — including the control MT1 already receives as features
    [10..16]. With one output a shared trunk has one consumer, so it is indirection, not sharing.

    Why the BLOCK STRUCTURE stayed: it is a real inductive bias, not decoration. The 37 features
    are three contiguous kinds — daily returns [0:10], decade momentum [10:17], vol+poly [17:37] —
    and keeping them apart for two layers stops layer one mixing 20 volatility features with 10
    daily returns. It is also CHEAPER than dense: 998 params per trunk versus 2,100 for a plain
    74->28. Two trunks (market, portfolio) preserve the same separation across the 74-feature
    input, which is [37 market || 37 portfolio].

      per trunk:  daily 10->6->4 | decade 7->5->4 | vol+poly 20->20->20  -> concat 28
      two trunks: 56
      tail:       56 -> 22 -> 10 -> 1        (was 4 layers; 3 because search is evolutionary,
                                              not gradient — depth costs more here)

    d2 takes 23 inputs, not 22. The 23rd is RESERVED, fed 0.0, held for (H-L)/A — range as a
    fraction of price, which measured +0.0338 incremental R2 against forward vol and is outside
    the span of these features by construction (all 37 derive from one cumulative close series,
    with no high or low anywhere). Injected at d2 rather than d1 so a single mutation can reach
    the output: under gradient-free search a feature buried behind three ReLUs is unlikely to be
    found. Reserving the slot now means filling it later changes no dimension, no offset and no
    file format — the same trick that let StockNN's slot 16 be filled in v0.6.1.0 with binary
    compatibility intact.

    3,501 params. Layer names/order MUST match MT1NET_LAYER_DEFS and the C++ offsets.
    """

    RESERVED_D2_INPUT = 22          # index of the held-open (H-L)/A slot

    def __init__(self):
        super().__init__()
        # market trunk
        self.m_a1 = nn.Linear(20, 20); self.m_a2 = nn.Linear(20, 20)
        self.m_b1 = nn.Linear(10, 6);  self.m_b2 = nn.Linear(6, 4)
        self.m_c1 = nn.Linear(7, 5);   self.m_c2 = nn.Linear(5, 4)
        # portfolio trunk
        self.p_a1 = nn.Linear(20, 20); self.p_a2 = nn.Linear(20, 20)
        self.p_b1 = nn.Linear(10, 6);  self.p_b2 = nn.Linear(6, 4)
        self.p_c1 = nn.Linear(7, 5);   self.p_c2 = nn.Linear(5, 4)
        # tail
        self.d1 = nn.Linear(56, 22)
        self.d2 = nn.Linear(23, 10)   # 22 + 1 reserved
        self.d3 = nn.Linear(10, 1)

    @staticmethod
    def _trunk(x, a1, a2, b1, b2, c1, c2):
        xb, xc, xa = x[:, 0:10], x[:, 10:17], x[:, 17:37]
        a = F.relu(a2(F.relu(a1(xa))))
        b = F.relu(b2(F.relu(b1(xb))))
        c = F.relu(c2(F.relu(c1(xc))))
        return torch.cat([a, b, c], dim=1)          # (batch, 28)

    def forward(self, x, extra=None):
        """x: (batch, 74) = [37 market || 37 portfolio]. Returns (batch, 1) RAW logit.

        `extra` is the reserved d2 input; None feeds 0.0, which is exactly inert (0 x w = 0).
        Decode with mt1_pred(): tanh(raw) * MT1_PRED_SCALE.
        """
        m = self._trunk(x[:, 0:37],  self.m_a1, self.m_a2, self.m_b1,
                        self.m_b2, self.m_c1, self.m_c2)
        p = self._trunk(x[:, 37:74], self.p_a1, self.p_a2, self.p_b1,
                        self.p_b2, self.p_c1, self.p_c2)
        h = F.relu(self.d1(torch.cat([m, p], dim=1)))           # (batch, 22)
        if extra is None:
            extra = torch.zeros(h.shape[0], 1, dtype=h.dtype, device=h.device)
        h = F.relu(self.d2(torch.cat([h, extra], dim=1)))       # (batch, 10)
        return self.d3(h)                                       # (batch, 1) raw


# (prefix, out_size, in_size) — mirrors the C++ offsets, layers in order, each as weights then bias.
MT1NET_LAYER_DEFS = [
    ('m_a1', 20, 20), ('m_a2', 20, 20), ('m_b1', 6, 10),
    ('m_b2', 4, 6),   ('m_c1', 5, 7),   ('m_c2', 4, 5),
    ('p_a1', 20, 20), ('p_a2', 20, 20), ('p_b1', 6, 10),
    ('p_b2', 4, 6),   ('p_c1', 5, 7),   ('p_c2', 4, 5),
    ('d1', 22, 56), ('d2', 10, 23), ('d3', 1, 10),
]

MT1NET_PARAMS = sum(o * i + o for _, o, i in MT1NET_LAYER_DEFS)   # 3,501


class MT1CNet(nn.Module):
    """MT1 competitor: TODAY only, no history of any kind.

    MT1Net sees 74 features, every one of them derived from a trailing value curve — 10 daily
    returns, 7 ten-day buckets, a 20-day vol, and polynomial fits over 5/10/30/60/90 days. It sees
    no individual symbol and nothing about what StockNN intends to do.

    This sees the opposite: everything available at today's close and nothing before it.

        per symbol (x12)   O H L C                                              4
                           close_pos, close_vs_wap, (H-L)/A                     3
                           holdings                                             1
                           buy_qty, buy_price_frac, sell_all_price_frac, sell_qty  4
        per industry       cash, book value                                     2
                                                                        total = 146

    Volume is deliberately excluded: without history there is no norm to read it against, so 1M
    shares carries no information about whether that is high or low for the symbol.

    Same target as MT1Net — that industry's next-session book P&L — and the same pool, scoring and
    lifecycle, so the comparison isolates the FEATURE SET rather than the machinery.

    The intents are causal: limit prices are `low_t + frac * span_t`, anchored to today's bar, and
    the quantities come straight off StockNN's forward pass on today's data. Executed volume would
    not be, since a fill is decided by the next day's bar.
    """

    N_SYMS = 12
    PER_SYM = 12
    N_IN = N_SYMS * PER_SYM + 2          # 146

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(self.N_IN, 64)
        self.l2 = nn.Linear(64, 24)
        self.l3 = nn.Linear(24, 8)
        self.l4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)                 # raw logit; tanh x MT1_PRED_SCALE at decode


# (prefix, out_size, in_size) — mirrors the C++ CN_* offsets, same convention as MT1NET_LAYER_DEFS.
MT1CNET_LAYER_DEFS = [
    ('l1', 64, MT1CNet.N_IN), ('l2', 24, 64), ('l3', 8, 24), ('l4', 1, 8),
]
MT1CNET_PARAMS = sum(o * i + o for _, o, i in MT1CNET_LAYER_DEFS)   # 11,177


class MT2INet(nn.Module):
    """The INDEPENDENT allocator: all 12 industries in, one industry's P&L out. No MT1, no MT1C.

    Input is the whole 888-feature master vector — every industry's 74 — so it can see the
    cross-section that MT1Net and MT1CNet cannot. Output is a single value for the industry being
    asked about, exactly like them, so it drops into the same pool, scoring and target and the
    three-way comparison isolates the FEATURE SET.

    Why a per-industry output rather than the joint (12,4) tier map MT2NN emits: training signal.
    A joint ranking makes each DAY one example (1,238 of them); a per-industry value makes each
    INDUSTRY-DAY one (14,856). That 12x is structural and no architecture recovers it. Ranking is
    then arithmetic — sort the 12 outputs and hand them to tiers_to_alloc — which is the half of
    the job that never needed a network.

    Sized by MEMORY, not by preference. 888 inputs make the first layer dominate, and at 32 wide
    the pool wanted 278 MB across 12 industries — which pushed a 1.9 GB box into swap and more
    than doubled the trainer's wall time (32 -> 80 s/day) with 936 MB paged out. At 16 wide it is
    138 MB and fits. samples/param is 1.03 against the MT1 stack's 0.354, so a loss still cannot
    be blamed on capacity relative to data; the honest caveat is the 16-wide bottleneck on 888
    inputs, which is a real constraint on how much of the cross-section it can carry.
    """

    N_IN = 888

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(self.N_IN, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 4)
        self.l4 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)                 # raw logit; tanh x MT1_PRED_SCALE at decode


MT2INET_LAYER_DEFS = [
    ('l1', 16, MT2INet.N_IN), ('l2', 8, 16), ('l3', 4, 8), ('l4', 1, 4),
]
MT2INET_PARAMS = sum(o * i + o for _, o, i in MT2INET_LAYER_DEFS)   # 14,401


class MT2NN(nn.Module):
    """
    Cross-industry tier allocator — replaces MasterNN.

    Input: (1, 12) — one MT1 prediction per industry, in INDUSTRY_NAMES order: that industry's
           predicted next-session StockNN P&L in dollars. No normalization — the dollar magnitude
           IS the allocation signal, and its sign is the direction call.

           This was 48 (4 channels × 12) until the MT1 rebuild. Of those four channels one was
           ungraded and fed a constant, one was the magnitude with its sign deliberately discarded,
           and one graded itself against another's residual. A single signed prediction carries
           what all four were reaching for, and MT2 can no longer fit structure that isn't there.

    FC branch (projects 12→36, then holds width):
      FC1:  12 → 36  ReLU
      FC2:  36 → 36  ReLU          output: 36

    LSTM branch (12 steps × 1 feature — one industry per step):
      LSTM layer 1: input=1,  hidden=36
      LSTM layer 2: input=36, hidden=36   output: 36 (final hidden state)

    Concatenate: [FC_out ‖ LSTM_out] = 72

    Taper FC layers (step = 6):
      T1:   72 → 66  ReLU
      T2:   66 → 60  ReLU
      T3:   60 → 54  ReLU
      fc_out: 54 → 48  (raw logits)

    Output (1, 48): raw logits, reshape to (12, 4) → argmax per industry → tier ∈ {0,1,2,3}

    Total params: 32,844
    """

    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(12, 36)
        self.fc2    = nn.Linear(36, 36)
        self.lstm   = nn.LSTM(input_size=1, hidden_size=36, num_layers=2, batch_first=True)
        self.taper1 = nn.Linear(72, 66)
        self.taper2 = nn.Linear(66, 60)
        self.taper3 = nn.Linear(60, 54)
        self.fc_out = nn.Linear(54, 48)

    def forward(self, x):
        # x: (batch, 12)
        fc = F.relu(self.fc1(x))
        fc = F.relu(self.fc2(fc))                       # (batch, 36)

        lstm_in = x.view(x.size(0), 12, 1)             # (batch, 12 steps, 1 feature)
        _, (h_n, _) = self.lstm(lstm_in)               # h_n: (2, batch, 36)
        lstm_out = h_n[-1]                              # last layer final hidden: (batch, 36)

        combined = torch.cat([fc, lstm_out], dim=1)    # (batch, 72)
        combined = F.relu(self.taper1(combined))
        combined = F.relu(self.taper2(combined))
        combined = F.relu(self.taper3(combined))
        return self.fc_out(combined)                    # (batch, 48) raw logits
