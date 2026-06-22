"""
models.py — Shared neural network definitions.

Single source of truth for StockNN, MasterNN, MT1NN, and MT2NN.
All training scripts, production_v2.py, and inspect_trades.py import from here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StockNN(nn.Module):
    """
    FC injection architecture — no LSTM.

    history: (1, 15, 60)  — 15 days oldest→newest, OHLCV × 12 stocks
    today:   (1, 208)     — current day full features:
               OHLCV×12 (60) + ΔOHLCV×12 (60) + price_pos×12 (12)
               + momentum×12 (12) + volatility×12 (12) + vol_ratio×12 (12)
               + dvol_ratio×12 (12) + ind_agg (15) + state (13) = 208

    Seed   (day 15):  60              → FC → 120
    Inject (×14):     (180+5i)+60     → FC → 125+5i  (grows 120→190)
    Today:            190+208=398     → FC → 300
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
        self.fc_today  = nn.Linear(398, 300)
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


class MT1NN(nn.Module):
    """
    Per-industry preprocessor — one pool per industry (12 total).

    Input: (1, 37) — one industry's slice of the 444-feature vector:
             18 delta lookbacks + 3 poly-2 coefs + 16 poly-3 coefs

    FC1:    37 → 37  ReLU  (width = n_inputs)
    FC2:    37 → 29  ReLU  (taper step ≈ 8)
    FC3:    29 → 20  ReLU
    FC4:    20 → 12  ReLU
    fc_out: 12 →  4  (raw logits decoded at score time)

    Output (1, 4) raw logits:
      out[0] → sigmoid → P(positive return), direction confidence ∈ [0,1]
      out[1] → tanh × $10K → expected dollar P&L (ceiling ≈ 5yr max swing)
      out[2] → softplus → range as % of effective_delta (dimensionless)
      out[3] → sigmoid → calibrated confidence (how well range covers actual)

    Total params: 3,412
    """

    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(37, 37)
        self.fc2    = nn.Linear(37, 29)
        self.fc3    = nn.Linear(29, 20)
        self.fc4    = nn.Linear(20, 12)
        self.fc_out = nn.Linear(12,  4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc_out(x)   # (1, 4) raw logits


class MT2NN(nn.Module):
    """
    Cross-industry tier allocator — replaces MasterNN.

    Input: (1, 48) — 4 MT1 slot0 raw activations × 12 industries, no normalization.
           [conf0, delta_tanh, range_pct, conf4] per industry in INDUSTRY_NAMES order.
           Reshaped to (1, 12, 4) for the LSTM branch (12 steps × 4 features).

    FC branch (projects 48→36, then holds width):
      FC1:  48 → 36  ReLU
      FC2:  36 → 36  ReLU          output: 36

    LSTM branch (12 steps × 4 features):
      LSTM layer 1: input=4,  hidden=36
      LSTM layer 2: input=36, hidden=36   output: 36 (final hidden state)

    Concatenate: [FC_out ‖ LSTM_out] = 72

    Taper FC layers (step = 6):
      T1:   72 → 66  ReLU
      T2:   66 → 60  ReLU
      T3:   60 → 54  ReLU
      fc_out: 54 → 48  (raw logits)

    Output (1, 48): raw logits, reshape to (12, 4) → argmax per industry → tier ∈ {0,1,2,3}

    Total params: 34,572
    """

    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(48, 36)
        self.fc2    = nn.Linear(36, 36)
        self.lstm   = nn.LSTM(input_size=4, hidden_size=36, num_layers=2, batch_first=True)
        self.taper1 = nn.Linear(72, 66)
        self.taper2 = nn.Linear(66, 60)
        self.taper3 = nn.Linear(60, 54)
        self.fc_out = nn.Linear(54, 48)

    def forward(self, x):
        # x: (batch, 48)
        fc = F.relu(self.fc1(x))
        fc = F.relu(self.fc2(fc))                       # (batch, 36)

        lstm_in = x.view(x.size(0), 12, 4)             # (batch, 12 steps, 4 features)
        _, (h_n, _) = self.lstm(lstm_in)               # h_n: (2, batch, 36)
        lstm_out = h_n[-1]                              # last layer final hidden: (batch, 36)

        combined = torch.cat([fc, lstm_out], dim=1)    # (batch, 72)
        combined = F.relu(self.taper1(combined))
        combined = F.relu(self.taper2(combined))
        combined = F.relu(self.taper3(combined))
        return self.fc_out(combined)                    # (batch, 48) raw logits
