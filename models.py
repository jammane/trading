"""
models.py — Shared neural network definitions.

Single source of truth for StockNN and MasterNN.
All training scripts (training_v2.py, training_v3.py, training_v4.py), production_v2.py,
and inspect_trades.py import from this module rather than defining their own copies.
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
    FC injection architecture — no LSTM.

    history: (1, 15, 61)  — 15 days oldest→newest,
               avg of 5 delta features × 12 industries + flat_cos for that day
    today:   (1, 229)     — current day full features:
               (max/min/avg of 5 deltas)×12 (180) + vol×12 (12)
               + momentum×12 (12) + correlation×12 (12) + state (13) = 229

    Seed   (day 15):  61              → FC → 120
    Inject (×14):     (181+5i)+61     → FC → 125+5i  (grows 120→190)
    Today:            190+229=419     → FC → 300
    Flat:             300             → FC → 300  (×2)
    Funnel:           300→234→168→102→36

    Output (1,36):
      [:12]   Softmax  → allocation weights (sum to 1)
      [12:24] Sigmoid  → liquidation depth per industry (0=hold, 1=liquidate to floor)
      [24:36] Sigmoid  → liquidation trigger per industry (>0.5 = execute)
    """

    def __init__(self):
        super().__init__()
        self.fc_seed   = nn.Linear(61,  120)
        self.fc_inject = nn.ModuleList([
            nn.Linear(181 + 5 * i, 125 + 5 * i) for i in range(14)
        ])
        self.fc_today  = nn.Linear(419, 300)
        self.fc_flat1  = nn.Linear(300, 300)
        self.fc_flat2  = nn.Linear(300, 300)
        self.fc_fc1    = nn.Linear(300, 234)
        self.fc_fc2    = nn.Linear(234, 168)
        self.fc_fc3    = nn.Linear(168, 102)
        self.fc_out    = nn.Linear(102,  36)

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
        out = self.fc_out(x)
        alloc   = F.softmax(out[:, :12],  dim=-1)
        depth   = torch.sigmoid(out[:, 12:24])
        trigger = torch.sigmoid(out[:, 24:36])
        return torch.cat([alloc, depth, trigger], dim=-1)
