"""Keep three vetted replacement symbols ready for every industry.

When a symbol has to leave the universe (its data stopped arriving, or its industry's prices
drifted out of band), the Friday training pass swaps in the first ready replacement. Choosing one
then, under time pressure, is how a poor pick gets made, so the list is maintained ahead of time
and refreshed early in the week: any concern about a candidate is raised, and the candidate
rejected, before Friday can choose it.

A candidate is screened from the Yahoo industries that at least two of the industry's members
belong to (one member Yahoo files elsewhere -- a utility it calls asset management -- must not
pull a whole unrelated industry in), is never a second share class of a company already in the
universe or already offered, and must pass every check the hand-made swaps were researched on:

  exchange     a major US exchange (Nasdaq, NYSE, NYSE American); no OTC listings
  price        last close inside the universe band, $30-$90
  history      at least 1230 clean daily bars in 5 years (the trainer's window is 1255; a
               5-year download of a long-listed stock returns ~1253, and missing bars are
               handled as any missing bar is)
  volatility   annualised volatility within 0.8x the quietest and 1.2x the most volatile member
  liquidity    60-day mean dollar volume at least half the least-traded member's
  tradable     Alpaca lists it as an active, tradable asset
  new          not in any universe, not removed in an earlier swap, not on the rejected list

Ready candidates are ranked by how closely they resemble the industry: the distance of their
volatility from the members' median, plus the distance of their price from the members' median
(both as log ratios). Each symbol is offered to one industry only.

The list lives in universe_watchlist.json under "replacements"; "rejected" ({SYMBOL: reason}) is
where a concern is recorded, and a rejected symbol is never offered again.

    python replacements.py refresh      # re-screen, re-vet, rewrite the list and print a report
    python replacements.py refresh --commit   # the Monday job: pull dev first, commit and push the list
    python replacements.py show         # print the current list
"""
import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys

import numpy as np

BAND = (30.0, 90.0)
N_READY = 3
MIN_HISTORY = 1230      # Yahoo's 5-year download returns ~1253 clean sessions for a listed stock
EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE"}
VOL_LO, VOL_HI = 0.8, 1.2
LIQ_FRAC = 0.5
SCREEN_SIZE = 60
WATCHLIST = "universe_watchlist.json"
STOCK_DATA = "stock_data"


def bar_stats(closes, volumes):
    """-> dict(price, vol (annualised, last 252 sessions), dollar_vol (60-session mean))."""
    c = np.asarray(closes, float)
    v = np.asarray(volumes, float)
    r = np.diff(np.log(c[-253:]))
    return dict(price=float(c[-1]), vol=float(r.std() * math.sqrt(252)) if len(r) > 1 else 0.0,
                dollar_vol=float((c[-60:] * v[-60:]).mean()))


def clean_bars(bars):
    """Keep bars whose OHLC are finite and positive. bars: [dict(open, high, low, close, volume)]."""
    return [b for b in bars if all(isinstance(b.get(k), (int, float)) and math.isfinite(b[k])
                                   and b[k] > 0 for k in ("open", "high", "low", "close"))]


def local_bars(sym, data_dir=STOCK_DATA):
    with open(os.path.join(data_dir, f"{sym}.json")) as f:
        return clean_bars(json.load(f).get("days", []))


def peer_profile(symbols, bars_fn=local_bars):
    """The industry's members, as the vetting bounds: per-member stats plus their ranges."""
    stats = {}
    for s in symbols:
        try:
            b = bars_fn(s)
            if len(b) > 60:
                stats[s] = bar_stats([x["close"] for x in b], [x["volume"] for x in b])
        except (OSError, ValueError):
            pass
    if not stats:
        return None
    vols = [x["vol"] for x in stats.values()]
    return dict(members=stats, vol_min=min(vols), vol_max=max(vols),
                vol_med=float(np.median(vols)),
                price_med=float(np.median([x["price"] for x in stats.values()])),
                liq_min=min(x["dollar_vol"] for x in stats.values()))


def vet(sym, exchange, bars, peers, tradable):
    """-> (stats, None) if `sym` passes every check, else (stats or None, the first reason it
    fails). bars: its daily bars, oldest first; tradable: Alpaca's verdict (None = not checked)."""
    if exchange not in EXCHANGES:
        return None, f"exchange {exchange} is not a major US exchange"
    bars = clean_bars(bars)
    if len(bars) < MIN_HISTORY:
        return None, f"{len(bars)} clean sessions of history, needs {MIN_HISTORY}"
    st = bar_stats([b["close"] for b in bars], [b["volume"] for b in bars])
    if not BAND[0] <= st["price"] <= BAND[1]:
        return st, f"price ${st['price']:.2f} outside ${BAND[0]:.0f}-${BAND[1]:.0f}"
    lo, hi = VOL_LO * peers["vol_min"], VOL_HI * peers["vol_max"]
    if not lo <= st["vol"] <= hi:
        return st, f"volatility {st['vol']:.0%} outside the industry's {lo:.0%}-{hi:.0%}"
    if st["dollar_vol"] < LIQ_FRAC * peers["liq_min"]:
        return st, (f"${st['dollar_vol'] / 1e6:.1f}M/day traded, under half the least-traded "
                    f"member's ${peers['liq_min'] / 1e6:.1f}M")
    if tradable is False:
        return st, "not an active, tradable asset on Alpaca"
    return st, None


def resemblance(st, peers):
    """Lower is closer: |log vol ratio| + |log price ratio| against the members' medians."""
    return (abs(math.log(max(st["vol"], 1e-9) / max(peers["vol_med"], 1e-9)))
            + abs(math.log(st["price"] / peers["price_med"])))


def excluded(watchlist, universe_syms):
    """Symbols never to offer: current members, earlier swaps' removed symbols, rejected ones."""
    out = set(universe_syms) | set(watchlist.get("completed", {})) | set(watchlist.get("rejected", {}))
    return out


def refresh(universe, watchlist, profile, screen, history, tradable, bars_fn=local_bars,
            today=None):
    """Re-screen and re-vet every industry. -> (new "replacements" dict, report lines).

    profile(sym) -> (its Yahoo industry, company name); screen(yahoo_industry) ->
    [(sym, exchange, company name)]; history(sym) -> its 5-year daily bars;
    tradable(sym) -> bool or None."""
    today = today or dt.date.today().isoformat()
    old = watchlist.get("replacements", {})
    skip = excluded(watchlist, [s for syms in universe.values() for s in syms])
    taken, out, report = set(), {}, []
    prof = {s: profile(s) for syms in universe.values() for s in syms}
    companies = {name for _, name in prof.values() if name}
    for ind, syms in universe.items():
        peers = peer_profile(syms, bars_fn)
        if peers is None:
            report.append(f"{ind}: CONCERN no member data to vet against")
            out[ind] = []
            continue
        held = [prof[s][0] for s in syms if prof[s][0]]
        yinds = sorted({y for y in held if held.count(y) >= 2}) or sorted(set(held))
        seen, ready, why = set(), [], {}
        for yi in yinds:
            try:
                found = screen(yi)
            except Exception as e:
                report.append(f"{ind}: CONCERN could not screen {yi!r} ({e})")
                continue
            for sym, exch, name in found:
                if sym in seen or sym in skip or sym in taken or (name and name in companies):
                    continue
                seen.add(sym)
                try:
                    bars = history(sym)
                    st, reason = vet(sym, exch, bars, peers, None)
                    if reason is None:                  # Alpaca last: only for real contenders
                        st, reason = vet(sym, exch, bars, peers, tradable(sym))
                except Exception as e:          # a failed fetch is a failed check, not a crash
                    st, reason = None, f"could not be checked ({e})"
                if reason is None:
                    ready.append(dict(symbol=sym, exchange=exch, yahoo_industry=yi, name=name,
                                      price=round(st["price"], 2), vol=round(st["vol"], 4),
                                      dollar_vol=round(st["dollar_vol"]),
                                      score=round(resemblance(st, peers), 4), checked=today))
                else:
                    why[sym] = reason
        ready.sort(key=lambda r: r["score"])
        out[ind] = []
        for r in ready:                                  # one share class per company
            if len(out[ind]) < N_READY and r["name"] not in {x["name"] for x in out[ind]}:
                out[ind].append(r)
        taken |= {r["symbol"] for r in out[ind]}
        companies |= {r["name"] for r in out[ind] if r["name"]}
        before = [r["symbol"] for r in old.get(ind, [])]
        now = [r["symbol"] for r in out[ind]]
        line = f"{ind}: " + (", ".join(f"{r['symbol']} ${r['price']:.2f} vol {r['vol']:.0%}"
                                       for r in out[ind]) or "none")
        if len(now) < N_READY:
            line += f"   CONCERN only {len(now)} of {N_READY} ready ({len(seen)} screened)"
        for s in before:
            if s not in now:
                line += f"\n    dropped {s}: {why.get(s, 'outranked or offered elsewhere')}"
        for s in now:
            if s not in before:
                line += f"\n    new {s}"
        report.append(line)
    return out, report


def ready(watchlist, industry):
    """The industry's ready replacements, best first."""
    return list(watchlist.get("replacements", {}).get(industry, []))


# ── Live data sources (yfinance, Alpaca) ─────────────────────────────────────────────────────────

def _yf_profile(sym):
    import yfinance as yf
    info = yf.Ticker(sym).info
    return info.get("industry"), info.get("longName") or info.get("shortName")


def screener_name(yahoo_industry, vocabulary):
    """Ticker.info spells "Software - Application" where the screener wants "Software—Application"
    (an em dash). -> the screener's name, or None if it has no such industry."""
    for name in (yahoo_industry, yahoo_industry.replace(" - ", "\u2014")):
        if name in vocabulary:
            return name
    return None


def _yf_screen(yahoo_industry):
    import yfinance as yf
    from yfinance import EquityQuery, const
    vocab = {v for vs in const.EQUITY_SCREENER_EQ_MAP["industry"].values() for v in vs}
    name = screener_name(yahoo_industry, vocab)
    if name is None:
        raise ValueError(f"the screener has no industry named {yahoo_industry!r}")
    q = EquityQuery("and", [EquityQuery("eq", ["region", "us"]),
                            EquityQuery("eq", ["industry", name]),
                            EquityQuery("btwn", ["intradayprice", BAND[0], BAND[1]])])
    r = yf.screen(q, size=SCREEN_SIZE, sortField="intradaymarketcap", sortAsc=False)
    return [(x["symbol"], x.get("exchange", ""), x.get("longName") or x.get("shortName"))
            for x in r.get("quotes", [])]


_HISTORY = {}


def _yf_history(sym):
    """5 years of daily bars, fetched once per run."""
    _cache = _HISTORY
    if sym not in _cache:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="5y", interval="1d", auto_adjust=True)
        _cache[sym] = [dict(open=float(r.Open), high=float(r.High), low=float(r.Low),
                            close=float(r.Close), volume=float(r.Volume))
                       for r in h.itertuples()]
    return _cache[sym]


def _alpaca_tradable():
    """sym -> True/False from Alpaca, or a function returning None when no keys are set."""
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        return lambda sym: None
    from alpaca.trading.client import TradingClient
    client = TradingClient(key, secret, paper=True)

    def check(sym):
        try:
            a = client.get_asset(sym)
            return bool(a.tradable) and str(getattr(a.status, "value", a.status)) == "active"
        except Exception:
            return False
    return check


def load_watchlist(path=WATCHLIST):
    with open(path) as f:
        return json.load(f)


def save_watchlist(wl, path=WATCHLIST):
    with open(path + ".tmp", "w") as f:
        json.dump(wl, f, indent=2)
        f.write("\n")
    os.replace(path + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("refresh", "show"))
    ap.add_argument("--watchlist", default=WATCHLIST)
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--commit", action="store_true",
                    help="pull dev first (picking up any rejections), then commit and push the list")
    a = ap.parse_args(argv)
    if a.commit:
        subprocess.run(["git", "pull", "-q", "--rebase", "origin", "dev"], check=True)
    wl = load_watchlist(a.watchlist)
    if a.cmd == "show":
        for ind, rs in wl.get("replacements", {}).items():
            print(f"{ind}: " + ", ".join(f"{r['symbol']} ${r['price']:.2f}" for r in rs))
        return 0
    with open(a.universe) as f:
        universe = json.load(f)
    tradable = _alpaca_tradable()
    if tradable("SPY") is None:
        print("NOTE: no Alpaca keys in the environment -- tradability not checked")
    new, report = refresh(universe, wl, _yf_profile, _yf_screen, _yf_history, tradable)
    wl["replacements"] = new
    wl["replacements_checked"] = dt.date.today().isoformat()
    save_watchlist(wl, a.watchlist)
    print(f"Replacement list refreshed {wl['replacements_checked']} "
          f"({N_READY} per industry; reject one by adding it to \"rejected\" in {a.watchlist})")
    print("\n".join(report))
    if a.commit:
        subprocess.run(["git", "add", a.watchlist], check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
            subprocess.run(["git", "commit", "-q", "-m",
                            f"Replacement list refreshed {wl['replacements_checked']}"], check=True)
            subprocess.run("git pull -q --rebase origin dev && git push -q origin dev",
                           shell=True, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
