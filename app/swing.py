"""Swing: per aandeel de strategie kiezen die buiten de keuzeperiode bleef werken, en daaruit een trade maken.

Werkwijze per onderliggend aandeel (dagkoersen, 5 jaar):

1. Alle strategieën één keer backtesten, met kosten. Alleen swingstrategieën tellen: een trade duurt gemiddeld
   2 tot 30 handelsdagen (dagen tot weken, geen jarenlang vasthouden en geen daghandel).
2. Elke trade krijgt dezelfde stop-loss en take-profit als op Hyperliquid: SL op 2 × ATR(14) onder de instap, TP op
   een veelvoud van de ATR dat uit de eerdere winnende trades van die strategie komt. Komen SL en TP op dezelfde dag,
   dan rekenen we de SL (voorzichtig).
3. Score = t-waarde van de trade-rendementen (gemiddelde / spreiding × √aantal) × hoe regelmatig de jaren winst gaven.
   Weinig trades of grote spreiding geven dus vanzelf een lage score; de hoogste winst alleen is niet genoeg.
   Een strategie moet ook beter doen dan het aandeel gewoon vasthouden over dezelfde dagen (geen meeliften op de trend).
4. Walk-forward: voor elk van de laatste 3 jaren kiezen we de beste strategie met alleen de jaren ervoor en kijken
   hoe die gekozen strategie het daarna deed. Dat is het eerlijke cijfer: zo goed werkte "de beste kiezen" echt.
5. Een voorstel komt er alleen als een van de best scorende strategieën nu een vers koopsignaal heeft.

Alles is long-only: de strategieën van de backtestserver kopen of staan cash.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .engine import run_backtest
from .strategies import CATEGORIES, STRATEGIES

YEARS = 5            # zoveel jaar trades
WF_YEARS = 3         # de laatste 3 jaar elk apart als testjaar
MIN_TRADES = 10      # minimum aantal afgesloten trades over de 5 jaar
SWING_BARS = (2, 30) # gemiddelde duur van een trade in handelsdagen
SL_ATR = 2.0         # stop-loss in ATR
TP_RANGE = (2.0, 6.0)
MIN_PF = 1.2         # winst / verlies over alle trades
FRESH_BARS = 5       # signaal hooguit zoveel handelsdagen oud
WF_MIN_N, WF_MIN_PF, WF_MIN_YEARS = 10, 1.3, 2  # walk-forward: trades, winstfactor en jaren met winst voor een voorstel
TOP = 5


def atr_pct(prices: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range als fractie van de slotkoers."""
    h, l, c = prices["high"], prices["low"], prices["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return (tr.rolling(n, min_periods=n).mean() / c).bfill()


def tp_multiple(trades: list[dict]) -> float:
    """TP-afstand in ATR: de mediaan van wat de winnende trades haalden (zonder SL/TP), begrensd op 2 tot 6."""
    wins = [t["raw"] / t["atr"] for t in trades if t["raw"] > 0 and t["atr"] > 0]
    if len(wins) < 3:
        return 3.0
    return float(min(TP_RANGE[1], max(TP_RANGE[0], np.median(wins))))


def overlay(prices: pd.DataFrame, trade: dict, sl_k: float, tp_k: float, fee: float) -> dict:
    """Een backtest-trade opnieuw afspelen met SL en TP op dagbasis. fee = kosten per kant (fractie)."""
    o, h, l = prices["open"].to_numpy(), prices["high"].to_numpy(), prices["low"].to_numpy()
    i0, i1, entry, atr = trade["i0"], trade["i1"], trade["entry"], trade["atr"]
    sl, tp = entry * (1 - sl_k * atr), entry * (1 + tp_k * atr)
    exit_px, why, last = None, "signaal", i1
    for d in range(i0, i1):
        if d > i0 and o[d] <= sl:   # gap onder de stop: uit op de opening
            exit_px, why, last = o[d], "sl", d
            break
        if d > i0 and o[d] >= tp:
            exit_px, why, last = o[d], "tp", d
            break
        if l[d] <= sl:
            exit_px, why, last = sl, "sl", d
            break
        if h[d] >= tp:
            exit_px, why, last = tp, "tp", d
            break
    if exit_px is None:
        exit_px = o[i1]
    return {"date": trade["date"], "exit": prices.index[last], "ret": float(exit_px / entry - 1 - 2 * fee), "why": why,
            "bars": int(max(1, last - i0)), "raw": trade["raw"], "atr": atr}


def stats(rets: list[float]) -> dict:
    n = len(rets)
    if not n:
        return {"n": 0, "win": None, "mean": None, "pf": None, "sum": 0.0, "t": None}
    a = np.array(rets)
    pos, neg = a[a > 0].sum(), -a[a <= 0].sum()
    sd = a.std(ddof=1) if n > 1 else 0.0
    return {"n": n, "win": float((a > 0).mean()), "mean": float(a.mean()), "pf": float(pos / neg) if neg > 0 else None,
            "sum": float(a.sum()), "t": float(a.mean() / sd * math.sqrt(n)) if sd > 0 else None,
            "sd": float(sd), "var": float(a.var(ddof=1)) if n > 1 else None}


def score(trades: list[dict], drift: float, min_n: int) -> dict | None:
    """Robuuste score van een lijst trades (na SL/TP en kosten), of None als ze niet door de filters komen."""
    if len(trades) < min_n:
        return None
    rets = [t["ret"] for t in trades]
    s = stats(rets)
    if not s["t"] or s["mean"] <= 0 or (s["pf"] is not None and s["pf"] < MIN_PF):
        return None
    alpha = float(np.mean([t["ret"] - drift * t["bars"] for t in trades]))
    if alpha <= 0:
        return None
    years: dict[int, float] = {}
    for t in trades:
        years[t["date"].year] = years.get(t["date"].year, 0.0) + t["ret"]
    cons = sum(1 for v in years.values() if v > 0) / len(years)
    return s | {"alpha": alpha, "cons": cons, "score": s["t"] * (0.5 + 0.5 * cons)}


def _raw_trades(prices: pd.DataFrame, res, atr: pd.Series) -> list[dict]:
    idx = prices.index
    out = []
    for t in res.trades:
        if t["exit_date"] is None:
            continue
        d0, d1 = pd.Timestamp(t["entry_date"]), pd.Timestamp(t["exit_date"])
        i0, i1 = idx.get_loc(d0), idx.get_loc(d1)
        out.append({"date": d0, "exit": d1, "i0": i0, "i1": i1, "entry": t["entry_price"], "raw": t["return"],
                    "atr": float(atr.iloc[i0 - 1] if i0 > 0 else atr.iloc[0]), "bars": t["bars"]})
    return out


def analyze(prices: pd.DataFrame, fee_bps: float = 9.0, today: pd.Timestamp | None = None) -> dict[str, Any]:
    """Volledige analyse van één onderliggend aandeel (prices met opwarmperiode, dagkoersen)."""
    end = pd.Timestamp(today) if today is not None else prices.index[-1]
    prices = prices[prices.index <= end]
    start = end - pd.DateOffset(years=YEARS)
    if prices.index[0] > start - pd.Timedelta(days=200):
        start = prices.index[0] + pd.Timedelta(days=300)  # kortere historie: wat opwarmtijd houden
    if (end - start).days < 3 * 365:
        raise ValueError("Te korte koershistorie (minder dan 3 jaar)")
    fee = fee_bps / 1e4
    atr = atr_pct(prices)
    traded = prices[prices.index >= start]
    drift = float(traded["close"].pct_change().mean())
    bounds = [end - pd.DateOffset(years=k) for k in range(WF_YEARS, 0, -1)] + [end + pd.Timedelta(days=1)]

    cands: dict[str, dict] = {}
    signals: dict[str, dict] = {}
    for name, cls in STRATEGIES.items():
        if cls.category == "benchmark":
            continue
        try:
            res = run_backtest(prices, cls(), 10_000.0, fee_bps, start=start)
        except Exception:  # noqa: BLE001
            continue
        m = res.metrics
        signals[name] = {"open": bool(m.get("signal_open")), "bars": m.get("bars_since_buy"), "date": m.get("last_buy_signal")}
        raw = _raw_trades(prices, res, atr)
        if len(raw) < MIN_TRADES // 2:
            continue
        avg_bars = float(np.mean([t["bars"] for t in raw]))
        if not SWING_BARS[0] <= avg_bars <= SWING_BARS[1]:
            continue
        cands[name] = {"raw": raw, "avg_bars": avg_bars}

    def played(name: str, before: pd.Timestamp | None) -> tuple[list[dict], float]:
        raw = cands[name]["raw"]
        learn = [t for t in raw if before is None or t["exit"] < before]
        tp_k = tp_multiple(learn)
        return [overlay(prices, t, SL_ATR, tp_k, fee) for t in raw], tp_k

    # ---- walk-forward: kiezen met de jaren ervoor, beoordelen in het testjaar
    wf_trades, wf_years = [], []
    for k in range(WF_YEARS):
        a, b = bounds[k], bounds[k + 1]
        best = None
        min_n = max(5, round(MIN_TRADES * (a - start).days / (end - start).days))
        for name in cands:
            tr, tp_k = played(name, a)
            sc = score([t for t in tr if t["exit"] < a], drift, min_n)
            if sc and (best is None or sc["score"] > best[1]["score"]):
                best = (name, sc, [t for t in tr if a <= t["date"] < b])
        if best is None:
            wf_years.append({"year_start": a.date().isoformat(), "strategy": None, "n": 0, "sum": 0.0})
            continue
        test = best[2]
        wf_trades += test
        wf_years.append({"year_start": a.date().isoformat(), "strategy": best[0], "n": len(test),
                         "sum": float(sum(t["ret"] for t in test)), "train_score": round(best[1]["score"], 2)})
    wf = stats([t["ret"] for t in wf_trades]) | {"years": wf_years}

    # ---- eindkeuze op alle 5 jaar (zo kiest de walk-forward ook, dus het WF-cijfer hoort bij deze werkwijze)
    rows = []
    for name, c in cands.items():
        tr, tp_k = played(name, None)
        sc = score(tr, drift, MIN_TRADES)
        if not sc:
            continue
        last = [t for t in tr if t["date"] >= bounds[-2]]
        sig = signals[name]
        cls = STRATEGIES[name]
        rows.append({"strategy": name, "category": CATEGORIES.get(cls.category, cls.category),
                     "description": cls.description, "score": round(sc["score"], 3), "n": sc["n"],
                     "win": sc["win"], "mean": sc["mean"], "pf": sc["pf"], "alpha": sc["alpha"], "cons": sc["cons"],
                     "var": sc["var"], "tp_atr": round(tp_k, 2), "avg_bars": round(c["avg_bars"], 1),
                     "exits": {w: sum(1 for t in tr if t["why"] == w) for w in ("sl", "tp", "signaal")},
                     "last12": stats([t["ret"] for t in last]), "open": sig["open"], "bars_since_buy": sig["bars"],
                     "last_buy": sig["date"]})
    rows.sort(key=lambda r: -r["score"])
    top = rows[:TOP]

    close = prices["close"]
    pick = next((r for r in top if r["open"] and r["bars_since_buy"] is not None and r["bars_since_buy"] <= FRESH_BARS), None)
    pick_info = None
    if pick:
        d = pd.Timestamp(pick["last_buy"])
        after = prices.index[prices.index > d]
        entry_ref = float(prices.loc[after[0], "open"]) if len(after) else float(close.iloc[-1])
        a_now = float(atr.iloc[-1])
        pick_info = pick | {"entry_ref": entry_ref, "run_up": float(close.iloc[-1] / entry_ref - 1),
                            "run_up_atr": float((close.iloc[-1] / entry_ref - 1) / a_now) if a_now else None}
    return {
        "last_date": close.index[-1].date().isoformat(), "last_close": float(close.iloc[-1]),
        "atr_pct": float(atr.iloc[-1]), "drift": drift, "tested": len(signals), "swing_candidates": len(cands),
        "passed": len(rows), "top": top, "wf": wf, "pick": pick_info,
        "agree": sum(1 for r in top if r["open"]),
        "signals": {n: s["open"] for n, s in signals.items()},
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
    }


# --------------------------------------------------------------------------- van analyse naar voorstel

def size(pick: dict, wf: dict, sl_dist: float, cost_frac: float, equity: float, cfg: dict, max_lev_mkt: float) -> dict:
    """Positiegrootte: kwart-Kelly op de walk-forward trades (anders de 5-jaarscijfers, gehalveerd), begrensd door
    het risico per trade (verlies bij SL), de marge per positie en de hefboom. cost_frac = funding over de houdduur."""
    if wf.get("n", 0) >= 6 and wf.get("var"):
        mu, var, src = wf["mean"], wf["var"], "walk-forward"
        shrink = 1.0
    else:
        mu, var, src = pick["mean"], pick.get("var") or 0.0, "5 jaar (gehalveerd: te weinig walk-forward trades)"
        shrink = 0.5
    mu_net = mu - cost_frac
    if mu_net <= 0 or not var:
        return {"ok": False, "why": f"verwachte winst per trade na funding ≤ 0 ({mu_net * 100:.2f}%)", "mu_net": mu_net}
    kelly = cfg["kelly"] * shrink * mu_net / var           # positie als fractie van het kapitaal
    risk_cap = cfg["risk_pct"] / 100 / sl_dist              # verlies bij SL hooguit risk_pct
    frac = min(kelly, risk_cap)
    margin_cap = cfg["margin_pct"] / 100
    lev_cap = int(max(1, min(cfg["max_lev"], max_lev_mkt or cfg["max_lev"])))
    lev = int(min(lev_cap, max(1, math.ceil(frac / margin_cap))))
    # Liquidatie moet ruim voorbij de stop liggen: afstand ≈ 1/hefboom - onderhoudsmarge (helft van 1/max hefboom).
    maint = 0.5 / (max_lev_mkt or 10)
    while lev > 1 and (1 / lev - maint) < 1.5 * sl_dist:
        lev -= 1
    margin = min(frac / lev, margin_cap)
    frac = margin * lev
    liq = 1 / lev - maint if lev > 1 else None
    want = min(kelly, risk_cap)
    limited = "marge/hefboom" if frac < want - 1e-9 else ("kelly" if kelly <= risk_cap else "risico per trade")
    return {"ok": True, "frac": frac, "notional": frac * equity, "margin": margin * equity, "lev": lev,
            "kelly": kelly, "risk_cap": risk_cap, "limited_by": limited,
            "liq_dist": liq, "mu_net": mu_net, "src": src, "risk_usd": frac * equity * sl_dist}


def proposal(coin: str, a: dict, mkt: dict, funding_h: float, equity: float, cfg: dict) -> dict:
    """Maak van een analyse een voorstel voor de Hyperliquid-markt `coin` (status + reden in gewone taal)."""
    px = mkt["px"]
    out: dict[str, Any] = {"coin": coin, "px": px, "dex": mkt.get("dex"), "volume": mkt.get("volume"),
                           "spread": mkt.get("spread"), "funding_h": funding_h, "status": "geen signaal", "why": []}
    wf, pick = a["wf"], a["pick"]
    out["wf"] = {k: wf.get(k) for k in ("n", "win", "mean", "pf", "sum", "years")}
    if not a["top"]:
        out["why"].append("geen enkele swingstrategie haalt de filters (genoeg trades, PF ≥ 1,2, beter dan vasthouden)")
        return out
    best = a["top"][0]
    out["best"] = {k: best[k] for k in ("strategy", "category", "n", "win", "pf", "open", "bars_since_buy")}
    if not pick:
        out["why"].append(f"beste strategie {best['strategy']} heeft nu geen vers koopsignaal")
        return out
    out["strategy"] = pick["strategy"]
    out["category"] = pick["category"]
    out["pick"] = {k: pick[k] for k in ("strategy", "category", "description", "score", "n", "win", "mean", "pf", "alpha",
                                        "cons", "tp_atr", "avg_bars", "exits", "last12", "bars_since_buy", "last_buy",
                                        "run_up", "run_up_atr")}
    sl_dist = SL_ATR * a["atr_pct"]
    tp_dist = pick["tp_atr"] * a["atr_pct"]
    hold_days = pick["avg_bars"] * 7 / 5
    funding_frac = funding_h * 24 * hold_days      # positief = long betaalt
    sz = size(pick, wf, sl_dist, max(0.0, funding_frac), equity, cfg, mkt.get("maxLeverage") or 10)
    out |= {"sl": px * (1 - sl_dist), "tp": px * (1 + tp_dist), "sl_dist": sl_dist, "tp_dist": tp_dist,
            "hold_days": hold_days, "max_days": max(3.0, math.ceil(2 * hold_days)), "funding_frac": funding_frac,
            "size": sz, "agree": a["agree"], "atr_pct": a["atr_pct"]}
    if sz.get("ok"):
        out["funding_usd"] = sz["notional"] * funding_frac
        out["liq"] = px * (1 - sz["liq_dist"]) if sz.get("liq_dist") else None
    # ---- poorten
    why = out["why"]
    if pick["run_up_atr"] is not None and pick["run_up_atr"] > 1.0:
        out["status"] = "loopt al"
        why.append(f"koers al {pick['run_up'] * 100:.1f}% boven de signaalinstap (>1 ATR): te laat")
        return out
    good_years = sum(1 for y in wf.get("years") or [] if y.get("sum", 0) > 0)
    if (wf.get("n", 0) < WF_MIN_N or not wf.get("pf") or wf["pf"] < WF_MIN_PF or (wf.get("mean") or 0) <= 0
            or good_years < WF_MIN_YEARS):
        out["status"] = "te weinig bewijs"
        pf = f"PF {wf['pf']:.2f}" if wf.get("pf") else "geen bruikbare PF"
        why.append(f"walk-forward: {wf.get('n', 0)} trades, {pf}, {good_years}/{len(wf.get('years') or [])} jaren winst "
                   f"(nodig: {WF_MIN_N} trades, PF {WF_MIN_PF}, {WF_MIN_YEARS} jaren)")
        return out
    if not sz.get("ok"):
        out["status"] = "te duur"
        why.append(sz["why"])
        return out
    if (mkt.get("volume") or 0) < cfg.get("min_volume", 250_000):
        out["status"] = "te illiquide"
        why.append(f"volume ${mkt.get('volume', 0) / 1e3:.0f}k per dag op Hyperliquid")
        return out
    out["status"] = "voorstel"
    why.append(f"{pick['strategy']} ({pick['category']}): {pick['n']} trades in 5 jaar, {pick['win'] * 100:.0f}% winst, "
               f"PF {pick['pf']:.2f}" if pick["pf"] else f"{pick['strategy']}: {pick['n']} trades")
    why.append(f"walk-forward (kiezen zonder te spieken): {wf['n']} trades, {wf['win'] * 100:.0f}% winst, PF {wf['pf']:.2f}, "
               f"{good_years}/{len(wf['years'])} jaren winst")
    why.append(f"signaal {pick['bars_since_buy']} handelsdag(en) oud, {a['agree']}/{len(a['top'])} topstrategieën long")
    if funding_frac > 0:
        why.append(f"funding kost ±{funding_frac * 100:.2f}% over {hold_days:.0f} dagen")
    return out
