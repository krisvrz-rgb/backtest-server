"""Automatische strategiekeuze per markttoestand, eerlijk getest met walk-forward.

Achteraf de beste strategie kiezen lukt altijd, maar dat is kijken naar de uitslag (overfitting).
Hier kiest de server op elk beslismoment alleen met wat hij toen kon weten:

1. Markttoestand bepalen op de beslisdag: stijgend, zijwaarts of dalend (koers t.o.v. SMA 200 en de
   richting van die SMA) en of de beweeglijkheid hoog is.
2. In de leerperiode ervoor (standaard 3 jaar) kijken welke strategie het best werkte op de dagen met
   diezelfde markttoestand.
3. Die strategie volgen tot het volgende beslismoment (standaard elk kwartaal).

Het resultaat van al die stukjes samen is een echte test buiten de leerperiode: zo had het gelopen als
je deze aanpak vanaf de startdatum had gevolgd.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import _simulate, compute_metrics, last_signals
from .strategies import CATEGORIES, STRATEGIES

REGIMES = {"up": "Stijgend", "side": "Zijwaarts", "down": "Dalend"}
OBJECTIVES = ("total_return", "sharpe")


def regimes(prices: pd.DataFrame, length: int = 200, slope: int = 20) -> pd.DataFrame:
    """Markttoestand per dag (alleen op basis van koersen tot en met die dag)."""
    close = prices["close"]
    sma = close.rolling(length, min_periods=length // 2).mean()
    rising = sma > sma.shift(slope)
    trend = pd.Series("side", index=close.index)
    trend[(close > sma) & rising] = "up"
    trend[(close < sma) & ~rising] = "down"
    vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    high_vol = vol > vol.rolling(252, min_periods=60).median() * 1.5
    return pd.DataFrame({"trend": trend.where(sma.notna()), "high_vol": high_vol.fillna(False), "volatility": vol})


def _strategy_paths(prices: pd.DataFrame, active: pd.Series, categories: list[str] | None):
    """Positie (na uitvoering op de volgende open) en dagrendement per strategie over de hele historie."""
    open_, close = prices["open"], prices["close"]
    # Rendement van één dag belegd zijn, met dezelfde opbouw als de engine: overnight + intraday.
    overnight = (open_ / close.shift(1) - 1).fillna(0.0)
    intraday = close / open_ - 1
    out = {}
    for name, cls in STRATEGIES.items():
        if cls.category == "benchmark" or (categories and cls.category not in categories):
            continue
        try:
            sig = cls().signals(prices, active).reindex(prices.index).fillna(0.0).clip(0, 1)
        except Exception:  # noqa: BLE001
            continue
        pos = sig.shift(1).fillna(0.0)
        ret = (1 + pos.shift(1).fillna(0.0) * overnight) * (1 + pos * intraday) - 1
        out[name] = (sig, pos, ret)
    return out


def _score(rets: pd.Series, objective: str) -> float:
    if len(rets) < 20:
        return float("-inf")
    if objective == "sharpe":
        sd = rets.std()
        return float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    return float(np.prod(1 + rets.to_numpy()) - 1)


def walk_forward(prices: pd.DataFrame, start: str, initial_capital: float = 10_000.0, fee_bps: float = 5.0,
                 train_years: float = 3.0, step: str = "QS", objective: str = "total_return",
                 by_regime: bool = True, categories: list[str] | None = None) -> dict:
    """Kies op elk beslismoment de strategie die in de leerperiode het best werkte (bij dezelfde markttoestand)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Doel moet een van {OBJECTIVES} zijn")
    start_ts = pd.Timestamp(start)
    train = pd.Timedelta(days=int(train_years * 365.25))
    reg = regimes(prices)
    # Signalen over de hele historie, zodat de leerperiode vóór de startdatum ook telt.
    active = pd.Series(True, index=prices.index)
    paths = _strategy_paths(prices, active, categories)
    if not paths:
        raise ValueError("Geen strategieën om uit te kiezen")
    period = prices.index[prices.index >= start_ts]
    if len(period) < 30:
        raise ValueError("Te weinig koersen na de startdatum")
    decision_days = pd.date_range(period[0], period[-1], freq=step)
    decisions = sorted({period[period.searchsorted(d)] for d in decision_days if period.searchsorted(d) < len(period)}
                       | {period[0]})

    chosen = pd.Series(index=period, dtype=object)
    steps = []
    for i, day in enumerate(decisions):
        end = decisions[i + 1] if i + 1 < len(decisions) else None
        # Beslissen op de slot van de dag ervoor: alleen data tot en met gisteren.
        loc = prices.index.get_loc(day)
        if loc == 0:
            continue
        known = prices.index[loc - 1]
        window = prices.index[(prices.index > known - train) & (prices.index <= known)]
        state = reg.loc[known]
        trend = state["trend"] if isinstance(state["trend"], str) else "side"
        mask = pd.Series(True, index=window)
        if by_regime:
            same = reg.loc[window, "trend"] == trend
            if same.sum() >= 60:  # genoeg vergelijkbare dagen, anders de hele leerperiode
                mask = same
        best, best_score = None, float("-inf")
        for name, (_, _, ret) in paths.items():
            sc = _score(ret.loc[window][mask.to_numpy()], objective)
            if sc > best_score:
                best, best_score = name, sc
        seg = period[(period >= day) & (period < end)] if end is not None else period[period >= day]
        chosen.loc[seg] = best
        steps.append({"date": day.date().isoformat(), "regime": trend, "regime_label": REGIMES.get(trend, trend),
                      "high_vol": bool(state["high_vol"]), "strategy": best,
                      "category_label": CATEGORIES.get(STRATEGIES[best].category, ""),
                      "train_score": best_score, "train_days": int(mask.sum()), "bars": len(seg)})

    chosen = chosen.ffill().bfill()
    sig = pd.Series([paths[n][0].loc[d] for d, n in chosen.items()], index=period, dtype=float)
    p = prices.loc[period]
    position = sig.shift(1).fillna(0.0)
    equity, _ = _simulate(p["open"], p["close"], position, initial_capital, fee_bps)
    hold = pd.Series(1.0, index=period)
    hold.iloc[0] = 0.0
    bench, _ = _simulate(p["open"], p["close"], hold, initial_capital, fee_bps)

    # Rendement per stap (buiten de leerperiode) voor de tabel.
    for s in steps:
        seg = equity[equity.index >= pd.Timestamp(s["date"])]
        nxt = next((x["date"] for x in steps if x["date"] > s["date"]), None)
        if nxt:
            seg = seg[seg.index < pd.Timestamp(nxt)]
        prev = equity[equity.index < pd.Timestamp(s["date"])]
        base = prev.iloc[-1] if len(prev) else initial_capital
        s["return"] = float(seg.iloc[-1] / base - 1) if len(seg) else 0.0
        bseg = bench[bench.index >= pd.Timestamp(s["date"])]
        if nxt:
            bseg = bseg[bseg.index < pd.Timestamp(nxt)]
        bprev = bench[bench.index < pd.Timestamp(s["date"])]
        bbase = bprev.iloc[-1] if len(bprev) else initial_capital
        s["buy_and_hold_return"] = float(bseg.iloc[-1] / bbase - 1) if len(bseg) else 0.0

    metrics = compute_metrics(equity)
    metrics.update({"final_equity": float(equity.iloc[-1]), "net_profit": float(equity.iloc[-1] - initial_capital),
                    "exposure": float((position != 0).mean()), "switches": int((chosen != chosen.shift(1)).sum() - 1),
                    **last_signals(sig)})
    bm = compute_metrics(bench) | {"net_profit": float(bench.iloc[-1] - initial_capital)}
    usage = chosen.value_counts()
    current = reg.iloc[-1]
    return {
        "period": {"start": period[0].date().isoformat(), "end": period[-1].date().isoformat()},
        "settings": {"train_years": train_years, "step": step, "objective": objective, "by_regime": by_regime},
        "metrics": metrics, "buy_and_hold": bm,
        "beats_buy_and_hold": metrics["total_return"] > bm["total_return"],
        "beats_bh_risk": metrics["sharpe"] > bm["sharpe"] and metrics["max_drawdown"] > bm["max_drawdown"],
        "current": {"regime": current["trend"], "regime_label": REGIMES.get(current["trend"], current["trend"]),
                    "high_vol": bool(current["high_vol"]), "strategy": chosen.iloc[-1],
                    "in_position": bool(sig.iloc[-1] > 0)},
        "regime_share": {REGIMES[k]: float((reg.loc[period, "trend"] == k).mean()) for k in REGIMES},
        "usage": [{"strategy": n, "share": float(c / len(chosen))} for n, c in usage.head(10).items()],
        "steps": steps,
        "equity_curve": [{"date": d.date().isoformat(), "equity": round(float(e), 2), "benchmark": round(float(b), 2)}
                         for d, e, b in zip(equity.index, equity, bench)],
    }
