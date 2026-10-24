"""Houdt de beste strategie stand? En welke strategie was elke maand de beste?

Per ticker draaien alle strategieën één keer over de laatste 3 jaar. Uit die kapitaalcurves halen we:

* Houdbaarheid (out-of-sample): kies de beste strategie op de eerste 2 jaar (de leerperiode) en kijk hoe
  ze het deed in het laatste jaar, dat ze bij die keuze niet gezien heeft. Zo zie je of de winnaar echt
  werkt of toevallig goed paste op het verleden.
* Per maand: de rendementen van elke strategie per kalendermaand over de laatste 24 maanden. Posities
  lopen gewoon door over de maandgrens, zoals in een echte backtest; per maand wordt de winst in die
  maand vergeleken.

De rendementen per strategie worden ook bewaard, zodat de server over alle tickers samen kan tellen welke
strategieën op veel tickers tegelijk werken (zie `cross_ticker`).
"""
from __future__ import annotations

from datetime import date
from statistics import median
from typing import Any

import pandas as pd

from .data import WARMUP_DAYS
from .engine import run_backtest
from .strategies import CATEGORIES, STRATEGIES

TRAIN_YEARS, TEST_YEARS, MONTHS = 2, 1, 24
MIN_TRAIN_DAYS = 180


def _ret(eq: pd.Series, a: pd.Timestamp, b: pd.Timestamp | None = None) -> float | None:
    """Rendement van de curve tussen de slot vóór `a` en de laatste slot vóór `b` (of het einde)."""
    before = eq[eq.index < a]
    seg = eq[eq.index < b] if b is not None else eq
    if not len(seg) or seg.index[-1] < a:
        return None
    base = before.iloc[-1] if len(before) else eq.iloc[0]
    return float(seg.iloc[-1] / base - 1)


def analyze(prices: pd.DataFrame, fee_bps: float = 5.0, today: date | None = None) -> dict[str, Any]:
    """Houdbaarheidstest en beste strategie per maand voor één ticker (prices inclusief opwarmperiode)."""
    today = today or prices.index[-1].date()
    end = pd.Timestamp(today)
    test_start = end - pd.DateOffset(years=TEST_YEARS)
    run_start = max(end - pd.DateOffset(years=TRAIN_YEARS + TEST_YEARS),
                    prices.index[0] + pd.Timedelta(days=WARMUP_DAYS // 2))
    p = prices[prices.index >= run_start - pd.Timedelta(days=WARMUP_DAYS)]
    if not (p.index >= run_start).any():
        raise ValueError("Te weinig koersen")

    curves: dict[str, pd.Series] = {}
    bh = None
    for name, cls in STRATEGIES.items():
        try:
            res = run_backtest(p, cls(), 10_000.0, fee_bps, start=run_start)
        except Exception:  # noqa: BLE001
            continue
        if cls.category == "benchmark":
            bh = res.benchmark
        else:
            curves[name] = res.equity
    if bh is None or not curves:
        raise ValueError("Geen strategieën gelukt")
    first = bh.index[0]

    # ---- houdbaarheid: kiezen op de leerperiode, beoordelen op het testjaar
    train = {n: _ret(c, first, test_start) for n, c in curves.items()}
    test = {n: _ret(c, test_start) for n, c in curves.items()}
    oos: dict[str, Any]
    train_days = (test_start - first).days
    if train_days < MIN_TRAIN_DAYS or bh.index[-1] < test_start:
        oos = {"error": "Te korte historie voor een leer- en testperiode"}
    else:
        bh_train, bh_test = _ret(bh, first, test_start), _ret(bh, test_start)
        ranked_train = sorted((n for n in curves if train[n] is not None), key=lambda n: -train[n])
        ranked_test = sorted((n for n in curves if test[n] is not None), key=lambda n: -test[n])
        best = ranked_train[0]
        top5 = ranked_train[:5]
        test_rank = ranked_test.index(best) + 1
        oos = {
            "train": {"start": first.date().isoformat(), "end": (test_start - pd.Timedelta(days=1)).date().isoformat()},
            "test": {"start": test_start.date().isoformat(), "end": bh.index[-1].date().isoformat()},
            "strategy": best, "category_label": CATEGORIES.get(STRATEGIES[best].category, ""),
            "train_return": train[best], "test_return": test[best],
            "bh_train": bh_train, "bh_test": bh_test,
            "test_rank": test_rank, "of": len(ranked_test),
            "beats_bh_test": test[best] > bh_test,
            # Gemiddelde van de top 5 uit de leerperiode in het testjaar: minder afhankelijk van één toevalstreffer.
            "top5": [{"strategy": n, "train_return": train[n], "test_return": test[n]} for n in top5],
            "top5_avg_test": float(sum(test[n] for n in top5) / len(top5)),
            "test_best": {"strategy": ranked_test[0], "test_return": test[ranked_test[0]],
                          "train_rank": ranked_train.index(ranked_test[0]) + 1},
        }
        # Oordeel in gewone taal.
        if oos["beats_bh_test"] and test_rank <= max(1, len(ranked_test) // 4):
            oos["verdict"], oos["verdict_label"] = "good", "Houdt stand"
        elif oos["beats_bh_test"] or test[best] > 0:
            oos["verdict"], oos["verdict_label"] = "mixed", "Deels"
        else:
            oos["verdict"], oos["verdict_label"] = "bad", "Hield geen stand"

    # ---- per maand (laatste 24 kalendermaanden, de lopende maand inbegrepen)
    months = pd.period_range(end=end.to_period("M"), periods=MONTHS, freq="M")
    monthly = []
    for m in months:
        a, b = m.start_time, (m + 1).start_time
        if bh.index[-1] < a or bh.index[0] >= b:
            continue
        rets = {n: r for n, c in curves.items() if (r := _ret(c, a, b)) is not None}
        if not rets:
            continue
        order = sorted(rets, key=lambda n: -rets[n])
        bh_m = _ret(bh, a, b)
        monthly.append({
            "month": str(m), "best": order[0], "best_return": rets[order[0]],
            "category_label": CATEGORIES.get(STRATEGIES[order[0]].category, ""),
            "bh_return": bh_m, "beat_count": sum(1 for r in rets.values() if bh_m is not None and r > bh_m),
            "top3": [{"strategy": n, "return": rets[n]} for n in order[:3]],
            "partial": m == pd.Timestamp(date.today()).to_period("M"),
        })
    # Welke strategieën het vaakst de maand wonnen.
    wins: dict[str, int] = {}
    for row in monthly:
        wins[row["best"]] = wins.get(row["best"], 0) + 1
    month_wins = [{"strategy": n, "months": k} for n, k in sorted(wins.items(), key=lambda x: -x[1])[:10]]

    return {
        "oos": oos, "monthly": monthly, "month_wins": month_wins,
        # Compact per strategie: leer- en testrendement (voor de vergelijking over alle tickers).
        "oos_returns": {n: [round(train[n], 4) if train[n] is not None else None,
                            round(test[n], 4) if test[n] is not None else None] for n in curves},
    }


def oos_summary(oos: dict[str, Any] | None) -> dict[str, Any] | None:
    """Wat de watchlist-tabel nodig heeft (de rest haalt de pagina op wanneer je een ticker openklapt)."""
    if not oos or "error" in oos:
        return oos
    return {k: oos[k] for k in ("strategy", "train_return", "test_return", "bh_train", "bh_test", "test_rank", "of",
                                "beats_bh_test", "verdict", "verdict_label", "top5_avg_test", "test")}


def cross_ticker(results: dict[str, dict[str, Any]], symbols: list[str], period: str) -> dict[str, Any]:
    """Welke strategieën werken op veel tickers tegelijk?

    Per strategie, over de gekozen tickers: op hoeveel tickers ze buy & hold verslaat in de gekozen periode,
    het mediane en gemiddelde rendement, de gemiddelde plaats (als percentiel, 0% = altijd de beste) en of ze
    ook buiten de leerperiode bleef winnen (verslaat buy & hold in zowel het leer- als het testjaar).
    """
    stats: dict[str, dict[str, Any]] = {}
    used = oos_used = 0
    for sym in symbols:
        r = results.get(sym) or {}
        per = (r.get("periods") or {}).get(period) or {}
        rets = (r.get("returns") or {}).get(period)
        if rets and "error" not in per:
            used += 1
            bh = per.get("buy_and_hold_return", 0.0)
            order = sorted(rets, key=lambda n: -rets[n])
            n_strats = len(order)
            for i, name in enumerate(order):
                s = stats.setdefault(name, {"returns": [], "beats": 0, "pct": [], "top10": 0, "oos_both": 0, "oos_n": 0,
                                            "oos_test_beats": 0})
                s["returns"].append(rets[name])
                s["beats"] += rets[name] > bh
                s["pct"].append(i / max(1, n_strats - 1))
                s["top10"] += i < max(1, n_strats // 10)
        o, oret = r.get("oos") or {}, r.get("oos_returns")
        if oret and "error" not in o and o.get("bh_train") is not None:
            oos_used += 1
            for name, (tr, te) in oret.items():
                if tr is None or te is None:
                    continue
                s = stats.setdefault(name, {"returns": [], "beats": 0, "pct": [], "top10": 0, "oos_both": 0, "oos_n": 0,
                                            "oos_test_beats": 0})
                s["oos_n"] += 1
                s["oos_test_beats"] += te > o["bh_test"]
                s["oos_both"] += tr > o["bh_train"] and te > o["bh_test"]
    rows = []
    for name, s in stats.items():
        cls = STRATEGIES.get(name)
        n = len(s["returns"])
        rows.append({
            "strategy": name, "category_label": CATEGORIES.get(cls.category, "") if cls else "",
            "tickers": n, "beats": s["beats"], "beat_share": s["beats"] / n if n else None,
            "median_return": median(s["returns"]) if n else None,
            "avg_return": sum(s["returns"]) / n if n else None,
            "avg_place": sum(s["pct"]) / n if n else None, "top10": s["top10"],
            "oos_n": s["oos_n"], "oos_both": s["oos_both"], "oos_test_beats": s["oos_test_beats"],
            "oos_share": s["oos_both"] / s["oos_n"] if s["oos_n"] else None,
        })
    rows.sort(key=lambda x: (x["beat_share"] or 0, x["median_return"] or 0), reverse=True)
    return {"period": period, "tickers": used, "oos_tickers": oos_used, "rows": rows}
