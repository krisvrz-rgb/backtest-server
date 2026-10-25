"""Aanraders: welke tickers hebben nu een koopsignaal van een strategie die het verdient om gevolgd te worden?

Dit is geen beleggingsadvies, enkel een rangschikking van backtestresultaten. Per ticker bekijken we elke
strategie die op dit moment een open koopsignaal heeft en geven punten voor bewijs dat ze werkt:

* houdbaar: verslaat buy & hold in de leerperiode (2 jaar) én in het jaar erna dat ze niet zag;
* verslaat buy & hold over de laatste 2 jaar, het laatste jaar en de laatste 6 maanden;
* werkt op veel tickers van dezelfde lijst (minder kans dat het toeval is);
* het signaal is vers (je stapt niet pas in als de beweging al gebeurd is);
* de trade loopt nog niet te ver weg van de instapkoers.

Daarbovenop een fijnere verdeling: de voorsprong op buy & hold in het testjaar en over 2 jaar, en het
percentage winnende trades. De strategie met de meeste punten bepaalt de score van de ticker. Hoeveel andere goede strategieën
tegelijk een open koopsignaal hebben, staat erbij als "eensgezindheid".
"""
from __future__ import annotations

import math
from typing import Any

from .robust import cross_ticker
from .strategies import CATEGORIES, STRATEGIES

PERIOD_POINTS = {"2j": 1.5, "1j": 1.0, "6m": 0.5}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _beats(r: dict[str, Any], period: str, strategy: str) -> tuple[bool | None, float | None, float | None]:
    per = (r.get("periods") or {}).get(period) or {}
    ret = ((r.get("returns") or {}).get(period) or {}).get(strategy)
    if ret is None or "error" in per:
        return None, None, None
    bh = per.get("buy_and_hold_return", 0.0)
    return ret > bh, ret, bh


def score_strategy(r: dict[str, Any], name: str, sig: dict[str, Any], robust: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pts, why = 0.0, []
    o, oret = r.get("oos") or {}, (r.get("oos_returns") or {}).get(name)
    if oret and "error" not in o and None not in oret and o.get("bh_train") is not None:
        if oret[0] > o["bh_train"] and oret[1] > o["bh_test"]:
            pts += 3
            why.append(("good", "houdt stand"))
        elif oret[1] > o["bh_test"]:
            pts += 1
            why.append(("mixed", "testjaar beter dan B&H"))
        else:
            why.append(("bad", "testjaar slechter dan B&H"))
    periods = {}
    for p, w in PERIOD_POINTS.items():
        b, ret, bh = _beats(r, p, name)
        periods[p] = {"return": ret, "bh": bh, "beats": b}
        if b:
            pts += w
    beaten = [p for p in PERIOD_POINTS if periods[p]["beats"]]
    if beaten:
        why.append(("good", "beter dan B&H: " + ", ".join(beaten)))
    if (periods["2j"]["return"] or 0) <= 0:
        pts -= 2
        why.append(("bad", "verlies over 2 jaar"))
    rb = robust.get(name)
    if rb and rb["tickers"] >= 3:
        share = rb["beat_share"] or 0
        if share >= 0.5:
            pts += 1.5
            why.append(("good", f"werkt op {round(share * 100)}% van de tickers"))
        elif share >= 0.3:
            pts += 0.5
    bars = sig.get("bars")
    if bars is not None:
        if bars <= 3:
            pts += 1.5
            why.append(("new", f"vers signaal ({bars} d)"))
        elif bars <= 10:
            pts += 0.75
        elif bars > 40:
            pts -= 0.5
    tr = sig.get("trade_return")
    if tr is not None and tr > 0.25:
        pts -= 1
        why.append(("mixed", f"al {round(tr * 100)}% gestegen sinds het signaal"))
    if (sig.get("trades") or 0) < 3:
        pts -= 0.5
        why.append(("mixed", "weinig trades in de test"))
    # Fijnere verdeling (zodat niet tien tickers gelijk op het maximum staan): hoe groot de voorsprong op buy & hold
    # was in het testjaar en over 2 jaar, en hoe vaak de trades winst gaven.
    if oret and None not in oret and o.get("bh_test") is not None:
        pts += _clamp(oret[1] - o["bh_test"], -0.5, 1.0)
    r2, bh2 = periods["2j"]["return"], periods["2j"]["bh"]
    if r2 is not None and bh2 is not None and r2 > -1 and bh2 > -1:
        pts += 0.5 * _clamp(math.log((1 + r2) / (1 + bh2)), -1.0, 1.0)
    if sig.get("win_rate") is not None:
        pts += sig["win_rate"] - 0.5
    cls = STRATEGIES.get(name)
    return {"strategy": name, "category_label": CATEGORIES.get(cls.category, "") if cls else "", "score": round(pts, 2),
            "why": [{"kind": k, "text": t} for k, t in why], "signal": sig, "periods": periods,
            "oos": {"train": oret[0], "test": oret[1]} if oret else None}


def picks(results: dict[str, dict[str, Any]], tickers: list[dict[str, Any]], min_score: float = 0.0) -> dict[str, Any]:
    """tickers: [{yahoo, tv, group}] uit de watchlist. Geeft de tickers gesorteerd op score."""
    syms = [t["yahoo"] for t in tickers]
    robust = {x["strategy"]: x for x in cross_ticker(results, syms, "2j")["rows"]}
    rows, missing = [], 0
    for t in tickers:
        r = results.get(t["yahoo"]) or {}
        if r.get("open_now") is None:
            missing += 1
            continue
        scored = sorted((score_strategy(r, n, sig, robust) for n, sig in r["open_now"].items()),
                        key=lambda x: -x["score"])
        good = [x for x in scored if x["score"] >= 3]
        if not scored or scored[0]["score"] < min_score:
            continue
        per = (r.get("periods") or {}).get("2j") or {}
        rows.append({
            **t, "score": scored[0]["score"], "best": scored[0], "others": scored[1:4],
            "open_count": len(scored), "agree": len(good),
            "last_close": per.get("last_close") or r.get("last_close"), "last_date": per.get("last_date") or r.get("last_date"),
            "change_pct": per.get("change_pct"), "oos_verdict": (r.get("oos") or {}).get("verdict_label"),
            "bh_2j": per.get("buy_and_hold_return"),
        })
    rows.sort(key=lambda x: (-x["score"], -x["agree"]))
    return {"rows": rows, "tickers": len(tickers), "missing": missing}
