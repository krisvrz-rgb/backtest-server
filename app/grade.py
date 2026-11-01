"""Eén cijfer per ticker, van A+ tot F: werken de strategieën voor dit aandeel, en hoeveel ervan staan nu op koop?

Dit is geen beleggingsadvies, enkel een samenvatting van backtestresultaten.

De formule (dezelfde in de walk-forward check van grade_check.py, zodat die check eerlijk is):

1. Elke strategie wordt beoordeeld zoals in de houdbaarheidstest: rendement in de leerperiode (2 jaar) en in het
   testjaar erna. Een strategie *werkt* voor deze ticker als ze in allebei buy & hold verslaat én in het testjaar
   zelf winst maakte. De beste in de leerperiode kiezen is dus niet genoeg: ze moet ook het jaar erna blijven werken.
2. Gewicht van een strategie die werkt = voorsprong op buy & hold in het testjaar (log, max 0,5) + de helft van de
   jaarlijkse voorsprong in de leerperiode (max 0,5). Hoe groter en hoe recenter de voorsprong, hoe meer ze telt.
3. Stemming = gewogen aandeel van die strategieën dat nu een open koopsignaal heeft (0 tot 1). Elke strategie zegt
   "koop" (open) of "cash" (geen signaal); dit is het gewogen gemiddelde van hun voorspelling.
4. Vertrouwen = 1 − e^(−totaalgewicht / 0,75): met één zwakke strategie is de stemming weinig waard,
   met veel sterke strategieën veel.
5. Score = 100 × stemming × (0,3 + 0,7 × vertrouwen). Grade: A+ ≥ 80, A ≥ 65, B ≥ 50, C ≥ 35, D ≥ 20, F daaronder.

Hoe goed die grade in het verleden de maand erna voorspelde, meet grade_check.py (walk-forward, zonder vooruitkijken).
"""
from __future__ import annotations

import math
from typing import Any

from .strategies import CATEGORIES, STRATEGIES

GRADES = [("A+", 80), ("A", 65), ("B", 50), ("C", 35), ("D", 20), ("F", 0)]
MAX_EDGE = 0.5
CONF_SCALE = 0.75
BASE_CONF = 0.3

FORMULA = [
    "Een strategie werkt als ze buy & hold verslaat in de leerperiode (2 jaar) én in het testjaar erna, en in dat "
    "testjaar winst maakte. De beste in het verleden kiezen telt dus niet: ze moet ook daarna blijven werken.",
    "Gewicht = voorsprong op buy & hold in het testjaar (max 0,5) + de helft van de jaarlijkse voorsprong in de "
    "leerperiode (max 0,5).",
    "Stemming = gewogen aandeel van de werkende strategieën dat nu een open koopsignaal heeft.",
    "Vertrouwen groeit met het totaalgewicht: 1 − e^(−gewicht / 0,75).",
    "Score = 100 × stemming × (0,3 + 0,7 × vertrouwen). A+ ≥ 80, A ≥ 65, B ≥ 50, C ≥ 35, D ≥ 20, anders F.",
]


def letter(score: float) -> str:
    return next(g for g, lo in GRADES if score >= lo)


def _lex(r: float, bh: float) -> float:
    """Voorsprong op buy & hold als verschil in log-rendement (symmetrisch, en grote uitschieters wegen minder)."""
    return math.log1p(max(r, -0.999)) - math.log1p(max(bh, -0.999))


def weight(train: float, test: float, bh_train: float, bh_test: float, train_years: float = 2.0) -> float:
    """Gewicht van een strategie (0 = werkt niet voor deze ticker)."""
    ex_te = _lex(test, bh_test)
    ex_tr = _lex(train, bh_train) / max(train_years, 0.5)
    if ex_te <= 0 or ex_tr <= 0 or test <= 0:
        return 0.0
    return min(ex_te, MAX_EDGE) + 0.5 * min(ex_tr, MAX_EDGE)


def score(weights: list[float], open_flags: list[bool]) -> dict[str, Any]:
    """Score en grade uit de gewichten van de strategieën en of ze nu open staan (gedeeld met de walk-forward check)."""
    total = sum(weights)
    if total <= 0:
        return {"score": 0.0, "grade": "F", "vote": 0.0, "confidence": 0.0, "weight": 0.0}
    w_open = sum(w for w, o in zip(weights, open_flags) if o)
    vote = w_open / total
    conf = 1 - math.exp(-total / CONF_SCALE)
    s = 100 * vote * (BASE_CONF + (1 - BASE_CONF) * conf)
    return {"score": round(s, 1), "grade": letter(s), "vote": round(vote, 3), "confidence": round(conf, 3),
            "weight": round(total, 3)}


def _reason(g: dict[str, Any]) -> str:
    if g.get("working", 0) == 0:
        return "Geen enkele strategie bleef voor deze ticker werken buiten de leerperiode."
    if g["working"] == 1:
        return ("De enige werkende strategie staat op koop, maar met één strategie is er weinig vertrouwen."
                if g["open_working"] else "Maar één strategie werkt, en die staat niet op koop.")
    if g["open_working"] == 0:
        return f"{g['working']} strategieën werken, maar geen enkele staat nu op koop."
    return (f"{g['open_working']} van de {g['working']} werkende strategieën staan op koop "
            f"({round(g['vote'] * 100)}% van het gewicht), vertrouwen {round(g['confidence'] * 100)}%.")


def grade_result(r: dict[str, Any] | None, detail: bool = False) -> dict[str, Any] | None:
    """Grade van één ticker uit zijn scanresultaat (data/scan.json). None als er geen houdbaarheidstest is."""
    if not r:
        return None
    o, oret = r.get("oos") or {}, r.get("oos_returns")
    if not oret or "error" in o or o.get("bh_train") is None or o.get("bh_test") is None:
        return None
    try:
        from datetime import date
        tr = o.get("train") or {}
        train_years = (date.fromisoformat(tr["end"]) - date.fromisoformat(tr["start"])).days / 365.25
    except (KeyError, TypeError, ValueError):
        train_years = 2.0
    open_now = r.get("open_now") or {}
    rows = []
    for name, pair in oret.items():
        if not pair or pair[0] is None or pair[1] is None:
            continue
        w = weight(pair[0], pair[1], o["bh_train"], o["bh_test"], train_years)
        rows.append((name, pair[0], pair[1], w, name in open_now))
    working = [x for x in rows if x[3] > 0]
    g = score([x[3] for x in working], [x[4] for x in working])
    g |= {"working": len(working), "open_working": sum(1 for x in working if x[4]), "strategies": len(rows),
          "open_total": len(open_now), "scanned_at": r.get("scanned_at"), "test": o.get("test")}
    g["reason"] = _reason(g)
    if not detail:
        return g

    def info(name: str, train: float, test: float, w: float, is_open: bool) -> dict[str, Any]:
        cls = STRATEGIES.get(name)
        sig = open_now.get(name) or {}
        return {"strategy": name, "category_label": CATEGORIES.get(cls.category, "") if cls else "",
                "train_return": train, "test_return": test, "weight": round(w, 4),
                "share": round(w / g["weight"], 4) if g["weight"] and w else 0.0, "open": is_open,
                "since": sig.get("date"), "bars": sig.get("bars"), "entry": sig.get("entry"),
                "trade_return": sig.get("trade_return"), "win_rate": sig.get("win_rate"), "trades": sig.get("trades"),
                "return_2j": ((r.get("returns") or {}).get("2j") or {}).get(name)}

    working.sort(key=lambda x: (not x[4], -x[3]))
    g["working_list"] = [info(*x) for x in working]
    # Open koopsignalen van strategieën die (nog) niet bewezen werken: tellen niet mee, wel tonen.
    g["open_other"] = sorted((info(*x) for x in rows if x[3] == 0 and x[4]), key=lambda x: -(x["test_return"] or 0))
    g["bh_train"], g["bh_test"], g["train"] = o["bh_train"], o["bh_test"], o.get("train")
    return g


def grades(results: dict[str, dict[str, Any]], symbols: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Grade per ticker (zonder details) voor alle of de gekozen tickers."""
    syms = symbols if symbols is not None else list(results)
    out = {}
    for s in syms:
        g = grade_result(results.get(s))
        if g:
            out[s] = {k: g[k] for k in ("grade", "score", "vote", "confidence", "working", "open_working",
                                        "open_total", "reason", "scanned_at")}
    return out
