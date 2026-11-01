"""API voor de ticker-grade (A+ tot F). Eigen router, zodat main.py maar één regel nodig heeft.

GET-antwoorden krijgen `Access-Control-Allow-Origin: *`, zodat ook de portfolio server (poort 5000) ze vanuit de
browser kan ophalen.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from .grade import FORMULA, grade_result, grades
from .grade_check import CHECKER, load_check
from .watchlist import list_name, load_watchlist, scan_cached, tv_to_yahoo

router = APIRouter(tags=["grade"])


def _cors(resp: Response) -> None:
    resp.headers["Access-Control-Allow-Origin"] = "*"


def _aliases() -> dict[str, str]:
    """TradingView-symbool en kale ticker -> Yahoo-ticker, uit de watchlist."""
    out: dict[str, str] = {}
    for g in load_watchlist():
        for s in g["symbols"]:
            y = s.get("yahoo")
            if not y:
                continue
            tv = (s.get("tv") or "").upper()
            out.setdefault(tv, y)
            if ":" in tv:
                out.setdefault(tv.split(":", 1)[1], y)
            out.setdefault(y.upper(), y)
    return out


def resolve(symbol: str, results: dict[str, Any]) -> str:
    s = symbol.strip()
    if s in results:
        return s
    al = _aliases()
    if s.upper() in al:
        return al[s.upper()]
    return tv_to_yahoo(s) or s.upper()


def _check_lite() -> dict[str, Any] | None:
    c = load_check()
    if not c or "summary" not in c:
        return None
    s = c["summary"]
    return {"finished": c.get("finished"), "verdict": s.get("verdict"), "ic": s.get("ic"), "spread": s.get("spread"),
            "observations": s.get("observations"), "tickers": s.get("tickers"), "from": s.get("from"), "to": s.get("to"),
            "all": s.get("all"), "by_grade": s.get("by_grade"), "horizon_days": s.get("horizon_days")}


@router.get("/api/grades")
def get_grades(response: Response, symbols: str | None = Query(None, description="Komma-gescheiden; leeg = alle"),
               list_: str | None = Query(None, alias="list"), group: str | None = None):
    """Grade per ticker (A+ tot F) voor alle gescande tickers, of een lijst/groep/selectie.

    `aliases` koppelt TradingView-symbolen en kale tickers (zoals de portfolio server ze gebruikt) aan de Yahoo-ticker."""
    _cors(response)
    results = scan_cached()["results"]
    syms = None
    if symbols:
        syms = [resolve(x, results) for x in symbols.split(",") if x.strip()]
    elif list_ or group:
        syms = list(dict.fromkeys(s["yahoo"] for g in load_watchlist()
                                  if (not list_ or list_name(g) == list_) and (not group or g["name"] == group)
                                  for s in g["symbols"] if s.get("yahoo")))
    out = grades(results, syms)
    al = {k: v for k, v in _aliases().items() if v in out}
    return {"grades": out, "aliases": al, "formula": FORMULA, "check": _check_lite()}


@router.get("/api/grade/check")
def get_check(response: Response):
    """Walk-forward check: hoe goed voorspelde de grade de maand erna, per grade (geen vooruitkijken)."""
    _cors(response)
    return {"result": load_check(), "status": CHECKER.status()}


@router.post("/api/grade/check")
def start_check(list_: str | None = Query(None, alias="list"), fee_bps: float = 5.0):
    """Start de walk-forward check op de tickers van de watchlist (of één lijst) op de achtergrond."""
    syms = list(dict.fromkeys(s["yahoo"] for g in load_watchlist() if not list_ or list_name(g) == list_
                              for s in g["symbols"] if s.get("yahoo")))
    if not syms:
        raise HTTPException(400, "Geen tickers in de watchlist")
    if not CHECKER.start(syms, fee_bps):
        raise HTTPException(409, "De check loopt al")
    return {"started": True, "tickers": len(syms)}


@router.get("/api/grade/{symbol:path}")
def get_grade(symbol: str, response: Response):
    """Grade van één ticker met alle werkende strategieën, welke er nu open staan en hun cijfers."""
    _cors(response)
    results = scan_cached()["results"]
    y = resolve(symbol, results)
    g = grade_result(results.get(y), detail=True)
    if g is None:
        raise HTTPException(404, f"Geen grade voor {symbol}: nog niet volledig gescand (vernieuw de watchlist)")
    return {"symbol": y, **g, "formula": FORMULA, "check": _check_lite()}
