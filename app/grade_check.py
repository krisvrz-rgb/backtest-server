"""Walk-forward check van de grade: hoe goed voorspelde ze in het verleden de maand erna?

Per ticker draaien alle strategieën één keer over de hele historie. Daarna doen we alsof we op een reeks datums
in het verleden staan (de laatste handelsdag van elke maand): met alleen de koersen tot die dag berekenen we de grade precies zoals
grade.py dat vandaag doet (leerperiode = de 2 jaar tot een jaar geleden, testjaar = het laatste jaar, open signalen
op die dag). Dan kijken we wat het aandeel de 21 handelsdagen erna deed, gekocht op de opening van de dag erna.

Zo zie je per grade het gemiddelde rendement van de maand erna en hoe vaak dat positief was, vergeleken met alle
maanden samen. Ook per ticker vergeleken met het eigen gemiddelde van die ticker (zodat een lijst met vooral
stijgers de grade niet beter doet lijken dan ze is), en de rangcorrelatie (IC) tussen score en rendement erna
over de tickers op dezelfde datum.

Gebruik op de server: `python -m app.grade_check` of POST /api/grade/check. Resultaat in data/grade-check.json.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .data import DATA_DIR, WARMUP_DAYS, get_prices
from .engine import run_backtest
from .grade import GRADES, NONE, score, weight
from .strategies import STRATEGIES

CHECK_FILE = DATA_DIR / "grade-check.json"
START = "2015-01-01"
HORIZON = 21    # rendement over de 21 handelsdagen erna
HORIZON_LONG = 63
CAP_LO, CAP_HI = 0.5, 1.0


def ticker_observations(prices: pd.DataFrame, fee_bps: float = 5.0) -> list[dict[str, Any]]:
    """Meetpunten (datum, score, grade, rendement erna) voor één ticker. prices inclusief opwarmperiode."""
    run_start = prices.index[0] + pd.Timedelta(days=WARMUP_DAYS)
    eqs, opens, bh = [], [], None
    for name, cls in STRATEGIES.items():
        try:
            res = run_backtest(prices, cls(), 10_000.0, fee_bps, start=run_start)
        except Exception:  # noqa: BLE001
            continue
        if cls.category == "benchmark":
            bh = res.benchmark
            continue
        eqs.append(res.equity.to_numpy())
        # Open op dag t = positie de dag erna (het signaal op de slot van t wordt op de open van t+1 gevuld).
        opens.append(res.position.shift(-1).fillna(0).to_numpy() > 0)
    if bh is None or not eqs:
        return []
    idx = bh.index
    E, O, B = np.vstack(eqs), np.vstack(opens), bh.to_numpy()
    p = prices.loc[idx]
    o_px, c_px = p["open"].to_numpy(), p["close"].to_numpy()
    first_eval = idx.searchsorted(idx[0] + pd.DateOffset(years=3))
    obs = []
    # Meetpunt = laatste handelsdag van elke maand (voor alle tickers dezelfde datums, zodat ze vergelijkbaar zijn).
    month_end = np.flatnonzero(idx.to_period("M")[1:] != idx.to_period("M")[:-1])
    for i in month_end[(month_end >= first_eval) & (month_end < len(idx) - HORIZON - 1)]:
        t = idx[i]
        f = idx.searchsorted(t - pd.DateOffset(years=3))
        ts = idx.searchsorted(t - pd.DateOffset(years=1))
        base_f = max(f - 1, 0)
        if ts - 1 <= base_f:
            continue
        train = E[:, ts - 1] / E[:, base_f] - 1
        test = E[:, i] / E[:, ts - 1] - 1
        bh_tr, bh_te = B[ts - 1] / B[base_f] - 1, B[i] / B[ts - 1] - 1
        years = (idx[ts - 1] - idx[base_f]).days / 365.25
        ws = [weight(a, b, bh_tr, bh_te, years) for a, b in zip(train, test)]
        work = [k for k, w in enumerate(ws) if w > 0]
        g = score([ws[k] for k in work], [bool(O[k, i]) for k in work])
        entry = o_px[i + 1]
        row = {"date": t.date().isoformat(), "score": g["score"], "grade": g["grade"], "working": len(work),
               "fwd": float(c_px[i + HORIZON] / entry - 1)}
        if i + HORIZON_LONG < len(idx):
            row["fwd3"] = float(c_px[i + HORIZON_LONG] / entry - 1)
        obs.append(row)
    return obs


def _one(yahoo: str, fee_bps: float, source: str) -> list[dict[str, Any]]:
    prices = get_prices(yahoo, source, START, None, warmup=True)
    return ticker_observations(prices, fee_bps)


def _stats(vals: list[float]) -> dict[str, Any]:
    a = np.array(vals, dtype=float)
    if not len(a):
        return {"n": 0}
    # Gemiddelde met afgetopte uitschieters (−50% tot +100% per maand): een paar foute koersen bij Yahoo
    # zouden anders het hele gemiddelde bepalen. De mediaan en % positief zijn ongevoelig voor uitschieters.
    return {"n": int(len(a)), "avg": float(np.clip(a, -CAP_LO, CAP_HI).mean()), "median": float(np.median(a)),
            "hit": float((a > 0).mean())}


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5:
        return None
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(per_ticker: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per grade: rendement de maand (en 3 maanden) erna, absoluut en t.o.v. het eigen gemiddelde van de ticker."""
    rows = []
    for sym, obs in per_ticker.items():
        if not obs:
            continue
        mean = float(np.mean(np.clip([o["fwd"] for o in obs], -CAP_LO, CAP_HI)))
        rows += [o | {"sym": sym, "rel": o["fwd"] - mean} for o in obs]
    if not rows:
        return {"observations": 0}
    by_grade = []
    for g in [x for x, _ in GRADES] + [NONE]:
        sel = [r for r in rows if r["grade"] == g]
        by_grade.append({"grade": g, **_stats([r["fwd"] for r in sel]),
                         "rel": float(np.mean([min(max(r["fwd"], -CAP_LO), CAP_HI) - (r["fwd"] - r["rel"]) for r in sel])) if sel else None,
                         "fwd3": _stats([r["fwd3"] for r in sel if "fwd3" in r]).get("avg")})
    # Rangcorrelatie per datum over de tickers, dan het gemiddelde (IC) en hoe vaak die positief was.
    dates: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        dates.setdefault(r["date"][:7], []).append(r)
    ics = [ic for d in dates.values() if (ic := _spearman([r["score"] for r in d], [r["fwd"] for r in d])) is not None]
    top = [r["fwd"] for r in rows if r["grade"] in ("A+", "A")]
    bottom = [r["fwd"] for r in rows if r["grade"] == "F"]
    out = {
        "observations": len(rows), "tickers": len([s for s, o in per_ticker.items() if o]),
        "from": min(r["date"] for r in rows), "to": max(r["date"] for r in rows),
        "all": _stats([r["fwd"] for r in rows]), "by_grade": by_grade,
        "ic": float(np.mean(ics)) if ics else None, "ic_positive": float(np.mean([i > 0 for i in ics])) if ics else None,
        "ic_months": len(ics),
        # Verschil in mediaan rendement de maand erna tussen A+/A en F (robuust tegen foute koersen).
        "spread": float(np.median(top) - np.median(bottom)) if top and bottom else None,
        "horizon_days": HORIZON, "step": "maandeinde",
    }
    out["verdict"] = verdict(out)
    return out


def _nl(x: float, fmt: str) -> str:
    return format(x, fmt).replace(".", ",")


def verdict(s: dict[str, Any]) -> str:
    """Eerlijk oordeel in gewone taal."""
    ic, spread = s.get("ic"), s.get("spread")
    if ic is None or spread is None:
        return "Te weinig meetpunten voor een oordeel."
    rel = {b["grade"]: b["rel"] for b in s["by_grade"] if b.get("rel") is not None}
    order_ok = rel.get("A+", rel.get("A", 0)) >= rel.get("F", 0)
    if ic >= 0.05 and spread > 0.005 and order_ok:
        return (f"Voorspelde iets: A+/A deden de maand erna (mediaan) {_nl(spread * 100, '+.1f')} procentpunt beter dan F, "
                f"rangcorrelatie {_nl(ic, '.2f')}. Bruikbaar als filter, niet als zekerheid.")
    if ic > 0 and spread > 0:
        return (f"Zwak: A+/A deden de maand erna (mediaan) {_nl(spread * 100, '+.1f')} procentpunt beter dan F, maar de rangcorrelatie "
                f"is maar {_nl(ic, '.2f')}. Grotendeels ruis; gebruik het hooguit als één signaal tussen andere.")
    return (f"Voorspelde niets: A+/A deden de maand erna niet beter dan F (mediaan {_nl(spread * 100, '+.1f')} procentpunt, "
            f"rangcorrelatie {_nl(ic, '.2f')}). Een hoge grade betekent hier vooral dat er een trend bezig was.")


class Checker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {"running": False, "done": 0, "total": 0, "errors": 0}

    def status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def start(self, symbols: list[str], fee_bps: float = 5.0, source: str = "yahoo") -> bool:
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "done": 0, "total": len(symbols), "errors": 0,
                          "started": datetime.now().isoformat(timespec="seconds")}
        threading.Thread(target=self.run, args=(symbols, fee_bps, source), daemon=True, name="grade-check").start()
        return True

    def run(self, symbols: list[str], fee_bps: float = 5.0, source: str = "yahoo", workers: int | None = None) -> dict:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        per: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, str] = {}
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futs = {pool.submit(_one, s, fee_bps, source): s for s in symbols}
                for fut in as_completed(futs):
                    s = futs[fut]
                    try:
                        per[s] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        errors[s] = str(e)[:200]
                    with self.lock:
                        self.state["done"] += 1
                        self.state["errors"] = len(errors)
            out = {"summary": summarize(per), "finished": datetime.now().isoformat(timespec="seconds"),
                   "fee_bps": fee_bps, "errors": errors,
                   "tickers": {s: [[o["date"], o["score"], round(o["fwd"], 4)] for o in obs] for s, obs in per.items()}}
            DATA_DIR.mkdir(exist_ok=True)
            tmp = CHECK_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, CHECK_FILE)
            return out
        finally:
            with self.lock:
                self.state["running"] = False
                self.state["finished"] = datetime.now().isoformat(timespec="seconds")


CHECKER = Checker()


def load_check() -> dict[str, Any] | None:
    if not CHECK_FILE.exists():
        return None
    data = json.loads(CHECK_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "tickers"}


if __name__ == "__main__":
    from .watchlist import load_watchlist

    syms = list(dict.fromkeys(s["yahoo"] for g in load_watchlist() for s in g["symbols"] if s.get("yahoo")))
    print(f"Grade-check op {len(syms)} tickers …")
    res = CHECKER.run(syms)
    print(json.dumps(res["summary"], indent=1, ensure_ascii=False))
