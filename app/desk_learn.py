"""Leren van de zwerm: welke kenmerken voorspellen echt iets, en hoeveel moeten ze wegen.

Geen taalmodel: een kleine logistische regressie (met L2) op de CPU, getraind op de eigen LEER-vensters.
Elk venster bewaart de kenmerken op het beslismoment, de kans die het vaste model toen gaf, en of de koers
daarna steeg of daalde. Het geleerde model wordt alleen gebruikt als het op recente data die het niet gezien
heeft beter voorspelt (lagere log-loss) dan het vaste model. Per groep: crypto en aandelen (HIP-3) apart.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from .data import DATA_DIR

FEATS = ("flow30", "flow10", "book", "bookDeep", "mom")
NAMES = ("koopdruk 3 min", "koopdruk 30 s", "orderboek top 5", "orderboek top 20", "momentum")
LEARN_FILE = DATA_DIR / "desk-learn.jsonl"
TRADES_FILE = DATA_DIR / "desk-trades.jsonl"
STATE_FILE = DATA_DIR / "desk-learn-state.json"
MIN_ROWS = 300      # minder vensters dan dit: niet trainen, te weinig om iets te bewijzen
MARGIN = 0.002      # het geleerde model moet minstens zoveel log-loss winnen op de testset

STATE: dict = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def group(coin: str) -> str:
    return "stock" if ":" in coin else "crypto"


def fvec(f: dict) -> list[float]:
    return [f.get("flow30", 0.0), f.get("flow10", 0.0), f.get("book", 0.0), f.get("bookDeep", 0.0),
            max(-1.0, min(1.0, f.get("mom", 0.0) / 2))]


def _append(path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def record(row: dict) -> None:
    """Eén afgesloten LEER-venster: t, coin, window (s), f (kenmerken), p (kans vast model), y (uitkomst in σ)."""
    _append(LEARN_FILE, row)


def record_trade(h: dict) -> None:
    _append(TRADES_FILE, h)


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def trades(since_ms: int = 0) -> list[dict]:
    return [h for h in _read(TRADES_FILE) if h.get("t", 0) >= since_ms]


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 2.0) -> np.ndarray:
    """Logistische regressie met L2 via Newton-stappen (IRLS). Laatste kolom van X is de constante."""
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    reg[-1, -1] = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + reg + np.eye(X.shape[1]) * 1e-9
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r["f"] + [1.0] for r in rows], dtype=float)
    y = np.array([1.0 if r["y"] > 0 else 0.0 for r in rows])
    base = np.array([r["p"] for r in rows], dtype=float)
    return X, y, base


def train(window: int) -> dict:
    """Train per groep op de vensters van deze lengte. Oudste 70% om te leren, nieuwste 30% om te toetsen."""
    rows = [r for r in _read(LEARN_FILE) if r.get("window") == window and r.get("y") and r.get("f") and r.get("p") is not None]
    rows.sort(key=lambda r: r["t"])
    res = {"window": window, "n": len(rows), "at": int(time.time() * 1000), "groups": {}}
    for g in ("crypto", "stock"):
        rs = [r for r in rows if group(r["coin"]) == g]
        info: dict = {"n": len(rs), "use": False}
        if len(rs) >= MIN_ROWS:
            cut = int(len(rs) * .7)
            Xtr, ytr, _ = _design(rs[:cut])
            Xte, yte, bte = _design(rs[cut:])
            w = _fit(Xtr, ytr)
            pte = 1 / (1 + np.exp(-(Xte @ w)))
            info.update(ll_learned=round(_ll(pte, yte), 4), ll_base=round(_ll(bte, yte), 4),
                        acc_learned=round(float(np.mean((pte > .5) == (yte > .5))), 3),
                        acc_base=round(float(np.mean((bte > .5) == (yte > .5))), 3), n_test=len(rs) - cut)
            info["use"] = info["ll_learned"] < info["ll_base"] - MARGIN and info["acc_learned"] >= info["acc_base"]
            Xall, yall, _ = _design(rs)
            w = _fit(Xall, yall)  # wat gebruikt wordt: getraind op alles
            info["w"] = [round(float(v), 4) for v in w[:-1]]
            info["b"] = round(float(w[-1]), 4)
        res["groups"][g] = info
    STATE[str(window)] = res
    STATE_FILE.write_text(json.dumps(STATE))
    return res


def predict(window: int, coin: str, f: dict) -> float | None:
    """Kans dat de koers stijgt volgens het geleerde model, of None als het (nog) niet beter is dan het vaste."""
    g = STATE.get(str(window), {}).get("groups", {}).get(group(coin))
    if not g or not g.get("use") or "w" not in g:
        return None
    z = sum(w * x for w, x in zip(g["w"], fvec(f))) + g["b"]
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


_count = {"t": 0.0, "n": {}}


def summary(window: int) -> dict:
    s = STATE.get(str(window))
    if not s:
        if time.time() - _count["t"] > 60:  # niet bij elke poll het hele bestand lezen
            n: dict = {}
            for r in _read(LEARN_FILE):
                n[r.get("window")] = n.get(r.get("window"), 0) + 1
            _count.update(t=time.time(), n=n)
        return {"window": window, "n": _count["n"].get(window, 0), "min": MIN_ROWS, "groups": {}, "names": NAMES}
    return {**s, "min": MIN_ROWS, "names": NAMES}
