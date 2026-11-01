"""De strategieën van de backtestserver als stem voor de zwerm, op de candles van het venster zelf (15m, 1u, 4u).

Per markt: alle strategieën backtesten op de Hyperliquid-candles van dat venster (tot 5000 bars), met kosten.
Een strategie mag alleen meestemmen als ze winst maakt op de oudste 70% én op de nieuwste 30% die ze bij het
kiezen niet zag, en daar ook beter doet dan gewoon vasthouden. Hun stem nu: zit de strategie in een positie terwijl ze dat gewoonlijk niet doet, dan is dat
een koopstem; staat ze plat terwijl ze gewoonlijk long is, een verkoopstem. Uitkomst tussen -1 en +1.
"""
from __future__ import annotations

import json
import threading
import time

import pandas as pd

from . import desk
from .data import DATA_DIR
from .engine import run_backtest
from .strategies import STRATEGIES

STATE_FILE = DATA_DIR / "desk-strats.json"
FEE_BPS = 3.0        # per positiewissel: maker erin, mix maker/taker eruit
MIN_TRADES = 5       # op het testdeel, anders bewijst het niets
TOP = 10             # zoveel strategieën per markt mogen meestemmen
INTERVAL = {900: "15m", 3600: "1h", 14400: "4h"}
_lock = threading.Lock()
STATE: dict = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def _key(coin: str, window: int) -> str:
    return f"{coin}|{window}"


def _frame(coin: str, window: int) -> pd.DataFrame:
    iv = INTERVAL.get(window, "15m")
    cs = desk.candles(coin, iv, 5000 * window // 60)
    df = pd.DataFrame(cs[:-1])  # de lopende candle niet: die is nog niet gesloten
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df.index = pd.to_datetime(df["t"], unit="ms")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def select(coin: str, window: int) -> dict:
    """Alle strategieën op dit venster backtesten en de bewezen beste kiezen. Duurt een paar seconden per markt."""
    df = _frame(coin, window)
    if len(df) < 400:
        raise ValueError(f"te weinig candles ({len(df)})")
    cut = df.index[int(len(df) * .7)]
    rows = []
    for name, cls in STRATEGIES.items():
        if name == "buy_and_hold":
            continue
        try:
            s = cls()
            ins = run_backtest(df[df.index < cut], s, fee_bps=FEE_BPS)
            oos = run_backtest(df, s, fee_bps=FEE_BPS, start=cut)  # opwarmen op het begin, handelen alleen op het testdeel
        except Exception:
            continue
        mi, mo = ins.metrics, oos.metrics
        if (mi["total_return"] > 0 and mo["total_return"] > 0 and mo["excess_return"] > 0  # ook beter dan gewoon vasthouden
                and mo["num_trades"] >= MIN_TRADES):
            rows.append({"name": name, "ret_in": round(mi["total_return"], 4), "ret_out": round(mo["total_return"], 4),
                         "sharpe_out": round(mo["sharpe"], 2), "trades_out": mo["num_trades"],
                         "win_out": round(mo["win_rate"], 3), "exposure": round(float(oos.position.mean()), 3),
                         "bh_out": round(oos.benchmark_metrics["total_return"], 4)})
    rows.sort(key=lambda r: r["sharpe_out"], reverse=True)
    res = {"at": int(time.time()), "coin": coin, "window": window, "bars": len(df), "tested": len(STRATEGIES) - 1,
           "passed": len(rows), "picked": rows[:TOP], "vote": None}
    res["vote"] = vote(res, df)
    with _lock:
        STATE[_key(coin, window)] = res
        STATE_FILE.write_text(json.dumps(STATE))
    return res


def vote(res: dict, df: pd.DataFrame | None = None) -> dict | None:
    """De stem van de gekozen strategieën nu, gewogen naar hun sharpe op het testdeel."""
    if not res.get("picked"):
        return None
    if df is None:
        df = _frame(res["coin"], res["window"])
    tot = wsum = 0.0
    longs = []
    for r in res["picked"]:
        try:
            pos = float(STRATEGIES[r["name"]]().signals(df).fillna(0).iloc[-1] > 0)
        except Exception:
            continue
        w = max(r["sharpe_out"], .1)
        tot += w * (pos - r["exposure"]) / max(r["exposure"], 1 - r["exposure"], .1)  # tov hoe vaak ze gewoonlijk long is
        wsum += w
        if pos:
            longs.append(r["name"])
    if not wsum:
        return None
    return {"v": round(max(-1.0, min(1.0, tot / wsum)), 3), "long": longs, "n": len(res["picked"]), "at": int(time.time())}


def get(coin: str, window: int) -> dict | None:
    return STATE.get(_key(coin, window))


def refresh_votes(coins: list[str], window: int) -> None:
    """Elke gesloten candle: de stem van de al gekozen strategieën opnieuw berekenen (snel, enkel de top 10)."""
    for c in coins:
        res = get(c, window)
        if not res or not res.get("picked"):
            continue
        try:
            res["vote"] = vote(res)
        except Exception:
            pass
        time.sleep(.5)
    with _lock:
        STATE_FILE.write_text(json.dumps(STATE))
