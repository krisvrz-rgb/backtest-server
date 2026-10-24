"""Watchlist met groepen en een achtergrond-scanner die alle strategieën per ticker test.

De watchlist komt uit TradingView (export of de secties van een lijst) en wordt bewaard in
data/watchlist.json. De scanner draait alle strategieën op elke ticker en bewaart per ticker een
samenvatting in data/scan.json, zodat de watchlist-pagina meteen laadt en je alleen opnieuw hoeft
te scannen als je verse cijfers wilt.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .data import DATA_DIR, get_prices
from .ranking import rank_strategies

WATCHLIST_FILE = DATA_DIR / "watchlist.json"
SCAN_FILE = DATA_DIR / "scan.json"
DEFAULT_FILE = Path(__file__).parent / "default_watchlist.json"

# TradingView-beurs -> Yahoo-achtervoegsel. None = Amerikaanse beurs, geen achtervoegsel.
EXCHANGE_SUFFIX = {
    "NASDAQ": "", "NYSE": "", "AMEX": "", "NYSEARCA": "", "ARCA": "", "BATS": "", "CBOE": "", "OTC": "",
    "XETR": ".DE", "GETTEX": ".DE", "FWB": ".F", "LSE": ".L", "MIL": ".MI", "OMXSTO": ".ST", "TSX": ".TO",
    "TSXV": ".V", "EURONEXT": ".AS", "SIX": ".SW", "HKEX": ".HK",
}
# Euronext-aandelen die niet in Amsterdam noteren.
EURONEXT_OVERRIDES = {"SOF": "SOF.BR", "ACKB": "ACKB.BR", "UCB": "UCB.BR", "ABI": "ABI.BR", "KBC": "KBC.BR"}
CRYPTO_EXCHANGES = {"BITSTAMP", "COINBASE", "BINANCE", "KRAKEN", "BYBIT"}
NO_DATA_PREFIXES = {"TVC", "ECONOMICS", "FX_IDC", "FX", "CME", "CME_MINI", "ICEEUR", "ICEENDEX", "SPREADEX", "USI",
                    "EASYMARKETS", "OANDA", "CAPITALCOM"}


def tv_to_yahoo(tv: str) -> str | None:
    """Zet een TradingView-symbool (EXCHANGE:TICKER) om naar een Yahoo-ticker; None als er geen koersdata is."""
    tv = tv.strip()
    if not tv or "/" in tv or "!" in tv:
        return None
    if ":" not in tv:
        return tv.upper()
    exchange, ticker = tv.upper().split(":", 1)
    if exchange in NO_DATA_PREFIXES:
        return None
    if exchange in CRYPTO_EXCHANGES:
        for quote in ("USDT", "USDC", "USD", "EUR"):
            if ticker.endswith(quote) and len(ticker) > len(quote):
                return f"{ticker[:-len(quote)]}-{'USD' if quote.startswith('USD') else quote}"
        return None
    if exchange == "EURONEXT" and ticker in EURONEXT_OVERRIDES:
        return EURONEXT_OVERRIDES[ticker]
    if exchange in EXCHANGE_SUFFIX:
        return ticker.replace(".", "-") + EXCHANGE_SUFFIX[exchange]
    return None


def parse_tradingview(text: str, default_group: str = "WATCHLIST") -> list[dict[str, Any]]:
    """Leest een TradingView-watchlist-export ("###SECTIE,NASDAQ:AAPL,...") of een lijst tickers per regel."""
    groups: list[dict[str, Any]] = []
    current = {"name": default_group, "symbols": []}
    for raw in text.replace("\n", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if item.startswith("###"):
            if current["symbols"]:
                groups.append(current)
            current = {"name": item.lstrip("#").strip() or default_group, "symbols": []}
            continue
        if item not in [s["tv"] for s in current["symbols"]]:
            current["symbols"].append({"tv": item, "yahoo": tv_to_yahoo(item)})
    if current["symbols"]:
        groups.append(current)
    return groups


def load_watchlist() -> list[dict[str, Any]]:
    for f in (WATCHLIST_FILE, DEFAULT_FILE):
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))["groups"]
    return []


def save_watchlist(groups: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps({"groups": groups}, indent=1, ensure_ascii=False), encoding="utf-8")


def load_scan() -> dict[str, Any]:
    if SCAN_FILE.exists():
        return json.loads(SCAN_FILE.read_text(encoding="utf-8"))
    return {"results": {}, "settings": None, "finished": None}


def _save_scan(scan: dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tmp = SCAN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(scan, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, SCAN_FILE)


# --------------------------------------------------------------------------- één ticker scannen


def summarize(rank: dict[str, Any], prices) -> dict[str, Any]:
    """Watchlist-samenvatting van een ranglijst: hoeveel strategieën verslaan buy & hold, beste strategie, laatste buy."""
    bh = rank["buy_and_hold"]
    bh_ret = bh["total_return"] if bh else 0.0
    strategies = [x for x in rank["results"] if x["category"] != "benchmark"]
    beat = [x for x in strategies if x["beats_buy_and_hold"]]
    best = strategies[0] if strategies else None

    # Laatste buy-signaal dat nog open staat: eerst bij de strategieën die buy & hold verslaan, anders bij de top 20.
    pool = [x for x in beat if x["signal_open"]] or [x for x in strategies[:20] if x["signal_open"]]
    latest = min(pool, key=lambda x: (x["bars_since_buy"], -x["total_return"])) if pool else None
    close = prices["close"]
    best_info = None
    if best:
        best_info = {k: best[k] for k in ("strategy", "category_label", "total_return", "sharpe", "percent_profitable",
                                          "num_trades", "max_drawdown")}
        # "Hoeveel keer beter dan buy & hold": eindkapitaal strategie / eindkapitaal vasthouden.
        best_info["times_bh"] = (1 + best["total_return"]) / (1 + bh_ret) if bh_ret > -1 else None
    return {
        "last_close": float(close.iloc[-1]),
        "change_pct": float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0.0,
        "last_date": close.index[-1].date().isoformat(),
        "period": best["period"] if best else None,
        "strategies": len(strategies),
        "beat_count": len(beat),
        "buy_and_hold_return": bh_ret,
        "best": best_info,
        "top3": [x["strategy"] for x in strategies[:3]],
        "latest_buy": {k: latest[k] for k in ("strategy", "last_buy_signal", "bars_since_buy", "days_since_buy",
                                              "total_return", "beats_buy_and_hold")} if latest else None,
        "open_signals": len([x for x in beat if x["signal_open"]]),
        "recent_buys": len([x for x in beat if x["signal_open"] and x["bars_since_buy"] is not None
                            and x["bars_since_buy"] <= 5]),
    }


def scan_one(yahoo: str, start: str, fee_bps: float, source: str = "yahoo") -> dict[str, Any]:
    """Draait in een apart proces: koersen ophalen, alle strategieën testen, samenvatten."""
    prices = get_prices(yahoo, source, start, None, warmup=True)
    rank = rank_strategies(prices, start, fee_bps=fee_bps)
    return summarize(rank, prices[prices.index >= start])


# --------------------------------------------------------------------------- achtergrond-scanner


class Scanner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {"running": False, "done": 0, "total": 0, "errors": 0, "started": None,
                                      "finished": None, "current": None}
        self._stop = False

    def status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def stop(self) -> None:
        self._stop = True

    def start(self, symbols: list[str], start: str, fee_bps: float = 5.0, workers: int | None = None,
              source: str = "yahoo") -> bool:
        with self.lock:
            if self.state["running"]:
                return False
            self.state = {"running": True, "done": 0, "total": len(symbols), "errors": 0,
                          "started": datetime.now().isoformat(timespec="seconds"), "finished": None, "current": None}
        self._stop = False
        threading.Thread(target=self._run, args=(symbols, start, fee_bps, workers, source), daemon=True).start()
        return True

    def _record(self, scan: dict[str, Any], yahoo: str, result: dict[str, Any]) -> None:
        with self.lock:
            scan["results"][yahoo] = result
            self.state["done"] += 1
            if "error" in result:
                self.state["errors"] += 1
            _save_scan(scan)

    def _run(self, symbols: list[str], start: str, fee_bps: float, workers: int | None, source: str) -> None:
        scan = load_scan()
        scan["settings"] = {"start": start, "fee_bps": fee_bps, "source": source}
        workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(scan_one, y, start, fee_bps, source): y for y in symbols}
                for fut in as_completed(futures):
                    y = futures[fut]
                    if self._stop:
                        for f in futures:
                            f.cancel()
                        break
                    try:
                        result = fut.result() | {"scanned_at": datetime.now().isoformat(timespec="seconds")}
                    except Exception as e:  # noqa: BLE001 - één ticker zonder data stopt de scan niet
                        result = {"error": str(e)[:200], "scanned_at": datetime.now().isoformat(timespec="seconds")}
                    with self.lock:
                        self.state["current"] = y
                    self._record(scan, y, result)
        finally:
            with self.lock:
                scan["finished"] = datetime.now().isoformat(timespec="seconds")
                _save_scan(scan)
                self.state["running"] = False
                self.state["finished"] = scan["finished"]


SCANNER = Scanner()


def watchlist_view() -> dict[str, Any]:
    scan = load_scan()
    groups = load_watchlist()
    out = []
    for g in groups:
        out.append({"name": g["name"], "symbols": [s | {"result": scan["results"].get(s["yahoo"]) if s.get("yahoo") else None}
                                                   for s in g["symbols"]]})
    return {"groups": out, "scan": SCANNER.status(), "settings": scan.get("settings"), "finished": scan.get("finished"),
            "generated": time.time()}
