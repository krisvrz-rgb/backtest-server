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
from datetime import date, datetime
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
# Euronext-noteringen die niet in Amsterdam zitten (Brussel .BR, Parijs .PA).
EURONEXT_OVERRIDES = {"SOF": "SOF.BR", "ACKB": "ACKB.BR", "UCB": "UCB.BR", "ABI": "ABI.BR", "KBC": "KBC.BR",
                      "NRJ": "NRJ.PA", "VAC": "VAC.PA"}
# Yahoo gebruikt voor sommige munten een id-achtervoegsel.
CRYPTO_OVERRIDES = {"UNI": "UNI7083", "HYPE": "HYPE32196", "SUI": "SUI20947", "TAO": "TAO22974"}
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
                base = ticker[:-len(quote)]
                return f"{CRYPTO_OVERRIDES.get(base, base)}-{'USD' if quote.startswith('USD') else quote}"
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


# --------------------------------------------------------------------------- portfolio server

PORTFOLIO_URL = os.getenv("PORTFOLIO_URL", "http://localhost:5000")
TV_WATCHLIST_FILE = Path(os.getenv("TV_WATCHLIST_FILE", r"C:\portfolio-data\tradingview_watchlist.json"))
PORTFOLIO_GROUPS = [("kristof", "KRISTOF"), ("freya", "FREYA"), ("cezanne", "CÉZANNE")]


def _tv_lookup() -> dict[str, str]:
    """Kale ticker -> TradingView-symbool, uit de TradingView-export van de portfolio server."""
    if not TV_WATCHLIST_FILE.exists():
        return {}
    data = json.loads(TV_WATCHLIST_FILE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for lst in data.get("lijsten", []):
        for sym in lst.get("symbolen", []):
            if ":" in sym and not sym.startswith("###"):
                out.setdefault(sym.split(":", 1)[1].upper(), sym)
    return out


def _resolve(ticker: str, currency: str | None, lookup: dict[str, str]) -> dict[str, Any]:
    t = ticker.strip().upper()
    if ":" in t:
        return {"tv": t, "yahoo": tv_to_yahoo(t)}
    if "." in t or "-" in t:  # al een Yahoo-ticker (bv. SOF.BR, BTC-USD)
        return {"tv": t, "yahoo": t}
    if t in lookup:
        return {"tv": lookup[t], "yahoo": tv_to_yahoo(lookup[t])}
    if currency and currency.upper() == "EUR":
        return {"tv": f"EURONEXT:{t}", "yahoo": f"{t}.AS"}
    return {"tv": t, "yahoo": t}


def import_from_portfolio(base_url: str = PORTFOLIO_URL) -> list[dict[str, Any]]:
    """Bouwt dezelfde groepen als de watchlist van de portfolio server: GEZIN, KRISTOF, FREYA, CÉZANNE, WATCHLIST."""
    import httpx

    lookup = _tv_lookup()
    data = httpx.get(f"{base_url}/api/data", params={"snel": 1}, timeout=60).json()
    groups, family = [], {}
    for key, label in PORTFOLIO_GROUPS:
        holdings = data.get("portfolios", {}).get(key, {}).get("holdings", [])
        syms = []
        for h in holdings:
            if not h.get("ticker") or not (h.get("waarde") or 0) > 0:
                continue
            item = _resolve(h["ticker"], h.get("currency"), lookup) | {"name": h.get("naam")}
            if item["tv"] not in [x["tv"] for x in syms]:
                syms.append(item)
            family.setdefault(item["tv"], item)
        groups.append({"name": label, "symbols": syms})
    groups.insert(0, {"name": "GEZIN", "symbols": list(family.values())})
    panel = httpx.get(f"{base_url}/api/wlpaneel", timeout=60).json()
    wl = []
    for lst in panel.get("lijsten", []):
        if lst.get("naam") in ("Eigen watchlist", "Watchlist"):
            for t in lst.get("tickers", []):
                item = _resolve(t, None, lookup)
                if item["tv"] not in family and item["tv"] not in [x["tv"] for x in wl]:
                    wl.append(item)
    groups.append({"name": "WATCHLIST", "symbols": wl})
    return [g | {"list": PORTFOLIO_LIST} for g in groups if g["symbols"]]


# --------------------------------------------------------------------------- TradingView-lijsten

PORTFOLIO_LIST = "Portfolio server"
TV_SNAPSHOT_FILE = Path(__file__).parent / "tradingview_lists.json"
FLAG_COLORS = {"red": "rood", "blue": "blauw", "cyan": "cyaan", "orange": "oranje", "green": "groen",
               "purple": "paars", "pink": "roze"}
TV_PREFIX = "TradingView · "


def list_name(group: dict[str, Any]) -> str:
    """Lijst waartoe een groep hoort; groepen van vóór de lijsten horen bij de portfolio server."""
    return group.get("list") or PORTFOLIO_LIST


def import_from_tradingview(data: dict[str, Any] | None = None, path: Path | None = None) -> list[dict[str, Any]]:
    """Elke TradingView-watchlist wordt een lijst; de ###secties erin worden groepen.

    Bron: de export van de portfolio server ({lijsten: [{naam, kleur, symbolen}]}) of hetzelfde formaat als data.
    """
    if data is None:
        path = path or TV_WATCHLIST_FILE
        if not path.exists():
            if path != TV_WATCHLIST_FILE or not TV_SNAPSHOT_FILE.exists():
                raise FileNotFoundError(f"TradingView-export niet gevonden: {path}")
            path = TV_SNAPSHOT_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
    lists = data.get("lijsten") or data.get("watchlists") or []
    # Actieve lijst eerst, dan de benoemde lijsten, dan de gekleurde vlaglijsten.
    lists = sorted(lists, key=lambda x: (not x.get("actief", x.get("active", False)),
                                         not (x.get("naam") or x.get("name"))))
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lst in lists:
        syms = [x for x in lst.get("symbolen", lst.get("symbols", [])) if isinstance(x, str)]
        color = lst.get("kleur") or lst.get("color")
        name = (lst.get("naam") or lst.get("name") or "").strip() or f"Vlag {FLAG_COLORS.get(color, color or lst.get('id'))}"
        while name in seen:
            name += " (2)"
        seen.add(name)
        for g in parse_tradingview(",".join(syms), default_group=name.upper()):
            if any(s["yahoo"] for s in g["symbols"]):  # secties met alleen indexen/rentes/futures overslaan
                groups.append(g | {"list": TV_PREFIX + name})
    return groups


def replace_lists(groups: list[dict[str, Any]], new: list[dict[str, Any]], match) -> list[dict[str, Any]]:
    """Vervangt de groepen waarvan de lijst voldoet aan `match` door `new`; andere lijsten blijven staan."""
    return [g for g in groups if not match(list_name(g))] + new


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
                                          "num_trades", "max_drawdown", "signal_open", "bars_since_buy",
                                          "last_buy_signal", "beats_bh_risk")}
        best_info["vs_bh"] = best["total_return"] - bh_ret
        # "Hoeveel keer beter dan buy & hold": eindkapitaal strategie / eindkapitaal vasthouden.
        best_info["times_bh"] = (1 + best["total_return"]) / (1 + bh_ret) if bh_ret > -1 else None
    # Plaats van buy & hold tussen alle strategieën (1 = niets doet beter dan gewoon vasthouden).
    bh_rank = 1 + len([x for x in strategies if x["total_return"] > bh_ret])
    profitable = sorted([x for x in strategies if x["total_return"] > 0], key=lambda x: -x["total_return"])
    open_prof = [x for x in profitable if x["signal_open"]]
    latest_prof = min(open_prof, key=lambda x: (x["bars_since_buy"], -x["total_return"])) if open_prof else None
    return {
        "bh_rank": bh_rank,
        "profitable_count": len(profitable),
        "profitable": [{k: x[k] for k in ("strategy", "category_label", "total_return", "beats_buy_and_hold",
                                          "signal_open", "bars_since_buy", "last_buy_signal", "sharpe", "max_drawdown")}
                       for x in profitable[:12]],
        "latest_buy_profitable": {k: latest_prof[k] for k in ("strategy", "last_buy_signal", "bars_since_buy",
                                                              "total_return", "beats_buy_and_hold")}
        if latest_prof else None,
        "open_profitable": len(open_prof),
        "last_close": float(close.iloc[-1]),
        "change_pct": float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0.0,
        "last_date": close.index[-1].date().isoformat(),
        "period": best["period"] if best else None,
        "strategies": len(strategies),
        "beat_count": len(beat),
        # Strategieën die buy & hold verslaan én zelf winst maakten: zegt meer als het aandeel zelf daalde.
        "beat_profitable_count": len([x for x in beat if x["total_return"] > 0]),
        "history_days": (close.index[-1] - close.index[0]).days,
        "buy_and_hold_return": bh_ret,
        "best": best_info,
        "top3": [x["strategy"] for x in strategies[:3]],
        "latest_buy": {k: latest[k] for k in ("strategy", "last_buy_signal", "bars_since_buy", "days_since_buy",
                                              "total_return", "beats_buy_and_hold")} if latest else None,
        "open_signals": len([x for x in beat if x["signal_open"]]),
        "recent_buys": len([x for x in beat if x["signal_open"] and x["bars_since_buy"] is not None
                            and x["bars_since_buy"] <= 5]),
    }


def default_periods(today: date | None = None) -> dict[str, dict[str, Any]]:
    """Standaard testperiodes: vanaf 2015, 2020 en 2024 tot nu, de laatste 2 jaar, vorig jaar en dit jaar."""
    today = today or date.today()
    y = today.year
    two = today.replace(year=y - 2) if not (today.month == 2 and today.day == 29) else today.replace(year=y - 2, day=28)
    return {
        "2015": {"start": "2015-01-01", "label": "2015 – nu"},
        "2020": {"start": "2020-01-01", "label": "2020 – nu"},
        "2024": {"start": "2024-01-01", "label": "2024 – nu"},
        "2j": {"start": two.isoformat(), "label": "Laatste 2 jaar"},
        str(y - 1): {"start": f"{y - 1}-01-01", "end": f"{y - 1}-12-31", "label": f"Jaar {y - 1}"},
        str(y): {"start": f"{y}-01-01", "label": f"{y} tot nu"},
    }


PERIODS = default_periods()


def _period(v: dict[str, Any] | str) -> tuple[str, str | None]:
    return (v, None) if isinstance(v, str) else (v["start"], v.get("end"))


def scan_one(yahoo: str, periods: dict[str, Any] | str, fee_bps: float, source: str = "yahoo") -> dict[str, Any]:
    """Draait in een apart proces: koersen één keer ophalen, alle strategieën testen per periode, samenvatten.

    Het resultaat bevat de samenvatting per periode onder "periods"; de eerste periode staat ook bovenaan
    (zoals vroeger), zodat oudere code en pagina's blijven werken.
    """
    if isinstance(periods, str):
        periods = {periods[:4]: periods}
    first = min(_period(v)[0] for v in periods.values())
    prices = get_prices(yahoo, source, first, None, warmup=True)
    out: dict[str, Any] = {"periods": {}}
    for key, v in periods.items():
        start, end = _period(v)
        try:
            p = prices[prices.index <= end] if end else prices
            if not (p.index >= start).any():
                raise ValueError("Geen koersen in deze periode")
            rank = rank_strategies(p, start, fee_bps=fee_bps)
            out["periods"][key] = summarize(rank, p[p.index >= start]) | {"start": start, "end": end}
        except Exception as e:  # noqa: BLE001 - bv. geen koersen in deze periode
            out["periods"][key] = {"error": str(e)[:200], "start": start, "end": end}
    ok = [v for v in out["periods"].values() if "error" not in v]
    if not ok:
        raise ValueError(next(iter(out["periods"].values()))["error"])
    return ok[0] | out


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

    def start(self, symbols: list[str], start: dict[str, Any] | str, fee_bps: float = 5.0, workers: int | None = None,
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

    def _run(self, symbols: list[str], start: dict[str, Any] | str, fee_bps: float, workers: int | None, source: str) -> None:
        scan = load_scan()
        periods = start if isinstance(start, dict) else {start[:4]: start}
        scan["settings"] = {"start": min(_period(v)[0] for v in periods.values()), "periods": periods,
                            "fee_bps": fee_bps, "source": source}
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
        # Tickers zonder koersdata (indexen, futures, rentes) tonen we niet.
        syms = [s | {"result": scan["results"].get(s["yahoo"])} for s in g["symbols"] if s.get("yahoo")]
        if syms:
            out.append({"name": g["name"], "list": list_name(g), "symbols": syms})
    periods = [{"key": k, **v} for k, v in default_periods().items()]
    return {"groups": out, "periods": periods, "scan": SCANNER.status(), "settings": scan.get("settings"), "finished": scan.get("finished"),
            "generated": time.time()}
