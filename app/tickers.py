"""Tickers herkennen in OCR-tekst en controleren tegen de lijst van Amerikaanse beurssymbolen."""
from __future__ import annotations

import re
import time

from .data import DATA_DIR

SYMBOL_FILE = DATA_DIR / "symbols.txt"
NASDAQ_URLS = {
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
# Beurscode in otherlisted.txt -> prefix die TradingView gebruikt
EXCHANGES = {"Q": "NASDAQ", "N": "NYSE", "A": "AMEX", "P": "AMEX", "Z": "CBOE", "V": "IEX"}

# Woorden die op het scherm van de scanner staan en toevallig ook een geldig symbool zijn
STOPWORDS = {
    "AM", "PM", "ET", "EST", "EDT", "HOD", "LOD", "USD", "EUR", "DJIA", "SPX", "NDX", "VIX", "OIL",
    "NEWS", "HALT", "LULD", "NAME", "TIME", "PRICE", "FLOAT", "GAP", "UP", "DOWN", "MOMO", "MIN", "MAX",
    "VOL", "AVG", "HIGH", "LOW", "LIVE", "NEW", "ALL", "OFF", "IN", "OUT", "AT",
    "TO", "OF", "OR", "AND", "THE", "FOR", "BY", "BE", "IT", "IS", "SO", "NO", "GO", "DO", "ME", "MY", "WE",
    "ATM", "CEO", "CFO", "FDA", "SEC", "IPO", "ETF", "EPS", "AI", "PT", "TP", "GM", "LOL", "OK", "YES",
    "NASDAQ", "NYSE", "AMEX", "OTC", "ARCA", "TV", "RES", "PRE", "POST", "BUY", "SELL", "LONG", "SHORT",
}
TOKEN = re.compile(r"(?<![A-Za-z])\$?([A-Z]{2,5})(?![A-Za-z])")

_symbols: dict[str, dict] | None = None
_loaded_at = 0.0


def _parse(text: str, kind: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    lines = text.strip().splitlines()
    header = lines[0].split("|")
    for line in lines[1:]:
        if line.startswith("File Creation Time"):
            continue
        row = dict(zip(header, line.split("|")))
        if row.get("Test Issue") == "Y":
            continue
        sym = row.get("Symbol") or row.get("ACT Symbol") or ""
        if not sym.isalpha():
            continue
        exch = "NASDAQ" if kind == "nasdaq" else EXCHANGES.get(row.get("Exchange", ""), "")
        out[sym] = {"exchange": exch, "name": row.get("Security Name", "")}
    return out


def download_symbols() -> dict[str, dict]:
    import httpx

    symbols: dict[str, dict] = {}
    for kind, url in NASDAQ_URLS.items():
        r = httpx.get(url, timeout=30)
        r.raise_for_status()
        symbols.update(_parse(r.text, kind))
    DATA_DIR.mkdir(exist_ok=True)
    SYMBOL_FILE.write_text("\n".join(f"{s}|{v['exchange']}|{v['name']}" for s, v in sorted(symbols.items())),
                           encoding="utf-8")
    return symbols


def load_symbols(max_age_days: float = 7) -> dict[str, dict] | None:
    """Symbolenlijst uit de cache, wekelijks ververst. None als hij niet te krijgen is."""
    global _symbols, _loaded_at
    if _symbols is not None and time.time() - _loaded_at < 3600:
        return _symbols
    fresh = SYMBOL_FILE.exists() and time.time() - SYMBOL_FILE.stat().st_mtime < max_age_days * 86400
    if not fresh:
        try:
            _symbols, _loaded_at = download_symbols(), time.time()
            return _symbols
        except Exception:  # noqa: BLE001 - val terug op de oude cache
            pass
    if SYMBOL_FILE.exists():
        _symbols = {}
        for line in SYMBOL_FILE.read_text(encoding="utf-8").splitlines():
            sym, exch, name = (line.split("|", 2) + ["", ""])[:3]
            _symbols[sym] = {"exchange": exch, "name": name}
        _loaded_at = time.time()
    return _symbols


def extract_tickers(texts: list[str], symbols: dict[str, dict] | None) -> list[str]:
    """Unieke tickers uit een lijst OCR-regels, in volgorde van voorkomen.

    Met een symbolenlijst telt alleen wat echt genoteerd is; zonder lijst vallen we terug op
    3 tot 5 hoofdletters, wat meer ruis geeft.
    """
    found: list[str] = []
    for text in texts:
        for m in TOKEN.finditer(text):
            tok = m.group(1)
            if tok in STOPWORDS or tok in found:
                continue
            if symbols is not None and tok not in symbols:
                continue
            if symbols is None and len(tok) < 3:
                continue
            found.append(tok)
    return found
