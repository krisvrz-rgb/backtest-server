"""Koersdata ophalen: Yahoo Finance, IBKR, Hyperliquid, lokale CSV-bestanden of synthetische testdata."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Ontbrekende kolommen in koersdata: {sorted(missing)}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[COLUMNS].sort_index().dropna(subset=["close"])


def load_csv(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Geen CSV gevonden voor {symbol} in {DATA_DIR}")
    return _normalize(pd.read_csv(path, index_col=0, parse_dates=True))


def load_yahoo(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Geen data van Yahoo Finance voor {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _normalize(df)


def load_ibkr(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Dagkoersen via TWS of IB Gateway (moet lokaal draaien met de API aan).

    Verbinding via omgevingsvariabelen IBKR_HOST (127.0.0.1), IBKR_PORT (7497 = TWS paper,
    7496 = TWS live, 4002/4001 = Gateway) en IBKR_CLIENT_ID (17).
    Symboolformaat: 'AAPL' (SMART/USD) of 'ASML:AEB:EUR' (ticker:exchange:valuta).
    """
    import os

    from ib_async import IB, Stock, util

    ticker, exchange, currency = (symbol.split(":") + ["SMART", "USD"])[:3] if ":" in symbol else (symbol, "SMART", "USD")
    years = max(1, int(np.ceil((pd.Timestamp(end or pd.Timestamp.today()) - pd.Timestamp(start or "2015-01-01")).days / 365.25)))
    ib = IB()
    ib.connect(os.getenv("IBKR_HOST", "127.0.0.1"), int(os.getenv("IBKR_PORT", "7497")),
               clientId=int(os.getenv("IBKR_CLIENT_ID", "17")), readonly=True, timeout=10)
    try:
        bars = ib.reqHistoricalData(Stock(ticker, exchange, currency), endDateTime="", durationStr=f"{years} Y",
                                    barSizeSetting="1 day", whatToShow="ADJUSTED_LAST", useRTH=True)
    finally:
        ib.disconnect()
    if not bars:
        raise ValueError(f"Geen data van IBKR voor {symbol}")
    return _normalize(util.df(bars).set_index("date"))


HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"


def load_hyperliquid(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Dagcandles van Hyperliquid (openbare API, geen account nodig). Symbool = coin, bv. 'BTC'."""
    import httpx

    start_ms = int(pd.Timestamp(start or "2023-01-01").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end or pd.Timestamp.today()).timestamp() * 1000)
    rows: list[dict] = []
    while start_ms < end_ms:  # max 5000 candles per call
        resp = httpx.post(HYPERLIQUID_URL, timeout=20, json={"type": "candleSnapshot", "req": {
            "coin": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms}})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows += batch
        start_ms = batch[-1]["t"] + 1
        if len(batch) < 5000:
            break
    if not rows:
        raise ValueError(f"Geen data van Hyperliquid voor {symbol}")
    df = pd.DataFrame(rows).drop_duplicates("t")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df.index = pd.to_datetime(df["t"], unit="ms")
    return _normalize(df[COLUMNS].astype(float))


def synthetic(symbol: str, start: str | None = None, end: str | None = None, seed: int | None = None) -> pd.DataFrame:
    """Geometrische Brownse beweging; handig om de server zonder internet te testen."""
    idx = pd.bdate_range(start or "2010-01-01", end or "2024-12-31")
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.upper().encode()))
    rets = rng.normal(0.0004, 0.015, len(idx))
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, len(idx))))
    vol = rng.integers(1_000_000, 5_000_000, len(idx))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    ).rename_axis("date")


WARMUP_DAYS = 550  # ruim genoeg voor indicatoren tot ~252 handelsdagen (bv. 52-weeks hoogtepunt, SMA 200)


def get_prices(symbol: str, source: str = "yahoo", start: str | None = None, end: str | None = None,
               warmup: bool = False) -> pd.DataFrame:
    """Koersdata vanaf `start`; met warmup=True ook ~1,5 jaar ervoor om indicatoren op te warmen."""
    if warmup and start:
        start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).date().isoformat()
    if symbol.upper().startswith("HL:"):  # Hyperliquid-perp uit de zwerm, ongeacht de gekozen bron
        df = load_hyperliquid(symbol[3:].upper(), start, end)
    elif source == "csv":
        df = load_csv(symbol)
    elif source == "yahoo":
        df = load_yahoo(symbol, start, end)
    elif source == "ibkr":
        df = load_ibkr(symbol, start, end)
    elif source == "hyperliquid":
        df = load_hyperliquid(symbol, start, end)
    elif source == "synthetic":
        df = synthetic(symbol, "2010-01-01", end)
    else:
        raise ValueError(f"Onbekende databron: {source}")
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 2:
        raise ValueError("Te weinig koersdata in de gekozen periode")
    return df
