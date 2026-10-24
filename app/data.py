"""Koersdata ophalen: Yahoo Finance, lokale CSV-bestanden of synthetische testdata."""
from __future__ import annotations

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


def synthetic(symbol: str, start: str | None = None, end: str | None = None, seed: int | None = None) -> pd.DataFrame:
    """Geometrische Brownse beweging; handig om de server zonder internet te testen."""
    idx = pd.bdate_range(start or "2018-01-01", end or "2024-12-31")
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % 2**32)
    rets = rng.normal(0.0004, 0.015, len(idx))
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, len(idx))))
    vol = rng.integers(1_000_000, 5_000_000, len(idx))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    ).rename_axis("date")


def get_prices(symbol: str, source: str = "yahoo", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if source == "csv":
        df = load_csv(symbol)
    elif source == "yahoo":
        df = load_yahoo(symbol, start, end)
    elif source == "synthetic":
        df = synthetic(symbol, start, end)
    else:
        raise ValueError(f"Onbekende databron: {source}")
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 2:
        raise ValueError("Te weinig koersdata in de gekozen periode")
    return df
