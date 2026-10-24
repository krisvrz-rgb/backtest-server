from __future__ import annotations

import os
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .data import DATA_DIR, _normalize, get_prices
from .engine import run_backtest
from .live_scanner import board_for, days, scanner
from .pine import to_pine
from .strategies import STRATEGIES, get_strategy



@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("SCANNER_AUTOSTART", "1") == "1":
        scanner.start()
    yield
    scanner.stop()


app = FastAPI(title="Backtest Server", version="0.1.0", lifespan=lifespan,
              description="Backtest handelsstrategieën op aandelen en exporteer ze naar TradingView Pine Script.")


Source = Literal["csv", "yahoo", "ibkr", "hyperliquid", "synthetic"]


class BacktestRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    strategy: str = Field(..., examples=["sma_crossover"])
    params: dict[str, Any] = Field(default_factory=dict, examples=[{"fast": 50, "slow": 200}])
    source: Source = "csv"
    start: str | None = Field(None, examples=["2015-01-01"])
    end: str | None = None
    initial_capital: float = 10_000.0
    fee_bps: float = Field(5.0, description="Kosten per transactie in basispunten (5 = 0,05%)")
    slippage_bps: float = 0.0
    allow_short: bool = False
    include_equity: bool = True


class CompareRequest(BaseModel):
    symbols: list[str]
    strategies: list[dict[str, Any]] = Field(..., examples=[[{"name": "sma_crossover", "params": {"fast": 20, "slow": 100}},
                                                            {"name": "buy_and_hold"}]])
    source: Source = "csv"
    start: str | None = None
    end: str | None = None
    fee_bps: float = 5.0


def _run(req: BacktestRequest) -> dict:
    try:
        prices = get_prices(req.symbol, req.source, req.start, req.end)
        strat = get_strategy(req.strategy, **req.params)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    res = run_backtest(prices, strat, req.initial_capital, req.fee_bps, req.slippage_bps, req.allow_short)
    out = {
        "symbol": req.symbol.upper(), "strategy": strat.name, "params": strat.p,
        "period": {"start": prices.index[0].date().isoformat(), "end": prices.index[-1].date().isoformat()},
        "metrics": res.metrics, "benchmark_metrics": res.benchmark_metrics, "trades": res.trades,
    }
    if req.include_equity:
        out["equity_curve"] = [{"date": d.date().isoformat(), "equity": round(e, 2), "benchmark": round(b, 2)}
                               for d, e, b in zip(res.equity.index, res.equity, res.benchmark)]
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/strategies")
def list_strategies():
    return [{"name": n, "description": c.description, "params": c.params} for n, c in STRATEGIES.items()]


@app.post("/backtest")
def backtest(req: BacktestRequest):
    return _run(req)


@app.post("/compare")
def compare(req: CompareRequest):
    rows = []
    for sym in req.symbols:
        for s in req.strategies:
            r = _run(BacktestRequest(symbol=sym, strategy=s["name"], params=s.get("params", {}), source=req.source,
                                     start=req.start, end=req.end, fee_bps=req.fee_bps, include_equity=False))
            rows.append({"symbol": r["symbol"], "strategy": r["strategy"], "params": r["params"], **r["metrics"]})
    rows.sort(key=lambda x: x["sharpe"], reverse=True)
    return rows


@app.post("/pine", response_class=PlainTextResponse)
def pine(req: BacktestRequest):
    """Pine Script v6 voor de gekozen strategie en parameters, klaar voor de Pine Editor."""
    try:
        strat = get_strategy(req.strategy, **req.params)
        return to_pine(strat, req.initial_capital, req.fee_bps, req.start or "2015-01-01")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/data")
def list_data():
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


@app.post("/data/{symbol}")
async def upload_data(symbol: str, file: UploadFile = File(...)):
    """Upload een CSV met kolommen date, open, high, low, close, volume (bijv. export uit TradingView)."""
    raw = (await file.read()).decode()
    try:
        df = pd.read_csv(StringIO(raw))
        df.columns = [c.lower() for c in df.columns]
        datecol = "date" if "date" in df.columns else "time"
        if datecol == "time" and pd.api.types.is_numeric_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], unit="s")
        df = _normalize(df.set_index(datecol))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Kon CSV niet lezen: {e}")
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / f"{symbol.upper()}.csv")
    return {"symbol": symbol.upper(), "rows": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date())}


# ---- Live scanner: tickers uit de YouTube-livestream ----

SCANNER_PAGE = Path(__file__).parent / "static" / "scanner.html"


@app.get("/scanner", response_class=HTMLResponse, include_in_schema=False)
def scanner_page():
    return SCANNER_PAGE.read_text(encoding="utf-8")


@app.get("/scanner/tickers")
def scanner_tickers(day: str | None = None, min_count: int = 1):
    """Tickers van een handelsdag (standaard vandaag), laatst gezien bovenaan."""
    board = board_for(day)
    return {"day": board.day, "tickers": board.rows(min_count)}


@app.get("/scanner/status")
def scanner_status():
    return {**scanner.status, "url": scanner.url, "interval": scanner.interval, "day": scanner.board.day,
            "days": days()}


@app.post("/scanner/start")
def scanner_start():
    scanner.start()
    return scanner_status()


@app.post("/scanner/stop")
def scanner_stop():
    scanner.stop()
    return scanner_status()


@app.post("/scanner/image")
async def scanner_image(file: UploadFile = File(...)):
    """Screenshot uploaden en de tickers erop meetellen; handig om de herkenning te testen."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Geen geldig beeldbestand")
    return scanner.process_image(img)
