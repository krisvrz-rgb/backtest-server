from __future__ import annotations

from io import StringIO
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pathlib import Path

from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import DATA_DIR, _normalize, get_prices
from .engine import compute_metrics, monthly_returns, run_backtest
from .pine import to_pine
from .strategies import STRATEGIES, get_strategy

app = FastAPI(title="Backtest Server", version="0.1.0",
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
    include_prices: bool = Field(False, description="Koersdata (OHLCV) meesturen voor de candlestick-grafiek")
    market_benchmark: str | None = Field("SPY", description="Marktbenchmark om mee te vergelijken (S&P 500 via SPY); leeg = geen")


class CompareRequest(BaseModel):
    symbols: list[str]
    strategies: list[dict[str, Any]] = Field(..., examples=[[{"name": "sma_crossover", "params": {"fast": 20, "slow": 100}},
                                                            {"name": "buy_and_hold"}]])
    source: Source = "csv"
    start: str | None = None
    end: str | None = None
    fee_bps: float = 5.0


def _market_curve(req: BacktestRequest, index: pd.DatetimeIndex) -> pd.Series | None:
    """Koers van de marktbenchmark op dezelfde dagen als het aandeel, herschaald naar het startkapitaal."""
    if not req.market_benchmark or req.market_benchmark.upper() == req.symbol.upper() or req.source == "hyperliquid":
        return None
    try:
        bench = get_prices(req.market_benchmark, req.source, req.start, req.end)["close"]
    except Exception:  # noqa: BLE001 - benchmark is optioneel, de backtest moet altijd lukken
        return None
    bench = bench.reindex(index.union(bench.index)).ffill().reindex(index).dropna()
    if len(bench) < 2:
        return None
    return req.initial_capital * bench / bench.iloc[0]


def _run(req: BacktestRequest) -> dict:
    try:
        prices = get_prices(req.symbol, req.source, req.start, req.end)
        strat = get_strategy(req.strategy, **req.params)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    res = run_backtest(prices, strat, req.initial_capital, req.fee_bps, req.slippage_bps, req.allow_short)
    market = _market_curve(req, prices.index)
    out = {
        "symbol": req.symbol.upper(), "strategy": strat.name, "params": strat.p,
        "period": {"start": prices.index[0].date().isoformat(), "end": prices.index[-1].date().isoformat()},
        "metrics": res.metrics, "benchmark_metrics": res.benchmark_metrics, "trades": res.trades,
        "market_benchmark": req.market_benchmark.upper() if market is not None else None,
        "market_metrics": compute_metrics(market) if market is not None else None,
        "monthly_returns": monthly_returns(res.equity, req.initial_capital),
    }
    if req.include_equity:
        mk = market.reindex(res.equity.index) if market is not None else None
        out["equity_curve"] = [
            {"date": d.date().isoformat(), "equity": round(e, 2), "benchmark": round(b, 2),
             "market": None if mk is None or pd.isna(mk[d]) else round(float(mk[d]), 2)}
            for d, e, b in zip(res.equity.index, res.equity, res.benchmark)]
    if req.include_prices:
        out["prices"] = [{"date": d.date().isoformat(), "open": round(r.open, 4), "high": round(r.high, 4),
                          "low": round(r.low, 4), "close": round(r.close, 4), "volume": float(r.volume)}
                         for d, r in zip(prices.index, prices.itertuples())]
    return out


STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


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
