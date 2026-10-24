from __future__ import annotations

from io import StringIO
from typing import Any, Literal

import pandas as pd
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import DATA_DIR, _normalize, get_prices
from .engine import compute_metrics, monthly_returns, run_backtest
from .optimize import OBJECTIVES, optimize
from .pine import to_pine
from .strategies import CATEGORIES, STRATEGIES, get_strategy

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


def _load(symbol: str, source: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        return get_prices(symbol, source, start, end, warmup=True)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


def _run(req: BacktestRequest, prices: pd.DataFrame | None = None) -> dict:
    if prices is None:
        prices = _load(req.symbol, req.source, req.start, req.end)
    try:
        strat = get_strategy(req.strategy, **req.params)
    except ValueError as e:
        raise HTTPException(400, str(e))
    res = run_backtest(prices, strat, req.initial_capital, req.fee_bps, req.slippage_bps, req.allow_short,
                       start=req.start)
    prices = prices.loc[res.equity.index]
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
    return [{"name": n, "category": c.category, "category_label": CATEGORIES.get(c.category, c.category),
             "description": c.description, "params": c.params} for n, c in STRATEGIES.items()]


@app.get("/categories")
def list_categories():
    return [{"key": k, "label": v, "count": sum(c.category == k for c in STRATEGIES.values())}
            for k, v in CATEGORIES.items()]


SortKey = Literal["total_return", "net_profit", "sharpe", "percent_profitable", "max_drawdown", "cagr", "profit_factor"]


class RankRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    source: Source = "csv"
    start: str | None = Field(None, examples=["2015-01-01"])
    end: str | None = None
    initial_capital: float = 10_000.0
    fee_bps: float = 5.0
    categories: list[str] | None = Field(None, description="Alleen deze categorieën; leeg = alle")
    sort_by: SortKey = "total_return"


@app.post("/rank")
def rank(req: RankRequest):
    """Test alle strategieën (met standaardinstellingen) op één ticker en rangschik ze."""
    prices = _load(req.symbol, req.source, req.start, req.end)
    rows = []
    for name, cls in STRATEGIES.items():
        if req.categories and cls.category not in req.categories and cls.category != "benchmark":
            continue
        try:
            r = _run(BacktestRequest(symbol=req.symbol, strategy=name, source=req.source, start=req.start, end=req.end,
                                     initial_capital=req.initial_capital, fee_bps=req.fee_bps,
                                     include_equity=False, market_benchmark=None), prices)
        except Exception as e:  # noqa: BLE001 - één kapotte strategie mag de ranglijst niet breken
            rows.append({"strategy": name, "category": cls.category, "error": str(e)})
            continue
        m = r["metrics"]
        rows.append({
            "strategy": name, "category": cls.category, "category_label": CATEGORIES.get(cls.category, cls.category),
            "description": cls.description, "params": r["params"],
            **{k: m[k] for k in ("total_return", "cagr", "net_profit", "sharpe", "max_drawdown", "percent_profitable",
                                 "num_trades", "closed_trades", "winning_trades", "profit_factor", "exposure")},
            "beats_buy_and_hold": m["total_return"] > r["benchmark_metrics"]["total_return"],
            "period": r["period"],
        })
    ok = [x for x in rows if "error" not in x]
    # Hoogste eerst; bij max drawdown de minst diepe (dichtst bij 0) eerst.
    ok.sort(key=lambda x: x[req.sort_by] if x[req.sort_by] is not None else float("-inf"), reverse=True)
    bh = next((x for x in ok if x["strategy"] == "buy_and_hold"), None)
    return {"symbol": req.symbol.upper(), "sort_by": req.sort_by, "count": len(ok),
            "buy_and_hold": bh, "results": ok, "errors": [x for x in rows if "error" in x]}


@app.post("/backtest")
def backtest(req: BacktestRequest):
    return _run(req)


@app.post("/compare")
def compare(req: CompareRequest):
    rows = []
    for sym in req.symbols:
        for s in req.strategies:
            r = _run(BacktestRequest(symbol=sym, strategy=s["name"], params=s.get("params", {}), source=req.source,
                                     start=req.start, end=req.end, fee_bps=req.fee_bps, include_equity=False,
                                     market_benchmark=None))
            rows.append({"symbol": r["symbol"], "strategy": r["strategy"], "params": r["params"], **r["metrics"]})
    rows.sort(key=lambda x: x["sharpe"], reverse=True)
    return rows


class OptimizeRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    strategy: str = Field(..., examples=["rsi_mean_reversion"])
    source: Source = "csv"
    start: str | None = Field(None, examples=["2015-01-01"])
    end: str | None = None
    initial_capital: float = 10_000.0
    fee_bps: float = 5.0
    objective: Literal[OBJECTIVES] = "total_return"  # type: ignore[valid-type]
    train_pct: float = Field(0.7, ge=0.3, le=0.9, description="Deel van de periode om op te leren; de rest is controle")
    max_combos: int = Field(150, ge=1, le=500)


@app.post("/optimize")
def optimize_endpoint(req: OptimizeRequest):
    """Zoek betere instellingen op het eerste deel van de periode en controleer ze op het laatste deel."""
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"Onbekende strategie '{req.strategy}'")
    prices = _load(req.symbol, req.source, req.start, req.end)
    try:
        return optimize(prices, req.strategy, req.start, req.objective, req.train_pct, req.max_combos,
                        req.initial_capital, req.fee_bps)
    except ValueError as e:
        raise HTTPException(400, str(e))


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
