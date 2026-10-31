from __future__ import annotations

import threading
from io import StringIO
from typing import Any, Literal

import pandas as pd
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import DATA_DIR, _normalize, get_prices
from .adaptive import walk_forward
from .schedule import load_schedule, next_run, run_now, save_schedule, start_scheduler
from .engine import compute_metrics, monthly_returns, run_backtest
from .optimize import OBJECTIVES, optimize
from .pine import to_pine
from .ranking import rank_strategies
from .watchlist import (PORTFOLIO_LIST, default_periods, PORTFOLIO_URL, SCANNER, TV_PREFIX, import_from_portfolio, import_from_tradingview,
                        list_name, load_watchlist, parse_tradingview, replace_lists, save_watchlist, tv_to_yahoo,
                        scan_cached, ticker_detail, watchlist_view)
from .robust import cross_ticker
from . import desk, desk_coach
from .desk_auto import swarm
from .picks import picks, strategy_scores
from .strategies import CATEGORIES, STRATEGIES, get_strategy
from .strategies.plot import indicators
from .strategies.explain import explain_strategy, indicator_info

app = FastAPI(title="Backtest Server", version="0.1.0",
              description="Backtest handelsstrategieën op aandelen en exporteer ze naar TradingView Pine Script.")
# Grote JSON-antwoorden (watchlist) gecomprimeerd versturen.
app.add_middleware(GZipMiddleware, minimum_size=2000)


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
    full_prices = prices
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
        try:
            entry, exit_ = strat.rules()
            ind = indicators(entry, exit_, full_prices, prices.index)
            # Uitleg in gewone taal bij elk indicatorvenster en elke lijn op de koersgrafiek.
            for pane in ind["panes"]:
                pane["info"] = indicator_info(pane["title"])
            for ov in ind["overlays"]:
                ov["info"] = indicator_info(ov["label"])
            out["indicators"] = ind
        except Exception:  # tekenen mag een backtest nooit doen mislukken
            out["indicators"] = {"overlays": [], "panes": []}
    return out


STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def overview_page():
    return FileResponse(STATIC_DIR / "overview.html")


@app.get("/tool", include_in_schema=False)
def tool_page():
    return FileResponse(INDEX_HTML)


@app.get("/watchlist", include_in_schema=False)
def watchlist_page():
    return FileResponse(STATIC_DIR / "watchlist.html")


@app.get("/aanraders", include_in_schema=False)
def picks_page():
    return FileResponse(STATIC_DIR / "picks.html")


@app.get("/desk", include_in_schema=False)
def desk_page():
    return FileResponse(STATIC_DIR / "desk.html")


class DeskOrder(BaseModel):
    coin: str
    side: Literal["long", "short"] | None = None
    usd: float = 0
    token: str | None = None


@app.get("/api/desk/status")
def desk_status():
    return desk.status()


@app.get("/api/desk/candles")
def desk_candles(coin: str = "BTC", interval: str = "1m", minutes: int = Query(180, ge=10, le=5000)):
    try:
        return desk.candles(coin, interval, minutes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Hyperliquid niet bereikbaar: {e}")


@app.get("/api/desk/scan")
def desk_scan():
    """Voorselectie van Hyperliquid-perps (ook tokenized aandelen) die het meest bewegen tegenover de kosten."""
    try:
        return desk.scan_cached(top=swarm.scan_n)
    except Exception as e:
        raise HTTPException(502, f"Hyperliquid niet bereikbaar: {e}")


@app.get("/api/desk/account")
def desk_account():
    try:
        return desk.account()
    except Exception as e:
        raise HTTPException(502, f"Hyperliquid niet bereikbaar: {e}")


@app.post("/api/desk/open")
def desk_open(req: DeskOrder):
    if req.side is None:
        raise HTTPException(400, "side ontbreekt")
    try:
        return desk.open_position(req.coin, req.side, req.usd, req.token)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/desk/close")
def desk_close(req: DeskOrder):
    try:
        return desk.close_position(req.coin, req.token)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


class DeskAuto(BaseModel):
    token: str | None = None
    on: bool | None = None
    confirm: bool | None = None
    coins: dict[str, bool] | None = None
    window_min: int | None = None
    reset_paper: float | None = None
    min_edge: float | None = None
    model: dict | None = None
    max_lev: float | None = None
    auto_scan: bool | None = None
    scan_n: int | None = None
    kelly_frac: float | None = None
    max_margin_pct: float | None = None
    sl_sigma: float | None = None
    tp_sigma: float | None = None
    hold_max: int | None = None
    learn: bool | None = None
    risk_pct: float | None = None


class DeskApproval(BaseModel):
    token: str | None = None
    id: int
    yes: bool


@app.get("/api/desk/auto")
def desk_auto_state(since: int = 0):
    return swarm.snapshot(since)


@app.post("/api/desk/auto")
def desk_auto_set(req: DeskAuto):
    try:
        if desk.status()["live"]:  # zodra echt geld mogelijk is, vraagt elke wijziging het token
            desk.check_token(req.token)
        elif req.on is not None or req.confirm is not None:
            raise PermissionError("Live handel staat uit; alleen markten volgen op papier kan")
        swarm.set(on=req.on, confirm=req.confirm, coins=req.coins,
                  window_min=req.window_min, reset_paper=req.reset_paper, min_edge=req.min_edge, model=req.model,
                  max_lev=req.max_lev, risk_pct=req.risk_pct, auto_scan=req.auto_scan, scan_n=req.scan_n,
                  kelly_frac=req.kelly_frac, max_margin_pct=req.max_margin_pct,
                  sl_sigma=req.sl_sigma, tp_sigma=req.tp_sigma, hold_max=req.hold_max, learn=req.learn)
        return swarm.snapshot(swarm.seq)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


class DeskLevel(BaseModel):
    token: str | None = None
    coin: str
    kind: str
    price: float | None = None


@app.post("/api/desk/auto/level")
def desk_auto_level(req: DeskLevel):
    try:
        if desk.status()["live"]:
            desk.check_token(req.token)
        return swarm.set_level(req.coin, req.kind, req.price)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


class DeskCoach(BaseModel):
    token: str = ""
    url: str | None = None
    model: str | None = None
    hour: int | None = Field(None, ge=0, le=23)
    enabled: bool | None = None
    detect: bool = False


@app.get("/api/desk/pnl")
def desk_pnl():
    """Alle afgesloten trades, compact, voor de PnL-grafieken per uur/dag/week/maand/jaar."""
    from . import desk_learn
    seen, rows = set(), []
    for h in list(desk_learn.trades()) + list(swarm.history):
        k = (h.get("t"), h.get("coin"))
        if k in seen or h.get("pnl") is None:
            continue
        seen.add(k)
        rows.append([h["t"], round(h["pnl"], 4), bool(h.get("paper", True)), h.get("coin"), h.get("side"),
                     h.get("reason") or "venster"])
    rows.sort()
    return {"trades": rows, "paper_start": swarm.paper.get("start")}


@app.get("/api/desk/coach")
def desk_coach_get():
    """Je lokale taalmodel als dagelijkse coach: instellingen, of hij bezig is, en de laatste rapporten."""
    return {"config": desk_coach.config(), "status": desk_coach.STATUS, "reports": desk_coach.reports()[-7:]}


@app.post("/api/desk/coach/config")
def desk_coach_config(req: DeskCoach):
    try:
        if desk.status()["live"]:
            desk.check_token(req.token)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if req.detect:
        found = desk_coach.detect()
        if not found:
            raise HTTPException(404, "Geen lokaal taalmodel gevonden op " + ", ".join(desk_coach.CANDIDATES))
        return {"config": desk_coach.save_config(url=found["url"], model=found["model"]), "models": found["models"]}
    if req.url and not req.url.startswith(("http://127.0.0.1", "http://localhost", "http://192.168.")):
        raise HTTPException(400, "Alleen een taalmodel op deze machine of in je thuisnetwerk")
    return {"config": desk_coach.save_config(url=req.url, model=req.model, hour=req.hour, enabled=req.enabled)}


@app.post("/api/desk/coach/run")
def desk_coach_run():
    if desk_coach.STATUS["running"]:
        raise HTTPException(409, "De coach is al bezig")
    desk_coach.start_background_run()
    return {"ok": True}


@app.post("/api/desk/auto/approve")
def desk_auto_approve(req: DeskApproval):
    try:
        desk.check_token(req.token)
        return swarm.approve(req.id, req.yes)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/strategies")
def list_strategies():
    return [{"name": n, "category": c.category, "category_label": CATEGORIES.get(c.category, c.category),
             "description": c.description, "params": c.params, "explain": explain_strategy(n, c.category)}
            for n, c in STRATEGIES.items()]


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
    out = rank_strategies(prices, req.start, req.initial_capital, req.fee_bps, req.categories, req.sort_by)
    return {"symbol": req.symbol.upper(), **out}


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


class AdaptiveRequest(BaseModel):
    symbol: str = Field(..., examples=["AMZN"])
    source: Source = "csv"
    start: str = Field("2020-01-01", description="Vanaf hier telt het resultaat; de leerperiode ligt ervoor")
    end: str | None = None
    initial_capital: float = 10_000.0
    fee_bps: float = 5.0
    train_years: float = Field(3.0, ge=1, le=10, description="Lengte van de leerperiode vóór elk beslismoment")
    step: Literal["MS", "QS", "YS"] = Field("QS", description="Opnieuw kiezen per maand, kwartaal of jaar")
    objective: Literal["total_return", "sharpe"] = "total_return"
    by_regime: bool = Field(True, description="Alleen dagen met dezelfde markttoestand gebruiken om te kiezen")
    categories: list[str] | None = None


@app.post("/adaptive")
def adaptive_endpoint(req: AdaptiveRequest):
    """Automatische strategiekeuze per markttoestand, getest met walk-forward (geen kijken naar de uitslag)."""
    first = (pd.Timestamp(req.start) - pd.Timedelta(days=int(req.train_years * 365.25))).date().isoformat()
    prices = _load(req.symbol, req.source, first, req.end)
    try:
        out = walk_forward(prices, req.start, req.initial_capital, req.fee_bps, req.train_years, req.step,
                           req.objective, req.by_regime, req.categories)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"symbol": req.symbol.upper(), **out}


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


# --------------------------------------------------------------------------- watchlist


@app.get("/api/watchlist")
def get_watchlist():
    """Watchlist met groepen en per ticker de laatste scanresultaten."""
    return watchlist_view()


@app.get("/api/watchlist/detail")
def get_watchlist_detail(symbol: str):
    """Per ticker: houdbaarheid van de beste strategie en de beste strategie per maand (laatste 24 maanden)."""
    return ticker_detail(symbol)


@app.get("/api/picks")
def get_picks(list_: str | None = Query(None, alias="list"), group: str | None = None, min_score: float = 0.0):
    """Aanraders: tickers met nu een open koopsignaal van een strategie met bewijs dat ze werkt (geen advies)."""
    seen, tickers = set(), []
    for g in load_watchlist():
        if (list_ and list_name(g) != list_) or (group and g["name"] != group):
            continue
        for s in g["symbols"]:
            if s.get("yahoo") and s["yahoo"] not in seen:
                seen.add(s["yahoo"])
                tickers.append({"yahoo": s["yahoo"], "tv": s["tv"], "group": g["name"], "list": list_name(g)})
    return picks(scan_cached()["results"], tickers, min_score)


@app.get("/api/scores")
def get_scores(symbol: str):
    """Score per strategie voor één ticker uit de laatste scan (vergeleken met de andere tickers in dezelfde lijsten)."""
    scan = scan_cached()["results"]
    lists = {list_name(g) for g in load_watchlist() if any(s.get("yahoo") == symbol for s in g["symbols"])}
    peers = list(dict.fromkeys(s["yahoo"] for g in load_watchlist() if list_name(g) in lists
                               for s in g["symbols"] if s.get("yahoo"))) or [symbol]
    return strategy_scores(scan, symbol, peers)


@app.get("/api/robust")
def get_robust(period: str = "2j", list_: str | None = Query(None, alias="list"), group: str | None = None):
    """Welke strategieën werken op veel tickers tegelijk (over een lijst of groep uit de watchlist)."""
    syms = [s["yahoo"] for g in load_watchlist() if (not list_ or list_name(g) == list_) and (not group or g["name"] == group)
            for s in g["symbols"] if s.get("yahoo")]
    return cross_ticker(scan_cached()["results"], list(dict.fromkeys(syms)), period)


class WatchlistImport(BaseModel):
    text: str = Field(..., description="TradingView-export (###SECTIE,NASDAQ:AAPL,...) of één ticker per regel")
    mode: Literal["replace", "merge"] = "replace"
    group: str = Field("WATCHLIST", description="Groepsnaam voor tickers zonder ###SECTIE")


@app.post("/api/watchlist/import")
def import_watchlist(req: WatchlistImport):
    parsed = [g | {"list": TV_PREFIX + "Geplakt"} for g in parse_tradingview(req.text, req.group)]
    if req.mode == "replace":
        new = replace_lists(load_watchlist(), parsed, lambda name: name == TV_PREFIX + "Geplakt")
    else:
        groups = load_watchlist()
        for g in parsed:
            target = next((x for x in groups if x["name"] == g["name"]), None)
            if target is None:
                groups.append(g)
            else:
                known = {s["tv"] for s in target["symbols"]}
                target["symbols"] += [s for s in g["symbols"] if s["tv"] not in known]
        new = groups
    save_watchlist(new)
    return {"groups": len(parsed), "symbols": sum(len(g["symbols"]) for g in parsed),
            "without_data": [s["tv"] for g in parsed for s in g["symbols"] if not s["yahoo"]]}


class PortfolioImport(BaseModel):
    url: str = Field(PORTFOLIO_URL, description="Adres van de portfolio server")


@app.post("/api/watchlist/import-portfolio")
def import_portfolio(req: PortfolioImport):
    """Neem de groepen over van de portfolio server (GEZIN, KRISTOF, FREYA, CÉZANNE, WATCHLIST)."""
    try:
        groups = import_from_portfolio(req.url.rstrip("/"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Portfolio server niet bereikbaar of onverwacht antwoord: {e}")
    save_watchlist(replace_lists(load_watchlist(), groups, lambda name: name == PORTFOLIO_LIST))
    return {"groups": [{"name": g["name"], "count": len(g["symbols"])} for g in groups],
            "symbols": sum(len(g["symbols"]) for g in groups)}


class TradingViewImport(BaseModel):
    path: str | None = Field(None, description="Pad naar de TradingView-export; leeg = TV_WATCHLIST_FILE")
    data: dict[str, Any] | None = Field(None, description="Of de export zelf: {lijsten: [{naam, kleur, symbolen}]}")


@app.post("/api/watchlist/import-tradingview")
def import_tradingview(req: TradingViewImport):
    """Neem alle TradingView-watchlists over als lijsten (secties worden groepen)."""
    try:
        groups = import_from_tradingview(req.data, Path(req.path) if req.path else None)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Kon TradingView-watchlists niet lezen: {e}")
    if not groups:
        raise HTTPException(400, "Geen TradingView-lijsten met symbolen gevonden")
    save_watchlist(replace_lists(load_watchlist(), groups, lambda name: name.startswith(TV_PREFIX)))
    lists: dict[str, int] = {}
    for g in groups:
        lists[g["list"]] = lists.get(g["list"], 0) + len(g["symbols"])
    return {"lists": [{"name": k, "count": v} for k, v in lists.items()], "groups": len(groups),
            "without_data": [s["tv"] for g in groups for s in g["symbols"] if not s["yahoo"]]}

class WatchlistSymbol(BaseModel):
    symbol: str = Field(..., examples=["NASDAQ:AAPL"])
    group: str = "WATCHLIST"
    yahoo: str | None = Field(None, description="Yahoo-ticker als de automatische omzetting niet klopt (bv. SOF.BR)")


@app.post("/api/watchlist/symbols")
def add_symbol(req: WatchlistSymbol):
    groups = load_watchlist()
    target = next((g for g in groups if g["name"] == req.group), None)
    if target is None:
        target = {"name": req.group, "symbols": []}
        groups.append(target)
    target["symbols"] = [s for s in target["symbols"] if s["tv"] != req.symbol]
    target["symbols"].append({"tv": req.symbol, "yahoo": req.yahoo or tv_to_yahoo(req.symbol)})
    save_watchlist(groups)
    return target


@app.delete("/api/watchlist/symbols")
def remove_symbol(group: str, symbol: str, list: str | None = None):
    groups = load_watchlist()
    for g in groups:
        if g["name"] == group and (list is None or list_name(g) == list):
            g["symbols"] = [s for s in g["symbols"] if s["tv"] != symbol]
    save_watchlist([g for g in groups if g["symbols"]])
    return {"ok": True}


class ScanRequest(BaseModel):
    periods: dict[str, Any] = Field(default_factory=lambda: default_periods(),
                                    description="Testperiodes: naam -> startdatum of {start, end}; zonder end = tot vandaag")
    start: str | None = Field(None, description="Eén startdatum in plaats van de periodes")
    fee_bps: float = 5.0
    groups: list[str] | None = Field(None, description="Alleen deze groepen; leeg = alles")
    lists: list[str] | None = Field(None, description="Alleen deze lijsten (bv. 'TradingView · Watchlist')")
    symbols: list[str] | None = Field(None, description="Alleen deze Yahoo-tickers")
    source: Source = "yahoo"


@app.post("/api/watchlist/scan")
def start_scan(req: ScanRequest):
    """Start de scanner op de achtergrond: alle strategieën op elke ticker."""
    syms = req.symbols or list(dict.fromkeys(
        s["yahoo"] for g in load_watchlist()
        if (not req.groups or g["name"] in req.groups) and (not req.lists or list_name(g) in req.lists)
        for s in g["symbols"] if s.get("yahoo")))
    if not syms:
        raise HTTPException(400, "Geen tickers om te scannen")
    periods = {req.start[:4]: req.start} if req.start else req.periods
    if not SCANNER.start(syms, periods, req.fee_bps, source=req.source):
        raise HTTPException(409, "Er loopt al een scan")
    return SCANNER.status()


@app.get("/api/watchlist/scan")
def scan_status():
    return SCANNER.status()


@app.post("/api/watchlist/scan/stop")
def stop_scan():
    SCANNER.stop()
    return SCANNER.status()


# --------------------------------------------------------------------------- automatisch vernieuwen


@app.on_event("startup")
def _start_scheduler():
    start_scheduler()
    swarm.start()  # op papier tot live aan staat; volgt alle aangezette markten en leert 24/7
    threading.Thread(target=desk_coach.scheduler, daemon=True, name="desk-coach").start()


class ScheduleSettings(BaseModel):
    enabled: bool = False
    time: str = Field("22:30", pattern=r"^\d{1,2}:\d{2}$", description="Lokale tijd van de server (UU:MM)")
    days: list[int] = Field([0, 1, 2, 3, 4], description="0 = maandag ... 6 = zondag")
    lists: list[str] = Field(default_factory=list, description="Leeg = alle lijsten")


@app.get("/api/schedule")
def get_schedule():
    cfg = load_schedule()
    return cfg | {"next_run": next_run(cfg)}


@app.post("/api/schedule")
def set_schedule(req: ScheduleSettings):
    cfg = save_schedule(load_schedule() | req.model_dump())
    return cfg | {"next_run": next_run(cfg)}


@app.post("/api/schedule/run")
def run_schedule_now():
    if not run_now():
        raise HTTPException(409, "Er loopt al een scan of er zijn geen tickers")
    return SCANNER.status()
