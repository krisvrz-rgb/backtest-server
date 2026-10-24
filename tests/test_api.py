import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.data import synthetic
from app.engine import run_backtest
from app.main import app
from app.pine import to_pine
from app.strategies import STRATEGIES, get_strategy

client = TestClient(app)


def test_buy_and_hold_matches_price_from_first_open():
    prices = synthetic("X", "2020-01-01", "2021-12-31", seed=1)
    res = run_backtest(prices, get_strategy("buy_and_hold"), 10_000, fee_bps=0)
    expected = 10_000 * prices["close"].iloc[-1] / prices["open"].iloc[1]
    assert np.isclose(res.equity.iloc[-1], expected)


def test_no_lookahead():
    prices = synthetic("X", "2020-01-01", "2020-12-31", seed=2)
    res = run_backtest(prices, get_strategy("sma_crossover", fast=5, slow=20), fee_bps=0)
    assert res.position.iloc[0] == 0


def test_fees_reduce_return():
    prices = synthetic("X", seed=3)
    s = get_strategy("rsi_mean_reversion")
    assert run_backtest(prices, s, fee_bps=50).equity.iloc[-1] < run_backtest(prices, s, fee_bps=0).equity.iloc[-1]


def test_all_strategies_have_pine():
    for name in STRATEGIES:
        code = to_pine(get_strategy(name))
        assert code.startswith("//@version=6") and "strategy(" in code


def test_api_backtest_and_compare():
    r = client.post("/backtest", json={"symbol": "TEST", "strategy": "sma_crossover", "source": "synthetic",
                                       "params": {"fast": 20, "slow": 50}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"sharpe", "max_drawdown", "cagr"} <= body["metrics"].keys()
    r = client.post("/compare", json={"symbols": ["A", "B"], "source": "synthetic",
                                      "strategies": [{"name": "buy_and_hold"}, {"name": "momentum"}]})
    assert r.status_code == 200 and len(r.json()) == 4


def test_api_rejects_unknown():
    r = client.post("/backtest", json={"symbol": "T", "strategy": "nope", "source": "synthetic"})
    assert r.status_code == 400


def test_upload_tradingview_csv(tmp_path, monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "DATA_DIR", tmp_path)
    df = synthetic("X", "2023-01-01", "2023-06-30", seed=4).reset_index()
    df["time"] = df["date"].astype("int64") // 10**9
    csv = df.drop(columns="date")[["time", "open", "high", "low", "close", "volume"]].to_csv(index=False)
    r = client.post("/data/xyz", files={"file": ("x.csv", csv)})
    assert r.status_code == 200, r.text
    assert (tmp_path / "XYZ.csv").exists()


def test_hyperliquid_parsing(monkeypatch):
    import httpx

    from app import data

    candles = [{"t": 1704067200000 + i * 86_400_000, "o": "100", "h": "110", "l": "90", "c": str(100 + i), "v": "5"}
               for i in range(10)]

    class Resp:
        def raise_for_status(self): ...
        def json(self): return candles

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp())
    df = data.load_hyperliquid("BTC", "2024-01-01", "2024-01-10")
    assert len(df) == 10 and df["close"].iloc[-1] == 109.0


def test_start_page():
    r = client.get("/")
    assert r.status_code == 200 and "Backtest Server" in r.text


def test_trade_pnl_adds_up_to_net_profit():
    prices = synthetic("X", "2015-01-01", "2024-12-31", seed=5)
    res = run_backtest(prices, get_strategy("rsi_mean_reversion"), 10_000, fee_bps=10)
    total = sum(t["pnl"] for t in res.trades)
    assert np.isclose(total, res.metrics["net_profit"], rtol=1e-9, atol=1e-6)
    m = res.metrics
    assert m["winning_trades"] + m["losing_trades"] == m["closed_trades"]
    assert np.isclose(m["realized_pnl"] + m["open_pnl"], m["net_profit"])


def test_api_includes_prices_market_and_monthly():
    r = client.post("/backtest", json={"symbol": "TEST", "strategy": "sma_crossover", "source": "synthetic",
                                       "params": {"fast": 20, "slow": 50}, "include_prices": True,
                                       "start": "2018-01-01"})
    body = r.json()
    assert body["market_benchmark"] == "SPY" and body["market_metrics"]["sharpe"] is not None
    assert body["equity_curve"][0]["market"] == 10_000
    assert len(body["prices"]) == len(body["equity_curve"])
    assert body["monthly_returns"][0]["year"] == 2018


def test_warmup_lets_indicators_trade_from_start():
    from app.data import get_prices
    prices = get_prices("X", "synthetic", "2018-01-01", "2020-12-31", warmup=True)
    res = run_backtest(prices, get_strategy("sma_crossover"), start="2018-01-01")
    assert res.equity.index[0] >= pd.Timestamp("2018-01-01")
    assert prices.index[0] < pd.Timestamp("2017-01-01")


def test_every_strategy_runs_and_has_pine():
    prices = synthetic("X", "2012-01-01", "2020-12-31", seed=7)
    for name, cls in STRATEGIES.items():
        res = run_backtest(prices, get_strategy(name), start="2014-01-01")
        assert np.isfinite(res.equity.iloc[-1]), name
        code = to_pine(get_strategy(name))
        assert "entryCond = " in code and "exitCond = " in code, name
    assert len(STRATEGIES) >= 100


def test_rank_sorts_by_return():
    r = client.post("/rank", json={"symbol": "TEST", "source": "synthetic", "start": "2016-01-01",
                                   "categories": ["trend", "breakout"]})
    assert r.status_code == 200, r.text
    body = r.json()
    rets = [x["total_return"] for x in body["results"]]
    assert rets == sorted(rets, reverse=True) and not body["errors"]
    assert {x["category"] for x in body["results"]} <= {"trend", "breakout", "benchmark"}
    assert body["buy_and_hold"] is not None and not body["buy_and_hold"]["beats_buy_and_hold"]


def test_buy_and_hold_strategy_equals_benchmark():
    prices = synthetic("X", "2015-01-01", "2020-12-31", seed=8)
    res = run_backtest(prices, get_strategy("buy_and_hold"), fee_bps=5)
    assert np.allclose(res.equity, res.benchmark)


def test_excess_return_vs_buy_and_hold():
    prices = synthetic("X", "2015-01-01", "2020-12-31", seed=9)
    res = run_backtest(prices, get_strategy("sma_crossover", fast=20, slow=50))
    m = res.metrics
    assert np.isclose(m["excess_profit"], res.equity.iloc[-1] - res.benchmark.iloc[-1])
    assert np.isclose(m["excess_return"], m["total_return"] - res.benchmark_metrics["total_return"])


def test_optimize_splits_train_and_test():
    r = client.post("/optimize", json={"symbol": "T", "source": "synthetic", "start": "2014-01-01",
                                       "strategy": "sma_crossover", "max_combos": 20})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["train_period"]["end"] < d["test_period"]["start"]
    assert d["default"]["is_default"] and d["combos_tested"] <= 21
    scores = [x["score"] for x in d["results"] if x["score"] is not None]
    assert scores == sorted(scores, reverse=True)
    assert all(x["params"]["fast"] < x["params"]["slow"] for x in d["results"])


def test_last_signals():
    from app.engine import last_signals
    idx = pd.bdate_range("2024-01-01", periods=10)
    s = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 1, 1], index=idx, dtype=float)
    r = last_signals(s)
    assert r["last_buy_signal"] == idx[5].date().isoformat() and r["signal_open"] and r["bars_since_buy"] == 4
    assert r["last_sell_signal"] == idx[3].date().isoformat()
    r = last_signals(pd.Series([0, 1, 0], index=idx[:3], dtype=float))
    assert not r["signal_open"] and r["bars_since_buy"] == 1


def test_tradingview_symbol_mapping_and_parsing():
    from app.watchlist import parse_tradingview, tv_to_yahoo
    assert tv_to_yahoo("NASDAQ:AAPL") == "AAPL" and tv_to_yahoo("NYSE:BRK.B") == "BRK-B"
    assert tv_to_yahoo("XETR:AIXA") == "AIXA.DE" and tv_to_yahoo("EURONEXT:SOF") == "SOF.BR"
    assert tv_to_yahoo("BITSTAMP:BTCUSD") == "BTC-USD" and tv_to_yahoo("COINBASE:SOLEUR") == "SOL-EUR"
    assert tv_to_yahoo("TVC:VIX") is None and tv_to_yahoo("CME_MINI:NQ1!") is None
    groups = parse_tradingview("###AANDELEN,NASDAQ:MU,NYSE:ANET,###CRYPTO,BITSTAMP:BTCUSD")
    assert [g["name"] for g in groups] == ["AANDELEN", "CRYPTO"] and len(groups[0]["symbols"]) == 2


def test_watchlist_scan_end_to_end(tmp_path, monkeypatch):
    import time as _t
    import app.watchlist as wl
    monkeypatch.setattr(wl, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    monkeypatch.setattr(wl, "SCAN_FILE", tmp_path / "scan.json")
    monkeypatch.setattr(wl, "DATA_DIR", tmp_path)
    r = client.post("/api/watchlist/import", json={"text": "###TEST,NASDAQ:AAA,NASDAQ:BBB"})
    assert r.json()["symbols"] == 2
    r = client.post("/api/watchlist/scan", json={"start": "2019-01-01", "source": "synthetic"})
    assert r.status_code == 200, r.text
    for _ in range(120):
        if not client.get("/api/watchlist/scan").json()["running"]:
            break
        _t.sleep(0.5)
    view = client.get("/api/watchlist").json()
    res = [s["result"] for g in view["groups"] for s in g["symbols"]]
    assert all(x and "error" not in x for x in res), res
    first = res[0]
    assert 0 <= first["beat_count"] <= first["strategies"] and first["best"]["times_bh"] > 0


def test_import_from_portfolio(monkeypatch, tmp_path):
    import httpx
    import app.watchlist as wl
    tv = tmp_path / "tv.json"
    tv.write_text('{"lijsten":[{"naam":"Watchlist","symbolen":["EURONEXT:IWDA","NASDAQ:POET","XETR:AIXA"]}]}')
    monkeypatch.setattr(wl, "TV_WATCHLIST_FILE", tv)
    responses = {
        "/api/data": {"portfolios": {
            "kristof": {"holdings": [{"ticker": "POET", "waarde": 100, "currency": "USD"}, {"ticker": "X", "waarde": 0}]},
            "freya": {"holdings": [{"ticker": "IWDA", "waarde": 50, "currency": "EUR"}, {"ticker": "POET", "waarde": 5}]},
            "cezanne": {"holdings": [{"ticker": "ASML", "waarde": 10, "currency": "EUR"}]}}},
        "/api/wlpaneel": {"lijsten": [{"naam": "Watchlist", "tickers": ["AIXA", "POET", "SHOP"]}, {"naam": "M", "tickers": ["BBAI"]}]},
    }

    class R:
        def __init__(self, d): self.d = d
        def json(self): return self.d

    monkeypatch.setattr(httpx, "get", lambda url, **k: R(next(v for p, v in responses.items() if url.endswith(p))))
    groups = {g["name"]: g["symbols"] for g in wl.import_from_portfolio("http://pf")}
    assert list(groups) == ["GEZIN", "KRISTOF", "FREYA", "CÉZANNE", "WATCHLIST"]
    assert len(groups["GEZIN"]) == 3 and len(groups["KRISTOF"]) == 1
    assert {s["yahoo"] for s in groups["FREYA"]} == {"IWDA.AS", "POET"}
    assert groups["CÉZANNE"][0]["yahoo"] == "ASML.AS"
    assert [s["yahoo"] for s in groups["WATCHLIST"]] == ["AIXA.DE", "SHOP"]
