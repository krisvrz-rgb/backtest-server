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
                                       "params": {"fast": 20, "slow": 50}, "include_prices": True})
    body = r.json()
    assert body["market_benchmark"] == "SPY" and body["market_metrics"]["sharpe"] is not None
    assert body["equity_curve"][0]["market"] == 10_000
    assert len(body["prices"]) == len(body["equity_curve"])
    assert body["monthly_returns"][0]["year"] == 2018
