"""Vectorized backtest-engine met transactiekosten en standaard performance-metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import Strategy

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    position: pd.Series
    trades: list[dict]
    metrics: dict
    benchmark_metrics: dict


def compute_metrics(equity: pd.Series, risk_free: float = 0.0) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    excess = rets - risk_free / TRADING_DAYS
    sharpe = excess.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (excess.mean() * TRADING_DAYS) / downside if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def _extract_trades(open_: pd.Series, close: pd.Series, position: pd.Series) -> list[dict]:
    trades, entry = [], None
    prev = 0.0
    for date, pos in position.items():
        if pos != prev:
            if prev != 0 and entry is not None:
                ret = (open_[date] / entry[1] - 1) * np.sign(prev)
                trades.append({"entry_date": entry[0].date().isoformat(), "exit_date": date.date().isoformat(),
                               "side": "long" if prev > 0 else "short",
                               "entry_price": float(entry[1]), "exit_price": float(open_[date]),
                               "return": float(ret)})
                entry = None
            if pos != 0:
                entry = (date, open_[date])
        prev = pos
    if entry is not None:
        last = close.index[-1]
        trades.append({"entry_date": entry[0].date().isoformat(), "exit_date": None,
                       "side": "long" if prev > 0 else "short",
                       "entry_price": float(entry[1]), "exit_price": float(close[last]),
                       "return": float((close[last] / entry[1] - 1) * np.sign(prev))})
    return trades


def run_backtest(prices: pd.DataFrame, strategy: Strategy, initial_capital: float = 10_000.0,
                 fee_bps: float = 5.0, slippage_bps: float = 0.0, allow_short: bool = False) -> BacktestResult:
    close, open_ = prices["close"], prices["open"]
    signal = strategy.signals(prices).reindex(close.index).fillna(0.0).clip(-1, 1)
    if not allow_short:
        signal = signal.clip(lower=0)
    # Zelfde uitvoering als TradingView: signaal op slot van bar t, order gevuld op open van bar t+1.
    position = signal.shift(1).fillna(0.0)
    prev_pos = position.shift(1).fillna(0.0)
    overnight = (open_ / close.shift(1) - 1).fillna(0.0)
    intraday = close / open_ - 1
    turnover = (position - prev_pos).abs()
    costs = turnover * (fee_bps + slippage_bps) / 10_000
    strat_rets = (1 + prev_pos * overnight) * (1 + position * intraday) * (1 - costs) - 1
    equity = initial_capital * (1 + strat_rets).cumprod()
    benchmark = initial_capital * (1 + close.pct_change().fillna(0.0)).cumprod()

    trades = _extract_trades(open_, close, position)
    metrics = compute_metrics(equity)
    closed = [t["return"] for t in trades]
    metrics.update({
        "num_trades": len(trades),
        "win_rate": float(np.mean([r > 0 for r in closed])) if closed else 0.0,
        "avg_trade_return": float(np.mean(closed)) if closed else 0.0,
        "exposure": float((position != 0).mean()),
        "final_equity": float(equity.iloc[-1]),
    })
    return BacktestResult(equity, benchmark, position, trades, metrics, compute_metrics(benchmark))
