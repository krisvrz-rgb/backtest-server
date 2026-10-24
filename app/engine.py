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


def _extract_trades(open_: pd.Series, close: pd.Series, position: pd.Series,
                    equity: pd.Series, equity_open: pd.Series) -> list[dict]:
    """Trades met prijsrendement en echte winst/verlies in geld (na kosten).

    Instap en uitstap gebeuren op de opening. De P&L is het verschil in kapitaal tussen de slotkoers
    voor de instap en de opening van de uitstap, dus inclusief kosten van beide transacties.
    """
    trades, entry = [], None
    dates = position.index
    prev = 0.0

    def make(entry, exit_date, exit_price, exit_equity, side_sign):
        i, entry_date, entry_price = entry
        start_equity = equity.iloc[i - 1] if i > 0 else equity.iloc[0]
        pnl = float(exit_equity - start_equity)
        return {
            "entry_date": entry_date.date().isoformat(),
            "exit_date": exit_date.date().isoformat() if exit_date is not None else None,
            "side": "long" if side_sign > 0 else "short",
            "entry_price": float(entry_price), "exit_price": float(exit_price),
            "return": float((exit_price / entry_price - 1) * side_sign),
            "pnl": pnl, "pnl_pct": pnl / float(start_equity),
            "bars": int((dates.get_loc(exit_date) if exit_date is not None else len(dates) - 1) - i),
        }

    for i, (date, pos) in enumerate(position.items()):
        if pos != prev:
            if prev != 0 and entry is not None:
                trades.append(make(entry, date, open_[date], equity_open[date], np.sign(prev)))
                entry = None
            if pos != 0:
                entry = (i, date, open_[date])
        prev = pos
    if entry is not None:
        trades.append(make(entry, None, close.iloc[-1], equity.iloc[-1], np.sign(prev)))
    return trades


def trade_stats(trades: list[dict], initial_capital: float) -> dict:
    """Statistieken over afgesloten trades; een nog openstaande trade telt niet mee."""
    closed = [t for t in trades if t["exit_date"] is not None]
    pnls = np.array([t["pnl"] for t in closed])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    open_trade = next((t for t in trades if t["exit_date"] is None), None)
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "closed_trades": len(closed),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "percent_profitable": float(len(wins) / len(closed)) if closed else 0.0,
        "realized_pnl": float(pnls.sum()),
        "realized_return": float(pnls.sum() / initial_capital),
        "open_pnl": float(open_trade["pnl"]) if open_trade else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "largest_win": float(wins.max()) if len(wins) else 0.0,
        "largest_loss": float(losses.min()) if len(losses) else 0.0,
        "avg_bars_in_trade": float(np.mean([t["bars"] for t in closed])) if closed else 0.0,
    }


def monthly_returns(equity: pd.Series, initial_capital: float) -> list[dict]:
    monthly = equity.groupby([equity.index.year, equity.index.month]).last()
    rets = monthly.pct_change()
    rets.iloc[0] = monthly.iloc[0] / initial_capital - 1
    return [{"year": int(y), "month": int(m), "return": float(r)} for (y, m), r in rets.items()]


def _simulate(open_: pd.Series, close: pd.Series, position: pd.Series, initial_capital: float,
              cost_bps: float) -> tuple[pd.Series, pd.Series]:
    """Kapitaal per dag (op de slot) en bij de opening, voor een positie die op de open ingaat."""
    prev_pos = position.shift(1).fillna(0.0)
    overnight = (open_ / close.shift(1) - 1).fillna(0.0)
    intraday = close / open_ - 1
    costs = (position - prev_pos).abs() * cost_bps / 10_000
    open_factor = (1 + prev_pos * overnight) * (1 - costs)
    equity = initial_capital * (open_factor * (1 + position * intraday)).cumprod()
    equity_open = equity.shift(1).fillna(initial_capital) * open_factor
    return equity, equity_open


def run_backtest(prices: pd.DataFrame, strategy: Strategy, initial_capital: float = 10_000.0,
                 fee_bps: float = 5.0, slippage_bps: float = 0.0, allow_short: bool = False,
                 start: str | pd.Timestamp | None = None) -> BacktestResult:
    """Backtest vanaf `start`. Koersen voor `start` dienen alleen om de indicatoren op te warmen,
    net zoals TradingView de hele historie gebruikt maar pas vanaf de startdatum handelt."""
    active = prices.index >= pd.Timestamp(start) if start is not None else None
    raw = strategy.signals(prices, pd.Series(active, index=prices.index) if active is not None else None)
    signal = raw.reindex(prices.index).fillna(0.0).clip(-1, 1)
    if active is not None:
        prices, signal = prices[active], signal[active]
    close, open_ = prices["close"], prices["open"]
    if not allow_short:
        signal = signal.clip(lower=0)
    # Zelfde uitvoering als TradingView: signaal op slot van bar t, order gevuld op open van bar t+1.
    position = signal.shift(1).fillna(0.0)
    equity, equity_open = _simulate(open_, close, position, initial_capital, fee_bps + slippage_bps)
    # Buy & hold als benchmark: zelfde uitvoering en kosten (koop op de open van de tweede bar),
    # zodat de strategie "buy_and_hold" precies gelijk is aan de benchmark.
    hold = pd.Series(1.0, index=close.index)
    hold.iloc[0] = 0.0
    benchmark, _ = _simulate(open_, close, hold, initial_capital, fee_bps + slippage_bps)

    trades = _extract_trades(open_, close, position, equity, equity_open)
    metrics = compute_metrics(equity)
    stats = trade_stats(trades, initial_capital)
    metrics.update({
        "num_trades": len(trades),
        "win_rate": stats["percent_profitable"],
        "avg_trade_return": float(np.mean([t["return"] for t in trades])) if trades else 0.0,
        "exposure": float((position != 0).mean()),
        "final_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - initial_capital),
        # Wat de strategie méér (of minder) opleverde dan gewoon het aandeel vasthouden.
        "excess_profit": float(equity.iloc[-1] - benchmark.iloc[-1]),
        "excess_return": float((equity.iloc[-1] - benchmark.iloc[-1]) / initial_capital),
        **{k: v for k, v in stats.items() if k not in metrics},
    })
    bench_metrics = compute_metrics(benchmark)
    bench_metrics["net_profit"] = float(benchmark.iloc[-1] - initial_capital)
    return BacktestResult(equity, benchmark, position, trades, metrics, bench_metrics)
