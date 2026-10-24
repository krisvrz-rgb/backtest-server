"""Alle strategieën op één koersreeks testen en rangschikken (gebruikt door /rank en de watchlist-scanner)."""
from __future__ import annotations

import pandas as pd

from .engine import run_backtest
from .strategies import CATEGORIES, STRATEGIES

RANK_FIELDS = ("total_return", "cagr", "net_profit", "sharpe", "max_drawdown", "percent_profitable", "num_trades",
               "closed_trades", "winning_trades", "profit_factor", "exposure", "last_buy_signal", "last_sell_signal",
               "signal_open", "bars_since_buy", "days_since_buy")


def rank_strategies(prices: pd.DataFrame, start: str | None, initial_capital: float = 10_000.0, fee_bps: float = 5.0,
                    categories: list[str] | None = None, sort_by: str = "total_return") -> dict:
    rows, errors = [], []
    for name, cls in STRATEGIES.items():
        if categories and cls.category not in categories and cls.category != "benchmark":
            continue
        try:
            res = run_backtest(prices, cls(), initial_capital, fee_bps, start=start)
        except Exception as e:  # noqa: BLE001 - één kapotte strategie mag de ranglijst niet breken
            errors.append({"strategy": name, "category": cls.category, "error": str(e)})
            continue
        m = res.metrics
        rows.append({
            "strategy": name, "category": cls.category, "category_label": CATEGORIES.get(cls.category, cls.category),
            "description": cls.description, "params": dict(cls.params),
            **{k: m[k] for k in RANK_FIELDS},
            "beats_buy_and_hold": m["total_return"] > res.benchmark_metrics["total_return"],
            "period": {"start": res.equity.index[0].date().isoformat(), "end": res.equity.index[-1].date().isoformat()},
        })
    # Hoogste eerst; bij max drawdown de minst diepe (dichtst bij 0) eerst.
    rows.sort(key=lambda x: x[sort_by] if x[sort_by] is not None else float("-inf"), reverse=True)
    bh = next((x for x in rows if x["strategy"] == "buy_and_hold"), None)
    return {"sort_by": sort_by, "count": len(rows), "buy_and_hold": bh, "results": rows, "errors": errors}
