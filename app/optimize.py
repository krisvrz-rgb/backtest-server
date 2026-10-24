"""Parameters fine-tunen met een eenvoudige walk-forward-controle.

De periode wordt gesplitst: op het eerste deel ("leren", in-sample) zoeken we de beste instellingen,
op het laatste deel ("controle", out-of-sample) kijken we of ze daar ook werken. Een instelling die
alleen in-sample uitblinkt, is waarschijnlijk toevallig op het verleden afgestemd (overfitting).
"""
from __future__ import annotations

import itertools
import random
from typing import Any

import pandas as pd

from .engine import run_backtest
from .strategies import STRATEGIES

# Parameters die een niveau/drempel zijn (bv. RSI 30): die verschuiven we in plaats van te vermenigvuldigen.
LEVEL_HINTS = ("level", "oversold", "overbought", "threshold", "adx_min", "z_", "pct")
LENGTH_STEPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
FLOAT_STEPS = (0.6, 0.8, 1.0, 1.2, 1.5)
OBJECTIVES = ("total_return", "sharpe", "profit_factor", "percent_profitable", "net_profit")


def candidates(name: str, default: Any) -> list[Any]:
    if isinstance(default, bool):
        return [default]
    if any(h in name for h in LEVEL_HINTS):
        span = max(abs(default) * 0.33, 1 if isinstance(default, int) else 0.5)
        vals = [default + k * span for k in (-1, -0.5, 0, 0.5, 1)]
        vals = [round(v) if isinstance(default, int) else round(v, 2) for v in vals]
        if "oversold" in name or "overbought" in name or name in ("low_level", "high_level", "entry_level", "exit_level", "level"):
            vals = [v for v in vals if -100 <= v <= 100]
        return sorted(set(vals))
    if isinstance(default, int):
        return sorted({max(1, round(default * s)) for s in LENGTH_STEPS})
    return sorted({round(default * s, 3) for s in FLOAT_STEPS})


def grid(strategy: str, max_combos: int, seed: int = 42) -> list[dict[str, Any]]:
    defaults = STRATEGIES[strategy].params
    if not defaults:
        return [{}]
    keys = list(defaults)
    values = [candidates(k, defaults[k]) for k in keys]
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combos = [c for c in combos if _valid(c)]
    if len(combos) > max_combos:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_combos - 1)
    if defaults not in combos:
        combos.insert(0, dict(defaults))
    return combos


def _valid(c: dict[str, Any]) -> bool:
    """Combinaties die geen zin hebben overslaan (snel >= traag, oversold >= overbought, ...)."""
    pairs = [("fast", "slow"), ("short", "long_len"), ("short", "mid"), ("mid", "long"), ("tenkan", "kijun"),
             ("kijun", "senkou"), ("oversold", "overbought"), ("low_level", "high_level"), ("exit_len", "entry_len")]
    return all(c[a] < c[b] for a, b in pairs if a in c and b in c)


def _summary(res) -> dict[str, Any]:
    m = res.metrics
    return {k: m[k] for k in ("total_return", "cagr", "net_profit", "sharpe", "max_drawdown", "percent_profitable",
                              "num_trades", "closed_trades", "winning_trades", "profit_factor", "excess_return")} | {
        "buy_and_hold_return": res.benchmark_metrics["total_return"]}


def _score(s: dict[str, Any], objective: str, min_trades: int) -> float:
    if s["num_trades"] < min_trades:
        return float("-inf")
    v = s[objective]
    return float("-inf") if v is None else float(v)


def optimize(prices: pd.DataFrame, strategy: str, start: str | None, objective: str = "total_return",
             train_pct: float = 0.7, max_combos: int = 150, initial_capital: float = 10_000.0, fee_bps: float = 5.0,
             min_trades: int = 5, top: int = 15) -> dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValueError(f"Onbekend doel '{objective}'. Kies uit: {', '.join(OBJECTIVES)}")
    period = prices.index[prices.index >= pd.Timestamp(start)] if start else prices.index
    if len(period) < 100:
        raise ValueError("Te weinig koersdata om te optimaliseren (minstens 100 dagen nodig)")
    split = period[int(len(period) * train_pct)]
    train_prices = prices[prices.index < split]
    cls = STRATEGIES[strategy]

    rows = []
    for params in grid(strategy, max_combos):
        strat = cls(**params)
        train = _summary(run_backtest(train_prices, strat, initial_capital, fee_bps, start=period[0]))
        test = _summary(run_backtest(prices, strat, initial_capital, fee_bps, start=split))
        rows.append({"params": params, "is_default": params == cls.params, "train": train, "test": test,
                     "score": _score(train, objective, min_trades)})

    rows.sort(key=lambda r: r["score"], reverse=True)
    default = next(r for r in rows if r["is_default"])
    best = rows[0]
    for r in rows:
        r["score"] = None if r["score"] == float("-inf") else r["score"]
    # Robuust = beste instelling doet het in de controleperiode minstens zo goed als de standaardinstelling.
    robust = best["test"][objective] is not None and default["test"][objective] is not None and \
        best["test"][objective] >= default["test"][objective]
    return {
        "strategy": strategy, "objective": objective, "combos_tested": len(rows),
        "train_period": {"start": period[0].date().isoformat(), "end": train_prices.index[-1].date().isoformat()},
        "test_period": {"start": split.date().isoformat(), "end": prices.index[-1].date().isoformat()},
        "default": default, "best": best, "robust": robust, "results": rows[:top],
    }
