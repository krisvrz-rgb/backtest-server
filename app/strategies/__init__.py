"""Strategie-register, opgebouwd uit de bibliotheek in library.py."""
from __future__ import annotations

import pandas as pd

from .base import Strategy
from .dsl import Expr, Param, PineCtx, run_rules
from .library import CATEGORIES, SPECS, Spec, _P


class RuleStrategy(Strategy):
    """Strategie op basis van een instap- en uitstapregel uit de bibliotheek."""

    spec: Spec

    def rules(self) -> tuple[Expr, Expr]:
        return self.spec.build(_P(self.p))

    def signals(self, prices: pd.DataFrame, active: pd.Series | None = None) -> pd.Series:
        entry, exit_ = self.rules()
        return run_rules(prices, entry, exit_, active)

    def pine_rules(self) -> tuple[list[str], str, str]:
        ctx = PineCtx()
        entry, exit_ = self.rules()
        e, x = entry.pine(ctx), exit_.pine(ctx)
        return ctx.lines, e, x


def _make(spec: Spec) -> type[RuleStrategy]:
    return type(spec.name, (RuleStrategy,), {
        "name": spec.name, "category": spec.category, "description": spec.description,
        "params": dict(spec.params), "spec": spec,
    })


STRATEGIES: dict[str, type[RuleStrategy]] = {s.name: _make(s) for s in SPECS}


def get_strategy(name: str, **params) -> RuleStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Onbekende strategie '{name}'. Kies uit: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)


__all__ = ["CATEGORIES", "STRATEGIES", "Strategy", "RuleStrategy", "Param", "get_strategy"]
