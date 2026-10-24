"""Strategie-register. Nieuwe strategie? Maak een subclass van Strategy en zet hem in STRATEGIES."""
from .base import Strategy
from .builtin import BollingerBreakout, BuyAndHold, Momentum, RsiMeanReversion, SmaCrossover

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (BuyAndHold, SmaCrossover, RsiMeanReversion, Momentum, BollingerBreakout)
}


def get_strategy(name: str, **params) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Onbekende strategie '{name}'. Kies uit: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)
