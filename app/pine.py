"""Genereer Pine Script v6 die dezelfde regels draait als de Python-strategie.

Plak de output in de Pine Editor van TradingView en klik 'Add to chart' om de backtest
in de Strategy Tester te zien. Uitvoering (100% van equity, fill op open van de volgende bar,
commissie in %) is gelijk aan de Python-engine, zodat de resultaten vergelijkbaar zijn.
"""
from __future__ import annotations

from .strategies import CATEGORIES, RuleStrategy

HEADER = """//@version=6
// {category}: {description}
strategy("{title}", overlay=true, initial_capital={capital}, default_qty_type=strategy.percent_of_equity,
     default_qty_value=100, commission_type=strategy.commission.percent, commission_value={commission},
     slippage=0, process_orders_on_close=false)

startDate = input.time(timestamp("{start}"), "Start")
inRange = time >= startDate
"""

BODY = """
entryCond = {entry}
exitCond = {exit}

if inRange and entryCond and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
else if exitCond and strategy.position_size > 0
    strategy.close("Long")

plotshape(inRange and entryCond and strategy.position_size == 0, "Instapsignaal", shape.triangleup, location.belowbar, color.new(color.green, 60), size=size.tiny)
"""


def _input(name: str, value) -> str:
    if isinstance(value, bool):
        return f'{name} = input.bool({"true" if value else "false"}, "{name}")'
    if isinstance(value, int):
        return f'{name} = input.int({value}, "{name}")'
    return f'{name} = input.float({float(value)}, "{name}")'


def to_pine(strategy: RuleStrategy, initial_capital: float = 10_000.0, fee_bps: float = 5.0,
            start: str = "2015-01-01") -> str:
    params = ", ".join(f"{k}={v}" for k, v in strategy.p.items())
    title = strategy.name + (f" ({params})" if params else "")
    head = HEADER.format(title=title, capital=initial_capital, commission=fee_bps / 100, start=start,
                         category=CATEGORIES.get(strategy.category, strategy.category),
                         description=strategy.description)
    inputs = "\n".join(_input(k, v) for k, v in strategy.p.items())
    lines, entry, exit_ = strategy.pine_rules()
    out = head
    if inputs:
        out += "\n" + inputs + "\n"
    if lines:
        out += "\n" + "\n".join(lines) + "\n"
    return out + BODY.format(entry=entry, exit=exit_)
