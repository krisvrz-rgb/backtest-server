"""Genereer Pine Script v6 die dezelfde logica draait als de Python-strategie.

Plak de output in de Pine Editor van TradingView en klik 'Add to chart' om de backtest
in de Strategy Tester te zien. Uitvoering (100% van equity, fill op open van de volgende bar,
commissie in %) is gelijk aan de Python-engine, zodat de resultaten vergelijkbaar zijn.
"""
from __future__ import annotations

from .strategies import Strategy

HEADER = """//@version=6
strategy("{title}", overlay=true, initial_capital={capital}, default_qty_type=strategy.percent_of_equity,
     default_qty_value=100, commission_type=strategy.commission.percent, commission_value={commission},
     slippage=0, process_orders_on_close=false)

startDate = input.time(timestamp("{start}"), "Start")
inRange = time >= startDate
"""

BODIES: dict[str, str] = {
    "buy_and_hold": """
if inRange and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
""",
    "sma_crossover": """
fastLen = input.int({fast}, "Snelle SMA")
slowLen = input.int({slow}, "Trage SMA")
fast = ta.sma(close, fastLen)
slow = ta.sma(close, slowLen)
plot(fast, "Snel", color.orange)
plot(slow, "Traag", color.blue)
longCond = fast > slow
if inRange and longCond and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
if not longCond and strategy.position_size > 0
    strategy.close("Long")
""",
    "rsi_mean_reversion": """
len = input.int({period}, "RSI periode")
os = input.float({oversold}, "Oversold")
ob = input.float({overbought}, "Overbought")
r = ta.rsi(close, len)
if inRange and r < os and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
if r > ob and strategy.position_size > 0
    strategy.close("Long")
""",
    "momentum": """
lookback = input.int({lookback}, "Lookback")
mom = close / close[lookback] - 1
if inRange and mom > 0 and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
if mom <= 0 and strategy.position_size > 0
    strategy.close("Long")
""",
    "bollinger_reversion": """
len = input.int({window}, "Window")
mult = input.float({num_std}, "Std dev")
[mid, upper, lower] = ta.bb(close, len, mult)
plot(mid, "Midden", color.gray)
plot(lower, "Onder", color.teal)
if inRange and close < lower and strategy.position_size == 0
    strategy.entry("Long", strategy.long)
if close > mid and strategy.position_size > 0
    strategy.close("Long")
""",
}


def to_pine(strategy: Strategy, initial_capital: float = 10_000.0, fee_bps: float = 5.0,
            start: str = "2015-01-01") -> str:
    if strategy.name not in BODIES:
        raise ValueError(f"Geen Pine-template voor {strategy.name}")
    params = ", ".join(f"{k}={v}" for k, v in strategy.p.items())
    title = f"{strategy.name}" + (f" ({params})" if params else "")
    head = HEADER.format(title=title, capital=initial_capital, commission=fee_bps / 100, start=start)
    return head + BODIES[strategy.name].format(**strategy.p)
