from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    description = "Koop op dag 1 en houd vast. Nuttig als benchmark."
    params = {}

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index)


class SmaCrossover(Strategy):
    name = "sma_crossover"
    description = "Long als de snelle SMA boven de trage SMA staat, anders cash."
    params = {"fast": 50, "slow": 200}

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        fast = prices["close"].rolling(int(self.p["fast"])).mean()
        slow = prices["close"].rolling(int(self.p["slow"])).mean()
        return (fast > slow).astype(float)


class RsiMeanReversion(Strategy):
    name = "rsi_mean_reversion"
    description = "Koop als RSI onder 'oversold' zakt, verkoop als RSI boven 'overbought' komt."
    params = {"period": 14, "oversold": 30, "overbought": 70}

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        delta = prices["close"].diff()
        n = int(self.p["period"])
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        pos = pd.Series(np.nan, index=prices.index)
        pos[rsi < self.p["oversold"]] = 1.0
        pos[rsi > self.p["overbought"]] = 0.0
        return pos.ffill().fillna(0.0)


class Momentum(Strategy):
    name = "momentum"
    description = "Long als het rendement over 'lookback' dagen positief is, anders cash."
    params = {"lookback": 126}

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        return (prices["close"].pct_change(int(self.p["lookback"])) > 0).astype(float)


class BollingerBreakout(Strategy):
    name = "bollinger_reversion"
    description = "Koop onder de onderste Bollinger-band, verkoop boven het midden."
    params = {"window": 20, "num_std": 2.0}

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        c = prices["close"]
        mid = c.rolling(int(self.p["window"])).mean()
        lower = mid - self.p["num_std"] * c.rolling(int(self.p["window"])).std()
        pos = pd.Series(np.nan, index=prices.index)
        pos[c < lower] = 1.0
        pos[c > mid] = 0.0
        return pos.ffill().fillna(0.0)
