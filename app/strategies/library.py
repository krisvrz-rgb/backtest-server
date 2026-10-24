"""Strategiebibliotheek: ~100 strategieën in zes categorieën, elk met instelbare parameters.

Elke strategie is een regel voor instappen en een regel voor uitstappen (long-only). Dezelfde regels
worden in Python uitgerekend en naar Pine Script vertaald, zodat je elke strategie ook in TradingView ziet.
Nieuwe strategie toevoegen: één `add(...)`-regel onderaan, met een functie die (instap, uitstap) teruggeeft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .dsl import (
    ATR, CCI, CLOSE, CMO, DMI, EMA, HIGH, HL2, HLC3, HMA, LOW, MACD, MFI, MOM, OBV, OPEN, ROC, RSI, SAR, SMA, TSI,
    VOLUME, VWMA, WMA, WPR, Expr, FalseCond, Highest, Lowest, Param, StochD, StochK, Stdev, Supertrend, TrueCond,
    bollinger, donchian_mid, keltner,
)

CATEGORIES = {
    "trend_hold": "Vasthouden met trendfilter",
    "trend": "Trend volgen",
    "mean_reversion": "Mean reversion",
    "momentum": "Momentum",
    "breakout": "Breakout",
    "volatility": "Volatiliteit",
    "volume": "Volume",
    "benchmark": "Benchmark",
}


@dataclass
class Spec:
    name: str
    category: str
    description: str
    params: dict[str, Any]
    build: Callable[[Any], tuple[Expr, Expr]]


class _P:
    """Geeft parameters door als Param-objecten (Python-waarde + Pine-inputnaam)."""

    def __init__(self, values: dict[str, Any]):
        self._values = values

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return Param(name, self._values[name])


SPECS: list[Spec] = []


def add(name: str, category: str, description: str, params: dict[str, Any], build):
    SPECS.append(Spec(name, category, description, params, build))


# ============================================================================ benchmark

add("buy_and_hold", "benchmark", "Koop op dag 1 en houd vast. Nuttig als referentie.", {},
    lambda p: (TrueCond(), FalseCond()))

# ============================================================================ trend

for kind, cls, pairs in [
    ("sma", SMA, [(10, 30), (20, 50), (50, 200)]),
    ("ema", EMA, [(9, 21), (12, 26), (20, 50), (50, 200)]),
    ("wma", WMA, [(10, 30), (20, 50)]),
    ("hma", HMA, [(9, 21), (20, 50)]),
]:
    for f, s in pairs:
        name = "sma_crossover" if (kind, f, s) == ("sma", 50, 200) else f"{kind}_cross_{f}_{s}"
        add(name, "trend", f"Long zolang de snelle {kind.upper()} boven de trage {kind.upper()} staat.",
            {"fast": f, "slow": s},
            lambda p, cls=cls: (cls(CLOSE, p.fast) > cls(CLOSE, p.slow), cls(CLOSE, p.fast) < cls(CLOSE, p.slow)))

for kind, cls in [("sma", SMA), ("ema", EMA)]:
    for n in (20, 50, 100, 200):
        add(f"price_above_{kind}_{n}", "trend", f"Long zolang de slotkoers boven de {n}-daagse {kind.upper()} staat.",
            {"length": n}, lambda p, cls=cls: (CLOSE > cls(CLOSE, p.length), CLOSE < cls(CLOSE, p.length)))

add("triple_ema_alignment", "trend", "Long als EMA kort > midden > lang (trend in alle tijdframes), uit als kort onder midden zakt.",
    {"short": 10, "mid": 30, "long": 60},
    lambda p: ((EMA(CLOSE, p.short) > EMA(CLOSE, p.mid)) & (EMA(CLOSE, p.mid) > EMA(CLOSE, p.long)),
               EMA(CLOSE, p.short) < EMA(CLOSE, p.mid)))

add("golden_cross_filtered", "trend", "SMA 50/200 golden cross, maar alleen instappen als de koers ook boven de SMA 50 staat.",
    {"fast": 50, "slow": 200},
    lambda p: ((SMA(CLOSE, p.fast) > SMA(CLOSE, p.slow)) & (CLOSE > SMA(CLOSE, p.fast)),
               SMA(CLOSE, p.fast) < SMA(CLOSE, p.slow)))

for f, s, sig in [(12, 26, 9), (8, 17, 9), (5, 35, 5)]:
    add(f"macd_signal_cross_{f}_{s}_{sig}", "trend", "Koop als de MACD boven de signaallijn kruist, verkoop bij de kruising omlaag.",
        {"fast": f, "slow": s, "signal_len": sig},
        lambda p: (MACD(CLOSE, p.fast, p.slow, p.signal_len, "macd").crosses_above(MACD(CLOSE, p.fast, p.slow, p.signal_len, "signal")),
                   MACD(CLOSE, p.fast, p.slow, p.signal_len, "macd").crosses_below(MACD(CLOSE, p.fast, p.slow, p.signal_len, "signal"))))

add("macd_zero_line", "trend", "Long zolang de MACD-lijn boven nul staat.", {"fast": 12, "slow": 26, "signal_len": 9},
    lambda p: (MACD(CLOSE, p.fast, p.slow, p.signal_len) > 0, MACD(CLOSE, p.fast, p.slow, p.signal_len) < 0))

add("macd_histogram", "trend", "Long zolang het MACD-histogram positief is.", {"fast": 12, "slow": 26, "signal_len": 9},
    lambda p: (MACD(CLOSE, p.fast, p.slow, p.signal_len, "hist") > 0, MACD(CLOSE, p.fast, p.slow, p.signal_len, "hist") < 0))

add("macd_with_trend_filter", "trend", "MACD-kruising omhoog, alleen als de koers boven de SMA 200 staat.",
    {"fast": 12, "slow": 26, "signal_len": 9, "trend_len": 200},
    lambda p: (MACD(CLOSE, p.fast, p.slow, p.signal_len, "macd").crosses_above(MACD(CLOSE, p.fast, p.slow, p.signal_len, "signal"))
               & (CLOSE > SMA(CLOSE, p.trend_len)),
               MACD(CLOSE, p.fast, p.slow, p.signal_len, "macd").crosses_below(MACD(CLOSE, p.fast, p.slow, p.signal_len, "signal"))))

for factor, n in [(3.0, 10), (2.0, 7), (4.0, 14)]:
    add(f"supertrend_{n}_{int(factor)}", "trend", "Long zolang de Supertrend opwaarts is.", {"factor": factor, "atr_len": n},
        lambda p: (Supertrend(p.factor, p.atr_len, "dir") < 0, Supertrend(p.factor, p.atr_len, "dir") > 0))

add("parabolic_sar", "trend", "Long zolang de koers boven de Parabolic SAR staat.", {"sar_start": 0.02, "sar_inc": 0.02, "sar_max": 0.2},
    lambda p: (CLOSE > SAR(p.sar_start, p.sar_inc, p.sar_max), CLOSE < SAR(p.sar_start, p.sar_inc, p.sar_max)))

add("adx_di_cross", "trend", "Koop als +DI boven -DI staat en de ADX een sterke trend aangeeft, uit als -DI erboven komt.",
    {"di_len": 14, "adx_len": 14, "adx_min": 20},
    lambda p: ((DMI(p.di_len, p.adx_len, "plus") > DMI(p.di_len, p.adx_len, "minus")) & (DMI(p.di_len, p.adx_len, "adx") > p.adx_min),
               DMI(p.di_len, p.adx_len, "plus") < DMI(p.di_len, p.adx_len, "minus")))

add("adx_trend_ema", "trend", "Long als de koers boven de EMA staat en de ADX boven de drempel; uit onder de EMA.",
    {"ema_len": 50, "di_len": 14, "adx_len": 14, "adx_min": 25},
    lambda p: ((CLOSE > EMA(CLOSE, p.ema_len)) & (DMI(p.di_len, p.adx_len, "adx") > p.adx_min), CLOSE < EMA(CLOSE, p.ema_len)))

add("ichimoku_tk_cross", "trend", "Ichimoku: koop als Tenkan boven Kijun kruist en de koers boven de wolk staat.",
    {"tenkan": 9, "kijun": 26, "senkou": 52},
    lambda p: (donchian_mid(p.tenkan).crosses_above(donchian_mid(p.kijun))
               & (CLOSE > ((donchian_mid(p.tenkan) + donchian_mid(p.kijun)) / 2)[p.kijun])
               & (CLOSE > donchian_mid(p.senkou)[p.kijun]),
               donchian_mid(p.tenkan).crosses_below(donchian_mid(p.kijun))))

add("ichimoku_cloud", "trend", "Ichimoku: long zolang de koers boven de wolk staat.", {"tenkan": 9, "kijun": 26, "senkou": 52},
    lambda p: ((CLOSE > ((donchian_mid(p.tenkan) + donchian_mid(p.kijun)) / 2)[p.kijun]) & (CLOSE > donchian_mid(p.senkou)[p.kijun]),
               (CLOSE < ((donchian_mid(p.tenkan) + donchian_mid(p.kijun)) / 2)[p.kijun]) | (CLOSE < donchian_mid(p.senkou)[p.kijun])))

add("hma_slope", "trend", "Long zolang de Hull MA stijgt.", {"length": 20},
    lambda p: (HMA(CLOSE, p.length) > HMA(CLOSE, p.length)[1], HMA(CLOSE, p.length) < HMA(CLOSE, p.length)[1]))

add("ema_slope", "trend", "Long zolang de EMA hoger staat dan 5 dagen geleden.", {"length": 50, "lookback": 5},
    lambda p: (EMA(CLOSE, p.length) > EMA(CLOSE, p.length)[p.lookback], EMA(CLOSE, p.length) < EMA(CLOSE, p.length)[p.lookback]))

add("keltner_midline", "trend", "Long boven het midden van het Keltner-kanaal, uit onder de onderkant.", {"length": 20, "mult": 2.0},
    lambda p: (CLOSE > keltner(CLOSE, p.length, p.mult)[0], CLOSE < keltner(CLOSE, p.length, p.mult)[2]))

add("donchian_trend", "trend", "Long zolang de koers boven het midden van het Donchian-kanaal staat.", {"length": 50},
    lambda p: (CLOSE > donchian_mid(p.length), CLOSE < donchian_mid(p.length)))

add("vwma_sma_cross", "trend", "Long als de volumegewogen MA boven de gewone SMA staat (koopdruk).", {"length": 20},
    lambda p: (VWMA(CLOSE, p.length) > SMA(CLOSE, p.length), VWMA(CLOSE, p.length) < SMA(CLOSE, p.length)))

add("higher_highs_lows", "trend", "Long als zowel de hoogste als laagste koers van de laatste n dagen hoger ligt dan de periode ervoor.",
    {"length": 20},
    lambda p: ((Highest(HIGH, p.length) > Highest(HIGH, p.length)[p.length]) & (Lowest(LOW, p.length) > Lowest(LOW, p.length)[p.length]),
               Lowest(LOW, p.length) < Lowest(LOW, p.length)[p.length]))

# ============================================================================ mean reversion

for n, lo, hi in [(14, 30, 70), (14, 25, 75), (7, 20, 80), (21, 35, 65)]:
    name = "rsi_mean_reversion" if (n, lo, hi) == (14, 30, 70) else f"rsi_{n}_{lo}_{hi}"
    add(name, "mean_reversion", f"Koop als RSI({n}) onder {lo} zakt, verkoop boven {hi}.",
        {"period": n, "oversold": lo, "overbought": hi},
        lambda p: (RSI(CLOSE, p.period) < p.oversold, RSI(CLOSE, p.period) > p.overbought))

add("connors_rsi2", "mean_reversion", "Connors RSI(2): koop als RSI(2) < 10 boven de SMA 200, verkoop als de koers boven de SMA 5 sluit.",
    {"rsi_len": 2, "entry_level": 10, "trend_len": 200, "exit_len": 5},
    lambda p: ((RSI(CLOSE, p.rsi_len) < p.entry_level) & (CLOSE > SMA(CLOSE, p.trend_len)), CLOSE > SMA(CLOSE, p.exit_len)))

add("rsi2_extreme", "mean_reversion", "Koop als RSI(2) onder 5 zakt, verkoop boven 70.", {"rsi_len": 2, "entry_level": 5, "exit_level": 70},
    lambda p: (RSI(CLOSE, p.rsi_len) < p.entry_level, RSI(CLOSE, p.rsi_len) > p.exit_level))

add("rsi_trend_pullback", "mean_reversion", "Pullback in een uptrend: koop als RSI(14) < 40 boven de SMA 200, verkoop als RSI > 60.",
    {"period": 14, "entry_level": 40, "exit_level": 60, "trend_len": 200},
    lambda p: ((RSI(CLOSE, p.period) < p.entry_level) & (CLOSE > SMA(CLOSE, p.trend_len)), RSI(CLOSE, p.period) > p.exit_level))

for n, mult, exit_upper in [(20, 2.0, False), (20, 2.0, True), (20, 2.5, False), (10, 1.5, False)]:
    name = "bollinger_reversion" if (n, mult, exit_upper) == (20, 2.0, False) else \
        f"bollinger_{n}_{str(mult).replace('.', '')}" + ("_to_upper" if exit_upper else "")
    target = "bovenste band" if exit_upper else "midden"
    add(name, "mean_reversion", f"Koop onder de onderste Bollinger-band ({n}, {mult}), verkoop boven de {target}.",
        {"window": n, "num_std": mult},
        lambda p, up=exit_upper: (CLOSE < bollinger(CLOSE, p.window, p.num_std)[2],
                                  CLOSE > bollinger(CLOSE, p.window, p.num_std)[1 if up else 0]))

add("bollinger_trend_dip", "mean_reversion", "Koop onder de onderste Bollinger-band maar alleen boven de SMA 200; verkoop boven het midden.",
    {"window": 20, "num_std": 2.0, "trend_len": 200},
    lambda p: ((CLOSE < bollinger(CLOSE, p.window, p.num_std)[2]) & (CLOSE > SMA(CLOSE, p.trend_len)),
               CLOSE > bollinger(CLOSE, p.window, p.num_std)[0]))

add("stochastic_oversold", "mean_reversion", "Koop als %K boven %D kruist onder 20, verkoop boven 80.",
    {"k_len": 14, "k_smooth": 3, "d_len": 3, "low_level": 20, "high_level": 80},
    lambda p: (StochK(p.k_len, p.k_smooth).crosses_above(StochD(p.k_len, p.k_smooth, p.d_len)) & (StochK(p.k_len, p.k_smooth) < p.low_level),
               StochK(p.k_len, p.k_smooth) > p.high_level))

add("stochastic_fast", "mean_reversion", "Snelle stochastic: koop onder 10, verkoop boven 70.", {"k_len": 5, "k_smooth": 1, "low_level": 10, "high_level": 70},
    lambda p: (StochK(p.k_len, p.k_smooth) < p.low_level, StochK(p.k_len, p.k_smooth) > p.high_level))

add("williams_r", "mean_reversion", "Koop als Williams %R onder -80 zakt, verkoop boven -20.", {"length": 14, "low_level": -80, "high_level": -20},
    lambda p: (WPR(p.length) < p.low_level, WPR(p.length) > p.high_level))

add("williams_r_trend", "mean_reversion", "Williams %R(10) onder -90 in een uptrend (boven SMA 200), uit boven -30.",
    {"length": 10, "low_level": -90, "high_level": -30, "trend_len": 200},
    lambda p: ((WPR(p.length) < p.low_level) & (CLOSE > SMA(CLOSE, p.trend_len)), WPR(p.length) > p.high_level))

add("cci_reversion", "mean_reversion", "Koop als CCI onder -100 zakt, verkoop boven +100.", {"length": 20, "low_level": -100, "high_level": 100},
    lambda p: (CCI(HLC3, p.length) < p.low_level, CCI(HLC3, p.length) > p.high_level))

add("keltner_reversion", "mean_reversion", "Koop onder de onderkant van het Keltner-kanaal, verkoop boven het midden.", {"length": 20, "mult": 2.0},
    lambda p: (CLOSE < keltner(CLOSE, p.length, p.mult)[2], CLOSE > keltner(CLOSE, p.length, p.mult)[0]))

add("zscore_reversion", "mean_reversion", "Koop als de koers meer dan 2 standaardafwijkingen onder het gemiddelde ligt, verkoop bij het gemiddelde.",
    {"length": 20, "z_entry": -2.0, "z_exit": 0.0},
    lambda p: ((CLOSE - SMA(CLOSE, p.length)) / Stdev(CLOSE, p.length) < p.z_entry,
               (CLOSE - SMA(CLOSE, p.length)) / Stdev(CLOSE, p.length) > p.z_exit))

add("consecutive_down_days", "mean_reversion", "Koop na 3 dalende slotkoersen op rij, verkoop bij de eerste stijgende dag.",
    {"trend_len": 200},
    lambda p: ((CLOSE < CLOSE[1]) & (CLOSE[1] < CLOSE[2]) & (CLOSE[2] < CLOSE[3]) & (CLOSE > SMA(CLOSE, p.trend_len)),
               CLOSE > CLOSE[1]))

add("ibs_reversion", "mean_reversion", "Internal Bar Strength: koop als de koers in de onderste 20% van de dagrange sluit, verkoop in de bovenste 20%.",
    {"low_level": 0.2, "high_level": 0.8},
    lambda p: ((CLOSE - LOW) / (HIGH - LOW) < p.low_level, (CLOSE - LOW) / (HIGH - LOW) > p.high_level))

add("mfi_reversion", "mean_reversion", "Koop als de Money Flow Index onder 20 zakt, verkoop boven 80.", {"length": 14, "low_level": 20, "high_level": 80},
    lambda p: (MFI(HLC3, p.length) < p.low_level, MFI(HLC3, p.length) > p.high_level))

add("cmo_reversion", "mean_reversion", "Koop als de Chande Momentum Oscillator onder -50 zakt, verkoop boven +50.", {"length": 14, "low_level": -50, "high_level": 50},
    lambda p: (CMO(CLOSE, p.length) < p.low_level, CMO(CLOSE, p.length) > p.high_level))

add("pct_below_sma", "mean_reversion", "Koop als de koers meer dan 5% onder de SMA 20 staat, verkoop terug boven de SMA.",
    {"length": 20, "pct": 5.0},
    lambda p: (CLOSE < SMA(CLOSE, p.length) * (1 - p.pct / 100), CLOSE > SMA(CLOSE, p.length)))

add("lower_low_reversal", "mean_reversion", "Koop na een nieuw 10-daags dieptepunt in een uptrend, verkoop bij een nieuw 5-daags hoogtepunt.",
    {"low_len": 10, "exit_len": 5, "trend_len": 200},
    lambda p: ((LOW <= Lowest(LOW, p.low_len)) & (CLOSE > SMA(CLOSE, p.trend_len)), HIGH >= Highest(HIGH, p.exit_len)[1]))

add("gap_down_reversal", "mean_reversion", "Koop na een opening die meer dan 2% lager ligt dan de vorige slot, verkoop na een hogere slot.",
    {"gap_pct": 2.0},
    lambda p: (OPEN < CLOSE[1] * (1 - p.gap_pct / 100), CLOSE > CLOSE[1]))

# ============================================================================ momentum

for n in (20, 60, 126, 252):
    name = "momentum" if n == 126 else f"roc_{n}"
    add(name, "momentum", f"Long als het rendement over {n} dagen positief is.", {"lookback": n},
        lambda p: (ROC(CLOSE, p.lookback) > 0, ROC(CLOSE, p.lookback) < 0))

add("dual_momentum", "momentum", "Long als het 6-maands rendement positief is én de koers boven de SMA 200 staat.",
    {"lookback": 126, "trend_len": 200},
    lambda p: ((ROC(CLOSE, p.lookback) > 0) & (CLOSE > SMA(CLOSE, p.trend_len)), (ROC(CLOSE, p.lookback) < 0) | (CLOSE < SMA(CLOSE, p.trend_len))))

add("roc_threshold", "momentum", "Koop als het 20-daags rendement boven 10% komt, verkoop als het negatief wordt.", {"lookback": 20, "threshold": 10.0},
    lambda p: (ROC(CLOSE, p.lookback) > p.threshold, ROC(CLOSE, p.lookback) < 0))

add("rsi_50_cross", "momentum", "Long zolang RSI(14) boven 50 staat.", {"period": 14, "level": 50},
    lambda p: (RSI(CLOSE, p.period).crosses_above(p.level), RSI(CLOSE, p.period).crosses_below(p.level)))

add("rsi_momentum_band", "momentum", "Koop als RSI boven 55 komt, verkoop onder 45 (momentum met buffer).", {"period": 14, "entry_level": 55, "exit_level": 45},
    lambda p: (RSI(CLOSE, p.period) > p.entry_level, RSI(CLOSE, p.period) < p.exit_level))

add("rsi_strength_70", "momentum", "Koop sterke aandelen: RSI boven 70, verkoop als hij onder 50 zakt.", {"period": 14, "entry_level": 70, "exit_level": 50},
    lambda p: (RSI(CLOSE, p.period) > p.entry_level, RSI(CLOSE, p.period) < p.exit_level))

add("tsi_signal_cross", "momentum", "Koop als de True Strength Index boven zijn signaallijn kruist.", {"short": 13, "long_len": 25, "signal_len": 13},
    lambda p: (TSI(CLOSE, p.short, p.long_len).crosses_above(EMA(TSI(CLOSE, p.short, p.long_len), p.signal_len)),
               TSI(CLOSE, p.short, p.long_len).crosses_below(EMA(TSI(CLOSE, p.short, p.long_len), p.signal_len))))

add("tsi_zero", "momentum", "Long zolang de True Strength Index boven nul staat.", {"short": 13, "long_len": 25},
    lambda p: (TSI(CLOSE, p.short, p.long_len) > 0, TSI(CLOSE, p.short, p.long_len) < 0))

add("momentum_10", "momentum", "Long zolang de koers hoger staat dan 10 dagen geleden (ta.mom).", {"length": 10},
    lambda p: (MOM(CLOSE, p.length) > 0, MOM(CLOSE, p.length) < 0))

add("cci_trend", "momentum", "Koop als CCI boven +100 komt (sterk momentum), verkoop onder 0.", {"length": 20, "entry_level": 100, "exit_level": 0},
    lambda p: (CCI(HLC3, p.length) > p.entry_level, CCI(HLC3, p.length) < p.exit_level))

add("stochastic_momentum", "momentum", "Koop als %K boven 80 komt, verkoop onder 50.", {"k_len": 14, "k_smooth": 3, "entry_level": 80, "exit_level": 50},
    lambda p: (StochK(p.k_len, p.k_smooth) > p.entry_level, StochK(p.k_len, p.k_smooth) < p.exit_level))

add("near_52w_high", "momentum", "Long zolang de koers binnen 5% van het 52-weekse hoogtepunt staat.", {"length": 252, "pct": 5.0, "exit_pct": 15.0},
    lambda p: (CLOSE >= Highest(HIGH, p.length) * (1 - p.pct / 100), CLOSE < Highest(HIGH, p.length) * (1 - p.exit_pct / 100)))

add("acceleration", "momentum", "Koop als het 20-daags rendement hoger is dan het 60-daags rendement per maand (versnelling).",
    {"short": 20, "long_len": 60},
    lambda p: ((ROC(CLOSE, p.short) > ROC(CLOSE, p.long_len) / 3) & (ROC(CLOSE, p.short) > 0), ROC(CLOSE, p.short) < 0))

add("cmo_momentum", "momentum", "Long zolang de Chande Momentum Oscillator boven nul staat.", {"length": 20},
    lambda p: (CMO(CLOSE, p.length) > 0, CMO(CLOSE, p.length) < 0))

add("macd_rsi_combo", "momentum", "Koop als MACD-histogram positief is én RSI boven 50, verkoop als beide wegvallen.",
    {"fast": 12, "slow": 26, "signal_len": 9, "period": 14},
    lambda p: ((MACD(CLOSE, p.fast, p.slow, p.signal_len, "hist") > 0) & (RSI(CLOSE, p.period) > 50),
               (MACD(CLOSE, p.fast, p.slow, p.signal_len, "hist") < 0) & (RSI(CLOSE, p.period) < 50)))

# ============================================================================ breakout

for entry_n, exit_n in [(20, 10), (55, 20), (10, 5), (100, 50)]:
    add(f"donchian_{entry_n}_{exit_n}", "breakout",
        f"Turtle-stijl: koop bij een nieuw {entry_n}-daags hoogtepunt, verkoop bij een nieuw {exit_n}-daags dieptepunt.",
        {"entry_len": entry_n, "exit_len": exit_n},
        lambda p: (CLOSE > Highest(HIGH, p.entry_len)[1], CLOSE < Lowest(LOW, p.exit_len)[1]))

add("breakout_52w_high", "breakout", "Koop bij een nieuw 52-weeks hoogtepunt, verkoop onder de SMA 50.", {"length": 252, "exit_len": 50},
    lambda p: (CLOSE > Highest(HIGH, p.length)[1], CLOSE < SMA(CLOSE, p.exit_len)))

add("bollinger_breakout", "breakout", "Koop als de koers boven de bovenste Bollinger-band sluit, verkoop onder het midden.",
    {"window": 20, "num_std": 2.0},
    lambda p: (CLOSE > bollinger(CLOSE, p.window, p.num_std)[1], CLOSE < bollinger(CLOSE, p.window, p.num_std)[0]))

add("keltner_breakout", "breakout", "Koop boven de bovenkant van het Keltner-kanaal, verkoop onder het midden.", {"length": 20, "mult": 1.5},
    lambda p: (CLOSE > keltner(CLOSE, p.length, p.mult)[1], CLOSE < keltner(CLOSE, p.length, p.mult)[0]))

add("atr_channel_breakout", "breakout", "Koop als de koers meer dan 2x ATR boven de SMA 20 sluit, verkoop terug onder de SMA.",
    {"length": 20, "atr_len": 14, "mult": 2.0},
    lambda p: (CLOSE > SMA(CLOSE, p.length) + p.mult * ATR(p.atr_len), CLOSE < SMA(CLOSE, p.length)))

add("range_breakout_volume", "breakout", "Breakout boven het 20-daags hoogtepunt met minstens 1,5x het gemiddelde volume.",
    {"length": 20, "vol_len": 20, "vol_mult": 1.5, "exit_len": 10},
    lambda p: ((CLOSE > Highest(HIGH, p.length)[1]) & (VOLUME > SMA(VOLUME, p.vol_len) * p.vol_mult), CLOSE < Lowest(LOW, p.exit_len)[1]))

add("gap_up_breakout", "breakout", "Koop na een opening boven het hoogtepunt van gisteren, verkoop onder de EMA 10.", {"exit_len": 10},
    lambda p: (OPEN > HIGH[1], CLOSE < EMA(CLOSE, p.exit_len)))

add("inside_bar_breakout", "breakout", "Koop als de koers boven een inside bar (range binnen die van de dag ervoor) uitbreekt.",
    {"exit_len": 10},
    lambda p: ((HIGH[1] < HIGH[2]) & (LOW[1] > LOW[2]) & (CLOSE > HIGH[1]), CLOSE < Lowest(LOW, p.exit_len)[1]))

add("nr7_breakout", "breakout", "Koop als de koers boven de dag met de kleinste range van de laatste 7 dagen uitbreekt (NR7).",
    {"range_len": 7, "exit_len": 10},
    lambda p: (((HIGH - LOW)[1] <= Lowest(HIGH - LOW, p.range_len)[1]) & (CLOSE > HIGH[1]), CLOSE < Lowest(LOW, p.exit_len)[1]))

add("opening_range_trend", "breakout", "Koop als de slot boven het 5-daags hoogtepunt ligt en de SMA 50 stijgt.", {"length": 5, "trend_len": 50},
    lambda p: ((CLOSE > Highest(HIGH, p.length)[1]) & (SMA(CLOSE, p.trend_len) > SMA(CLOSE, p.trend_len)[5]), CLOSE < Lowest(LOW, p.length)[1]))

add("consolidation_breakout", "breakout", "Koop bij een uitbraak uit een smalle range (hoog/laag binnen 10% over 20 dagen).",
    {"length": 20, "max_range_pct": 10.0, "exit_len": 10},
    lambda p: ((Highest(HIGH, p.length)[1] / Lowest(LOW, p.length)[1] < 1 + p.max_range_pct / 100) & (CLOSE > Highest(HIGH, p.length)[1]),
               CLOSE < Lowest(LOW, p.exit_len)[1]))

add("ema_pullback_breakout", "breakout", "Na een pullback naar de EMA 20 in een uptrend: koop als de koers het hoogtepunt van gisteren breekt.",
    {"ema_len": 20, "trend_len": 50},
    lambda p: ((LOW[1] < EMA(CLOSE, p.ema_len)[1]) & (CLOSE > HIGH[1]) & (EMA(CLOSE, p.ema_len) > EMA(CLOSE, p.trend_len)),
               CLOSE < EMA(CLOSE, p.trend_len)))

# ============================================================================ volatiliteit

add("bb_squeeze", "volatility", "TTM-squeeze: koop als de Bollinger Bands uit het Keltner-kanaal breken en de koers stijgt.",
    {"length": 20, "bb_mult": 2.0, "kc_mult": 1.5},
    lambda p: ((bollinger(CLOSE, p.length, p.bb_mult)[1] > keltner(CLOSE, p.length, p.kc_mult)[1])
               & (bollinger(CLOSE, p.length, p.bb_mult)[1][1] < keltner(CLOSE, p.length, p.kc_mult)[1][1])
               & (CLOSE > SMA(CLOSE, p.length)),
               CLOSE < SMA(CLOSE, p.length)))

add("chandelier_exit", "volatility", "Long boven de EMA 50, uitstappen met een Chandelier-stop (hoogste koers 22 dagen min 3x ATR).",
    {"ema_len": 50, "length": 22, "mult": 3.0},
    lambda p: (CLOSE > EMA(CLOSE, p.ema_len), CLOSE < Highest(HIGH, p.length) - p.mult * ATR(p.length)))

add("atr_trailing_trend", "volatility", "Koop bij een nieuw 20-daags hoogtepunt, uitstappen als de koers 2x ATR onder de hoogste slot zakt.",
    {"length": 20, "atr_len": 14, "mult": 2.0},
    lambda p: (CLOSE > Highest(HIGH, p.length)[1], CLOSE < Highest(CLOSE, p.length) - p.mult * ATR(p.atr_len)))

add("low_volatility_trend", "volatility", "Long in een uptrend zolang de volatiliteit (ATR%) laag is.", {"trend_len": 200, "atr_len": 20, "max_atr_pct": 2.5},
    lambda p: ((CLOSE > SMA(CLOSE, p.trend_len)) & (ATR(p.atr_len) / CLOSE * 100 < p.max_atr_pct),
               (CLOSE < SMA(CLOSE, p.trend_len)) | (ATR(p.atr_len) / CLOSE * 100 > p.max_atr_pct * 1.5)))

def _bb_width(length, num_std):
    mid, upper, lower = bollinger(CLOSE, length, num_std)
    return (upper - lower) / mid


add("volatility_contraction", "volatility", "Koop als de Bollinger-breedte bij het laagste punt van 6 maanden staat en de koers uitbreekt.",
    {"length": 20, "num_std": 2.0, "lookback": 126},
    lambda p: ((_bb_width(p.length, p.num_std)[1] <= Lowest(_bb_width(p.length, p.num_std), p.lookback)[1] * 1.1)
               & (CLOSE > bollinger(CLOSE, p.length, p.num_std)[1]),
               CLOSE < SMA(CLOSE, p.length)))

add("atr_expansion_reversal", "volatility", "Koop na een grote dalende dag (range > 2x ATR) in een uptrend, verkoop na 1 hogere slot.",
    {"atr_len": 14, "mult": 2.0, "trend_len": 200},
    lambda p: (((HIGH - LOW) > p.mult * ATR(p.atr_len)) & (CLOSE < OPEN) & (CLOSE > SMA(CLOSE, p.trend_len)), CLOSE > CLOSE[1]))

add("supertrend_adx", "volatility", "Supertrend opwaarts, maar alleen instappen bij ADX > 20.", {"factor": 3.0, "atr_len": 10, "adx_min": 20},
    lambda p: ((Supertrend(p.factor, p.atr_len, "dir") < 0) & (DMI(14, 14, "adx") > p.adx_min), Supertrend(p.factor, p.atr_len, "dir") > 0))

add("keltner_bollinger_reversion", "volatility", "Koop onder zowel de onderste Bollinger- als Keltner-band (extreme uitschieter), verkoop bij het midden.",
    {"length": 20, "bb_mult": 2.0, "kc_mult": 2.0},
    lambda p: ((CLOSE < bollinger(CLOSE, p.length, p.bb_mult)[2]) & (CLOSE < keltner(CLOSE, p.length, p.kc_mult)[2]),
               CLOSE > SMA(CLOSE, p.length)))

# ============================================================================ volume

add("obv_above_sma", "volume", "Long zolang On-Balance Volume boven zijn 20-daags gemiddelde staat.", {"length": 20},
    lambda p: (OBV() > SMA(OBV(), p.length), OBV() < SMA(OBV(), p.length)))

add("obv_ema_cross", "volume", "Koop als OBV-EMA 10 boven OBV-EMA 30 kruist.", {"fast": 10, "slow": 30},
    lambda p: (EMA(OBV(), p.fast).crosses_above(EMA(OBV(), p.slow)), EMA(OBV(), p.fast).crosses_below(EMA(OBV(), p.slow))))

add("obv_price_trend", "volume", "Long als zowel de koers als OBV boven hun 50-daags gemiddelde staan.", {"length": 50},
    lambda p: ((CLOSE > SMA(CLOSE, p.length)) & (OBV() > SMA(OBV(), p.length)), CLOSE < SMA(CLOSE, p.length)))

add("mfi_50_cross", "volume", "Long zolang de Money Flow Index boven 50 staat.", {"length": 14, "level": 50},
    lambda p: (MFI(HLC3, p.length) > p.level, MFI(HLC3, p.length) < p.level))

add("volume_spike_up", "volume", "Koop na een stijgende dag met meer dan 2x het gemiddelde volume, verkoop onder de EMA 10.",
    {"vol_len": 20, "vol_mult": 2.0, "exit_len": 10},
    lambda p: ((VOLUME > SMA(VOLUME, p.vol_len) * p.vol_mult) & (CLOSE > OPEN), CLOSE < EMA(CLOSE, p.exit_len)))

add("volume_trend_confirm", "volume", "Koop boven de SMA 50 als het 10-daags volume boven het 50-daags volume ligt.",
    {"length": 50, "vol_fast": 10, "vol_slow": 50},
    lambda p: ((CLOSE > SMA(CLOSE, p.length)) & (SMA(VOLUME, p.vol_fast) > SMA(VOLUME, p.vol_slow)), CLOSE < SMA(CLOSE, p.length)))

add("vwma_price_cross", "volume", "Long zolang de koers boven de volumegewogen MA (VWMA 50) staat.", {"length": 50},
    lambda p: (CLOSE > VWMA(CLOSE, p.length), CLOSE < VWMA(CLOSE, p.length)))

add("accumulation_days", "volume", "Koop als de koers stijgt op hoger volume dan gisteren, 3 dagen op rij, verkoop onder de EMA 20.",
    {"exit_len": 20},
    lambda p: ((CLOSE > CLOSE[1]) & (VOLUME > VOLUME[1]) & (CLOSE[1] > CLOSE[2]) & (VOLUME[1] > VOLUME[2])
               & (CLOSE[2] > CLOSE[3]) & (VOLUME[2] > VOLUME[3]),
               CLOSE < EMA(CLOSE, p.exit_len)))

add("climax_reversal", "volume", "Koop na een sterk dalende dag op 3x het gemiddelde volume (capitulatie), verkoop na 5 dagen hoger.",
    {"vol_len": 20, "vol_mult": 3.0, "exit_len": 5},
    lambda p: ((VOLUME > SMA(VOLUME, p.vol_len) * p.vol_mult) & (CLOSE < OPEN) & (CLOSE < CLOSE[1]),
               CLOSE > Highest(CLOSE, p.exit_len)[1]))

add("mfi_rsi_combo", "volume", "Koop als zowel MFI als RSI oversold zijn (< 30), verkoop als RSI boven 60 komt.",
    {"length": 14, "low_level": 30, "exit_level": 60},
    lambda p: ((MFI(HLC3, p.length) < p.low_level) & (RSI(CLOSE, p.length) < p.low_level), RSI(CLOSE, p.length) > p.exit_level))

add("hl2_vwma_trend", "volume", "Long als de VWMA 20 van de middenkoers stijgt en de koers erboven staat.", {"length": 20},
    lambda p: ((CLOSE > VWMA(HL2, p.length)) & (VWMA(HL2, p.length) > VWMA(HL2, p.length)[1]), CLOSE < VWMA(HL2, p.length)))

# ============================================================================ vasthouden met trendfilter
# Zo goed als altijd belegd, zoals buy & hold, en alleen eruit in een duidelijke neerwaartse trend. In een
# sterk stijgend aandeel laten gewone strategieën te veel liggen omdat ze vaak cash staan; deze niet.

add("hold_above_sma_buffer", "trend_hold",
    "Vasthouden zolang de koers boven de SMA 200 staat; pas verkopen als ze er 3% onder zakt (minder valse signalen).",
    {"length": 200, "buffer": 3.0},
    lambda p: (CLOSE > SMA(CLOSE, p.length), CLOSE < SMA(CLOSE, p.length) * (1 - p.buffer / 100)))

add("hold_sma_confirm", "trend_hold",
    "Vasthouden; pas verkopen als de koers 5 dagen op rij onder de SMA 200 sluit, terugkopen erboven.",
    {"length": 200, "days": 5},
    lambda p: (CLOSE > SMA(CLOSE, p.length), Highest(CLOSE - SMA(CLOSE, p.length), p.days) < 0))

add("hold_death_cross_exit", "trend_hold",
    "Vasthouden boven de EMA 200; alleen verkopen bij een death cross (EMA 50 < EMA 200) én koers onder de EMA 200.",
    {"fast": 50, "slow": 200},
    lambda p: (CLOSE > EMA(CLOSE, p.slow), (EMA(CLOSE, p.fast) < EMA(CLOSE, p.slow)) & (CLOSE < EMA(CLOSE, p.slow))))

add("hold_drawdown_stop", "trend_hold",
    "Vasthouden; verkopen als de koers 20% onder de top van het jaar staat én onder de SMA 200, terugkopen boven de SMA 50.",
    {"drop": 20.0, "lookback": 252, "trend_len": 200, "reentry": 50},
    lambda p: (CLOSE > SMA(CLOSE, p.reentry),
               (CLOSE < Highest(CLOSE, p.lookback) * (1 - p.drop / 100)) & (CLOSE < SMA(CLOSE, p.trend_len))))

add("hold_stage_analysis", "trend_hold",
    "Weinstein-fase: vasthouden zolang de 30-weekse lijn (EMA 150) stijgt of de koers erboven staat.",
    {"length": 150, "slope": 20},
    lambda p: (CLOSE > EMA(CLOSE, p.length),
               (EMA(CLOSE, p.length) < EMA(CLOSE, p.length)[p.slope]) & (CLOSE < EMA(CLOSE, p.length))))

add("hold_supertrend_slow", "trend_hold",
    "Trage Supertrend (ATR 20, factor 5): blijft lang in de trend en stapt alleen uit bij een grote ommekeer.",
    {"atr_len": 20, "factor": 5.0},
    lambda p: (Supertrend(p.factor, p.atr_len, "dir") < 0, Supertrend(p.factor, p.atr_len, "dir") > 0))

add("hold_crash_filter", "trend_hold",
    "Vasthouden; alleen verkopen in paniek: koers onder de SMA 100 terwijl de dagelijkse beweeglijkheid (ATR) boven 5% ligt.",
    {"length": 100, "atr_len": 20, "vol_level": 5.0, "reentry": 50},
    lambda p: (CLOSE > SMA(CLOSE, p.reentry),
               (CLOSE < SMA(CLOSE, p.length)) & (ATR(p.atr_len) / CLOSE * 100 > p.vol_level)))

add("hold_dual_ma_buffer", "trend_hold",
    "Vasthouden zolang SMA 50 boven SMA 200 staat of de koers boven de SMA 200; uit als beide wegvallen met 2% marge.",
    {"fast": 50, "slow": 200, "buffer": 2.0},
    lambda p: ((SMA(CLOSE, p.fast) > SMA(CLOSE, p.slow)) | (CLOSE > SMA(CLOSE, p.slow)),
               (SMA(CLOSE, p.fast) < SMA(CLOSE, p.slow)) & (CLOSE < SMA(CLOSE, p.slow) * (1 - p.buffer / 100))))
