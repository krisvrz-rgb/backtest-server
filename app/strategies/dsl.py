"""Kleine taal om strategieën één keer te beschrijven en ze zowel in Python als in Pine Script te draaien.

Een expressie (Expr) weet hoe ze zichzelf berekent op een koers-DataFrame (`calc`) en hoe ze eruitziet
in Pine Script v6 (`pine`). De indicatoren volgen de definities van TradingView (`ta.*`), zodat de
resultaten van de server en de Strategy Tester zo dicht mogelijk bij elkaar liggen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- context


@dataclass
class PineCtx:
    """Verzamelt hulpregels (bv. tuple-uitkomsten van ta.macd) die boven de condities moeten staan."""
    lines: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)

    def declare(self, key: str, make_line) -> str:
        if key not in self.names:
            var = f"_v{len(self.names) + 1}"
            self.names[key] = var
            self.lines.append(make_line(var))
        return self.names[key]

    def hoist(self, expr: str) -> str:
        """Elke ta.*-aanroep krijgt een eigen variabele op globaal niveau. Pine v6 evalueert `and`/`or`
        lui, dus een ta.*-functie binnen een conditie zou anders niet op elke bar berekend worden."""
        return self.declare(expr, lambda var: f"{var} = {expr}")


class Param:
    """Instelbare parameter: in Python de waarde, in Pine de naam van de input."""

    def __init__(self, name: str, value: Any):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return f"Param({self.name}={self.value})"

    # Rekenen met een parameter levert een expressie op (bv. p.pct / 100).
    def __add__(self, o): return Const(self) + o
    def __radd__(self, o): return o + Const(self)
    def __sub__(self, o): return Const(self) - o
    def __rsub__(self, o): return _wrap(o) - Const(self)
    def __mul__(self, o): return Const(self) * o
    def __rmul__(self, o): return _wrap(o) * Const(self)
    def __truediv__(self, o): return Const(self) / o
    def __rtruediv__(self, o): return _wrap(o) / Const(self)


def _v(x):
    return x.value if isinstance(x, Param) else x


def _p(x) -> str:
    if isinstance(x, Param):
        return x.name
    if isinstance(x, bool):
        return "true" if x else "false"
    return repr(x) if isinstance(x, (int, float)) else str(x)


def _wrap(x) -> "Expr":
    return x if isinstance(x, Expr) else Const(x)


# --------------------------------------------------------------------------- basis


class Expr:
    def calc(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def pine(self, ctx: PineCtx) -> str:
        raise NotImplementedError

    # rekenkunde
    def __add__(self, o): return BinOp("+", self, _wrap(o))
    def __radd__(self, o): return BinOp("+", _wrap(o), self)
    def __sub__(self, o): return BinOp("-", self, _wrap(o))
    def __rsub__(self, o): return BinOp("-", _wrap(o), self)
    def __mul__(self, o): return BinOp("*", self, _wrap(o))
    def __rmul__(self, o): return BinOp("*", _wrap(o), self)
    def __truediv__(self, o): return BinOp("/", self, _wrap(o))
    def __neg__(self): return BinOp("-", Const(0), self)

    # vergelijkingen
    def __gt__(self, o): return Cmp(">", self, _wrap(o))
    def __ge__(self, o): return Cmp(">=", self, _wrap(o))
    def __lt__(self, o): return Cmp("<", self, _wrap(o))
    def __le__(self, o): return Cmp("<=", self, _wrap(o))

    # logica (alleen zinvol op condities)
    def __and__(self, o): return Logic("and", self, _wrap(o))
    def __or__(self, o): return Logic("or", self, _wrap(o))
    def __invert__(self): return Not(self)

    def __getitem__(self, n):
        return Shift(self, n)

    def crosses_above(self, o): return Cross(self, _wrap(o), up=True)
    def crosses_below(self, o): return Cross(self, _wrap(o), up=False)


class Const(Expr):
    def __init__(self, value):
        self.value = value

    def calc(self, df):
        v = _v(self.value)
        return pd.Series(float(v) if not isinstance(v, bool) else v, index=df.index)

    def pine(self, ctx):
        return _p(self.value)


class Src(Expr):
    """Koersreeks: open, high, low, close, volume of hl2/hlc3/ohlc4."""

    def __init__(self, name: str):
        self.name = name

    def calc(self, df):
        if self.name == "hl2":
            return (df["high"] + df["low"]) / 2
        if self.name == "hlc3":
            return (df["high"] + df["low"] + df["close"]) / 3
        if self.name == "ohlc4":
            return (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        return df[self.name].astype(float)

    def pine(self, ctx):
        return self.name


OPEN, HIGH, LOW, CLOSE, VOLUME = Src("open"), Src("high"), Src("low"), Src("close"), Src("volume")
HL2, HLC3 = Src("hl2"), Src("hlc3")


class BinOp(Expr):
    OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}

    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def calc(self, df):
        a, b = self.a.calc(df), self.b.calc(df)
        if self.op == "/":
            return a / b.replace(0, np.nan)
        return self.OPS[self.op](a, b)

    def pine(self, ctx):
        return f"({self.a.pine(ctx)} {self.op} {self.b.pine(ctx)})"


class Cmp(Expr):
    OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}

    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def calc(self, df):
        a, b = self.a.calc(df), self.b.calc(df)
        return (self.OPS[self.op](a, b) & a.notna() & b.notna()).astype(bool)

    def pine(self, ctx):
        return f"({self.a.pine(ctx)} {self.op} {self.b.pine(ctx)})"


class Logic(Expr):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def calc(self, df):
        a, b = self.a.calc(df).astype(bool), self.b.calc(df).astype(bool)
        return (a & b) if self.op == "and" else (a | b)

    def pine(self, ctx):
        return f"({self.a.pine(ctx)} {self.op} {self.b.pine(ctx)})"


class Not(Expr):
    def __init__(self, a):
        self.a = a

    def calc(self, df):
        return ~self.a.calc(df).astype(bool)

    def pine(self, ctx):
        return f"(not {self.a.pine(ctx)})"


class Shift(Expr):
    """Waarde van n bars geleden (Pine: expr[n])."""

    def __init__(self, a, n):
        self.a, self.n = a, n

    def calc(self, df):
        return self.a.calc(df).shift(int(_v(self.n)))

    def pine(self, ctx):
        return f"({self.a.pine(ctx)})[{_p(self.n)}]"


class Cross(Expr):
    def __init__(self, a, b, up):
        self.a, self.b, self.up = a, b, up

    def calc(self, df):
        a, b = self.a.calc(df), self.b.calc(df)
        d, d1 = a - b, (a - b).shift(1)
        res = (d > 0) & (d1 <= 0) if self.up else (d < 0) & (d1 >= 0)
        return res.fillna(False).astype(bool)

    def pine(self, ctx):
        fn = "ta.crossover" if self.up else "ta.crossunder"
        return ctx.hoist(f"{fn}({self.a.pine(ctx)}, {self.b.pine(ctx)})")


class TrueCond(Expr):
    def calc(self, df):
        return pd.Series(True, index=df.index)

    def pine(self, ctx):
        return "true"


class FalseCond(Expr):
    def calc(self, df):
        return pd.Series(False, index=df.index)

    def pine(self, ctx):
        return "false"


# --------------------------------------------------------------------------- hulpfuncties (Python)


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _tr(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


class Fn(Expr):
    """Indicator met een eenvoudige Pine-functieaanroep."""
    pine_fn: str = ""

    def __init__(self, *args):
        self.args = args

    def arg(self, i, df):
        a = self.args[i]
        return a.calc(df) if isinstance(a, Expr) else _v(a)

    def pine(self, ctx):
        return ctx.hoist(f"{self.pine_fn}({', '.join(a.pine(ctx) if isinstance(a, Expr) else _p(a) for a in self.args)})")


class SMA(Fn):
    pine_fn = "ta.sma"
    def calc(self, df): return self.arg(0, df).rolling(int(self.arg(1, df))).mean()


class EMA(Fn):
    pine_fn = "ta.ema"
    def calc(self, df): return _ema(self.arg(0, df), int(self.arg(1, df)))


class RMA(Fn):
    pine_fn = "ta.rma"
    def calc(self, df): return _rma(self.arg(0, df), int(self.arg(1, df)))


class WMA(Fn):
    pine_fn = "ta.wma"
    def calc(self, df): return _wma(self.arg(0, df), int(self.arg(1, df)))


class HMA(Fn):
    pine_fn = "ta.hma"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        return _wma(2 * _wma(s, max(n // 2, 1)) - _wma(s, n), max(int(np.sqrt(n)), 1))


class VWMA(Fn):
    pine_fn = "ta.vwma"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        return (s * df["volume"]).rolling(n).sum() / df["volume"].rolling(n).sum()


class Stdev(Fn):
    pine_fn = "ta.stdev"
    def calc(self, df): return self.arg(0, df).rolling(int(self.arg(1, df))).std(ddof=0)


class Highest(Fn):
    pine_fn = "ta.highest"
    def calc(self, df): return self.arg(0, df).rolling(int(self.arg(1, df))).max()


class Lowest(Fn):
    pine_fn = "ta.lowest"
    def calc(self, df): return self.arg(0, df).rolling(int(self.arg(1, df))).min()


class RSI(Fn):
    pine_fn = "ta.rsi"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        d = s.diff()
        up, down = _rma(d.clip(lower=0), n), _rma(-d.clip(upper=0), n)
        rsi = 100 - 100 / (1 + up / down)
        return rsi.where(down != 0, 100.0).where(up.notna())


class ROC(Fn):
    pine_fn = "ta.roc"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        return 100 * (s - s.shift(n)) / s.shift(n)


class MOM(Fn):
    pine_fn = "ta.mom"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        return s - s.shift(n)


class CCI(Fn):
    pine_fn = "ta.cci"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        ma = s.rolling(n).mean()
        dev = s.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (s - ma) / (0.015 * dev)


class CMO(Fn):
    pine_fn = "ta.cmo"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        d = s.diff()
        up, down = d.clip(lower=0).rolling(n).sum(), (-d.clip(upper=0)).rolling(n).sum()
        return 100 * (up - down) / (up + down)


class TSI(Fn):
    """True Strength Index, schaal -1..1 zoals ta.tsi."""
    pine_fn = "ta.tsi"

    def calc(self, df):
        s, short, long_ = self.arg(0, df), int(self.arg(1, df)), int(self.arg(2, df))
        m = s.diff()
        return _ema(_ema(m, long_), short) / _ema(_ema(m.abs(), long_), short)


class ATR(Fn):
    pine_fn = "ta.atr"
    def calc(self, df): return _rma(_tr(df), int(self.arg(0, df)))


class WPR(Fn):
    pine_fn = "ta.wpr"

    def calc(self, df):
        n = int(self.arg(0, df))
        hh, ll = df["high"].rolling(n).max(), df["low"].rolling(n).min()
        return 100 * (df["close"] - hh) / (hh - ll)


class MFI(Fn):
    pine_fn = "ta.mfi"

    def calc(self, df):
        s, n = self.arg(0, df), int(self.arg(1, df))
        mf = s * df["volume"]
        d = s.diff()
        pos = mf.where(d > 0, 0.0).rolling(n).sum()
        neg = mf.where(d < 0, 0.0).rolling(n).sum()
        return 100 - 100 / (1 + pos / neg)


class OBV(Expr):
    def calc(self, df):
        return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

    def pine(self, ctx):
        return ctx.hoist("ta.obv")


class StochK(Fn):
    """%K van de stochastic, afgevlakt met een SMA (zoals de standaard Stoch-indicator)."""

    def calc(self, df):
        n, smooth = int(self.arg(0, df)), int(self.arg(1, df))
        ll, hh = df["low"].rolling(n).min(), df["high"].rolling(n).max()
        k = 100 * (df["close"] - ll) / (hh - ll)
        return k.rolling(smooth).mean()

    def pine(self, ctx):
        n, smooth = (_p(a) for a in self.args)
        return ctx.hoist(f"ta.sma(ta.stoch(close, high, low, {n}), {smooth})")


class StochD(Fn):
    def calc(self, df):
        return StochK(self.args[0], self.args[1]).calc(df).rolling(int(self.arg(2, df))).mean()

    def pine(self, ctx):
        k = StochK(self.args[0], self.args[1]).pine(ctx)
        return ctx.hoist(f"ta.sma({k}, {_p(self.args[2])})")


class _Tuple(Expr):
    """Indicator waarvan Pine een tuple teruggeeft; we declareren hem één keer en kiezen een element."""
    index: int = 0

    def __init__(self, *args):
        self.args = args

    def call(self) -> tuple[str, int]:
        raise NotImplementedError

    def pine(self, ctx):
        call, size = self.call()
        names = ctx.declare(call, lambda var: f"[{', '.join(f'{var}_{i}' for i in range(size))}] = {call}")
        return f"{names}_{self.index}"


class MACD(_Tuple):
    """MACD(src, fast, slow, signal); part = 'macd' | 'signal' | 'hist'."""

    def __init__(self, src, fast, slow, signal, part="macd"):
        super().__init__(src, fast, slow, signal)
        self.part = part
        self.index = {"macd": 0, "signal": 1, "hist": 2}[part]

    def call(self):
        src, f, s, sig = self.args
        return f"ta.macd({src.pine(None)}, {_p(f)}, {_p(s)}, {_p(sig)})", 3

    def calc(self, df):
        src, f, s, sig = self.args
        x = src.calc(df)
        macd = _ema(x, int(_v(f))) - _ema(x, int(_v(s)))
        signal = _ema(macd, int(_v(sig)))
        return {"macd": macd, "signal": signal, "hist": macd - signal}[self.part]


class DMI(_Tuple):
    """DMI(di_len, adx_len); part = 'plus' | 'minus' | 'adx'."""

    def __init__(self, di_len, adx_len, part="adx"):
        super().__init__(di_len, adx_len)
        self.part = part
        self.index = {"plus": 0, "minus": 1, "adx": 2}[part]

    def call(self):
        return f"ta.dmi({_p(self.args[0])}, {_p(self.args[1])})", 3

    def calc(self, df):
        n, m = int(_v(self.args[0])), int(_v(self.args[1]))
        up, down = df["high"].diff(), -df["low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = _rma(_tr(df), n)
        plus = 100 * _rma(pd.Series(plus_dm, index=df.index), n) / tr
        minus = 100 * _rma(pd.Series(minus_dm, index=df.index), n) / tr
        s = plus + minus
        adx = 100 * _rma((plus - minus).abs() / s.where(s != 0, 1), m)
        return {"plus": plus, "minus": minus, "adx": adx}[self.part]


class Supertrend(_Tuple):
    """Supertrend(factor, atr_len); part = 'line' | 'dir' (dir < 0 = opwaartse trend, zoals in Pine)."""

    def __init__(self, factor, atr_len, part="dir"):
        super().__init__(factor, atr_len)
        self.part = part
        self.index = {"line": 0, "dir": 1}[part]

    def call(self):
        return f"ta.supertrend({_p(self.args[0])}, {_p(self.args[1])})", 2

    def calc(self, df):
        factor, n = float(_v(self.args[0])), int(_v(self.args[1]))
        atr = _rma(_tr(df), n).to_numpy()
        hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
        close = df["close"].to_numpy()
        upper, lower = hl2 + factor * atr, hl2 - factor * atr
        st = np.full(len(df), np.nan)
        direction = np.full(len(df), np.nan)
        for i in range(len(df)):
            if np.isnan(atr[i]):
                continue
            if i > 0 and not np.isnan(st[i - 1]):
                lower[i] = lower[i] if (lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
                upper[i] = upper[i] if (upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
                if st[i - 1] == upper[i - 1]:
                    direction[i] = -1 if close[i] > upper[i] else 1
                else:
                    direction[i] = 1 if close[i] < lower[i] else -1
            else:
                direction[i] = 1
            st[i] = lower[i] if direction[i] == -1 else upper[i]
        out = st if self.part == "line" else direction
        return pd.Series(out, index=df.index)


class SAR(Fn):
    """Parabolic SAR (ta.sar)."""
    pine_fn = "ta.sar"

    def calc(self, df):
        start, inc, mx = (float(self.arg(i, df)) for i in range(3))
        h, l = df["high"].to_numpy(), df["low"].to_numpy()
        n = len(df)
        sar = np.full(n, np.nan)
        if n < 2:
            return pd.Series(sar, index=df.index)
        up = h[1] >= h[0]
        af, ep = start, (h[1] if up else l[1])
        sar[1] = l[0] if up else h[0]
        for i in range(2, n):
            prev = sar[i - 1]
            s = prev + af * (ep - prev)
            if up:
                s = min(s, l[i - 1], l[i - 2])
                if l[i] < s:
                    up, s, ep, af = False, ep, l[i], start
                elif h[i] > ep:
                    ep, af = h[i], min(af + inc, mx)
            else:
                s = max(s, h[i - 1], h[i - 2])
                if h[i] > s:
                    up, s, ep, af = True, ep, h[i], start
                elif l[i] < ep:
                    ep, af = l[i], min(af + inc, mx)
            sar[i] = s
        return pd.Series(sar, index=df.index)


def keltner(src, n, mult):
    """(midden, boven, onder) van het Keltner-kanaal zoals ta.kc (true range)."""
    mid = EMA(src, n)
    rng = _TrEma(n)
    return mid, mid + mult * rng, mid - mult * rng


class _TrEma(Fn):
    def calc(self, df): return _ema(_tr(df), int(self.arg(0, df)))
    def pine(self, ctx): return ctx.hoist(f"ta.ema(ta.tr(true), {_p(self.args[0])})")


def bollinger(src, n, mult):
    """(midden, boven, onder) van de Bollinger Bands zoals ta.bb."""
    mid = SMA(src, n)
    sd = Stdev(src, n)
    return mid, mid + mult * sd, mid - mult * sd


def donchian_mid(n):
    return (Highest(HIGH, n) + Lowest(LOW, n)) / 2


# --------------------------------------------------------------------------- signalen


def run_rules(df: pd.DataFrame, entry: Expr, exit_: Expr, active: pd.Series | None = None) -> pd.Series:
    """Toestandsmachine identiek aan de Pine-export: instappen als je vlak staat, uitstappen als je long bent."""
    e = entry.calc(df).fillna(False).to_numpy(dtype=bool)
    x = exit_.calc(df).fillna(False).to_numpy(dtype=bool)
    act = np.ones(len(df), dtype=bool) if active is None else active.reindex(df.index).fillna(False).to_numpy(dtype=bool)
    pos = np.zeros(len(df))
    state = 0.0
    for i in range(len(df)):
        if state == 0 and e[i] and act[i]:
            state = 1.0
        elif state == 1 and x[i]:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)
