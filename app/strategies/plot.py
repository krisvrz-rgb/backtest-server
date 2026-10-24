"""Haalt uit de instap- en uitstapregel van een strategie welke indicatoren je moet tekenen.

Elke vergelijking (a > b, a kruist b) wordt bekeken: een getal of parameter wordt een horizontale lijn,
een koersreeks (SMA, EMA, Bollinger, SAR, ...) komt op de koersgrafiek en een oscillator (RSI, MACD,
Stoch, ...) krijgt een eigen paneel onder de koersgrafiek.
"""
from __future__ import annotations

import math

import pandas as pd

from .dsl import (
    ATR, CCI, CMO, DMI, EMA, HMA, MACD, MFI, MOM, OBV, RMA, ROC, RSI, SAR, SMA, TSI, VWMA, WMA, WPR,
    BinOp, Cmp, Const, Cross, Expr, Fn, Highest, Logic, Lowest, Not, Param, Shift, Src, StochD, StochK, Stdev,
    Supertrend, _TrEma, _p, _v,
)

PRICE_SRC = {"open", "high", "low", "close", "hl2", "hlc3", "ohlc4"}
SMOOTHERS = (SMA, EMA, RMA, WMA, HMA, VWMA, Highest, Lowest)
OSC_NAMES = {RSI: "RSI", CCI: "CCI", CMO: "CMO", TSI: "TSI", ROC: "ROC", MOM: "Momentum", WPR: "Williams %R",
             MFI: "MFI", ATR: "ATR", _TrEma: "ATR", Stdev: "Stdev", StochK: "Stochastic", StochD: "Stochastic",
             MACD: "MACD", DMI: "ADX / DI", OBV: "OBV"}
FN_LABEL = {SMA: "SMA", EMA: "EMA", RMA: "RMA", WMA: "WMA", HMA: "HMA", VWMA: "VWMA", Highest: "Hoogste",
            Lowest: "Laagste", RSI: "RSI", CCI: "CCI", CMO: "CMO", TSI: "TSI", ROC: "ROC", MOM: "Mom", WPR: "%R",
            MFI: "MFI", ATR: "ATR", _TrEma: "ATR", Stdev: "Stdev", SAR: "SAR"}
SKIP = "skip"
PRICE_SUM = "price_sum"


def _arg(a) -> str:
    if isinstance(a, Src):
        return a.name
    if isinstance(a, Expr):
        return label(a)
    v = _v(a)
    return f"{v:g}" if isinstance(v, float) else str(v)


def label(e: Expr) -> str:
    """Korte, leesbare naam zoals in de legende van TradingView (bv. 'SMA 50', 'RSI 14')."""
    if isinstance(e, Const):
        v = _v(e.value)
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)
    if isinstance(e, Src):
        return e.name
    if isinstance(e, Shift):
        inner = label(e.a)
        return f"({inner})[{_v(e.n)}]" if isinstance(e.a, BinOp) and _const_value(e.a) is None \
            and " " in inner and not _band(e.a) and not inner.startswith("Midden") else f"{inner}[{_v(e.n)}]"
    if isinstance(e, BinOp):
        return _binop_label(e)
    if isinstance(e, MACD):
        return {"macd": "MACD", "signal": "Signaal", "hist": "Histogram"}[e.part]
    if isinstance(e, DMI):
        return {"plus": "+DI", "minus": "-DI", "adx": "ADX"}[e.part]
    if isinstance(e, Supertrend):
        return f"Supertrend {_arg(e.args[1])}/{_arg(e.args[0])}"
    if isinstance(e, StochK):
        return f"%K {_arg(e.args[0])}"
    if isinstance(e, StochD):
        return f"%D {_arg(e.args[2])}"
    if isinstance(e, OBV):
        return "OBV"
    if isinstance(e, Fn):
        name = FN_LABEL.get(type(e), type(e).__name__)
        args = list(e.args)
        if args and isinstance(args[0], Src) and args[0].name in ("close", "high", "low") and type(e) in SMOOTHERS:
            src = args.pop(0).name
            name += "" if src == "close" else f" {src}"
        elif args and isinstance(args[0], Src) and args[0].name == "close":
            args.pop(0)
        elif args and isinstance(args[0], Src) and args[0].name == "volume":
            args.pop(0)
            name += " volume"
        return f"{name} {' '.join(_arg(a) for a in args)}".strip()
    return type(e).__name__


PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


def _const_value(e) -> float | None:
    """Waarde van een expressie die alleen uit getallen en parameters bestaat (bv. 1 - pct / 100)."""
    if isinstance(e, Const):
        v = _v(e.value)
        return float(v) if isinstance(v, (int, float)) else None
    if isinstance(e, BinOp):
        a, b = _const_value(e.a), _const_value(e.b)
        if a is None or b is None:
            return None
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else float("nan")}[e.op]
    return None


def _band(e: BinOp) -> str | None:
    """Herken Bollinger-, Keltner- en ATR-banden: midden ± k × (stdev | ATR)."""
    if e.op not in "+-" or not isinstance(e.b, BinOp) or e.b.op != "*":
        return None
    k, vol = (e.b.a, e.b.b) if _const_value(e.b.a) is not None else (e.b.b, e.b.a)
    kv = _const_value(k)
    if kv is None:
        return None
    side = "boven" if e.op == "+" else "onder"
    if isinstance(vol, Stdev) and isinstance(e.a, SMA):
        return f"BB {side} {_arg(vol.args[1])} {kv:g}"
    if isinstance(vol, _TrEma) and isinstance(e.a, EMA):
        return f"Keltner {side} {_arg(vol.args[0])} {kv:g}"
    if isinstance(vol, ATR):
        return f"{label(e.a)} {e.op} {kv:g}×ATR {_arg(vol.args[0])}"
    return None


def _binop_label(e: BinOp) -> str:
    c = _const_value(e)
    if c is not None:
        return f"{c:g}"
    band = _band(e)
    if band:
        return band
    if e.op == "/" and _const_value(e.b) == 2 and isinstance(e.a, BinOp) and e.a.op == "+" \
            and isinstance(e.a.a, Highest) and isinstance(e.a.b, Lowest):
        return f"Midden {_arg(e.a.a.args[1])}"

    def part(x, right=False):
        t = label(x)
        if isinstance(x, BinOp) and _const_value(x) is None and not _band(x) and not t.startswith("Midden"):
            if PREC[x.op] < PREC[e.op] or (right and PREC[x.op] == PREC[e.op] and e.op in "-/"):
                return f"({t})"
        return t
    return f"{part(e.a)} {e.op.replace('*', '×')} {part(e.b, right=True)}"


def scale(e) -> str:
    """Op welke schaal een reeks hoort: 'price', 'level' (vast getal), 'skip' of de naam van een paneel."""
    if not isinstance(e, Expr) or isinstance(e, Const):
        return "level"
    if isinstance(e, Src):
        return "price" if e.name in PRICE_SRC else "Volume"
    if isinstance(e, Shift):
        return scale(e.a)
    if isinstance(e, SMOOTHERS):
        return scale(e.args[0])
    if isinstance(e, (SAR,)):
        return "price"
    if isinstance(e, Supertrend):
        return "price" if e.part == "line" else SKIP
    if isinstance(e, BinOp):
        sa, sb = scale(e.a), scale(e.b)
        if SKIP in (sa, sb):
            return SKIP
        if sa == "level" or sb == "level":
            other = sb if sa == "level" else sa
            if other == PRICE_SUM:
                return "price" if e.op in "*/" else label(e)
            return other
        if e.op == "+" and sa == sb == "price":
            return PRICE_SUM  # (hoogste + laagste) / 2 is weer een koers
        if e.op in "+-":
            if sa == "price" and sb != "price":
                return "price"  # bv. midden + 2 × stdev, slot - ATR
            if sb == "price" and sa != "price":
                return "price"
            if sa == sb and sa != "price":
                return sa
        return label(e)  # verhouding of verschil: eigen paneel
    if isinstance(e, (Cmp, Cross, Logic, Not)):
        return SKIP
    for cls, name in OSC_NAMES.items():
        if isinstance(e, cls):
            return name
    return type(e).__name__


def _is_plain_price(e) -> bool:
    while isinstance(e, Shift):
        e = e.a
    return isinstance(e, Src) and e.name in PRICE_SRC


def _comparisons(e, out):
    if isinstance(e, (Cmp, Cross)):
        out.append((e.a, e.b))
        for side in (e.a, e.b):  # vergelijkingen binnen een rekenexpressie bestaan niet, maar wees veilig
            if isinstance(side, (Logic, Not, Cmp, Cross)):
                _comparisons(side, out)
    elif isinstance(e, Logic):
        _comparisons(e.a, out)
        _comparisons(e.b, out)
    elif isinstance(e, Not):
        _comparisons(e.a, out)


def _companions(e: Expr) -> list[Expr]:
    """Volledige set lijnen voor indicatoren die je normaal samen ziet (MACD, DMI, Stochastic)."""
    if isinstance(e, MACD):
        return [MACD(*e.args, part=p) for p in ("macd", "signal", "hist")]
    if isinstance(e, DMI):
        return [DMI(*e.args, part=p) for p in ("plus", "minus", "adx")]
    if isinstance(e, StochK):
        return [e, StochD(e.args[0], e.args[1], 3)]
    if isinstance(e, StochD):
        return [StochK(e.args[0], e.args[1]), e]
    return [e]


def _series(s: pd.Series, index) -> list:
    s = s.reindex(index)
    return [{"time": d.date().isoformat(), "value": round(float(v), 6)}
            for d, v in zip(s.index, s) if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _level_value(e, df) -> float | None:
    if isinstance(e, Param):
        return float(e.value)
    if not isinstance(e, Expr):
        return float(e)
    try:
        v = e.calc(df.iloc[-5:]).dropna()
        return float(v.iloc[-1]) if len(v) else None
    except Exception:
        return None


def _level_name(e) -> str:
    if isinstance(e, Const) and isinstance(e.value, Param):
        return e.value.name
    if isinstance(e, Param):
        return e.name
    return ""


def indicators(entry: Expr, exit_: Expr, df: pd.DataFrame, index) -> dict:
    """Overlays (op de koers) en panelen (oscillatoren) om bij een backtest te tekenen.

    `df` bevat de koersen inclusief opwarmperiode, `index` is de backtestperiode die getekend wordt.
    """
    pairs: list = []
    _comparisons(entry, pairs)
    _comparisons(exit_, pairs)

    overlays: dict[str, Expr] = {}
    panes: dict[str, dict] = {}

    def add(expr):
        if isinstance(expr, Supertrend):
            expr = Supertrend(*expr.args, part="line")
        k = scale(expr)
        if k == PRICE_SUM:
            k = label(expr)
        if k in (SKIP, "level") or _is_plain_price(expr):
            return None
        if k == "price":
            overlays.setdefault(label(expr), expr)
        else:
            pane = panes.setdefault(k, {"series": {}, "levels": {}})
            for c in _companions(expr):
                pane["series"].setdefault(label(c), c)
        return k

    for a, b in pairs:
        ka, kb = add(a), add(b)
        for key, other in ((ka, b), (kb, a)):
            if key and key != "price" and scale(other) == "level":
                v = _level_value(other, df)
                if v is not None:
                    panes[key]["levels"].setdefault(round(v, 6), _level_name(other))

    out_overlays = []
    for name, expr in overlays.items():
        try:
            out_overlays.append({"label": name, "data": _series(expr.calc(df), index)})
        except Exception:
            continue
    out_panes = []
    for title, pane in panes.items():
        series = []
        if len(title) > 32:
            title = "Hulpreeks"
        for name, expr in pane["series"].items():
            try:
                data = _series(expr.calc(df), index)
            except Exception:
                continue
            kind = "histogram" if (isinstance(expr, MACD) and expr.part == "hist") or \
                (isinstance(expr, Src) and expr.name == "volume") else "line"
            series.append({"label": name, "type": kind, "data": data})
        if series:
            levels = [{"value": v, "label": n} for v, n in sorted(pane["levels"].items())]
            out_panes.append({"title": title, "series": series, "levels": levels})
    return {"overlays": out_overlays, "panes": out_panes}


__all__ = ["indicators", "label", "scale", "_p"]
