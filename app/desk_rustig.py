"""Rustig: de trage tegenhanger van de zwerm, op papier, om je desk eerlijk mee te vergelijken.

Idee (zoals "Helder"): minder rondes, kosten eerst. Elke REBALANCE uur rangschikt hij een vaste lijst liquide
Hyperliquid-perps op hun rendement over de laatste LOOKBACK uur (relatieve kracht). De sterkste N gaan long,
de zwakste N short, samen GROSS × de portefeuille. Kosten eerst:
- een munt die al in het boek zit blijft staan zolang hij binnen de top N + BUFFER blijft (geen nutteloze wissels);
- een nieuwe positie opent alleen als zijn signaal (|rendement over de lookback|) minstens COST_MULT × de kosten
  van een ronde is;
- posities worden niet bijgesteld op kleine afwijkingen.

Kosten: taker bij in- en uitstap (de zwerm-instelling DESK_FEE_BPS) + 1 bp slippage, plus de echte funding van
Hyperliquid zolang een positie open staat (als die op te halen is). Alles op 1-uurs candles, geen lookahead:
een beslissing op uur T gebruikt alleen candles tot en met T en vult op de slot van T.

Hij start op hetzelfde moment en met hetzelfde bedrag als je papierportefeuille, met parameters die vastliggen
vóór de start, dus elke dag die erbij komt is een echte papiertest vooruit. De tabel met andere herbalans-tijden
is alleen een gevoeligheidscheck, niet om de beste knop te kiezen.

Daarnaast: wat je eigen trades zeggen. Hoe lang zat elke trade in de markt, waar zit het verlies (per houdtijd en
per uitstapreden), hoeveel ging naar kosten, en wat als de desk alleen trades nam met minstens X¢ edge bij de
instap (achteraf op je eigen trades, dus optimistisch).
"""
from __future__ import annotations

import math
import statistics
import threading
import time

from . import desk

UNIVERSE = ("BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "SUI", "AVAX", "LINK", "BNB", "NEAR", "AAVE", "LTC", "ENA")
DEFAULT = {"rebalance": 72, "lookback": 120, "n": 2, "gross": 0.5, "buffer": 1, "cost_mult": 3.0}
GRID = (24, 48, 72)
HIST_DAYS = 55
EDGE_STEPS = (-2, 0, 1, 2, 4)
SLIP_BPS = 1.0
HOLD_BUCKETS = ((0, 1, "< 1 min"), (1, 5, "1-5 min"), (5, 15, "5-15 min"), (15, 60, "15-60 min"),
                (60, 240, "1-4 uur"), (240, 1440, "4-24 uur"), (1440, 1e12, "> 1 dag"))

_COINS: dict = {}  # munt -> {at, from, closes, funding}
PAUSE = 0.3
_LOCK = threading.Lock()


# ---------------------------------------------------------------- eigen trades
def held_min(h: dict) -> float | None:
    """Houdtijd van een afgesloten trade in minuten (vanaf de vulling van de instap tot de afrekening)."""
    if h.get("t") and h.get("opened"):
        return max(0.0, (h["t"] - h["opened"]) / 60000)
    ew = h.get("exit_why") or {}
    return ew.get("held")


def _fees(h: dict) -> float:
    ew = h.get("exit_why") or {}
    if ew.get("fees") is not None:
        return float(ew["fees"])
    cost = (h.get("why") or {}).get("cost") or 6.0
    return h.get("usd", 0) * cost / 1e4


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    pnl = [r["pnl"] for r in rows]
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
    fees = sum(_fees(r) for r in rows)
    aw, al = (sum(wins) / len(wins) if wins else 0.0), (-sum(losses) / len(losses) if losses else 0.0)
    return {"n": n, "pnl": round(sum(pnl), 2), "fees": round(fees, 2), "gross": round(sum(pnl) + fees, 2),
            "win": round(len(wins) / n * 100, 1), "payoff": round(aw / al, 2) if al else None,
            "avg": round(sum(pnl) / n, 3)}


def own_stats(history: list[dict], start_ms: int | None = None) -> dict:
    """Houdtijden, verdeling en waar het geld heen ging, voor de afgesloten papiertrades."""
    hs = [h for h in history if h.get("paper", True) and h.get("pnl") is not None and (not start_ms or h.get("t", 0) >= start_ms)]
    hs.sort(key=lambda h: h.get("t", 0))
    held = [(h, held_min(h)) for h in hs]
    known = [m for _, m in held if m is not None]
    buckets = []
    for lo, hi, name in HOLD_BUCKETS:
        rows = [h for h, m in held if m is not None and lo <= m < hi]
        buckets.append({"name": name, "lo": lo, "hi": hi if hi < 1e11 else None, **_agg(rows)})
    reasons: dict[str, list] = {}
    for h, m in held:
        reasons.setdefault(h.get("reason") or "venster", []).append(h)
    by_reason = []
    for k, rows in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        ms = [held_min(r) for r in rows if held_min(r) is not None]
        by_reason.append({"reason": k, "median_min": round(statistics.median(ms), 1) if ms else None, **_agg(rows)})
    filt = []  # alleen trades met minstens zoveel edge (¢ boven break-even na kosten) bij de instap
    for k in EDGE_STEPS:
        filt.append({"k": k, **_agg([h for h in hs if h.get("edge") is not None and h["edge"] >= k])})
    span_d = ((hs[-1]["t"] - hs[0].get("opened", hs[0]["t"])) / 864e5) if len(hs) > 1 else 0
    tot = _agg(hs)
    return {"total": tot, "per_day": round(len(hs) / span_d, 1) if span_d > 0 else None,
            "days": round(span_d, 1), "median_min": round(statistics.median(known), 1) if known else None,
            "under5": sum(1 for m in known if m < 5), "under5_pct": round(sum(1 for m in known if m < 5) / len(known) * 100, 1) if known else None,
            "buckets": buckets, "reasons": by_reason, "edge_filter": filt,
            "fee_share": round(tot["fees"] / tot["gross"] * 100, 0) if tot.get("n") and tot["gross"] > 0 else None}


def desk_curve(history: list[dict], start_cash: float, start_ms: int) -> list[list]:
    """Papiercurve van de desk uit de afgesloten trades: [ms, waarde]."""
    hs = sorted((h for h in history if h.get("paper", True) and h.get("pnl") is not None and h.get("t", 0) >= start_ms),
                key=lambda h: h["t"])
    out, v = [[start_ms, round(start_cash, 2)]], start_cash
    for h in hs:
        v += h["pnl"]
        out.append([h["t"], round(v, 2)])
    return out


def max_dd(curve: list[list]) -> float:
    peak, dd = -1e18, 0.0
    for _, v in curve:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak if peak > 0 else 0)
    return round(dd * 100, 2)


# ---------------------------------------------------------------- rustige strategie
def _call(payload: dict):
    """Eén Hyperliquid-call met backoff (2, 4, 8 s) bij 429 of netwerkfout: de desk bevraagt Hyperliquid zelf ook constant."""
    for wait in (2, 4, 8, None):
        try:
            return desk._info(payload)
        except Exception:
            if wait is None:
                raise
            time.sleep(wait)


def fetch_coin(c: str, start_ms: int, end_ms: int, old: dict | None) -> dict:
    """1-uurs slotkoersen en funding per uur voor één munt. Funding wordt aangevuld vanaf wat er al was."""
    rows = _call({"type": "candleSnapshot", "req": {"coin": c, "interval": "1h", "startTime": start_ms, "endTime": end_ms}})
    if not rows:
        raise RuntimeError("geen candles")
    closes = {int(r["t"]) // 3_600_000: float(r["c"]) for r in rows}
    f = dict(old["funding"]) if old and old.get("from", 1e18) <= start_ms else {}
    t = (max(f) + 1) * 3_600_000 if f else start_ms
    try:
        for _ in range(10):
            time.sleep(PAUSE)
            rr = _call({"type": "fundingHistory", "coin": c, "startTime": t, "endTime": end_ms})
            if not rr:
                break
            for r in rr:
                f[int(r["time"]) // 3_600_000] = float(r["fundingRate"])
            t = int(rr[-1]["time"]) + 1
            if len(rr) < 500:
                break
    except Exception:
        pass  # funding is een correctie; zonder funding toch de koersen bewaren
    return {"at": time.time(), "from": start_ms, "closes": closes, "funding": f}


def fetch(coins, start_ms: int, end_ms: int, max_age: float = 900, force: bool = False) -> tuple[dict, dict, list]:
    """Per munt bewaard: alleen wat ontbreekt of ouder is dan max_age wordt opnieuw opgehaald.
    Lukt een verversing niet, dan blijft de vorige (volledige) reeks staan."""
    missing = []
    for c in coins:
        old = _COINS.get(c)
        if old and not force and time.time() - old["at"] < max_age and old["from"] <= start_ms:
            continue
        try:
            _COINS[c] = fetch_coin(c, start_ms, end_ms, old)
        except Exception:
            if not old or old["from"] > start_ms:
                missing.append(c)
            break  # na 2+4+8 s nog altijd afgeremd: de rest bij de volgende vraag, de pagina niet minuten laten wachten
        time.sleep(PAUSE)
    missing += [c for c in coins if c not in missing and (c not in _COINS or _COINS[c]["from"] > start_ms)]
    have = [c for c in coins if c in _COINS and _COINS[c]["from"] <= start_ms]
    return {c: _COINS[c]["closes"] for c in have}, {c: _COINS[c]["funding"] for c in have}, missing


def simulate(closes: dict, funding: dict, start_h: int, end_h: int, cash: float, fee_bps: float,
             rebalance: int, lookback: int, n: int, gross: float, buffer: int, cost_mult: float) -> dict:
    """Relatieve kracht, N long en N short, herbalans elke `rebalance` uur. Uren zijn ms // 3.600.000."""
    side_cost = (fee_bps + SLIP_BPS) / 1e4
    round_bp = 2 * (fee_bps + SLIP_BPS)
    pos: dict[str, dict] = {}  # coin -> {side, usd, entry, opened(h), fund}
    eq_curve, trades, fees, fund_paid = [], [], 0.0, 0.0
    realized = cash

    def px(c, h):
        return closes.get(c, {}).get(h)

    def value(h):
        v = realized
        for c, p in pos.items():
            q = px(c, h) or p["last"]
            if q:
                v += p["usd"] * (1 if p["side"] == "long" else -1) * (q / p["entry"] - 1)
        return v

    def close(c, h, why):
        nonlocal realized, fees
        p = pos.pop(c)
        q = px(c, h) or p["last"]
        sign = 1 if p["side"] == "long" else -1
        fee = p["usd"] * side_cost
        pnl = p["usd"] * sign * (q / p["entry"] - 1) - fee
        realized += pnl
        fees += fee
        total = pnl - p["fee_in"] - p["fund"]
        trades.append({"coin": c, "side": p["side"], "usd": round(p["usd"], 2), "entry": p["entry"], "exit": q,
                       "opened": p["opened"] * 3_600_000, "t": h * 3_600_000, "held_h": h - p["opened"],
                       "pnl": round(total, 2), "fees": round(p["fee_in"] + fee, 3), "funding": round(p["fund"], 3), "why": why})

    for h in range(start_h, end_h + 1):
        # funding per uur op open posities (long betaalt bij positieve funding)
        for c, p in pos.items():
            r = funding.get(c, {}).get(h)
            q = px(c, h)
            if q:
                p["last"] = q
            if r is not None and q:
                notional = p["usd"] * q / p["entry"]
                pay = notional * r * (1 if p["side"] == "long" else -1)
                p["fund"] += pay
                realized -= pay
        if (h - start_h) % rebalance == 0:
            rets = {}
            for c in closes:
                a, b = px(c, h - lookback), px(c, h)
                if a and b:
                    rets[c] = b / a - 1
            if len(rets) >= 2 * n + 1:
                ranked = sorted(rets, key=lambda c: rets[c], reverse=True)
                keep_l = set(ranked[:n + buffer])
                keep_s = set(ranked[-(n + buffer):])
                for c in list(pos):
                    p = pos[c]
                    if c not in rets or (p["side"] == "long" and c not in keep_l) or (p["side"] == "short" and c not in keep_s):
                        close(c, h, "uit de top" if c in rets else "geen koers")
                eq = value(h)
                per = eq * gross / (2 * n)
                for side, cands in (("long", ranked[:n + buffer]), ("short", ranked[::-1][:n + buffer])):
                    have = sum(1 for p in pos.values() if p["side"] == side)
                    for c in cands:
                        if have >= n:
                            break
                        if c in pos:
                            continue
                        sig = rets[c] if side == "long" else -rets[c]
                        if sig * 1e4 < cost_mult * round_bp:  # signaal kleiner dan k × de kosten van een ronde: overslaan
                            continue
                        fee = per * side_cost
                        realized -= fee
                        fees += fee
                        pos[c] = {"side": side, "usd": per, "entry": px(c, h), "opened": h, "fund": 0.0, "fee_in": fee, "last": px(c, h)}
                        have += 1
        v = value(h)
        eq_curve.append([h * 3_600_000, round(v, 2)])
    fund_paid = sum(p["fund"] for p in pos.values())  # open posities alleen tonen, niet sluiten
    fund_paid += sum(t["funding"] for t in trades)
    closed = trades
    wins = [t["pnl"] for t in closed if t["pnl"] > 0]
    losses = [-t["pnl"] for t in closed if t["pnl"] <= 0]
    open_book = []
    for c, p in pos.items():
        q = px(c, end_h) or p["last"]
        sign = 1 if p["side"] == "long" else -1
        open_book.append({"coin": c, "side": p["side"], "usd": round(p["usd"], 2), "entry": p["entry"], "now": q,
                          "opened": p["opened"] * 3_600_000, "upl": round(p["usd"] * sign * (q / p["entry"] - 1), 2),
                          "held_h": end_h - p["opened"]})
    end_v = eq_curve[-1][1] if eq_curve else cash
    nxt = None
    if eq_curve:
        nxt = (start_h + ((end_h - start_h) // rebalance + 1) * rebalance) * 3_600_000
    return {"params": {"rebalance": rebalance, "lookback": lookback, "n": n, "gross": gross, "buffer": buffer,
                       "cost_mult": cost_mult, "fee_bps": fee_bps, "slip_bps": SLIP_BPS},
            "end": round(end_v, 2), "pnl": round(end_v - cash, 2), "closed": len(closed),
            "win": round(len(wins) / len(closed) * 100, 1) if closed else None,
            "payoff": round((sum(wins) / len(wins)) / (sum(losses) / len(losses)), 2) if wins and losses else None,
            "fees": round(fees, 2), "funding": round(fund_paid, 2), "max_dd": max_dd(eq_curve),
            "median_held_h": statistics.median([t["held_h"] for t in closed]) if closed else None,
            "curve": eq_curve, "trades": closed[-60:], "open": open_book, "next_rebalance": nxt}


def compare(history: list[dict], paper: dict, fee_bps: float, max_age: float = 900, force: bool = False) -> dict:
    """Je desk tegen Rustig. Drie blikken:
    - sinds de start van je papierdesk: zelfde start, zelfde bedrag (de echte papiertest vooruit);
    - de laatste ±55 dagen terug (backtest, want je desk heeft nog maar kort data);
    - hetzelfde Rustig-signaal maar elk uur herbalanceren: wat alleen al de omloop kost.
    Candles en funding worden per munt 15 min bewaard; zolang niet alle munten binnen zijn: geen vergelijking."""
    hs = [h for h in history if h.get("paper", True) and h.get("pnl") is not None]
    firsts = [h.get("opened") or h["t"] for h in hs] + [c[0] for c in (paper.get("curve") or [])[:1]]
    now_ms = int(time.time() * 1000)
    start_ms = int(min(firsts, default=now_ms - 864e5))
    start_cash = float(paper.get("start") or 1000.0)
    start_h, end_h = start_ms // 3_600_000 + 1, now_ms // 3_600_000 - 1
    look = max(DEFAULT["lookback"], 168)
    hist_from = min(start_h, end_h - HIST_DAYS * 24) - look - 2
    with _LOCK:
        closes, funding, missing = fetch(UNIVERSE, hist_from * 3_600_000, now_ms, max_age, force)
    if missing:  # met een half universum is de vergelijking niet eerlijk: dan liever niets tonen
        return {"ok": False, "error": f"{len(closes)} van {len(UNIVERSE)} munten binnen, nog niet: {', '.join(missing)}. "
                f"Hyperliquid remt even af; probeer het over een minuut opnieuw.", "coins": sorted(closes), "missing": missing}
    fetched_at = min(_COINS[c]["at"] for c in closes)
    h0 = hist_from + look + 2  # eerste uur met genoeg geschiedenis voor de lookback
    fwd = simulate(closes, funding, start_h, max(end_h, start_h + 1), start_cash, fee_bps, **DEFAULT)
    hist = simulate(closes, funding, h0, end_h, 1000.0, fee_bps, **DEFAULT)
    hourly = simulate(closes, funding, h0, end_h, 1000.0, fee_bps, **{**DEFAULT, "rebalance": 1, "buffer": 0, "cost_mult": 0})
    keys = ("pnl", "closed", "win", "payoff", "fees", "funding", "max_dd")
    grid = []
    for r in GRID:
        for lb in (72, 120):
            g = simulate(closes, funding, h0, end_h, 1000.0, fee_bps, **{**DEFAULT, "rebalance": r, "lookback": lb})
            grid.append({k: g[k] for k in keys} | {"rebalance": r, "lookback": lb})
    grid.append({k: hourly[k] for k in keys} | {"rebalance": 1, "lookback": DEFAULT["lookback"], "hourly": True})
    own = own_stats(history, start_ms)
    dc = desk_curve(history, start_cash, start_ms)
    b0, b1 = closes.get("BTC", {}).get(h0), closes.get("BTC", {}).get(end_h)
    slim = lambda r: {k: v for k, v in r.items() if k != "trades"}
    return {"ok": True, "start": start_ms, "start_cash": start_cash, "now": now_ms, "coins": sorted(closes),
            "funding_ok": any(funding.values()), "rustig": fwd, "hist": slim(hist), "hourly": slim(hourly),
            "hist_start": h0 * 3_600_000, "grid": grid, "own": own,
            "desk": {"curve": dc, "end": dc[-1][1], "pnl": round(dc[-1][1] - start_cash, 2), "max_dd": max_dd(dc)},
            "btc_hist": round((b1 / b0 - 1) * 100, 2) if b0 and b1 else None, "fetched_at": int(fetched_at * 1000), "universe": len(UNIVERSE)}
