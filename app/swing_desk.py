"""Swing Desk: de tokenized aandelen van Hyperliquid (HIP-3) gekoppeld aan de backtests, met een eigen papierportefeuille.

* Universum: elke HIP-3-markt (bv. xyz:NVDA) wordt gekoppeld aan het onderliggende aandeel op Yahoo. Staat een aandeel op
  meerdere HIP-3-beurzen, dan handelen we op de markt met het meeste volume. De koppeling telt alleen als de prijs op
  Hyperliquid binnen 20% van de laatste slotkoers ligt (zo glipt er geen verkeerd aandeel met dezelfde ticker tussen).
* Elke handelsdag na de slot in New York (standaard 22.40 lokale tijd) draait swing.analyze op elk aandeel en worden
  de voorstellen opnieuw gemaakt. Open posities sluiten als het signaal van hun strategie wegvalt.
* De papierportefeuille staat helemaal los van de scalping van de Swarm Desk (eigen bestand, eigen startkapitaal), zodat
  je beide kan vergelijken. Funding wordt elk uur aangerekend met de echte fundingrate van de markt.
* Er gaan GEEN echte orders uit deze module. Live handelen komt pas na expliciete toestemming.

Bestanden: data/swing-state.json (portefeuille en instellingen), data/swing-analysis.json (laatste analyse),
data/swing-trades.jsonl (alle afgesloten papiertrades).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from . import desk, swing
from .data import DATA_DIR, get_prices

STATE_FILE = DATA_DIR / "swing-state.json"
ANALYSIS_FILE = DATA_DIR / "swing-analysis.json"
TRADES_FILE = DATA_DIR / "swing-trades.jsonl"

# Naam op Hyperliquid -> Yahoo-ticker, waar die niet gewoon hetzelfde is.
OVERRIDES = {"XYZ100": "^NDX", "NDX": "^NDX", "US100": "^NDX", "SPX": "^GSPC", "US500": "^GSPC", "SP500": "^GSPC",
             "DJI": "^DJI", "US30": "^DJI", "GOLD": "GC=F", "XAU": "GC=F", "SILVER": "SI=F", "XAG": "SI=F",
             "CL": "CL=F", "OIL": "CL=F", "WTI": "CL=F", "BRENT": "BZ=F", "COPPER": "HG=F", "NATGAS": "NG=F",
             "PLATINUM": "PL=F", "PALLADIUM": "PA=F", "BRK": "BRK-B", "BRKB": "BRK-B", "BRK.B": "BRK-B"}
SKIP = {"EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "MXN", "CNH", "KRW", "DXY", "USD", "VIX"}
PRICE_TOLERANCE = (0.8, 1.25)

DEFAULTS = {
    "auto": True,          # voorstellen met status 'voorstel' automatisch op papier openen
    "kelly": 0.25,         # kwart-Kelly
    "risk_pct": 2.0,       # verlies bij de stop hooguit zoveel % van de portefeuille
    "margin_pct": 10.0,    # marge per positie hooguit zoveel % van de portefeuille
    "max_lev": 3,          # hoogste hefboom
    "max_open": 5,         # posities tegelijk
    "fee_bps": 9.0,        # kosten per kant (taker op HIP-3, voorzichtig gerekend)
    "min_volume": 250_000, # minimum dagvolume op Hyperliquid in dollar
    "refresh_at": "22:40", # lokale tijd, ma-vr (na de slot in New York)
}


def yahoo_for(coin: str) -> str | None:
    name = coin.split(":")[-1].upper()
    if name in SKIP:
        return None
    return OVERRIDES.get(name, name.replace(".", "-"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _load_prices(yahoo: str) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.DateOffset(years=swing.YEARS + 1)).date().isoformat()
    return get_prices(yahoo, "yahoo", start, None, warmup=True)


def funding_avg(coin: str, days: int = 7) -> float | None:
    """Gemiddelde fundingrate per uur over de laatste dagen (positief = long betaalt short)."""
    try:
        rows = desk._info({"type": "fundingHistory", "coin": coin, "startTime": _now_ms() - days * 86_400_000})
        rates = [float(r["fundingRate"]) for r in rows]
        return sum(rates) / len(rates) if rates else None
    except Exception:  # noqa: BLE001
        return None


def stock_markets(uni: dict) -> dict[str, dict]:
    """Yahoo-ticker -> {"market": meest verhandelde HIP-3-markt, "alts": andere markten}."""
    out: dict[str, dict] = {}
    for a in uni.values():
        if a.get("dex") in (None, "", "hyperliquid"):
            continue
        y = yahoo_for(a["coin"])
        if not y:
            continue
        g = out.setdefault(y, {"market": a, "alts": []})
        if a["volume"] > g["market"]["volume"]:
            g["alts"].append(g["market"]["coin"])
            g["market"] = a
        elif a is not g["market"]:
            g["alts"].append(a["coin"])
    return out


def stats(trades: list[dict], start: float, curve: list) -> dict:
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = -sum(p for p in pnls if p <= 0)
    peak, mdd = None, 0.0
    for _, v in curve:
        peak = v if peak is None else max(peak, v)
        mdd = min(mdd, v / peak - 1) if peak else mdd
    return {"n": len(pnls), "wins": len(wins), "win": len(wins) / len(pnls) if pnls else None, "pnl": sum(pnls),
            "ret": sum(pnls) / start if start else None, "pf": sum(wins) / loss if loss > 0 else None,
            "fees": sum(t.get("fees", 0) for t in trades), "funding": sum(t.get("funding", 0) for t in trades),
            "max_dd": mdd, "avg_hours": sum(t.get("hours", 0) for t in trades) / len(trades) if trades else None}


def trade_stats(pnls: list[float]) -> dict:
    """Zelfde vorm als trade_stats in main.py (Swarm): winrate, gem. winnaar/verliezer ($, positief), break-even-winrate
    en verwachting per trade. Een pnl ≤ 0 telt als verliezer. pnl is netto na kosten en funding."""
    w = [p for p in pnls if p > 0]
    l = [-p for p in pnls if p <= 0]
    aw = sum(w) / len(w) if w else 0.0
    al = sum(l) / len(l) if l else 0.0
    n = len(pnls)
    return {"n": n, "wins": len(w), "losses": len(l), "winrate": round(len(w) / n, 4) if n else None,
            "avg_win": round(aw, 2), "avg_loss": round(al, 2),
            "be_winrate": round(al / (aw + al), 4) if aw + al > 0 else None,
            "expectancy": round(sum(pnls) / n, 2) if n else None}


class SwingBook:
    def __init__(self, loader: Callable[[str], pd.DataFrame] | None = None) -> None:
        saved = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        start = float(os.getenv("SWING_PAPER_START", saved.get("start", 1000)))
        self.start: float = saved.get("start", start)
        self.cash: float = saved.get("cash", start)
        self.cfg: dict = DEFAULTS | saved.get("cfg", {})
        self.positions: list[dict] = saved.get("positions", [])
        self.history: list[dict] = saved.get("history", [])
        self.curve: list = saved.get("curve") or [[_now_ms(), start]]
        self.analysis: dict = json.loads(ANALYSIS_FILE.read_text()) if ANALYSIS_FILE.exists() else {}
        self.loader = loader or _load_prices
        self.lock = threading.RLock()
        self.progress: dict = {"running": False, "done": 0, "total": 0, "current": None, "error": None}
        self.last_curve = 0.0
        self.started = False

    # ---------- bewaren ----------
    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.curve = self.curve[-5000:]
            self.history = self.history[-2000:]
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"start": self.start, "cash": self.cash, "cfg": self.cfg, "positions": self.positions,
                                       "history": self.history, "curve": self.curve}))
            os.replace(tmp, STATE_FILE)

    def set_cfg(self, **kw: Any) -> dict:
        with self.lock:
            for k, v in kw.items():
                if k in DEFAULTS and v is not None:
                    self.cfg[k] = type(DEFAULTS[k])(v)
            self.cfg["max_lev"] = max(1, min(10, int(self.cfg["max_lev"])))
            self.cfg["max_open"] = max(1, min(20, int(self.cfg["max_open"])))
        self.save()
        return self.cfg

    # ---------- analyse ----------
    def refresh(self, symbols: list[str] | None = None) -> None:
        """Alle HIP-3-aandelen analyseren en de voorstellen opnieuw maken (duurt een paar minuten)."""
        with self.lock:
            if self.progress["running"]:
                return
            self.progress = {"running": True, "done": 0, "total": 0, "current": None, "error": None,
                             "started": datetime.now().isoformat(timespec="seconds")}
        try:
            uni = desk.universe(max_age=60)
            groups = stock_markets(uni)
            if symbols:
                groups = {k: v for k, v in groups.items() if k in symbols}
            self.progress["total"] = len(groups)
            results = {}
            for y, g in sorted(groups.items(), key=lambda kv: -kv[1]["market"]["volume"]):
                mkt = g["market"]
                self.progress["current"] = mkt["coin"]
                row: dict[str, Any] = {"yahoo": y, "coin": mkt["coin"], "alts": g["alts"]}
                try:
                    prices = self.loader(y)
                    ratio = mkt["px"] / float(prices["close"].iloc[-1])
                    row["ratio"] = ratio
                    if not PRICE_TOLERANCE[0] <= ratio <= PRICE_TOLERANCE[1]:
                        raise ValueError(f"koppeling onzeker: Hyperliquid {mkt['px']:.4g} vs {y} {prices['close'].iloc[-1]:.4g}")
                    row["analysis"] = swing.analyze(prices, self.cfg["fee_bps"])
                    if row["analysis"]["pick"]:
                        row["funding_h"] = funding_avg(mkt["coin"])
                except Exception as e:  # noqa: BLE001 - één aandeel zonder data stopt de rest niet
                    row["error"] = str(e)[:200]
                results[y] = row
                self.progress["done"] += 1
            with self.lock:
                self.analysis = {"at": _now_ms(), "date": datetime.now().date().isoformat(), "rows": results}
                tmp = ANALYSIS_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.analysis, default=str))
                os.replace(tmp, ANALYSIS_FILE)
            self.after_refresh(uni)
        except Exception as e:  # noqa: BLE001
            self.progress["error"] = str(e)[:300]
        finally:
            self.progress["running"] = False
            self.progress["finished"] = datetime.now().isoformat(timespec="seconds")

    def proposals(self, uni: dict | None = None) -> list[dict]:
        uni = uni if uni is not None else desk.universe()
        eq = self.equity(uni)
        out = []
        for y, row in (self.analysis.get("rows") or {}).items():
            mkt = uni.get(row["coin"]) or {}
            if "analysis" not in row or not mkt:
                out.append({"yahoo": y, "coin": row["coin"], "status": "fout" if "error" in row else "geen markt",
                            "why": [row.get("error") or "markt niet meer op Hyperliquid"]})
                continue
            fh = row.get("funding_h")
            p = swing.proposal(row["coin"], row["analysis"], mkt, fh if fh is not None else mkt.get("funding", 0.0), eq, self.cfg)
            p["yahoo"], p["alts"] = y, row.get("alts", [])
            p["held"] = any(x["yahoo"] == y for x in self.positions)
            out.append(p)
        order = {"voorstel": 0, "loopt al": 1, "te weinig bewijs": 2, "te duur": 3, "te illiquide": 4, "geen signaal": 5}
        out.sort(key=lambda p: (order.get(p["status"], 9), -((p.get("pick") or {}).get("score") or 0)))
        return out

    def after_refresh(self, uni: dict) -> None:
        rows = self.analysis.get("rows") or {}
        for pos in list(self.positions):
            a = (rows.get(pos["yahoo"]) or {}).get("analysis")
            if a and a["signals"].get(pos["strategy"]) is False:
                px = (uni.get(pos["coin"]) or {}).get("px")
                if px:
                    self.close(pos["id"], "signaal uit", px)
        if self.cfg["auto"]:
            for p in self.proposals(uni):
                if p["status"] == "voorstel" and not p["held"]:
                    try:
                        self.open(p, by="auto")
                    except ValueError:
                        pass

    # ---------- papierportefeuille ----------
    def upnl(self, pos: dict, px: float) -> float:
        return pos["notional"] * (px / pos["entry"] - 1)

    def equity(self, uni: dict | None = None) -> float:
        uni = uni or {}
        return self.cash + sum(self.upnl(p, (uni.get(p["coin"]) or {}).get("px") or p["px"]) - p["funding"]
                               for p in self.positions)

    def open_by_coin(self, coin: str, by: str = "mens") -> dict:
        uni = desk.universe(max_age=60)
        p = next((x for x in self.proposals(uni) if x["coin"] == coin), None)
        if not p:
            raise ValueError(f"geen analyse voor {coin}")
        return self.open(p, by=by, manual=True)

    def open(self, p: dict, by: str = "auto", manual: bool = False) -> dict:
        allowed = ("voorstel",) + (("te weinig bewijs", "loopt al") if manual else ())
        if p["status"] not in allowed or not (p.get("size") or {}).get("ok"):
            raise ValueError(f"{p['coin']}: {p['status']} ({'; '.join(p.get('why', []))})")
        with self.lock:
            if any(x["yahoo"] == p["yahoo"] for x in self.positions):
                raise ValueError(f"{p['coin']}: er loopt al een positie")
            if len(self.positions) >= self.cfg["max_open"]:
                raise ValueError(f"al {len(self.positions)} posities open (max {self.cfg['max_open']})")
            eq = self.equity()
            sz = p["size"]
            notional = sz["frac"] * eq
            margin = notional / sz["lev"]
            free = self.cash - sum(x["margin"] for x in self.positions)
            if margin > free:
                raise ValueError(f"te weinig vrije marge (${free:.2f} vrij, ${margin:.2f} nodig)")
            fee = notional * self.cfg["fee_bps"] / 1e4
            px = p["px"]
            pos = {"id": uuid.uuid4().hex[:8], "coin": p["coin"], "yahoo": p["yahoo"], "strategy": p["strategy"],
                   "category": p.get("category"), "opened": _now_ms(), "by": by, "entry": px, "px": px,
                   "notional": notional, "margin": margin, "lev": sz["lev"], "sl": p["sl"], "tp": p["tp"],
                   "liq": p.get("liq"), "max_until": _now_ms() + int(p["max_days"] * 86_400_000), "fees": fee,
                   "funding": 0.0, "fund_t": time.time(), "status_at_open": p["status"], "why": p.get("why", []),
                   "wf": {k: p["wf"].get(k) for k in ("n", "win", "pf")}, "best": px}
            self.cash -= fee
            self.positions.append(pos)
        self.save()
        return pos

    def close(self, pid: str, why: str, px: float | None = None) -> dict:
        with self.lock:
            pos = next((x for x in self.positions if x["id"] == pid), None)
            if not pos:
                raise ValueError("positie niet gevonden")
            px = px or pos["px"]
            fee = pos["notional"] * px / pos["entry"] * self.cfg["fee_bps"] / 1e4
            gross = self.upnl(pos, px)
            pnl = gross - pos["fees"] - fee - pos["funding"]
            self.cash += gross - fee - pos["funding"]
            self.positions.remove(pos)
            h = {k: pos[k] for k in ("id", "coin", "yahoo", "strategy", "category", "opened", "by", "entry", "notional",
                                     "lev", "sl", "tp", "funding", "wf")}
            h |= {"t": _now_ms(), "exit": px, "pnl": pnl, "fees": pos["fees"] + fee, "reason": why,
                  "ret": px / pos["entry"] - 1, "hours": (time.time() * 1000 - pos["opened"]) / 3.6e6, "paper": True}
            self.history.append(h)
            self.curve.append([_now_ms(), self.equity()])
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRADES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(h) + "\n")
        self.save()
        return h

    def tick(self, uni: dict | None = None) -> None:
        """Elke minuut: prijzen bijwerken, funding aanrekenen, SL/TP/maximale duur controleren."""
        if uni is None:
            uni = desk.universe(max_age=55)
        now = time.time()
        for pos in list(self.positions):
            m = uni.get(pos["coin"])
            if not m:
                continue
            px = m["px"]
            with self.lock:
                pos["px"] = px
                pos["best"] = max(pos.get("best", px), px)
                pos["funding"] += pos["notional"] * px / pos["entry"] * float(m.get("funding") or 0) * (now - pos["fund_t"]) / 3600
                pos["fund_t"] = now
            if px <= pos["sl"]:
                self.close(pos["id"], "sl", px)
            elif px >= pos["tp"]:
                self.close(pos["id"], "tp", px)
            elif now * 1000 > pos["max_until"]:
                self.close(pos["id"], "max duur", px)
        if now - self.last_curve > 900:
            self.last_curve = now
            with self.lock:
                self.curve.append([_now_ms(), round(self.equity(uni), 4)])
            self.save()

    def reset(self) -> None:
        with self.lock:
            self.cash = self.start
            self.positions, self.history, self.curve = [], [], [[_now_ms(), self.start]]
        self.save()

    def view(self) -> dict:
        try:
            uni = desk.universe()
        except Exception:  # noqa: BLE001
            uni = {}
        eq = self.equity(uni)
        pos = []
        for p in self.positions:
            px = (uni.get(p["coin"]) or {}).get("px") or p["px"]
            pos.append(p | {"px": px, "upnl": self.upnl(p, px) - p["funding"] - p["fees"],
                            "days": (time.time() * 1000 - p["opened"]) / 86_400_000})
        used = sum(p["margin"] for p in self.positions)
        return {"paper": True, "start": self.start, "cash": self.cash, "equity": eq, "used": used, "free": self.cash - used,
                "positions": pos, "history": self.history[-200:][::-1], "curve": self.curve[-1500:], "cfg": self.cfg,
                "stats": stats(self.history, self.start, self.curve + [[_now_ms(), eq]]),
                "analysis_at": self.analysis.get("at"), "progress": self.progress,
                "proposals": self.proposals(uni) if uni else []}

    def summary(self) -> dict:
        """Licht overzicht voor de statusbalk: waarde, open trades, winst/verlies per handelsdag, laatste analyse.
        Handelsdag zoals het dagjournaal van de Swarm Desk (22.00 tot 22.00 lokale tijd)."""
        from .desk_journal import DAY_END_HOUR, day_bounds
        try:
            uni = desk.universe()
        except Exception:  # noqa: BLE001
            uni = {}
        eq = self.equity(uni)
        days: dict[str, dict] = {}
        for h in self.history:
            # handelsdag = de datum waarop de dag om 22.00 eindigt
            day = datetime.fromtimestamp(h["t"] / 1000 + (24 - DAY_END_HOUR) * 3600).date().isoformat()
            r = days.setdefault(day, {"day": day, "wins": 0, "losses": 0, "pnl": 0.0})
            r["wins" if h["pnl"] > 0 else "losses"] += 1
            r["pnl"] += h["pnl"]
        t0, t1, today = day_bounds()
        return {"mode": "paper", "start": self.start, "equity": eq, "pnl": eq - self.start,
                "open": len(self.positions), "upnl": sum(self.upnl(p, (uni.get(p["coin"]) or {}).get("px") or p["px"])
                                                     - p["funding"] - p["fees"] for p in self.positions),
                "today": days.get(today, {"day": today, "wins": 0, "losses": 0, "pnl": 0.0}),
                "days": sorted(days.values(), key=lambda r: r["day"])[-30:],
                "analysis_at": self.analysis.get("at"), "analysis_running": self.progress["running"],
                "all": trade_stats([h["pnl"] for h in self.history])}

    # ---------- achtergrond ----------
    def due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        hh, mm = (int(x) for x in self.cfg["refresh_at"].split(":"))
        last = self.analysis.get("at")
        if last and time.time() * 1000 - last < 3600_000:
            return False
        if not last or time.time() * 1000 - last > 30 * 3600_000:
            return True  # nooit of al meer dan een dag niet meer gedraaid (bv. server stond uit)
        return now.weekday() < 5 and (now.hour, now.minute) >= (hh, mm) and self.analysis.get("date") != now.date().isoformat()

    def loop(self) -> None:
        time.sleep(120)  # eerst de Swarm Desk laten opwarmen
        while True:
            try:
                self.tick()
                if self.due() and not self.progress["running"]:
                    threading.Thread(target=self.refresh, daemon=True, name="swing-refresh").start()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(60)

    def start_background(self) -> None:
        if not self.started:
            self.started = True
            threading.Thread(target=self.loop, daemon=True, name="swing").start()


def compare(swing_view: dict, desk_portfolio: dict, desk_history: list[dict]) -> dict:
    """Swing naast scalping, allebei op papier, elk met hun eigen startkapitaal."""
    hist = [h for h in desk_history if h.get("paper", True) and h.get("pnl") is not None]
    d_curve = desk_portfolio.get("curve") or []
    d = stats(hist, desk_portfolio.get("start") or 1000, d_curve)
    d["equity"] = desk_portfolio.get("equity")
    d["fees"] = sum(h.get("fees") or 0 for h in hist) if any("fees" in h for h in hist) else None
    d["funding"] = None  # de scalping houdt posities hooguit enkele uren: funding wordt daar niet apart bijgehouden
    d["avg_hours"] = (sum(((h.get("t") or 0) - (h.get("opened") or h.get("t") or 0)) for h in hist) / len(hist) / 3.6e6) if hist else None
    d["since"] = hist[0]["t"] if hist else None
    s = swing_view["stats"] | {"equity": swing_view["equity"], "since": swing_view["curve"][0][0] if swing_view["curve"] else None}
    return {"swing": s, "scalp": d}


BOOK = SwingBook()
