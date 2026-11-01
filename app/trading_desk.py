"""Trading Desk: een handelsdesk met vaste rollen en een ticketproces, naar het model van HyperGrok (Galleon Labs, MIT).

Elke trade volgt dezelfde zeven stappen, in volgorde:

    idee -> bewijs -> risico -> jouw goedkeuring -> uitvoering -> afstemming -> review
    Desk Lead  Market/Research  Risk Manager  jij      Execution Trader  Execution Trader  Trade Reviewer

- Desk Lead opent een voorstel HG-JJJJMMDD-NN en houdt het proces in beweging.
- Market Analyst haalt live marktdata van Hyperliquid (/info, zonder sleutel): mid, mark, oracle, funding,
  open interest, volume, diepte binnen 5/10/25 bp, beperkingen van de markt, bereik en volatiliteit.
- Research Analyst zet er tegenbewijs naast: wat de Swarm en de dag-backtests van die munt vinden.
- Strategist zet signalen om in voorstellen (uit de Swarm), maar beslist niets.
- Risk Manager rekent de grootte uit je limieten en de live boekstand, met gestreste stopafstand, en geeft
  PASS met een exact ticket of REJECT met de ene poort die faalt.
- Jij keurt goed door letterlijk "approve HG-…" te typen, binnen de vervaltijd van het ticket.
- Execution Trader voert uit, alleen na PASS en goedkeuring, één keer. Hier: altijd op papier.
- Trade Reviewer schrijft het journaal en de review als de trade sluit (de lokale 35B als die er is).

Deze desk handelt NOOIT met echt geld: er is geen pad naar /exchange. Het is het 'testnet-niveau' van HyperGrok:
het hele proces met speelgeld, zodat je de tickets, goedkeuringen en reviews ziet voor er ooit een sleutel bestaat.
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import DATA_DIR

DIR = DATA_DIR / "trading-desk"
STATE_FILE = DIR / "state.json"

START_EQUITY = 1000.0
MAKER_BPS = 1.5
TAKER_BPS = 4.5
MIN_NOTIONAL = 10.0          # Hyperliquid weigert orders onder $10
EVIDENCE_MAX_AGE = 300       # s: ouder bewijs is niet bruikbaar voor een PASS
TICK_S = 5

# Plafonds van de desk: je limieten mogen alleen strenger zijn.
CEILINGS = {"risk_per_trade": 2.0, "max_open_risk": 6.0, "max_leverage": 20, "daily_loss_stop": 10.0}

DEFAULT_LIMITS = {
    "risk_per_trade": 0.5,       # % van de waarde, verlies als de stop geraakt wordt
    "max_open_risk": 2.0,        # % van de waarde: open posities plus gereserveerd risico van wachtende instappen
    "max_leverage": 3,
    "max_positions": 3,
    "allowed_markets": [],       # leeg = elke perp met genoeg volume
    "min_volume": 5e6,           # $ per 24 u, als allowed_markets leeg is
    "daily_loss_stop": 2.0,      # % van de waarde bij het begin van de dag: geen nieuw risico meer
    "max_slippage_bps": 10,      # Execution Trader stopt als de mid verder bewoog
    "cluster": ["BTC", "ETH", "SOL"],
    "cluster_max": 2,
    "ticket_expiry_min": 30,
}

ROLES = [
    {"key": "lead", "name": "Desk Lead", "job": "Opent voorstellen, houdt de stappen in volgorde, praat met jou."},
    {"key": "market", "name": "Market Analyst", "job": "Live marktdata van Hyperliquid, met bron en tijdstip."},
    {"key": "research", "name": "Research Analyst", "job": "Tegenbewijs: wat de Swarm en de backtests van de munt vinden."},
    {"key": "strategist", "name": "Strategist", "job": "Zet signalen om in toetsbare voorstellen. Beslist niets."},
    {"key": "risk", "name": "Risk Manager", "job": "Limieten, grootte, boek, veto. PASS met ticket of REJECT."},
    {"key": "exec", "name": "Execution Trader", "job": "De enige die uitvoert: na PASS en jouw goedkeuring, één keer."},
    {"key": "review", "name": "Trade Reviewer", "job": "Journaal en review na elke trade."},
]

_lock = threading.RLock()


# --------------------------------------------------------------------------- tijd en opslag
def _now() -> float:
    return time.time()


def _utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(t if t is not None else _now(), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _hm(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M UTC")


def _blank() -> dict:
    return {"start": START_EQUITY, "cash": START_EQUITY, "limits": [{"v": 1, "at": _now(), "by": "standaard", "why": "startwaarden van de desk",
                                                                      **DEFAULT_LIMITS}],
            "proposals": [], "floor": [], "day": None, "day_start_equity": START_EQUITY, "curve": [], "seq": {}}


_state: dict | None = None


def state() -> dict:
    global _state
    if _state is None:
        try:
            _state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else _blank()
        except Exception:
            _state = _blank()
    return _state


def _save() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state()))
    tmp.replace(STATE_FILE)


def reset_for_tests(path: Path | None = None) -> None:
    global _state, STATE_FILE
    if path is not None:
        STATE_FILE = path
    _state = _blank()


def limits() -> dict:
    return state()["limits"][-1]


# --------------------------------------------------------------------------- vloer: korte overdrachten tussen rollen
def floor(role: str, text: str, pid: str | None = None, to: str | None = None, nxt: str | None = None) -> None:
    st = state()
    st["floor"].append({"t": _now(), "role": role, "pid": pid, "to": to, "next": nxt, "text": text})
    st["floor"] = st["floor"][-400:]
    if pid:
        p = find(pid) or (find(pid[:-2]) if pid.endswith("-B") else None)
        if p is not None:
            p["log"].append({"t": _now(), "role": role, "to": to, "next": nxt, "text": text})


def find(pid: str) -> dict | None:
    for p in state()["proposals"]:
        if p["id"] == pid:
            return p
    return None


def _new_id() -> str:
    st = state()
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    n = st["seq"].get(day, 0) + 1
    st["seq"] = {day: n}
    return f"HG-{day}-{n:02d}"


# --------------------------------------------------------------------------- marktdata (Market Analyst)
def _info(payload: dict):
    from . import desk
    return desk._info(payload)


def _asset(coin: str) -> tuple[dict, dict, str]:
    """meta-regel en context van één markt, vers van /info (metaAndAssetCtxs)."""
    dex = coin.split(":", 1)[0] if ":" in coin else ""
    meta, ctxs = _info({"type": "metaAndAssetCtxs", **({"dex": dex} if dex else {})})
    for a, c in zip(meta["universe"], ctxs):
        if a["name"] == coin:
            return a, c, dex
    raise KeyError(f"{coin} staat niet op Hyperliquid")


def market_evidence(coin: str) -> dict:
    """Alles wat sizing en uitvoering nodig hebben, elk getal met bron en tijd. Wat niet lukt is 'niet beschikbaar'."""
    t = _now()
    ev: dict = {"coin": coin, "t": t, "at": _utc(t), "network": "mainnet (alleen lezen)", "unavailable": []}
    a, c, dex = _asset(coin)
    src = f"/info metaAndAssetCtxs{' dex=' + dex if dex else ''}"
    mark = float(c.get("markPx") or 0)
    mid = float(c.get("midPx") or mark)
    ev.update(mid=mid, mark=mark, oracle=float(c.get("oraclePx") or 0) or None, funding=float(c.get("funding") or 0),
              oi_usd=float(c.get("openInterest") or 0) * mark, vol24=float(c.get("dayNtlVlm") or 0),
              max_lev=int(a.get("maxLeverage") or 1), sz_dec=int(a.get("szDecimals") or 0), src=src, dex=dex or "hyperliquid")
    try:
        book = _info({"type": "l2Book", "coin": coin})
        bids, asks = book["levels"][0], book["levels"][1]
        best_bid, best_ask = float(bids[0]["px"]), float(asks[0]["px"])
        ev["spread_px"] = best_ask - best_bid
        ev["spread_bps"] = (best_ask - best_bid) / mid * 1e4 if mid else None
        depth = {}
        for bp in (5, 10, 25):
            depth[str(bp)] = {"bid": sum(float(l["px"]) * float(l["sz"]) for l in bids if float(l["px"]) >= mid * (1 - bp / 1e4)),
                         "ask": sum(float(l["px"]) * float(l["sz"]) for l in asks if float(l["px"]) <= mid * (1 + bp / 1e4))}
        ev["depth"] = depth
        ev["book_src"] = "/info l2Book"
    except Exception as e:  # zonder diepte: geen PASS, want de stop-slippage is dan niet te schatten
        ev["unavailable"].append(f"orderboek ({e.__class__.__name__})")
    try:
        end = int(t * 1000)
        cs = _info({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1h", "startTime": end - 72 * 3600_000, "endTime": end}})
        cs = [{"h": float(r["h"]), "l": float(r["l"]), "c": float(r["c"])} for r in cs]
        if len(cs) >= 15:
            trs = [max(cs[i]["h"] - cs[i]["l"], abs(cs[i]["h"] - cs[i - 1]["c"]), abs(cs[i]["l"] - cs[i - 1]["c"])) for i in range(1, len(cs))]
            ev["atr1h"] = sum(trs[-14:]) / 14
            last24 = cs[-24:]
            ev["hi24"], ev["lo24"] = max(x["h"] for x in last24), min(x["l"] for x in last24)
            ev["candle_src"] = "/info candleSnapshot 1h"
        else:
            ev["unavailable"].append("volatiliteit (te weinig candles)")
    except Exception as e:
        ev["unavailable"].append(f"volatiliteit ({e.__class__.__name__})")
    return ev


# Research Analyst: main.py zet hier een functie die {coin: ...} teruggeeft met wat de Swarm en de backtests vinden.
research_hook = None


def research(coin: str, side: str) -> dict:
    out = {"t": _now(), "items": [], "unavailable": ["nieuws en kalender (geen bron gekoppeld)"]}
    if research_hook is None:
        out["unavailable"].append("Swarm en backtests (niet gekoppeld)")
        return out
    try:
        r = research_hook(coin) or {}
    except Exception as e:
        out["unavailable"].append(f"Swarm ({e.__class__.__name__})")
        return out
    sw = r.get("swarm")
    if sw and sw.get("side"):
        agree = sw["side"] == side
        out["items"].append({"src": "Swarm-model", "agree": agree,
                             "text": f"Swarm ziet {sw['side']} met kans {sw.get('p', 0) * 100:.0f}% en edge {sw.get('edge', 0):+.1f}¢"
                                     + ("" if agree else " · TEGEN dit voorstel")})
    elif sw is not None:
        out["items"].append({"src": "Swarm-model", "agree": None, "text": "Swarm volgt deze munt niet of warmt nog op"})
    bt = r.get("bt")
    if bt and bt.get("prof"):
        lean = {"stijgend": "long", "zwak": "short"}.get(bt.get("trend"))
        out["items"].append({"src": "dag-backtests", "agree": None if lean is None else lean == side,
                             "text": f"{bt.get('open', 0)} van {bt['prof']} winstgevende strategieën hebben nu een koopsignaal (trend {bt.get('trend')})"})
    return out


# --------------------------------------------------------------------------- boek
def _book_mark(p: dict) -> float | None:
    return p.get("px")


def book() -> dict:
    """Waarde = kas + niet-gerealiseerde winst. Open risico = gestreste verliezen als alle stops geraakt worden,
    plus het gereserveerde risico van goedgekeurde instappen die nog wachten."""
    st = state()
    upnl, used, open_risk, pending_risk, pending_margin, pos, pend = 0.0, 0.0, 0.0, 0.0, 0.0, [], []
    for p in st["proposals"]:
        x = p.get("position")
        if p["status"] == "live" and x:
            px = x.get("px") or x["entry"]
            u = (px - x["entry"]) * x["size"] * (1 if p["side"] == "long" else -1)
            upnl += u
            used += x["entry"] * x["size"] / x["lev"]
            open_risk += p["ticket"]["risk_usd"]
            pos.append(p)
        elif p["status"] in ("approved", "resting"):
            pending_risk += p["ticket"]["risk_usd"]
            pending_margin += p["ticket"]["margin"]
            pend.append(p)
    equity = st["cash"] + upnl
    return {"equity": equity, "cash": st["cash"], "upnl": upnl, "used": used, "free": equity - used - pending_margin,
            "open_risk": open_risk, "pending_risk": pending_risk, "pending_margin": pending_margin,
            "positions": len(pos), "pending": len(pend), "coins": [p["coin"] for p in pos + pend]}


def _roll_day() -> None:
    from . import desk_journal
    st = state()
    day = desk_journal.day_bounds()[2]
    if st.get("day") != day:
        st["day"], st["day_start_equity"] = day, book()["equity"]


# --------------------------------------------------------------------------- voorstel openen (Desk Lead / Strategist / jij)
def propose(coin: str, side: str, stop: float, entry: float | None = None, tp_r: float | None = None,
            idea: str = "", by: str = "jij") -> dict:
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError("kant moet long of short zijn")
    with _lock:
        pid = _new_id()
        p = {"id": pid, "opened": _now(), "by": by, "coin": coin, "side": side, "entry": entry, "stop": float(stop),
             "tp_r": tp_r, "idea": idea or f"{side} {coin}", "status": "evidence", "evidence": None, "research": None,
             "risk": None, "ticket": None, "approval": None, "execution": [], "position": None, "review": None, "log": [],
             "adjust": None}
        state()["proposals"].append(p)
        floor("lead", f"voorstel geopend door {by}: {p['idea']}", pid, to="Market Analyst", nxt="Market Analyst")
        _save()
    return run_lifecycle(pid)


def run_lifecycle(pid: str) -> dict:
    """Desk Lead: bewijs -> risico. Stopt bij jouw goedkeuring; die kan niemand anders geven."""
    p = find(pid)
    if p is None:
        raise KeyError(pid)
    try:
        ev = market_evidence(p["coin"])
    except Exception as e:
        with _lock:
            p["status"] = "void"
            floor("market", f"marktdata niet beschikbaar ({e}). Zonder bewijs geen ticket.", pid, nxt="none")
            _save()
        return p
    rs = research(p["coin"], p["side"])
    with _lock:
        p["evidence"], p["research"] = ev, rs
        floor("market", f"mid {ev['mid']:g} · funding {ev['funding'] * 100:+.4f}%/u · OI ${ev['oi_usd'] / 1e6:,.1f}M · "
                        f"spread {ev.get('spread_bps') or 0:.1f} bp · bron {ev['src']} {_hm(ev['t'])}"
              + (f" · niet beschikbaar: {', '.join(ev['unavailable'])}" if ev["unavailable"] else ""), pid, to="Risk Manager", nxt="Risk Manager")
        for it in rs["items"]:
            floor("research", f"{it['src']}: {it['text']}", pid)
        p["status"] = "risk"
        risk_signoff(p)
        _save()
    return p


# --------------------------------------------------------------------------- Risk Manager
def _round_down(x: float, dec: int) -> float:
    f = 10 ** dec
    return math.floor(x * f + 1e-9) / f


def risk_signoff(p: dict) -> None:
    """PASS met exact ticket, of REJECT met de ene poort die faalt. Elke rekenregel staat in p['risk']['lines']."""
    lim, ev, bk = limits(), p["evidence"], book()
    _roll_day()
    st = state()
    lines: list[str] = []

    def reject(gate: str) -> None:
        p["risk"] = {"verdict": "REJECT", "gate": gate, "lines": lines, "t": _now(), "limits_v": lim["v"]}
        p["status"] = "rejected"
        floor("risk", f"REJECT · poort: {gate}", p["id"], nxt="jij (pas het idee aan of laat het)")

    if ev is None or _now() - ev["t"] > EVIDENCE_MAX_AGE:
        return reject("ontbrekende of verouderde marktdata")
    if any(u.startswith("orderboek") for u in ev["unavailable"]):
        return reject("orderboek niet beschikbaar, stop-slippage niet te schatten")
    for k, cap in CEILINGS.items():
        if lim[k] > cap:
            return reject(f"limieten v{lim['v']} gaan boven het plafond van de desk ({k} {lim[k]} > {cap})")
    coin, side = p["coin"], p["side"]
    allowed = lim.get("allowed_markets") or []
    if allowed and coin not in allowed:
        return reject(f"{coin} staat niet in je toegelaten markten")
    if not allowed and ev["vol24"] < lim.get("min_volume", 0):
        return reject(f"te weinig volume: ${ev['vol24'] / 1e6:,.1f}M per 24 u, minimum ${lim['min_volume'] / 1e6:,.0f}M")
    entry = p["entry"] or ev["mid"]
    stop = p["stop"]
    if (side == "long" and stop >= entry) or (side == "short" and stop <= entry):
        return reject("stop ligt aan de verkeerde kant van de instap")
    equity = bk["equity"]
    day_pnl_pct = (equity / st["day_start_equity"] - 1) * 100 if st["day_start_equity"] else 0
    if day_pnl_pct <= -lim["daily_loss_stop"]:
        return reject(f"dagverliesstop geraakt ({day_pnl_pct:.2f}% vandaag, grens −{lim['daily_loss_stop']}%)")
    risk_usd = equity * lim["risk_per_trade"] / 100
    stop_distance = abs(entry - stop)
    notional_guess = risk_usd / stop_distance * entry
    thin = ev["depth"]["25"]["bid" if side == "long" else "ask"] < 3 * notional_guess
    slip_stop = max(ev.get("spread_px") or 0, entry * (15 if thin else 5) / 1e4)
    stop_fill = stop - slip_stop if side == "long" else stop + slip_stop
    fees_per_unit = (entry + stop_fill) * TAKER_BPS / 1e4
    stressed = abs(entry - stop_fill) + fees_per_unit
    raw = risk_usd / stressed
    size = _round_down(raw, ev["sz_dec"])
    notional = size * entry
    lines += [f"risk_usd = waarde ${equity:,.2f} × {lim['risk_per_trade']}% = ${risk_usd:,.2f}",
              f"stopafstand = |{entry:g} − {stop:g}| = {stop_distance:g}",
              f"slip_stop = {slip_stop:g} ({'dun boek: 15 bp' if thin else '5 bp'}, minstens de spread)",
              f"stop_fill = {stop_fill:g}",
              f"fees per eenheid = ({entry:g} + {stop_fill:g}) × {TAKER_BPS} bp = {fees_per_unit:g}",
              f"gestreste afstand = {stressed:g}",
              f"grootte = {risk_usd:,.2f} / {stressed:g} = {raw:g} → afgerond naar beneden {size:g} {coin}",
              f"notional = ${notional:,.2f}"]
    if size <= 0:
        return reject("risicobudget te klein voor deze stop (minder dan 1 lot)")
    if notional < MIN_NOTIONAL:
        return reject(f"order van ${notional:,.2f} ligt onder het minimum van ${MIN_NOTIONAL:.0f}")
    max_lev = min(CEILINGS["max_leverage"], lim["max_leverage"], ev["max_lev"])
    headroom = bk["free"] * 0.8
    lev = max(1, min(max_lev, math.ceil(notional / max(headroom, 1e-9)))) if headroom > 0 else max_lev
    margin = notional / lev
    lines += [f"max hefboom hier = min(plafond 20x, jouw {lim['max_leverage']}x, beurs {ev['max_lev']}x) = {max_lev}x",
              f"hefboom = {lev}x (laagste die past) → marge ${margin:,.2f}, vrije marge × 0,8 = ${headroom:,.2f}"]
    if margin > headroom:
        return reject(f"marge ${margin:,.2f} past niet in 80% van de vrije marge (${headroom:,.2f})")
    open_after = bk["open_risk"] + bk["pending_risk"] + risk_usd
    lines.append(f"open risico na = ${bk['open_risk']:,.2f} open + ${bk['pending_risk']:,.2f} gereserveerd + ${risk_usd:,.2f} = ${open_after:,.2f}")
    if open_after > equity * lim["max_open_risk"] / 100 + 1e-9:
        return reject(f"totaal open risico ${open_after:,.2f} boven {lim['max_open_risk']}% van de waarde")
    if bk["positions"] + bk["pending"] + 1 > lim["max_positions"]:
        return reject(f"max {lim['max_positions']} posities (open plus wachtend)")
    cl = lim.get("cluster") or []
    if coin in cl and sum(1 for c in bk["coins"] if c in cl) + 1 > lim.get("cluster_max", 99):
        return reject(f"cluster {'/'.join(cl)} zit vol (max {lim['cluster_max']})")
    if coin in bk["coins"]:
        return reject(f"er staat al een positie of wachtende instap op {coin}")
    r_usd_unit = stop_distance
    tp = None
    if p.get("tp_r"):
        tp = entry + p["tp_r"] * r_usd_unit * (1 if side == "long" else -1)
    exp = _now() + lim["ticket_expiry_min"] * 60
    t = {"id": p["id"], "network": "papier", "coin": coin, "side": side, "is_buy": side == "long", "size": size,
         "notional": round(notional, 2), "entry": entry, "entry_type": "limiet" if p["entry"] else "markt",
         "stop": stop, "stop_fill": stop_fill, "slip_stop": slip_stop, "tp": tp, "tp_r": p.get("tp_r"), "lev": lev,
         "margin": round(margin, 2), "risk_usd": round(size * stressed, 2), "risk_pct": round(size * stressed / equity * 100, 3),
         "r_unit": r_usd_unit, "stressed": stressed, "slippage_bps": lim["max_slippage_bps"], "limits_v": lim["v"],
         "pass_at": _now(), "expires": exp, "equity": round(equity, 2)}
    t["text"] = ticket_text(t)
    p["ticket"] = t
    p["risk"] = {"verdict": "PASS", "lines": lines, "t": _now(), "limits_v": lim["v"]}
    p["status"] = "awaiting"
    floor("risk", f"PASS · {size:g} {coin} (${notional:,.0f}) · {lev}x · risico ${t['risk_usd']:.2f} = {t['risk_pct']:.2f}% · vervalt {_hm(exp)}",
          p["id"], to="Desk Lead", nxt="jij: typ \"approve " + p["id"] + "\"")


def ticket_text(t: dict) -> str:
    def f(x):
        return "geen" if x is None else f"{x:,.6g}"
    return "\n".join([
        f"TICKET {t['id']} | {t['network']}",
        f"markt: {t['coin']}-PERP      kant: {'buy' if t['is_buy'] else 'sell'} ({t['side']})      grootte: {t['size']:g} {t['coin']} (~${t['notional']:,.2f})",
        f"instap: {t['entry_type']} {f(t['entry'])}        reduce-only: nee",
        f"stop: {'sell' if t['is_buy'] else 'buy'} {t['size']:g} trigger {f(t['stop'])} markt (aangenomen vulling {f(t['stop_fill'])})",
        f"take-profit: {('%s (%gR)' % (f(t['tp']), t['tp_r'])) if t['tp'] else 'geen'}      hefboom: {t['lev']}x cross",
        f"slippage-tolerantie: {t['slippage_bps']} bp van de ticketprijs bij verzenden",
        f"risico: ${t['risk_usd']:.2f} = {t['risk_pct']:.2f}% van waarde ${t['equity']:,.2f}, R = {f(t['r_unit'])} per {t['coin']}",
        f"risico-akkoord: PASS {_hm(t['pass_at'])}, limieten v{t['limits_v']}",
        f"vervalt: {_hm(t['expires'])}",
        f"keur goed met: \"approve {t['id']}\""])


# --------------------------------------------------------------------------- jouw goedkeuring
def approve(text: str) -> dict:
    """Alleen de letterlijke zin met het ticketnummer telt. 'ja' of 'ok' niet."""
    words = (text or "").strip().split()
    if len(words) != 2 or words[0].lower() != "approve":
        raise ValueError('typ precies: approve HG-JJJJMMDD-NN')
    tid = words[1].upper()
    with _lock:
        base = tid[:-2] if tid.endswith("-B") else tid
        p = find(base)
        if p is None:
            raise KeyError(f"geen voorstel {tid}")
        if tid.endswith("-B"):
            return _approve_close(p, tid)
        t = p.get("ticket")
        if p["status"] != "awaiting" or not t:
            raise ValueError(f"{tid} wacht niet op goedkeuring (status: {p['status']})")
        if _now() > t["expires"]:
            p["status"] = "void"
            floor("lead", f"{tid} is vervallen om {_hm(t['expires'])}. Vraag een nieuw ticket.", tid, nxt="jij")
            _save()
            raise ValueError(f"{tid} is vervallen; vernieuw het ticket")
        p["approval"] = {"t": _now(), "line": f"approve {tid}"}
        p["status"] = "approved"
        floor("lead", f"goedgekeurd door jou om {_hm(_now())}", tid, to="Execution Trader", nxt="Execution Trader")
        _execute(p)
        _save()
        return p


def _mid(coin: str) -> float:
    dex = coin.split(":", 1)[0] if ":" in coin else ""
    mids = _info({"type": "allMids", **({"dex": dex} if dex else {})})
    return float(mids[coin])


def _execute(p: dict) -> None:
    """Execution Trader, op papier. Markt: vult nu aan de mid ± halve spread als de mid binnen de tolerantie ligt.
    Limiet: rust tot de prijs hem raakt (maker)."""
    t = p["ticket"]
    try:
        mid = _mid(p["coin"])
    except Exception as e:
        p["status"] = "void"
        floor("exec", f"geen prijs ({e}); niet verzonden", p["id"], nxt="Desk Lead")
        return
    p["execution"].append({"t": _now(), "what": "verzonden", "mid": mid})
    if t["entry_type"] == "markt":
        moved = abs(mid / t["entry"] - 1) * 1e4
        if moved > t["slippage_bps"]:
            p["status"] = "void"
            floor("exec", f"mid {mid:g} bewoog {moved:.1f} bp sinds het ticket (max {t['slippage_bps']}). Niet verzonden.", p["id"], nxt="Desk Lead")
            return
        half = (p["evidence"].get("spread_px") or 0) / 2
        _fill(p, mid + half if t["is_buy"] else mid - half, TAKER_BPS, "markt")
    else:
        p["status"] = "resting"
        floor("exec", f"limiet {t['entry']:g} rust op het boek (papier). Vult als de prijs hem raakt.", p["id"], nxt="Execution Trader")


def _fill(p: dict, px: float, fee_bps: float, how: str) -> None:
    t = p["ticket"]
    fee = px * t["size"] * fee_bps / 1e4
    state()["cash"] -= fee
    p["position"] = {"entry": px, "size": t["size"], "lev": t["lev"], "stop": t["stop"], "tp": t["tp"], "t": _now(),
                     "fees": fee, "px": px, "mae": 0.0, "mfe": 0.0}
    p["status"] = "live"
    p["execution"].append({"t": _now(), "what": f"gevuld ({how})", "px": px, "size": t["size"], "fee": round(fee, 4)})
    floor("exec", f"gevuld {t['size']:g} {p['coin']} @ {px:g} ({how}, fee ${fee:.3f}) · stop {t['stop']:g}"
                  + (f" · tp {t['tp']:g}" if t["tp"] else ""), p["id"], to="Trade Reviewer", nxt="Execution Trader (bewaakt)")


# --------------------------------------------------------------------------- uitstappen zijn ook tickets
def request_close(pid: str) -> dict:
    with _lock:
        p = find(pid)
        if p is None or p["status"] != "live":
            raise ValueError("geen open positie om te sluiten")
        x = p["position"]
        tid = p["id"] + "-B"
        exp = _now() + limits()["ticket_expiry_min"] * 60
        p["adjust"] = {"id": tid, "expires": exp, "text": "\n".join([
            f"TICKET {tid} | papier | aanpassing van {p['id']}: {p['idea']}",
            f"sluit {x['size']:g} {p['coin']} reduce-only, markt, slippage-grens {limits()['max_slippage_bps']} bp",
            f"vervalt: {_hm(exp)}", f"keur goed met: \"approve {tid}\""])}
        floor("lead", f"sluitticket {tid} klaar, wacht op jou", p["id"], nxt=f"jij: typ \"approve {tid}\"")
        _save()
        return p


def _approve_close(p: dict, tid: str) -> dict:
    a = p.get("adjust")
    if not a or a["id"] != tid or p["status"] != "live":
        raise ValueError(f"{tid} wacht niet op goedkeuring")
    if _now() > a["expires"]:
        p["adjust"] = None
        _save()
        raise ValueError(f"{tid} is vervallen; vraag opnieuw")
    mid = _mid(p["coin"])
    half = (p["evidence"].get("spread_px") or 0) / 2
    _close(p, mid - half if p["side"] == "long" else mid + half, "gesloten op jouw ticket " + tid, TAKER_BPS)
    p["adjust"] = None
    _save()
    return p


def void(pid: str) -> dict:
    with _lock:
        p = find(pid)
        if p is None or p["status"] not in ("awaiting", "resting", "approved", "rejected", "evidence", "risk"):
            raise ValueError("dit voorstel kan niet meer geannuleerd worden")
        p["status"] = "void"
        floor("lead", "geannuleerd door jou", pid, nxt="none")
        _save()
        return p


def _close(p: dict, px: float, why: str, fee_bps: float) -> None:
    x = p["position"]
    sgn = 1 if p["side"] == "long" else -1
    gross = (px - x["entry"]) * x["size"] * sgn
    fee = px * x["size"] * fee_bps / 1e4
    st = state()
    st["cash"] += gross - fee
    pnl = gross - fee - x["fees"]
    risk = p["ticket"]["risk_usd"] or 1e-9
    x.update(exit=px, exit_t=_now(), exit_fee=fee, pnl=pnl, r=pnl / risk, why=why)
    p["status"] = "closed"
    p["execution"].append({"t": _now(), "what": why, "px": px, "fee": round(fee, 4)})
    st["curve"].append([int(_now() * 1000), round(book()["equity"], 2)])
    st["curve"] = st["curve"][-2000:]
    floor("exec", f"{why} @ {px:g} · pnl ${pnl:+.2f} ({x['r']:+.2f}R)", p["id"], to="Trade Reviewer", nxt="Trade Reviewer")
    review(p)


# --------------------------------------------------------------------------- Trade Reviewer
review_llm = None  # main.py kan hier de lokale 35B koppelen: fn(prompt) -> tekst


def review(p: dict) -> None:
    x, t = p["position"], p["ticket"]
    plan_r = (t["tp_r"] or None)
    slip_bp = abs(x["entry"] / t["entry"] - 1) * 1e4 if t["entry"] else 0
    held = (x["exit_t"] - x["t"]) / 60
    facts = [f"uitkomst {x['pnl']:+.2f} USD = {x['r']:+.2f}R ({x['why']})",
             f"instap {x['entry']:g} tegen ticket {t['entry']:g} ({slip_bp:.1f} bp verschil)",
             f"grootste tegenbeweging {x['mae']:.2f}R, grootste meebeweging {x['mfe']:.2f}R",
             f"gehouden {held:.0f} min, fees ${x['fees'] + x['exit_fee']:.3f}"]
    lessons = []
    if x["r"] < 0 and x["mfe"] >= 1:
        lessons.append("stond minstens 1R in de plus en eindigde in verlies: overweeg een regel om de stop naar instap te zetten na 1R")
    if x["r"] <= -1.05:
        lessons.append("verlies groter dan 1R: de stop vulde slechter dan aangenomen; kijk naar de diepte van deze markt")
    if plan_r and x["mfe"] >= plan_r and "take-profit" not in x["why"]:
        lessons.append("het TP-doel werd intra-trade gehaald maar niet gevuld; controleer de TP-afstand")
    if not lessons:
        lessons.append("verliep volgens plan" if x["r"] > 0 else "verlies binnen het plan: de stop deed zijn werk")
    research_against = [i for i in (p.get("research") or {}).get("items", []) if i.get("agree") is False]
    if research_against:
        lessons.append("het tegenbewijs (" + "; ".join(i["src"] for i in research_against) + ") stond er al bij de instap")
    p["review"] = {"t": _now(), "facts": facts, "lessons": lessons, "ai": None}
    floor("review", f"review: {facts[0]} · {lessons[0]}", p["id"], nxt="none")
    if review_llm is not None:
        threading.Thread(target=_ai_review, args=(p["id"],), daemon=True, name="td-review").start()


def _ai_review(pid: str) -> None:
    p = find(pid)
    if p is None or review_llm is None:
        return
    prompt = ("Je bent de Trade Reviewer van een papieren handelsdesk. Schrijf in het Nederlands, maximaal 5 korte zinnen, "
              "zonder opsmuk: wat ging goed, wat ging fout, en één concrete les. Geen advies om de risicolimieten te verhogen.\n\n"
              f"Idee: {p['idea']}\nTicket:\n{p['ticket']['text']}\nFeiten: {'; '.join(p['review']['facts'])}\n"
              f"Bewijs bij instap: {json.dumps({k: p['evidence'].get(k) for k in ('mid', 'funding', 'spread_bps', 'atr1h', 'hi24', 'lo24')})}\n"
              f"Research: {'; '.join(i['text'] for i in p['research']['items']) or 'geen'}")
    try:
        txt = review_llm(prompt)
    except Exception as e:
        txt = None
        floor("review", f"35B-review niet beschikbaar ({e.__class__.__name__}); de regelreview blijft staan", pid)
    if txt:
        with _lock:
            p["review"]["ai"] = txt.strip()
            _save()


# --------------------------------------------------------------------------- Execution Trader bewaakt (elke paar seconden)
def tick(prices: dict | None = None) -> None:
    st = state()
    act = [p for p in st["proposals"] if p["status"] in ("resting", "live", "awaiting")]
    if not act:
        return
    if prices is None:
        prices = {}
        for dex in {(p["coin"].split(":", 1)[0] if ":" in p["coin"] else "") for p in act}:
            try:
                mids = _info({"type": "allMids", **({"dex": dex} if dex else {})})
                prices.update({k: float(v) for k, v in mids.items()})
            except Exception:
                pass
    with _lock:
        changed = False
        for p in act:
            px = prices.get(p["coin"])
            if p["status"] == "awaiting":
                if _now() > p["ticket"]["expires"]:
                    p["status"] = "void"
                    floor("lead", f"ticket vervallen om {_hm(p['ticket']['expires'])} zonder goedkeuring", p["id"], nxt="none")
                    changed = True
                continue
            if px is None:
                continue
            t = p["ticket"]
            if p["status"] == "resting":
                if (t["is_buy"] and px <= t["entry"]) or (not t["is_buy"] and px >= t["entry"]):
                    _fill(p, t["entry"], MAKER_BPS, "limiet geraakt, maker")
                    changed = True
                elif (t["is_buy"] and px <= t["stop"]) or (not t["is_buy"] and px >= t["stop"]):
                    p["status"] = "void"
                    floor("exec", "prijs ging voorbij de stop voor de limiet vulde; instap geannuleerd", p["id"], nxt="none")
                    changed = True
                continue
            x = p["position"]
            x["px"] = px
            sgn = 1 if p["side"] == "long" else -1
            r_now = (px - x["entry"]) * sgn / (t["r_unit"] or 1e-9)
            x["mae"], x["mfe"] = min(x["mae"], r_now), max(x["mfe"], r_now)
            if (sgn > 0 and px <= x["stop"]) or (sgn < 0 and px >= x["stop"]):
                _close(p, t["stop_fill"] if abs(px - x["stop"]) < t["slip_stop"] else px, "stop-loss", TAKER_BPS)
                changed = True
            elif x["tp"] and ((sgn > 0 and px >= x["tp"]) or (sgn < 0 and px <= x["tp"])):
                _close(p, x["tp"], "take-profit", MAKER_BPS)
                changed = True
        if changed:
            _save()


_thread: threading.Thread | None = None


def start() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return

    def loop():
        while True:
            try:
                tick()
            except Exception:
                pass
            time.sleep(TICK_S)
    _thread = threading.Thread(target=loop, daemon=True, name="trading-desk")
    _thread.start()


# --------------------------------------------------------------------------- limieten (alleen jij past ze aan)
def set_limits(new: dict, why: str = "") -> dict:
    cur = dict(limits())
    for k in DEFAULT_LIMITS:
        if k in new and new[k] is not None:
            cur[k] = new[k]
    for k, cap in CEILINGS.items():
        if float(cur[k]) > cap:
            raise ValueError(f"{k} = {cur[k]} gaat boven het plafond van de desk ({cap}). Alleen strenger mag.")
        if float(cur[k]) <= 0:
            raise ValueError(f"{k} moet groter dan 0 zijn")
    if int(cur["max_positions"]) < 1 or int(cur["ticket_expiry_min"]) < 1:
        raise ValueError("max posities en vervaltijd moeten minstens 1 zijn")
    cur["allowed_markets"] = [str(c).strip() for c in (cur.get("allowed_markets") or []) if str(c).strip()]
    cur["cluster"] = [str(c).strip() for c in (cur.get("cluster") or []) if str(c).strip()]
    with _lock:
        cur.update(v=limits()["v"] + 1, at=_now(), by="jij", why=why or "aangepast")
        state()["limits"].append(cur)
        floor("risk", f"limieten v{cur['v']} vastgelegd ({cur['why']})", nxt="none")
        _save()
    return cur


def reset_book() -> None:
    with _lock:
        if any(p["status"] in ("live", "resting", "approved") for p in state()["proposals"]):
            raise ValueError("sluit eerst de open posities en wachtende instappen")
        lim = state()["limits"]
        st = _blank()
        st["limits"] = lim
        globals()["_state"] = st
        floor("lead", "papierboek opnieuw gestart op $1000", nxt="none")
        _save()


# --------------------------------------------------------------------------- Strategist: voorstellen uit de Swarm
strategist_hook = None  # main.py: fn() -> lijst kandidaten {coin, side, p, edge, score}


def strategist(n: int = 1) -> list[dict]:
    """Neemt de sterkste Swarm-kandidaten en maakt er voorstellen van: stop op 1,5× ATR(1u), TP op 2R.
    Het blijft een voorstel: de Risk Manager rekent, jij keurt goed."""
    if strategist_hook is None:
        raise ValueError("Strategist is niet gekoppeld aan de Swarm")
    busy = set(book()["coins"]) | {p["coin"] for p in state()["proposals"] if p["status"] == "awaiting"}
    cands = [c for c in (strategist_hook() or []) if c["coin"] not in busy]
    out = []
    for c in cands[:n]:
        try:
            ev = market_evidence(c["coin"])
        except Exception:
            continue
        atr = ev.get("atr1h")
        if not atr:
            continue
        stop = ev["mid"] - 1.5 * atr if c["side"] == "long" else ev["mid"] + 1.5 * atr
        idea = (f"Strategist: Swarm ziet {c['side']} {c['coin']} (score {c.get('score', 0):+.2f}, kans {c.get('p', 0) * 100:.0f}%, "
                f"edge {c.get('edge', 0):+.1f}¢). Stop 1,5× ATR(1u) = {1.5 * atr:g}, doel 2R.")
        out.append(propose(c["coin"], c["side"], stop, None, 2.0, idea, by="Strategist"))
    if not out:
        floor("strategist", "geen Swarm-kandidaat met positieve edge die nog vrij is", nxt="none")
    return out


# --------------------------------------------------------------------------- weergave
def snapshot() -> dict:
    with _lock:
        _roll_day()
        st, bk = state(), book()
        props = sorted(st["proposals"], key=lambda p: -p["opened"])[:80]
        closed = [p for p in st["proposals"] if p["status"] == "closed"]
        return {"now": _now(), "level": "papier (oefenniveau, geen sleutel, geen echte orders)", "roles": ROLES,
                "ceilings": CEILINGS, "limits": limits(), "limits_history": st["limits"][-20:][::-1],
                "book": {**bk, "start": st["start"], "day_start_equity": st["day_start_equity"],
                         "day_pnl": bk["equity"] - st["day_start_equity"], "curve": st["curve"][-500:]},
                "stats": stats(closed), "proposals": props, "floor": st["floor"][-150:][::-1],
                "ai_review": review_llm is not None, "strategist": strategist_hook is not None}


def stats(closed: list[dict]) -> dict:
    pnls = [p["position"]["pnl"] for p in closed if p.get("position") and "pnl" in p["position"]]
    w = [x for x in pnls if x > 0]
    lo = [-x for x in pnls if x <= 0]
    aw = sum(w) / len(w) if w else 0.0
    al = sum(lo) / len(lo) if lo else 0.0
    n = len(pnls)
    rs = [p["position"]["r"] for p in closed if p.get("position") and "r" in p["position"]]
    return {"n": n, "wins": len(w), "losses": len(lo), "winrate": round(len(w) / n, 4) if n else None,
            "avg_win": round(aw, 2), "avg_loss": round(al, 2), "be_winrate": round(al / (aw + al), 4) if aw + al > 0 else None,
            "expectancy": round(sum(pnls) / n, 2) if n else None, "avg_r": round(sum(rs) / len(rs), 2) if rs else None}


def summary() -> dict:
    """Zelfde vorm als /api/desk/summary, voor de statusbalk."""
    from . import desk_journal
    with _lock:
        st, bk = state(), book()
        t0, t1, today = desk_journal.day_bounds()
        days: dict = {}
        for p in st["proposals"]:
            x = p.get("position") or {}
            if p["status"] != "closed" or "exit_t" not in x:
                continue
            day = datetime.fromtimestamp(x["exit_t"] + (24 - desk_journal.DAY_END_HOUR) * 3600).date().isoformat()
            r = days.setdefault(day, {"day": day, "wins": 0, "losses": 0, "pnl": 0.0})
            r["wins" if x["pnl"] > 0 else "losses"] += 1
            r["pnl"] = round(r["pnl"] + x["pnl"], 4)
        reviews = [p["review"]["t"] for p in st["proposals"] if p.get("review")]
        return {"mode": "paper", "start": st["start"], "equity": bk["equity"], "pnl": bk["equity"] - st["start"],
                "open": bk["positions"], "pending": bk["pending"] + sum(1 for p in st["proposals"] if p["status"] == "awaiting"),
                "upnl": bk["upnl"], "today": days.get(today, {"day": today, "wins": 0, "losses": 0, "pnl": 0.0}),
                "days": sorted(days.values(), key=lambda r: r["day"]), "analysis_at": int(max(reviews) * 1000) if reviews else None,
                "analysis_running": False, "on": True, "all": stats([p for p in st["proposals"] if p["status"] == "closed"])}
