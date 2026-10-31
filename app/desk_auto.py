"""De zwerm op de server: zes agents die elke gevolgde Hyperliquid-markt per venster beoordelen.

Hij draait altijd. Zolang live niet aan staat handelt hij op papier, uit één gedeelde papierportefeuille; zo leert
Prior 24/7, ook als de pagina dicht is. Live vraagt data/desk.env met DESK_AUTO=1 en de knop "Zwerm live" op /desk.

- Venster: DESK_WINDOW_MIN minuten (15, 60 of 240). Beslissen tussen 15% en 67% van het venster.
- Instappen met een maker-order (post-only limiet op de beste bied/laat). Niet gevuld binnen 10% van het venster:
  annuleren, geen trade. Uitstappen: maker-order vanaf 10% voor het einde, bij het einde zo nodig een marktorder.
- Eén portefeuille voor alle markten: de marge per trade komt uit de vrije marge, hoogstens DESK_MAX_OPEN posities
  tegelijk. Grenzen voor live: positie nooit boven DESK_MAX_USD, daglimiet DESK_DAILY_LOSS, na 3 mislukte orders uit.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone

from . import desk, desk_learn
from .data import DATA_DIR

APPROVE_SECONDS = 30
STATE_FILE = DATA_DIR / "desk-auto.json"
WS_URL = "wss://api.hyperliquid.xyz/ws"
WINDOWS = (15, 60, 240)
# Prijsmodel, instelbaar op /desk → Instellingen: gewichten van de Spotter-score en hoe snel Prior zijn β uit de data haalt.
BETA_MAX = 1.0       # het prijsmodel mag nooit zekerder worden dan Φ(β · score) met β ≤ 1
TRUST_TRADES = 200   # pas na zoveel afgesloten trades telt het volle vertrouwen mee in de hefboom
MAX_EXPOSURE = 2.0   # alle open posities samen hooguit 2× de portefeuille
MODEL_DEFAULT = {"beta0": 0.25, "k": 30, "w": {"flow30": .35, "flow10": .2, "book": .25, "bookDeep": .1, "mom": .1}}
MODEL: dict = json.loads(json.dumps(MODEL_DEFAULT))


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


EXIT_MAKER = {"share": .5}  # gemeten aandeel maker-uitstappen (laatste 50 trades), bijgewerkt bij elke afrekening


def costs(cfg: dict) -> dict:
    """Kosten per ronde in bp: maker bij instap, uitstap gewogen naar hoe vaak die echt als maker lukt."""
    maker, taker = desk.num(cfg, "DESK_MAKER_BPS", 1.5), desk.num(cfg, "DESK_FEE_BPS", 4.5)
    sh = EXIT_MAKER["share"]
    return {"maker": maker, "taker": taker, "round": maker + sh * maker + (1 - sh) * taker, "maker_share": sh}


def sr_levels(cs: list[dict], k: int = 6) -> list[dict]:
    """Steun en weerstand uit 5m-candles: draaipunten (hoogste/laagste van k candles links en rechts),
    gebundeld als ze dicht bij elkaar liggen. Meer raakpunten = sterker niveau."""
    if len(cs) < 2 * k + 5:
        return []
    rng = sorted(c["h"] - c["l"] for c in cs)
    tol = max(rng[len(rng) // 2] * .8, cs[-1]["c"] * 2e-4)
    piv = []
    for i in range(k, len(cs) - k):
        win = cs[i - k:i + k + 1]
        if cs[i]["h"] >= max(c["h"] for c in win):
            piv.append((cs[i]["h"], cs[i]["t"]))
        if cs[i]["l"] <= min(c["l"] for c in win):
            piv.append((cs[i]["l"], cs[i]["t"]))
    piv.sort()
    out: list[dict] = []
    for px, t in piv:
        if out and px - out[-1]["lo"] <= 1.5 * tol:  # niet laten doorketenen: een niveau is hooguit 1,5 tol breed
            o = out[-1]
            o["sum"] += px; o["n"] += 1; o["hi"] = px; o["last"] = max(o["last"], t)
        else:
            out.append({"sum": px, "n": 1, "hi": px, "lo": px, "last": t})
    return [{"px": o["sum"] / o["n"], "n": o["n"], "last": o["last"]} for o in out]


BT_LIST = "Zwerm (Hyperliquid)"


def bt_symbol(coin: str) -> str:
    """Symbool voor de backtestserver: aandelen (HIP-3, 'xyz:INTC') via Yahoo met lange historiek, crypto via Hyperliquid."""
    return coin.split(":", 1)[1].upper() if ":" in coin else f"HL:{coin.upper()}"


def bt_bias(coin: str) -> dict | None:
    """Wat zeggen de dag-backtests nu over deze ticker? Aandeel van de winstgevende strategieën met een open koopsignaal."""
    try:
        from . import watchlist as wl
        r = wl.scan_cached().get("results", {}).get(bt_symbol(coin))
    except Exception:
        return None
    if not r or "error" in r and not r.get("periods"):
        return None
    prof, openp = r.get("profitable_count") or 0, r.get("open_profitable") or 0
    best = r.get("best") or {}
    if prof < 3:
        trend = "onbekend"
    else:
        ratio = openp / prof
        trend = "stijgend" if ratio >= .6 else "zwak" if ratio <= .2 else "gemengd"
    return {"trend": trend, "prof": prof, "open": openp, "best": best.get("strategy"), "best_open": best.get("signal_open"),
            "best_ret": best.get("total_return"), "bh": r.get("buy_and_hold_return"), "at": r.get("scanned_at"),
            "period": r.get("period")}


LEARN = {"on": True}  # geleerde gewichten gebruiken als ze beter zijn dan de vaste (aan/uit vanaf /desk)


def trend_lines(cs: list[dict], k: int = 6) -> list[dict]:
    """Trendlijnen op 5m: door de laatste twee bodems als die stijgen (steunlijn) en door de laatste twee toppen
    als die dalen (weerstandslijn). Waarde op elk moment = lineair doorgetrokken."""
    if len(cs) < 2 * k + 5:
        return []
    hi, lo = [], []
    for i in range(k, len(cs) - k):
        win = cs[i - k:i + k + 1]
        if cs[i]["h"] >= max(c["h"] for c in win):
            hi.append((cs[i]["t"], cs[i]["h"]))
        if cs[i]["l"] <= min(c["l"] for c in win):
            lo.append((cs[i]["t"], cs[i]["l"]))
    out = []
    if len(lo) >= 2 and lo[-1][1] > lo[-2][1]:
        out.append({"kind": "sup", "t1": lo[-2][0], "p1": lo[-2][1], "t2": lo[-1][0], "p2": lo[-1][1]})
    if len(hi) >= 2 and hi[-1][1] < hi[-2][1]:
        out.append({"kind": "res", "t1": hi[-2][0], "p1": hi[-2][1], "t2": hi[-1][0], "p2": hi[-1][1]})
    return out


def trend_at(tl: dict, t_ms: float) -> float:
    return tl["p1"] + (tl["p2"] - tl["p1"]) * (t_ms - tl["t1"]) / max(1, tl["t2"] - tl["t1"])


class Market:
    """Koersdata, venster en ticket van één markt."""

    def __init__(self, coin: str) -> None:
        self.coin = coin
        self.px = 0.0
        self.last_px = 0.0
        self.var1 = 0.0
        self.trades: deque = deque()
        self.book = None
        self.win = 0
        self.open0 = 0.0
        self.decided = False
        self.sample = None
        self.feat: dict = {}
        self.model: dict = {}
        self.ticket: dict | None = None
        self.last_eval = 0
        self.levels: list[dict] = []
        self.trends: list[dict] = []
        self.levels_at = 0.0

    def prior_file(self, window: int):
        return DATA_DIR / f"desk-prior-{self.coin.replace(':', '_')}-{window // 60}m.json"

    def samples(self, window: int) -> list:
        f = self.prior_file(window)
        return json.loads(f.read_text()) if f.exists() else []

    def warm_up(self) -> None:
        """σ per seconde uit de 5m-candles van de afgelopen 2 dagen (ook nachten en weekends mee)."""
        for attempt in range(4):  # Hyperliquid geeft 429 als veel markten tegelijk opwarmen
            try:
                cs = desk.candles(self.coin, "5m", 2 * 24 * 60)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * 2 ** attempt)
        rs = [math.log(cs[i]["c"] / cs[i - 1]["c"]) for i in range(1, len(cs)) if cs[i - 1]["c"] > 0]
        self.var1 = sum(r * r for r in rs) / max(1, len(rs)) / 300
        self.levels, self.trends, self.levels_at = sr_levels(cs), trend_lines(cs), time.time()
        if not self.px and cs:
            self.px = cs[-1]["c"]

    def near_levels(self, px: float) -> tuple[dict | None, dict | None]:
        """Dichtstbijzijnde steun onder en weerstand boven deze prijs."""
        below = [l for l in self.levels if l["px"] < px]
        above = [l for l in self.levels if l["px"] > px]
        return (max(below, key=lambda l: l["px"]) if below else None,
                min(above, key=lambda l: l["px"]) if above else None)

    def best(self) -> tuple[float, float]:
        if self.book and self.book[0] and self.book[1]:
            return self.book[0][0][0], self.book[1][0][0]
        return self.px, self.px

    def features(self, now: float) -> dict:
        cut = now - 300
        while self.trades and self.trades[0][0] < cut:
            self.trades.popleft()

        def flow(sec: float) -> float:
            b = sum(sz for t, sz, buy, _ in self.trades if t > now - sec and buy)
            s = sum(sz for t, sz, buy, _ in self.trades if t > now - sec and not buy)
            return (b - s) / (b + s) if b + s else 0.0

        f = {"flow10": flow(30), "flow30": flow(180), "book": 0.0, "bookDeep": 0.0}
        if self.book:
            bids, asks = self.book

            def imb(n: int) -> float:
                b, a = sum(x[1] for x in bids[:n]), sum(x[1] for x in asks[:n])
                return (b - a) / (b + a) if b + a else 0.0

            f["book"], f["bookDeep"] = imb(5), imb(20)
        el = max(1.0, now - self.win)
        f["mom"] = math.log(self.px / self.open0) / math.sqrt(self.var1 * el) if self.open0 and self.var1 else 0.0
        w = MODEL["w"]
        f["score"] = _clamp(w["flow30"] * f["flow30"] + w["flow10"] * f["flow10"] + w["book"] * f["book"]
                            + w["bookDeep"] * f["bookDeep"] + w["mom"] * _clamp(f["mom"] / 2, -1, 1), -1, 1)
        return f

    def prior(self, window: int) -> dict:
        a = self.samples(window)
        n, beta0, k = len(a), MODEL["beta0"], MODEL["k"]
        sy = sum(s["s"] * s["y"] for s in a)
        ss = sum(s["s"] ** 2 for s in a)
        hit = sum(1 for s in a if (s["s"] > 0) == (s["y"] > 0)) / n if n else None
        bhat = _clamp(sy / ss, -BETA_MAX, BETA_MAX) if ss > 1e-9 else 0.0
        # begrensd: bij een handvol vensters sprong β tot 6 en het model tot 98% zeker (DOGE, n=6), terwijl het
        # in de praktijk maar 52% juist zat. Met β ≤ 1 is de kans hooguit Φ(1) = 84% bij een volle score.
        return {"beta": _clamp((n * bhat + k * beta0) / (n + k), -BETA_MAX, BETA_MAX), "n": n, "hit": hit}

    def compute(self, f: dict, now: float, cfg: dict, window: int) -> dict:
        left = max(1.0, window - (now - self.win))
        sd = math.sqrt(max(self.var1, 1e-12) * left)
        pr = self.prior(window)
        p = _phi(pr["beta"] * f["score"])
        src = "vast"
        if LEARN["on"]:
            pl = desk_learn.predict(window, self.coin, f)
            if pl is not None:
                p, src = pl, "geleerd"
        e_move = sd * math.sqrt(2 / math.pi) * 1e4
        be = .5 + costs(cfg)["round"] / (2 * max(e_move, .01))
        p_dir = max(p, 1 - p)
        return {"p": p, "pDir": p_dir, "side": "long" if p >= .5 else "short", "be": be,
                "edge": (p_dir - be) * 100, "sd": sd, "left": left, "prior": pr, "eMove": e_move, "src": src}


class Swarm:
    def __init__(self) -> None:
        saved = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        cfg = desk.config()
        self.on: bool = saved.get("on", False)
        self.confirm: bool = saved.get("confirm", True)
        self.coins: dict[str, bool] = {c: c == "BTC" for c in desk.COINS}
        self.coins.update(saved.get("coins", {}))
        self.window_min: int = int(saved.get("window_min") or desk.num(cfg, "DESK_WINDOW_MIN", 60))
        self.min_edge: float | None = saved.get("min_edge")
        self.auto_scan: bool = bool(saved.get("auto_scan"))  # elke 5 min de top 25 van de scanner volgen
        self.scan_at = 0.0
        self.scan_n: int = int(saved.get("scan_n", 25))  # 25, 50 of 0 = alle liquide markten met ratio ≥ 1
        self.drop: set = set()  # uit de top 25 gevallen, maar nog een open trade: gaat uit zodra die afgerekend is
        self.max_lev: float | None = saved.get("max_lev")    # None = DESK_MAX_LEV; live nooit boven DESK_MAX_LEV
        self.risk_pct: float | None = saved.get("risk_pct")  # None = DESK_RISK_PCT
        self.kelly_frac: float | None = saved.get("kelly_frac")
        self.kelly_info: dict = {}
        self.sl_sigma: float | None = saved.get("sl_sigma")
        self.tp_sigma: float | None = saved.get("tp_sigma")
        self.hold_max: int | None = saved.get("hold_max")
        LEARN["on"] = saved.get("learn", True)  # hoeveel vensters een trade mag doorlopen (None = 3)
        self.max_margin_pct: float | None = saved.get("max_margin_pct")
        if saved.get("model"):
            MODEL.update(saved["model"])  # drempel in ¢ boven break-even; None = DESK_MIN_EDGE
        start = desk.num(cfg, "DESK_PAPER_START", 1000)
        self.paper: dict = saved.get("paper") or {"start": start, "cash": start, "curve": [[int(time.time() * 1000), start]]}
        self.resubscribe = False
        self.day: str = saved.get("day", "")
        self.day_start: float | None = saved.get("day_start")
        self.halted: str | None = saved.get("halted") if saved.get("day") == _today() else None
        self.lock = threading.RLock()
        self.trace: deque = deque(maxlen=3000)
        self.seq = 0
        self.history: deque = deque(saved.get("history", []), maxlen=2000)
        self.update_maker_share()
        self.lev: dict[str, int] = {}
        self.markets = {c: Market(c) for c in self.coins}
        for t in saved.get("tickets", []):  # open papiertrades overleven een herstart (live: close_orphans)
            m = self.markets.get(t["coin"])
            if m and t.get("paper"):
                m.ticket, m.win, m.decided = t, t["win"], True
                m.open0 = t.get("entry") or t.get("limit")
        self.pending: dict | None = None
        self.errors = 0
        self.feed_ok = False
        self.last_account = 0.0
        self.account: dict = {}
        self.started = False

    @property
    def window(self) -> int:
        return self.window_min * 60

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.paper["curve"] = self.paper["curve"][-3000:]
        STATE_FILE.write_text(json.dumps({"on": self.on, "confirm": self.confirm, "coins": self.coins, "day": self.day,
                                          "day_start": self.day_start, "halted": self.halted, "window_min": self.window_min,
                                          "min_edge": self.min_edge, "auto_scan": self.auto_scan, "scan_n": self.scan_n, "max_lev": self.max_lev, "risk_pct": self.risk_pct, "kelly_frac": self.kelly_frac, "sl_sigma": self.sl_sigma, "tp_sigma": self.tp_sigma, "hold_max": self.hold_max, "learn": LEARN["on"],
                                          "max_margin_pct": self.max_margin_pct, "model": MODEL, "paper": self.paper, "history": list(self.history),
                                          "tickets": [m.ticket for m in self.markets.values() if m.ticket and m.ticket["paper"]]}))

    def log(self, agent: str, act: str, text: str, val: str = "", coin: str = "") -> None:
        self.seq += 1
        self.trace.append({"id": self.seq, "t": int(time.time() * 1000), "agent": agent, "act": act,
                           "text": f"{coin} · {text}" if coin else text, "val": val, "coin": coin})

    # ---------- marktdata: één websocket voor alle markten ----------
    def feed(self) -> None:
        from websockets.sync.client import connect

        while True:
            try:
                with connect(WS_URL, open_timeout=15, close_timeout=5) as ws:
                    self.resubscribe = False
                    for coin in list(self.markets):
                        for sub in ({"type": "trades", "coin": coin}, {"type": "l2Book", "coin": coin}):
                            ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
                    last_ping = time.time()
                    while not self.resubscribe:  # nieuwe markt erbij: opnieuw verbinden met de nieuwe lijst
                        if time.time() - last_ping > 30:
                            ws.send('{"method":"ping"}')
                            last_ping = time.time()
                        try:
                            msg = json.loads(ws.recv(timeout=5))
                        except TimeoutError:
                            continue
                        ch, data = msg.get("channel"), msg.get("data")
                        if ch == "trades":
                            self.feed_ok = True
                            with self.lock:
                                for t in data:
                                    m = self.markets.get(t["coin"])
                                    if m:
                                        m.px = float(t["px"])
                                        m.trades.append((t["time"] / 1000, float(t["sz"]), t["side"] == "B", m.px))
                        elif ch == "l2Book" and data.get("coin") in self.markets:
                            self.markets[data["coin"]].book = [[(float(l["px"]), float(l["sz"])) for l in side]
                                                               for side in data["levels"]]
            except Exception as e:  # opnieuw verbinden
                self.feed_ok = False
                self.log("spotter", "FEED", f"verbinding met Hyperliquid weg: {e}")
                time.sleep(5)

    # ---------- portefeuille ----------
    def is_live(self) -> bool:
        return self.on and desk.status()["auto_allowed"]

    def open_tickets(self) -> list[dict]:
        return [m.ticket for m in self.markets.values() if m.ticket]

    def portfolio(self) -> dict:
        """Eén portefeuille voor alle markten: papier, of het Hyperliquid-account als live aan staat."""
        tickets = []
        for m in self.markets.values():
            t = m.ticket
            if not t:
                continue
            upnl = 0.0
            if t["state"] in ("open", "exiting") and t.get("entry"):
                upnl = t["usd"] * (1 if t["side"] == "long" else -1) * (m.px - t["entry"]) / t["entry"]
            tickets.append({"coin": t["coin"], "side": t["side"], "usd": t["usd"], "lev": t["lev"], "margin": t["usd"] / t["lev"],
                            "state": t["state"], "entry": t.get("entry"), "px": m.px, "upnl": round(upnl, 4), "paper": t["paper"],
                            "sl": t.get("sl"), "tp": t.get("tp"), "why": t.get("why"), "plan": t.get("plan"),
                            "holds": t.get("holds", 0), "holds_why": t.get("holds_why", [])[-6:]})
        used = sum(t["margin"] for t in tickets)
        upnl = sum(t["upnl"] for t in tickets)
        live = self.is_live()
        if live and self.account.get("connected"):
            equity, free = self.account["value"], self.account["withdrawable"]
        else:
            equity, free = self.paper["cash"] + upnl, self.paper["cash"] - used
        hist = [h for h in self.history if bool(h.get("paper")) != live]
        wins = sum(1 for h in hist if h["pnl"] > 0)
        return {"mode": "live" if live else "paper", "start": self.paper["start"], "equity": equity, "used": used,
                "free": free, "upnl": upnl, "open": tickets, "curve": self.paper["curve"][-500:],
                "n": len(hist), "wins": wins, "pnl": sum(h["pnl"] for h in hist)}

    # ---------- de zwerm, per markt ----------
    def new_window(self, m: Market, now: float, cfg: dict) -> None:
        win = int(now // self.window) * self.window
        if win == m.win:
            return
        if m.ticket and m.ticket.get("hold_win") != win:
            self.settle(m, cfg)
        if m.sample and m.px:
            y = math.log(m.px / m.sample["px"]) / max(1e-9, m.sample["sd"])
            a = m.samples(self.window)
            a.append({"s": round(m.sample["s"], 4), "y": round(y, 4), "t": m.win})
            m.prior_file(self.window).write_text(json.dumps(a[-2000:]))
            if m.sample.get("f"):
                desk_learn.record({"t": m.win, "coin": m.coin, "window": self.window, "f": [round(v, 4) for v in m.sample["f"]],
                                   "s": round(m.sample["s"], 4), "p": round(m.sample["p"], 4), "y": round(y, 4), "src": m.sample.get("src")})
            if self.coins[m.coin]:
                self.log("prior", "LEER", f"score {m.sample['s']:+.2f} → uitkomst {y:+.2f}σ", f"n={len(a)}", m.coin)
        m.win, m.open0, m.decided, m.sample = win, m.px, False, None
        if self.pending and self.pending["coin"] == m.coin:
            self.pending = None
            self.log("taker", "NEE", "goedkeuring verlopen: venster voorbij", coin=m.coin)

    def decide(self, m: Market, now: float, cfg: dict) -> None:
        f, mo, el, W = m.feat, m.model, now - m.win, self.window
        if not m.sample and el >= .15 * W:
            m.sample = {"s": f["score"], "px": m.px, "sd": mo["sd"], "f": desk_learn.fvec(f), "p": mo["p"], "src": mo.get("src")}
        live = self.is_live()
        if not self.coins[m.coin] or (live and self.halted) or m.decided or m.ticket or (live and self.pending):
            return
        every = max(15, W // 40)
        if el < .15 * W or el > .67 * W or now - m.last_eval < every:
            return
        m.last_eval = now
        min_edge = self.edge_floor(cfg, live)
        self.log("spotter", "SCAN", f"flow {f['flow30']:+.2f} · boek {f['book']:+.2f} · score {f['score']:+.2f}", coin=m.coin)
        self.log("prior", "PRIJS", f"β {mo['prior']['beta']:.3f} · P(stijgt) {mo['p'] * 100:.1f}¢ · "
                                   f"verwachte beweging {mo['eMove']:.1f}bp · kosten {costs(cfg)['round']:.1f}bp", coin=m.coin)
        bt = bt_bias(m.coin)  # dag-backtests: tegen hun richting in vraagt de zwerm 1¢ extra edge
        against_bt = bool(bt) and ((mo["side"] == "long" and bt["trend"] == "zwak") or (mo["side"] == "short" and bt["trend"] == "stijgend"))
        if against_bt:
            min_edge += 1.0
        if mo["edge"] < min_edge:
            if against_bt and mo["edge"] >= min_edge - 1:
                self.log("edge", "PASS", f"{mo['side']} tegen de dag-backtests in ({bt['open']}/{bt['prof']} winstgevende strategieën "
                         f"met open koopsignaal): 1¢ extra edge nodig", f"{mo['edge']:+.1f}¢", m.coin)
                if el + every > .67 * W:
                    m.decided = True
                return
            self.log("edge", "PASS", f"{mo['side']} {mo['pDir'] * 100:.1f}¢ < break-even {mo['be'] * 100:.1f}¢ {min_edge:+g}¢",
                     f"{mo['edge']:+.1f}¢", m.coin)
            if el + every > .67 * W:
                m.decided = True
            return
        # ruimte tot het volgende niveau: long tegen een weerstand aan of short op een steun koopt weinig beweging
        sd = max(mo.get("sd", 0.0), 1e-5)
        sup, res = m.near_levels(m.px)
        wall = res if mo["side"] == "long" else sup
        room = abs(wall["px"] - m.px) / m.px / sd if wall else None
        if room is not None and room < .3:
            self.log("edge", "PASS", f"{mo['side']} maar {'weerstand' if mo['side'] == 'long' else 'steun'} op {wall['px']:,.6g} "
                     f"({wall['n']}x geraakt) ligt maar {room:.2f}σ verder, geen ruimte", f"{mo['edge']:+.1f}¢", m.coin)
            if el + every > .67 * W:
                m.decided = True
            return
        bid, ask = m.best()
        plan = self.exit_plan(m, mo["side"], bid if mo["side"] == "long" else ask, sd)
        if plan["rr"] < .7:
            self.log("edge", "PASS", f"{mo['side']}: winst tot de take-profit ({plan['tp_sigma']:.2f}σ) is te klein tegenover het "
                     f"risico tot de stop ({plan['sl_sigma']:.2f}σ), verhouding {plan['rr']:.2f}", f"{mo['edge']:+.1f}¢", m.coin)
            if el + every > .67 * W:
                m.decided = True
            return
        m.decided = True
        self.log("edge", "EDGE", f"{mo['side']} {mo['pDir'] * 100:.1f}¢ vs break-even {mo['be'] * 100:.1f}¢",
                 f"{mo['edge']:+.1f}¢", m.coin)
        self.log("closer", "PLAN", f"stop {plan['sl']:,.6g} ({plan['sl_why']}) · take-profit {plan['tp']:,.6g} ({plan['tp_why']}) · "
                 f"verhouding {plan['rr']:.2f}", coin=m.coin)
        sized = self.kelly(m, cfg, live, plan)
        if not sized:
            return
        lev, usd = sized
        w = MODEL["w"]  # waarom: welke kenmerken de score maakten, en wat Prior, Edge en Kelly ervan dachten
        parts = [("koopdruk 3 min", w["flow30"] * f["flow30"], f["flow30"]), ("koopdruk 30 s", w["flow10"] * f["flow10"], f["flow10"]),
                 ("orderboek top 5", w["book"] * f["book"], f["book"]), ("orderboek top 20", w["bookDeep"] * f["bookDeep"], f["bookDeep"]),
                 ("momentum", w["mom"] * _clamp(f["mom"] / 2, -1, 1), f["mom"])]
        parts.sort(key=lambda x: abs(x[1]), reverse=True)
        why = {"score": round(f["score"], 3), "parts": [[n_, round(c, 3), round(v, 3)] for n_, c, v in parts],
               "p": round(mo["pDir"], 4), "be": round(mo["be"], 4), "edge": round(mo["edge"], 2), "emove": round(mo["eMove"], 1),
               "beta": round(mo["prior"]["beta"], 3), "n": mo["prior"]["n"], "hit": mo["prior"]["hit"],
               "cost": costs(cfg)["round"], "src": mo.get("src"), **self.kelly_info,
               "room": round(room, 2) if room is not None else None, "wall": round(wall["px"], 8) if wall else None,
               "wall_n": wall["n"] if wall else None, "bt": bt, "bt_against": against_bt}
        bid, ask = m.best()
        limit = bid if mo["side"] == "long" else ask
        order = {"id": int(now * 1000), "coin": m.coin, "side": mo["side"], "usd": usd, "lev": lev, "edge": mo["edge"],
                 "p": mo["pDir"], "be": mo["be"], "win": m.win, "expires": now + APPROVE_SECONDS, "paper": not live,
                 "limit": limit, "why": why}
        if not live:
            self.enter(order, cfg)
        elif self.confirm:
            self.pending = order
            self.log("taker", "WACHT", f"{mo['side']} ${usd:g} ({lev}x) wacht op jouw goedkeuring", "mens", m.coin)
        else:
            self.enter(order, cfg)

    def edge_floor(self, cfg: dict, live: bool) -> float:
        """Drempel boven break-even. Op papier mag hij negatief (meer trades om van te leren); live nooit onder 0."""
        e = self.min_edge if self.min_edge is not None else desk.num(cfg, "DESK_MIN_EDGE", 2)
        return max(e, 0.0) if live else e

    def exit_limits(self) -> tuple[float, float]:
        """Stop-loss en take-profit in σ (van de rest van het venster bij instap); 0 = uit."""
        return (self.sl_sigma if self.sl_sigma is not None else 1.0, self.tp_sigma if self.tp_sigma is not None else 1.5)

    def size_limits(self) -> tuple[float, float]:
        """Kelly-fractie (0,25 = kwart-Kelly) en max marge per trade in % van de portefeuille."""
        return (self.kelly_frac if self.kelly_frac is not None else 0.25,
                self.max_margin_pct if self.max_margin_pct is not None else 5.0)

    def lev_limits(self, cfg: dict, live: bool) -> tuple[float, float]:
        """Max hefboom en risico per trade. Op papier vrij in te stellen; live nooit boven DESK_MAX_LEV uit desk.env."""
        env_cap = desk.num(cfg, "DESK_MAX_LEV", 3)
        cap = self.max_lev if self.max_lev is not None else max(env_cap, 10)  # papier: tot 10x, live altijd begrensd door desk.env
        risk = self.risk_pct if self.risk_pct is not None else desk.num(cfg, "DESK_RISK_PCT", 10)
        return (min(cap, env_cap) if live else cap), risk

    def trust(self, live: bool) -> tuple[float, str]:
        """Hoeveel de zwerm zichzelf mag geloven, uit wat hij al bewezen heeft. Geen vaste grens op de hefboom:
        weinig trades of verlies op de laatste 50 → voorzichtig, een lange winnende reeks → het volle vertrouwen."""
        hs = [h for h in self.history if h.get("paper", True) != live]
        n = len(hs)
        t = .3 + .7 * min(1.0, n / TRUST_TRADES)  # 0 trades: 30%, 200 trades: 100%
        why = f"{n}/{TRUST_TRADES} trades"
        if n >= 20:
            last = hs[-50:]
            pnl = sum(h["pnl"] for h in last)
            win = sum(1 for h in last if h["pnl"] > 0) / len(last)
            if pnl < 0:  # de laatste trades kosten geld: halveren
                t *= .5
                why += f", laatste {len(last)} {pnl:+.2f}$ (winrate {win * 100:.0f}%)"
        return _clamp(t, .1, 1.0), why

    def kelly(self, m: Market, cfg: dict, live: bool, plan: dict | None = None) -> tuple[int, float] | None:
        """Hefboom uit de volatiliteit: een tegenvaller van 2σ tot het einde van het venster mag hooguit
        risico % (10%) van de marge kosten, begrensd door de max hefboom en de max van de markt.
        Grootte uit Kelly: positie / portefeuille = fractie × μ / σ², met μ = (2p − 1)·E|beweging| − kosten.
        Marge = positie / hefboom, nooit meer dan max marge % van de portefeuille of een kwart van de vrije marge."""
        pf = self.portfolio()
        if len(pf["open"]) >= desk.num(cfg, "DESK_MAX_OPEN", 5):
            self.log("kelly", "VOL", f"al {len(pf['open'])} posities open, geen nieuwe", coin=m.coin)
            return None
        sd = max(m.model.get("sd", 0.0), 1e-5)
        cap, risk = self.lev_limits(cfg, live)
        market_max = desk.universe().get(m.coin, {}).get("maxLeverage")
        if market_max:
            cap = min(cap, market_max)
        mo, (kf, mpct) = m.model, self.size_limits()
        # hefboom per trade: de stop mag hooguit risico % van de marge kosten, en dat maal het vertrouwen
        # (edge 0¢ → 25% van die hefboom, edge ≥ 8¢ → 100%)
        sl_d = plan["sl_d"] if plan else 2 * sd
        conf = _clamp(mo["edge"] / 8, .25, 1.0)
        trust, trust_why = self.trust(live)
        # trefzekerheid van deze markt: hoe vaak de eigen voorspelling juist zat, boven 50%, gekrompen bij weinig vensters
        pr = mo.get("prior") or {}
        n_w, hit = pr.get("n", 0), pr.get("hit") or .5
        skill = .25 + .75 * max(0.0, (hit - .5) * 2) * n_w / (n_w + 75)
        want_risk = math.floor(risk / 100 / max(sl_d, 1e-6))
        want = max(1, math.floor(want_risk * conf * skill * trust))
        lev = int(_clamp(want, 1, max(1, cap)))
        mu = (2 * mo["pDir"] - 1) * sd * math.sqrt(2 / math.pi) - costs(cfg)["round"] / 1e4
        kelly_notional = max(0.0, kf * mu / (sd * sd)) * pf["equity"]
        cap_margin = min(pf["equity"] * mpct / 100, pf["free"] / 4)
        margin = min(kelly_notional / lev, cap_margin)
        if 0 < margin * lev < 10 <= cap_margin * lev:  # Hyperliquid wil minstens $10: kleinste order
            margin = 10 / lev
        usd = min(margin * lev, pf["equity"] * .5)  # één positie nooit groter dan de helft van de portefeuille
        room = pf["equity"] * MAX_EXPOSURE - sum(t.get("usd", 0) for t in pf["open"])
        if usd > room:  # samen nooit meer dan 2× de portefeuille
            if room < 10:
                self.log("kelly", "VOL", f"al ${pf['equity'] * MAX_EXPOSURE - room:,.0f} in posities, max {MAX_EXPOSURE:g}× de portefeuille", coin=m.coin)
                return None
            usd = room
        if live:
            usd = min(usd, desk.num(cfg, "DESK_MAX_USD", 25), desk.num(cfg, "DESK_ORDER_USD", 20) * lev)
        if usd < 10:  # Hyperliquid weigert orders onder $10
            self.log("kelly", "TE KLEIN", f"vrije marge ${pf['free']:,.2f} te klein voor een order", coin=m.coin)
            return None
        self.kelly_info = {"sd": round(sd * 1e4, 1), "lev": lev, "want_lev": want, "lev_cap": cap, "kelly": kf,
                           "lev_risk": want_risk, "lev_conf": round(conf, 2), "lev_trust": round(trust, 2), "lev_skill": round(skill, 2), "trust_why": trust_why, "sl_bp": round(sl_d * 1e4, 1), "risk_pct": risk,
                           "kelly_usd": round(kelly_notional, 2), "margin": round(usd / lev, 2), "usd": round(usd, 2),
                           "capped": "marge" if kelly_notional / lev > cap_margin else None, "max_margin_pct": mpct}
        self.log("kelly", "SIZE", f"stop op {sl_d * 1e4:.0f}bp, max {risk:g}% van de marge → {want_risk}x, × vertrouwen {conf:.2f} × trefzekerheid {skill:.2f} ({hit * 100:.0f}% juist op {n_w} vensters) × staat van dienst {trust:.2f} ({trust_why}) → hefboom {lev}x"
                                  + (f" (wil {want}x, max {cap:g}x)" if want > lev else "")
                                  + f" · Kelly ×{kf:g} wil ${kelly_notional:,.0f} positie"
                                  + (f", max marge {mpct:g}%" if kelly_notional / lev > cap_margin else "")
                                  + f" → marge ${usd / lev:,.2f} · positie ${usd:,.2f} (vrij ${pf['free']:,.0f})", coin=m.coin)
        return lev, round(usd, 2)

    # ---------- orders: instap en uitstap met maker-orders ----------
    def enter(self, order: dict, cfg: dict) -> None:
        m = self.markets[order["coin"]]
        t = {**order, "state": "entering", "opened": time.time(), "entry": None, "oid": None, "exit_oid": None,
             "maker_exit": False}
        if order["paper"]:
            m.ticket = t
            self.log("taker", "LIMIET", f"papier {order['side']} ${order['usd']:g} ({order['lev']}x) limiet {order['limit']:,.6g}",
                     coin=m.coin)
            self.save()
            return
        try:
            if self.lev.get(m.coin) != order["lev"]:
                lr = desk.set_leverage(cfg, m.coin, order["lev"])
                if not lr["ok"]:
                    raise RuntimeError("hefboom instellen mislukt: " + json.dumps(lr["result"])[:150])
                self.lev[m.coin] = order["lev"]
            size = desk.size_for(m.coin, order["usd"], order["limit"])
            r = desk.place_maker(cfg, m.coin, order["side"] == "long", size, order["limit"])
            if r["error"]:
                raise RuntimeError(r["error"])
            self.errors = 0
            t.update(oid=r["oid"], size=size, limit=r["px"])
            m.ticket = t
            if r["filled"]:
                self.filled_entry(m, r["px"], cfg)
            else:
                self.log("taker", "LIMIET", f"LIVE {order['side']} ${order['usd']:g} ({order['lev']}x) limiet {r['px']:,.6g}", coin=m.coin)
        except Exception as e:
            self.errors += 1
            self.log("taker", "FOUT", f"order mislukt ({self.errors}/3): {e}", coin=m.coin)
            if self.errors >= 3:
                self.on = False
                self.log("closer", "STOP", "3 mislukte orders op rij: autonome handel uit")
                self.save()

    def filled_entry(self, m: Market, px: float, cfg: dict | None = None) -> None:
        t = m.ticket
        t.update(state="open", entry=px, filled_at=time.time())
        # stop-loss en take-profit in σ van de rest van het venster, vanaf de instap
        self.plan_exits(m, t, px)
        if not t["paper"] and cfg is not None:
            self.place_tp(m, t, cfg)
        if t["paper"]:
            self.save()
        self.log("taker", "TAKE", f"{'papier' if t['paper'] else 'LIVE'} {t['side']} gevuld als maker @ {px:,.6g}",
                 f"{t['lev']}x ${t['usd']:g}", m.coin)

    @staticmethod
    def exit_plan(m: Market, side: str, px: float, sd: float) -> dict:
        """Stop-loss en take-profit per trade, uit de structuur van de markt:
        - stop net voorbij de dichtste steun (long) of weerstand (short) of trendlijn erachter, tussen 0,4σ en 2σ;
          niets in dat bereik: 1σ (de beweging die het venster normaal maakt);
        - take-profit net voor het dichtste niveau of de trendlijn ervoor, tussen 0,5σ en 3σ; niets: 1,5σ.
        σ = verwachte schommeling tot het einde van het venster."""
        long, sd = side == "long", max(sd, 1e-5)
        sign = 1 if long else -1
        now_ms = time.time() * 1000
        cand = [(l["px"], f"{'steun' if l['px'] < px else 'weerstand'} {l['px']:,.6g} ({l['n']}x geraakt)", l["n"]) for l in m.levels]
        cand += [(trend_at(tl, now_ms), f"{'stijgende steunlijn' if tl['kind'] == 'sup' else 'dalende weerstandslijn'} "
                  f"(nu {trend_at(tl, now_ms):,.6g})", 2) for tl in m.trends]
        d = lambda p: sign * (p - px) / px / sd  # >0 = in de richting van de trade, in σ
        back = sorted([c for c in cand if -2.0 <= d(c[0]) <= -.4], key=lambda c: -d(c[0]))
        front = sorted([c for c in cand if .5 <= d(c[0]) <= 3.0], key=lambda c: d(c[0]))
        if back:
            sl, sl_why = back[0][0] * (1 - sign * .15 * sd), "net voorbij " + back[0][1]
        else:
            sl, sl_why = px * (1 - sign * sd), "1σ: geen steun of weerstand binnen 2σ, dus de normale schommeling van het venster"
        if front:
            tp, tp_why = front[0][0] * (1 - sign * .05 * sd), "net voor " + front[0][1]
        else:
            tp, tp_why = px * (1 + sign * 1.5 * sd), "1,5σ: vrije baan tot 3σ, dus anderhalve keer de normale schommeling"
        sl_d, tp_d = abs(px - sl) / px, abs(tp - px) / px
        return {"sl": sl, "tp": tp, "sl_why": sl_why, "tp_why": tp_why, "sl_d": sl_d, "tp_d": tp_d,
                "rr": tp_d / max(sl_d, 1e-9), "sl_sigma": sl_d / sd, "tp_sigma": tp_d / sd}

    def plan_exits(self, m: Market, t: dict, px: float) -> None:
        """Stop-loss en take-profit bij de instap vastleggen, op de echte instapprijs."""
        e = self.exit_plan(m, t["side"], px, m.model.get("sd", 0.0))
        t["sl"], t["tp"] = e["sl"], e["tp"]
        t["plan"] = {"sl": e["sl_why"], "tp": e["tp_why"], "rr": round(e["rr"], 2),
                     "sl_sigma": round(e["sl_sigma"], 2), "tp_sigma": round(e["tp_sigma"], 2)}

    def place_tp(self, m: Market, t: dict, cfg: dict) -> None:
        """Take-profit meteen als limietorder in het boek (reduce-only, post-only): geraakt = maker, geen slippage."""
        try:
            size = abs(desk.position_size(cfg, m.coin))
            r = desk.place_maker(cfg, m.coin, t["side"] == "short", size, t["tp"], reduce_only=True)
            t["tp_oid"] = r["oid"]
            if r.get("error"):
                self.log("closer", "FOUT", f"TP-limiet geweigerd: {r['error']}; TP wordt een marktorder", coin=m.coin)
        except Exception as e:
            self.log("closer", "FOUT", f"TP-limiet plaatsen mislukt: {e}; TP wordt een marktorder", coin=m.coin)

    def cancel_tp(self, m: Market, t: dict, cfg: dict) -> None:
        if not t["paper"] and t.get("tp_oid"):
            try:
                desk.cancel(cfg, m.coin, t["tp_oid"])
            except Exception:
                pass
            t["tp_oid"] = None

    def maker_exit(self, m: Market, t: dict, now: float, cfg: dict, deadline: float) -> None:
        """Uitstap als maker op de beste prijs aan onze kant van het boek. Elke 20 s bijstellen naar de nieuwe beste
        prijs; niet gevuld voor de deadline: pas dan met een marktorder."""
        bid, ask = m.best()
        px = ask if t["side"] == "long" else bid
        if not px:
            return
        first = t["state"] != "exiting"
        if first:
            self.cancel_tp(m, t, cfg)  # de uitstap-limiet ligt dichter dan de TP; twee reduce-only orders botsen
        elif not t["paper"] and t.get("exit_oid"):
            try:
                desk.cancel(cfg, m.coin, t["exit_oid"])
            except Exception:
                pass
        t.update(state="exiting", exit_limit=px, exit_quoted=now, exit_oid=None)
        if first:
            t.update(exit_started=now, exit_deadline=deadline)
        if not t["paper"]:
            try:
                size = abs(desk.position_size(cfg, m.coin))
                r = desk.place_maker(cfg, m.coin, t["side"] == "short", size, px, reduce_only=True)
                t["exit_oid"] = r["oid"]
                if r["filled"]:
                    t["maker_exit"] = True
            except Exception as e:
                self.log("closer", "FOUT", f"maker-uitstap mislukt: {e}", coin=m.coin)
        self.log("closer", "LIMIET", f"uitstap als maker @ {px:,.6g}" + ("" if first else " (bijgesteld)")
                 + f", uiterlijk markt na {max(0, deadline - now):.0f} s", coin=m.coin)

    def manage(self, m: Market, now: float, cfg: dict) -> None:
        """Volgt een open ticket: maker-instap gevuld of verlopen, maker-uitstap op tijd plaatsen."""
        t = m.ticket
        if not t:
            return
        W = self.window
        wait = min(.1 * W, 300)
        if t["state"] == "entering":
            filled = False
            if t["paper"]:  # papier: gevuld zodra een trade door de limiet gaat
                filled = any(tt[0] >= t["opened"] and ((tt[3] <= t["limit"]) if t["side"] == "long" else (tt[3] >= t["limit"]))
                             for tt in m.trades)
            elif int(now) % 5 == 0:
                try:
                    filled = desk.order_filled(cfg, t["oid"])
                except Exception:
                    filled = False
            if filled:
                self.filled_entry(m, t["limit"], cfg)
            elif now - t["opened"] > wait:
                if not t["paper"]:
                    try:
                        desk.cancel(cfg, m.coin, t["oid"])
                        size = desk.position_size(cfg, m.coin)  # gedeeltelijk gevuld?
                    except Exception as e:
                        self.log("taker", "FOUT", f"annuleren mislukt: {e}", coin=m.coin)
                        size = 0.0
                    if size:
                        t["usd"] = abs(size) * t["limit"]
                        self.filled_entry(m, t["limit"], cfg)
                        return
                m.ticket = None
                self.log("taker", "VERLOPEN", f"limiet {t['limit']:,.6g} niet gevuld binnen {wait / 60:.0f} min, geen trade",
                         coin=m.coin)
            return
        if t["state"] in ("open", "exiting") and t.get("entry") and "sl" not in t and m.model.get("sd"):
            # trade van voor de stop-loss-versie: alsnog SL en TP zetten vanaf de instap
            self.plan_exits(m, t, t["entry"])
            if t["paper"]:
                self.save()
        if t["state"] == "open" and m.model.get("p") is not None and now - t.get("filled_at", now) > 60:
            # het model draait om: is de kans voor de andere kant zo groot dat hij die trade zelf zou openen,
            # met minstens 2¢ edge, en blijft dat 45 s zo, dan meteen sluiten in plaats van het venster uit te zitten
            mo = m.model
            # streng: 4 van 4 omgedraaid-trades verloren, telkens taker-kosten op een signaal dat even wiebelde
            against = mo["side"] != t["side"] and mo["edge"] >= 2
            if not against:
                t.pop("against_since", None)
            elif now - t.setdefault("against_since", now) >= 45:
                t["exit_reason"], t["maker_exit"] = "omgedraaid", False
                self.log("closer", "OMGEDRAAID", f"model wijst nu {mo['pDir'] * 100:.0f}¢ de andere kant op (break-even {mo['be'] * 100:.0f}¢): "
                         f"{t['side']} dicht, eerst als maker (30 s)", coin=m.coin)
                self.maker_exit(m, t, now, cfg, now + 30)
                return
        if t["state"] in ("open", "exiting") and m.px and t.get("tp"):  # take-profit: limiet in het boek, gevuld = maker
            long = t["side"] == "long"
            if t["paper"] or t.get("tp_oid"):
                if t["paper"]:
                    done = m.px >= t["tp"] if long else m.px <= t["tp"]  # er is door het TP-niveau gehandeld
                else:
                    done = False
                    if int(now) % 5 == 0:
                        try:
                            done = desk.order_filled(cfg, t["tp_oid"])
                        except Exception:
                            pass
                if done:
                    t.update(exit_reason="TP", maker_exit=True, exit_limit=t["tp"], tp_oid=None)
                    self.log("closer", "TP", f"take-profit gevuld als maker @ {t['tp']:,.6g}", coin=m.coin)
                    self.settle(m, cfg)
                    return
        if t["state"] in ("open", "exiting") and m.px:  # stop-loss geraakt (of TP zonder limiet in het boek): meteen markt
            long = t["side"] == "long"
            hit = ("SL" if t.get("sl") and (m.px <= t["sl"] if long else m.px >= t["sl"]) else
                   "TP" if t.get("tp") and not t["paper"] and not t.get("tp_oid") and (m.px >= t["tp"] if long else m.px <= t["tp"]) else None)
            if hit:
                t["exit_reason"], t["maker_exit"] = hit, False
                self.log("closer", hit, ("stop-loss" if hit == "SL" else "take-profit") + f" geraakt @ {m.px:,.6g} "
                         f"(niveau {t['sl' if hit == 'SL' else 'tp']:,.6g})", coin=m.coin)
                self.settle(m, cfg)
                return
        end = m.win + W
        lead = min(.25 * W, 600)  # uitstap-limiet al een kwart venster voor het einde: meer kans op een maker-vulling
        if t["state"] == "open" and now >= end - lead and t.get("hold_win") != m.win + W and self.try_hold(m, t, now, cfg):
            return
        if t["state"] == "open" and now >= end - lead and t.get("hold_win") != m.win + W:  # uitstap als maker proberen
            self.maker_exit(m, t, now, cfg, end - 5)
        elif t["state"] == "exiting" and not t.get("maker_exit"):
            if t["paper"]:
                t["maker_exit"] = any(tt[0] >= t.get("exit_quoted", t["exit_started"]) and ((tt[3] >= t["exit_limit"]) if t["side"] == "long"
                                                                      else (tt[3] <= t["exit_limit"])) for tt in m.trades)
            elif t["exit_oid"] and int(now) % 5 == 0:
                try:
                    t["maker_exit"] = desk.order_filled(cfg, t["exit_oid"])
                except Exception:
                    pass
            if t.get("maker_exit"):
                self.settle(m, cfg)
            elif now >= t.get("exit_deadline", end):
                self.log("closer", "MARKT", "maker-uitstap niet gevuld voor de deadline: dicht met een marktorder", coin=m.coin)
                self.settle(m, cfg)
            elif now - t.get("exit_quoted", now) >= 20:
                self.maker_exit(m, t, now, cfg, t.get("exit_deadline", end))

    def update_maker_share(self) -> None:
        last = [h for h in self.history if h.get("maker_exit") is not None][-50:]
        if len(last) >= 10:
            EXIT_MAKER["share"] = sum(1 for h in last if h["maker_exit"]) / len(last)

    def try_hold(self, m: Market, t: dict, now: float, cfg: dict) -> bool:
        """Einde venster: zou de zwerm nu opnieuw in dezelfde richting instappen voor een vol venster?
        Dan laat hij de trade doorlopen in plaats van te sluiten en opnieuw te openen (scheelt een ronde fees).
        Stop-loss en take-profit blijven staan."""
        hm = self.hold_max if self.hold_max is not None else 3
        n = t.get("holds", 0)
        if n >= hm or not m.feat or not m.px or not self.coins.get(m.coin):
            return False
        W = self.window
        mo = m.compute(m.feat, m.win, cfg, W)  # voorspelling voor een vol nieuw venster
        live = not t["paper"]
        same = mo["side"] == t["side"]
        sup, res = m.near_levels(m.px)
        wall = res if t["side"] == "long" else sup
        room = abs(wall["px"] - m.px) / m.px / max(mo["sd"], 1e-5) if wall else None
        ok = same and mo["edge"] >= self.edge_floor(cfg, live) and (room is None or room >= .3)
        why = {"t": int(now * 1000), "p": round(mo["pDir"] if same else 1 - mo["pDir"], 4), "be": round(mo["be"], 4),
               "edge": round(mo["edge"] if same else -mo["edge"], 2), "room": round(room, 2) if room is not None else None, "hold": ok}
        t.setdefault("holds_why", []).append(why)
        if not ok:
            reason = ("model wijst nu de andere kant op" if not same else
                      f"edge {mo['edge']:+.1f}¢ te klein" if mo["edge"] < self.edge_floor(cfg, live) else
                      f"{'weerstand' if t['side'] == 'long' else 'steun'} op {wall['px']:,.6g} te dichtbij")
            self.log("closer", "SLUIT", f"niet doorlopen: {reason}", coin=m.coin)
            return False
        t["holds"], t["hold_win"] = n + 1, m.win + W
        self.log("closer", "VERLENGD", f"{t['side']} loopt door in het volgende venster ({n + 1}/{hm}): {why['p'] * 100:.1f}¢ "
                 f"vs break-even {mo['be'] * 100:.1f}¢, edge {why['edge']:+.1f}¢", coin=m.coin)
        if t["paper"]:
            self.save()
        return True

    def settle(self, m: Market, cfg: dict) -> None:
        t, m.ticket = m.ticket, None
        if not t:
            return
        if t["state"] == "entering":  # nooit gevuld
            if not t["paper"] and t.get("oid"):
                try:
                    desk.cancel(cfg, m.coin, t["oid"])
                except Exception:
                    pass
            self.log("taker", "VERLOPEN", "venster voorbij voor de limiet gevuld werd", coin=m.coin)
            return
        try:
            c = costs(cfg)
            if t.get("maker_exit"):
                exit_px, exit_fee = t["exit_limit"], c["maker"]
            else:
                if not t["paper"]:
                    self.cancel_tp(m, t, cfg)
                    if t.get("exit_oid"):
                        try:
                            desk.cancel(cfg, m.coin, t["exit_oid"])
                        except Exception:
                            pass
                    desk.do_close(cfg, t["coin"], by="zwerm")
                bid, ask = m.best()
                exit_px, exit_fee = (bid if t["side"] == "long" else ask), c["taker"]
                if t["paper"] and m.px and bid and ask:
                    # het boek loopt soms achter op de laatste trade: rond de laatste prijs + halve spread afrekenen
                    half = min((ask - bid) / 2, m.px * 5e-4)
                    exit_px = m.px - half if t["side"] == "long" else m.px + half
            sign = 1 if t["side"] == "long" else -1
            pnl = t["usd"] * sign * (exit_px - t["entry"]) / t["entry"] - t["usd"] * (c["maker"] + exit_fee) / 1e4
            p_up = m.model.get("p")
            exit_why = {"reason": t.get("exit_reason") or "venster", "maker": bool(t.get("maker_exit")),
                        "conf": round(p_up if sign > 0 else 1 - p_up, 4) if p_up is not None else None,
                        "held": round((time.time() - t.get("filled_at", t["opened"])) / 60, 1),
                        "move": round(sign * (exit_px - t["entry"]) / t["entry"] / max((t.get("why") or {}).get("sd", 0) / 1e4 or m.model.get("sd", 0.0), 1e-5), 2),  # in σ bij instap
                        "window": self.window_min, "fees": round(t["usd"] * (c["maker"] + exit_fee) / 1e4, 4)}
            if t["paper"]:
                self.paper["cash"] += pnl
                self.paper["curve"].append([int(time.time() * 1000), round(self.paper["cash"], 4)])
            self.history.append({"t": int(time.time() * 1000), "opened": int(t.get("filled_at", t["opened"]) * 1000),
                                 "coin": t["coin"], "side": t["side"], "entry": t["entry"], "exit": exit_px, "usd": t["usd"],
                                 "lev": t["lev"], "pnl": round(pnl, 4), "edge": round(t["edge"], 2), "paper": t["paper"],
                                 "maker_exit": bool(t.get("maker_exit")), "reason": t.get("exit_reason") or "venster",
                                 "why": t.get("why"), "sl": t.get("sl"), "tp": t.get("tp"), "plan": t.get("plan"), "exit_why": exit_why,
                                 "holds": t.get("holds", 0), "holds_why": t.get("holds_why", [])[-12:]})
            desk_learn.record_trade(self.history[-1])
            self.update_maker_share()
            self.log("closer", "AFREK", f"{'papier' if t['paper'] else 'LIVE'} {t['side']} {t['entry']:,.6g} → {exit_px:,.6g} "
                                        f"({'maker' if t.get('maker_exit') else 'markt'})", f"{pnl:+.2f}", m.coin)
            self.save()
        except Exception as e:
            self.log("closer", "FOUT", f"sluiten mislukt: {e}. Controleer Hyperliquid!", coin=m.coin)

    def risk(self, cfg: dict) -> None:
        """Accountwaarde ophalen; daglimiet op verlies voor alle markten samen."""
        if time.time() - self.last_account < 30:
            return
        self.last_account = time.time()
        try:
            self.account = desk.account([desk.dex_of(c) for c in self.coins])
        except Exception:
            return
        if not self.account.get("connected"):
            return
        value = self.account["value"]
        if self.day != _today():
            self.day, self.day_start, self.halted = _today(), value, None
            self.log("prior", "DAG", f"nieuwe handelsdag, startwaarde ${value:,.2f}")
            self.save()
        limit = desk.num(cfg, "DESK_DAILY_LOSS", 10)
        if self.is_live() and self.day_start is not None and not self.halted and self.day_start - value > limit:
            self.halted = f"daglimiet geraakt: ${self.day_start - value:,.2f} verlies (limiet ${limit:g})"
            self.log("closer", "STOP", self.halted)
            self.pending = None
            for m in self.markets.values():
                if m.ticket and not m.ticket["paper"]:
                    m.ticket["exit_reason"] = "daglimiet"
                    self.settle(m, cfg)
            self.save()

    def close_orphans(self, cfg: dict) -> None:
        """Na een herstart: open posities op het aparte sub-account horen bij de zwerm en gaan dicht."""
        try:
            acc = desk.account([desk.dex_of(c) for c in self.coins])
            for p in acc.get("positions", []):
                if p["coin"] in self.coins and p["size"]:
                    if cfg.get("HL_SUBACCOUNT"):
                        desk.do_close(cfg, p["coin"], by="zwerm-herstart")
                        self.log("closer", "HERSTEL", "achtergebleven positie gesloten", coin=p["coin"])
                    else:
                        self.log("closer", "LET OP", "open positie op je hoofdaccount; die laat ik staan", coin=p["coin"])
        except Exception as e:
            self.log("closer", "FOUT", f"posities controleren mislukt: {e}")

    def loop(self) -> None:
        for m in list(self.markets.values()):
            time.sleep(.3)  # gespreid opwarmen
            try:
                m.warm_up()
            except Exception as e:
                self.log("prior", "FOUT", f"opwarmen mislukt: {e}", coin=m.coin)
        cfg = desk.config()
        if self.on and desk.status()["auto_allowed"]:
            self.close_orphans(cfg)
        actief = ", ".join(c for c, v in self.coins.items() if v) or "geen markten"
        self.log("prior", "START", f"zwerm gestart ({actief}), venster {self.window_min} min, {'live' if self.is_live() else 'papier'}")
        while True:
            time.sleep(1)
            try:
                self.step()
            except Exception as e:
                self.log("prior", "FOUT", f"interne fout: {e}")

    def step(self) -> None:
        now = time.time()
        with self.lock:
            st = desk.status()
            cfg = desk.config()
            if not st["auto_allowed"] and (self.on or any(not t["paper"] for t in self.open_tickets())):
                self.log("closer", "STOP", "live/DESK_AUTO staat niet meer aan in desk.env")
                self.on = False
                for m in self.markets.values():
                    if m.ticket and not m.ticket["paper"]:
                        m.ticket = None
                self.save()
            if self.pending and now > self.pending["expires"]:
                self.log("taker", "NEE", "goedkeuring verlopen", coin=self.pending["coin"])
                self.pending = None
            if st["live"]:
                self.risk(cfg)
            for m in self.markets.values():
                if not m.px:
                    continue
                if m.last_px and m.var1:
                    r = math.log(m.px / m.last_px)
                    m.var1 = .995 * m.var1 + .005 * r * r
                m.last_px = m.px
                self.new_window(m, now, cfg)
                m.feat = m.features(now)
                m.model = m.compute(m.feat, now, cfg, self.window)
                self.manage(m, now, cfg)
                if m.coin in self.drop and not m.ticket:  # uit de top 25 en de trade is afgerekend: nu uit
                    self.drop.discard(m.coin)
                    self.coins[m.coin] = False
                    self.log("spotter", "COIN", "uit de top 25, trade afgerekend: zwerm volgt niet meer", coin=m.coin)
                    self.save()
                self.decide(m, now, cfg)

    # ---------- bediening vanaf /desk ----------
    def start(self) -> None:
        if not self.started:
            self.started = True
            threading.Thread(target=self.feed, daemon=True, name="desk-feed").start()
            threading.Thread(target=self.loop, daemon=True, name="desk-swarm").start()
            threading.Thread(target=self.scanner, daemon=True, name="desk-scan").start()
            threading.Thread(target=self.level_refresher, daemon=True, name="desk-levels").start()
            threading.Thread(target=self.trainer, daemon=True, name="desk-learn").start()
            threading.Thread(target=self.backtest_sync, daemon=True, name="desk-backtest").start()

    def backtest_sync(self) -> None:
        """Elke ticker waar de zwerm op staat komt in de watchlist "Zwerm" van de backtestserver. Nieuwe tickers
        worden meteen gebacktest (dagcandles, alle strategieën); daarna doet de gewone planning het elke avond."""
        from . import watchlist as wl
        time.sleep(90)
        while True:
            try:
                with self.lock:
                    coins = [c for c, v in self.coins.items() if v]
                groups = wl.load_watchlist()
                grp = next((g for g in groups if wl.list_name(g) == BT_LIST), None)
                have = {x["yahoo"] for x in (grp or {}).get("symbols", [])}
                add = [bt_symbol(c) for c in coins if bt_symbol(c) not in have]
                if add:
                    syms = (grp or {}).get("symbols", []) + [{"tv": c if ":" in c else f"HYPERLIQUID:{c}USD", "yahoo": bt_symbol(c),
                                                              "name": c} for c in coins if bt_symbol(c) in add]
                    wl.save_watchlist(wl.replace_lists(groups, [{"name": "ZWERM", "list": BT_LIST, "symbols": syms}],
                                                       lambda n: n == BT_LIST))
                    self.log("prior", "BACKTEST", f"{len(add)} ticker(s) toegevoegd aan de backtestlijst: {', '.join(add[:8])}")
                done = wl.scan_cached().get("results", {})
                todo = [y for y in {bt_symbol(c) for c in coins} if y not in done]
                if todo and not wl.SCANNER.status()["running"]:
                    if wl.SCANNER.start(todo, wl.default_periods()):
                        self.log("prior", "BACKTEST", f"backtest gestart voor {len(todo)} nieuwe ticker(s)")
            except Exception as e:
                self.log("prior", "FOUT", f"backtestlijst bijwerken mislukt: {e}")
            time.sleep(600)

    def trainer(self) -> None:
        """Elk uur de gewichten opnieuw leren uit de LEER-vensters van het huidige venster."""
        time.sleep(60)
        while True:
            try:
                before = desk_learn.STATE.get(str(self.window), {}).get("groups", {})
                res = desk_learn.train(self.window)
                for g, i in res["groups"].items():
                    if "ll_learned" in i:
                        was = before.get(g, {}).get("use")
                        self.log("prior", "TRAIN", f"{g}: geleerd log-loss {i['ll_learned']:.4f} vs vast {i['ll_base']:.4f} op "
                                 f"{i['n_test']} nieuwe vensters → {'GELEERD in gebruik' if i['use'] else 'vast blijft'}"
                                 + (" (gewisseld)" if was is not None and was != i["use"] else ""), f"n={i['n']}")
            except Exception as e:
                self.log("prior", "FOUT", f"leren mislukt: {e}")
            time.sleep(3600)

    def level_refresher(self) -> None:
        """Steun en weerstand elk uur opnieuw berekenen, één markt per keer (anders 429 van Hyperliquid)."""
        while True:
            time.sleep(20)
            stale = [m for m in list(self.markets.values()) if time.time() - m.levels_at > 3600]
            for m in stale[:3]:
                try:
                    cs = desk.candles(m.coin, "5m", 2 * 24 * 60)
                    m.levels, m.trends, m.levels_at = sr_levels(cs), trend_lines(cs), time.time()
                except Exception:
                    m.levels_at = time.time() - 3000  # over 10 min opnieuw proberen
                time.sleep(3)

    def add_market(self, coin: str) -> Market:
        """Nieuwe markt (uit de scanner): opwarmen en op de feed zetten, nog niet volgen."""
        m = Market(coin)
        try:
            m.warm_up()
        except Exception as e:
            self.log("prior", "FOUT", f"opwarmen mislukt: {e}", coin=coin)
        self.markets[coin] = m
        self.coins[coin] = False
        self.resubscribe = True
        return m

    def scanner(self) -> None:
        """Auto-modus: elke 5 minuten volgt de zwerm precies de top 25 van de scanner. Wie eruit valt, krijgt
        geen nieuwe trades meer; een open trade daarop loopt gewoon af."""
        while True:
            time.sleep(10)
            if not self.auto_scan or time.time() - self.scan_at < 300:
                continue
            self.scan_at = time.time()
            try:
                top = [r["coin"] for r in desk.scan_cached(max_age=290, top=self.scan_n)["rows"]]
            except Exception as e:
                self.log("spotter", "FOUT", f"scanner mislukt: {e}")
                continue
            new = [c for c in top if c not in self.markets]
            fresh = {}
            for c in new:  # opwarmen buiten de lock: dat zijn HTTP-calls
                m = Market(c)
                try:
                    m.warm_up()
                except Exception:
                    pass
                fresh[c] = m
            with self.lock:
                if not self.auto_scan:
                    continue
                for c, m in fresh.items():
                    self.markets[c], self.coins[c] = m, False
                    self.resubscribe = True
                aan = [c for c in top if not self.coins[c]]
                uit = [c for c, v in self.coins.items() if v and c not in top and not self.markets[c].ticket]
                later = [c for c, v in self.coins.items() if v and c not in top and self.markets[c].ticket]
                self.drop = set(later)  # terug in de top 25 = blijft gewoon aan
                for c in aan:
                    self.coins[c] = True
                for c in uit:
                    self.coins[c] = False
                    if self.pending and self.pending["coin"] == c:
                        self.pending = None
                self.log("spotter", "TOP 25", f"scanner ({'alle liquide' if not self.scan_n else 'top ' + str(self.scan_n)}): {len(top)} markten gevolgd · {len(aan)} erbij"
                         + (f" ({', '.join(aan[:8])}{'…' if len(aan) > 8 else ''})" if aan else "")
                         + f" · {len(uit)} eraf" + (f" ({', '.join(uit[:8])}{'…' if len(uit) > 8 else ''})" if uit else "")
                         + (f" · {len(later)} gaan uit na hun trade ({', '.join(later)})" if later else ""))
                self.save()

    def set(self, on: bool | None = None, confirm: bool | None = None, coins: dict | None = None,
            window_min: int | None = None, reset_paper: float | None = None, min_edge: float | None = None,
            model: dict | None = None, auto_scan: bool | None = None, scan_n: int | None = None, max_lev: float | None = None, risk_pct: float | None = None,
            kelly_frac: float | None = None, max_margin_pct: float | None = None,
            sl_sigma: float | None = None, tp_sigma: float | None = None, hold_max: int | None = None,
            learn: bool | None = None) -> None:
        """Markten volgen, venster en papier mogen altijd; aan/uit en bevestigen vragen live-toestemming."""
        if on is not None or confirm is not None:
            if not desk.status()["auto_allowed"]:
                raise PermissionError("DESK_AUTO=1 ontbreekt in data/desk.env")
        cfg = desk.config()
        self.start()
        with self.lock:
            for coin, val in (coins or {}).items():
                coin = desk._coin(coin)
                if coin not in self.markets:  # nieuwe markt uit de scanner
                    self.add_market(coin)
                self.drop.discard(coin)  # handmatig gezet: de scanner-regel vervalt
                if bool(val) != self.coins[coin]:
                    self.coins[coin] = bool(val)
                    m = self.markets[coin]
                    if not val and self.pending and self.pending["coin"] == coin:
                        self.pending = None
                    self.log("prior", "COIN", "zwerm volgt nu " + ("wel" if val else "niet meer")
                             + (" (open trade loopt af)" if not val and m.ticket else ""), coin=coin)
            if window_min is not None and int(window_min) != self.window_min:
                if int(window_min) not in WINDOWS:
                    raise ValueError(f"Venster moet {', '.join(map(str, WINDOWS))} minuten zijn")
                for m in self.markets.values():
                    if m.ticket:
                        m.ticket["exit_reason"] = "venster gewijzigd"
                        self.settle(m, cfg)
                    m.win, m.sample = 0, None  # nieuwe vensters, zonder een half venster te laten leren
                self.window_min = int(window_min)
                self.log("prior", "VENSTER", f"vensters van {self.window_min} minuten")
            if min_edge is not None and float(min_edge) != self.min_edge:
                if not 0 <= float(min_edge) <= 10:  # onder 0 = bewust trades met verwacht verlies
                    raise ValueError("Drempel moet tussen 0¢ en 10¢ liggen")
                self.min_edge = float(min_edge)
                self.log("edge", "DREMPEL", f"edge-drempel {self.min_edge:+g}¢ boven break-even")
            if scan_n is not None and int(scan_n) != self.scan_n:
                if int(scan_n) not in (25, 50, 0):
                    raise ValueError("Scanner: top 25, top 50 of 0 (alle liquide markten)")
                self.scan_n, self.scan_at = int(scan_n), 0.0
                self.log("spotter", "AUTO", "scanner volgt " + ("alle liquide markten met ratio ≥ 1" if not self.scan_n else f"de top {self.scan_n}"))
            if auto_scan is not None and auto_scan != self.auto_scan:
                self.auto_scan, self.scan_at = auto_scan, 0.0
                self.drop.clear()
                self.log("spotter", "AUTO", "scanner-modus " + ("aan: elke 5 min de top 25" if auto_scan else "uit"))
            if learn is not None and learn != LEARN["on"]:
                LEARN["on"] = bool(learn)
                self.log("prior", "LEREN", "geleerde gewichten gebruiken als ze beter zijn" if learn else "alleen de vaste gewichten")
            if hold_max is not None:
                if not 0 <= int(hold_max) <= 12:
                    raise ValueError("Doorlopen: 0 (altijd sluiten op het einde van het venster) tot 12 vensters")
                self.hold_max = int(hold_max)
                self.log("closer", "INSTEL", f"trades mogen tot {self.hold_max} vensters doorlopen" if self.hold_max else "trades sluiten altijd op het einde van het venster")
            if sl_sigma is not None or tp_sigma is not None:
                for v in (sl_sigma, tp_sigma):
                    if v is not None and not (float(v) == 0 or 0.5 <= float(v) <= 6):
                        raise ValueError("Stop-loss en take-profit: 0 (uit) of 0,5σ tot 6σ")
                if sl_sigma is not None:
                    self.sl_sigma = float(sl_sigma)
                if tp_sigma is not None:
                    self.tp_sigma = float(tp_sigma)
                self.log("closer", "SL/TP", "stop-loss en take-profit worden nu per trade bepaald (structuur + verwachte beweging); "
                                            "deze instelling wordt niet meer gebruikt")
            if kelly_frac is not None or max_margin_pct is not None:
                if kelly_frac is not None and not 0.05 <= float(kelly_frac) <= 1:
                    raise ValueError("Kelly-fractie moet tussen 0,05 en 1 liggen")
                if max_margin_pct is not None and not 0.5 <= float(max_margin_pct) <= 25:
                    raise ValueError("Max marge per trade moet tussen 0,5% en 25% liggen")
                if kelly_frac is not None:
                    self.kelly_frac = float(kelly_frac)
                if max_margin_pct is not None:
                    self.max_margin_pct = float(max_margin_pct)
                kf, mp = self.size_limits()
                self.log("kelly", "GROOTTE", f"Kelly ×{kf:g} · max marge {mp:g}% van de portefeuille per trade")
            if max_lev is not None or risk_pct is not None:
                if max_lev is not None and not 1 <= float(max_lev) <= 20:
                    raise ValueError("Max hefboom moet tussen 1x en 20x liggen")
                if risk_pct is not None and not 1 <= float(risk_pct) <= 50:
                    raise ValueError("Risico per trade moet tussen 1% en 50% van de marge liggen")
                if max_lev is not None:
                    self.max_lev = float(max_lev)
                if risk_pct is not None:
                    self.risk_pct = float(risk_pct)
                cap, risk = self.lev_limits(cfg, False)
                self.log("kelly", "HEFBOOM", f"max {cap:g}x (live hooguit {desk.num(cfg, 'DESK_MAX_LEV', 3):g}x uit desk.env) · "
                                             f"risico {risk:g}% van de marge bij 2σ tegen")
            if model is not None:
                new = json.loads(json.dumps(MODEL_DEFAULT if model.get("reset") else MODEL))
                if not model.get("reset"):
                    if "beta0" in model:
                        new["beta0"] = float(model["beta0"])
                    if "k" in model:
                        new["k"] = float(model["k"])
                    for key, v in (model.get("w") or {}).items():
                        if key not in new["w"]:
                            raise ValueError(f"Onbekend kenmerk: {key}")
                        new["w"][key] = float(v)
                if not (0 <= new["beta0"] <= 2 and 1 <= new["k"] <= 1000 and all(0 <= v <= 1 for v in new["w"].values())
                        and sum(new["w"].values()) > 0):
                    raise ValueError("Start-β 0..2, krimp 1..1000, gewichten 0..1 (niet allemaal 0)")
                MODEL.clear()
                MODEL.update(new)
                ws = " · ".join(f"{k} {v:g}" for k, v in MODEL["w"].items())
                self.log("prior", "MODEL", f"{'standaard hersteld: ' if model.get('reset') else ''}start-β {MODEL['beta0']:g}, krimp {MODEL['k']:g}, gewichten {ws}")
            if reset_paper is not None:
                if any(t["paper"] for t in self.open_tickets()):
                    for m in self.markets.values():
                        if m.ticket and m.ticket["paper"]:
                            m.ticket = None
                amount = float(reset_paper) if reset_paper > 0 else desk.num(cfg, "DESK_PAPER_START", 1000)
                self.paper = {"start": amount, "cash": amount, "curve": [[int(time.time() * 1000), amount]]}
                self.history = deque([h for h in self.history if not h.get("paper")], maxlen=2000)
                self.log("prior", "RESET", f"papierportefeuille opnieuw op ${amount:,.0f}")
            if confirm is not None and confirm != self.confirm:
                self.confirm = confirm
                self.log("taker", "MODE", "bevestigen per order " + ("aan" if confirm else "uit: de zwerm handelt zelf"))
            if on is not None and on != self.on:
                self.on = on
                if not on:
                    self.pending = None
                    for m in self.markets.values():
                        if m.ticket and not m.ticket["paper"]:
                            m.ticket["exit_reason"] = "kill switch"
                            self.settle(m, cfg)
                    for c in list(self.coins):  # kill switch: alles dicht
                        try:
                            desk.do_close(desk.live_config(), c, by="kill")
                        except Exception as e:
                            self.log("closer", "FOUT", f"sluiten mislukt: {e}", coin=c)
                else:
                    for m in self.markets.values():  # papiertickets stoppen, live begint schoon
                        if m.ticket and m.ticket["paper"]:
                            self.settle(m, cfg)
                self.errors = 0
                self.log("closer", "AAN" if on else "UIT", "live handel " + ("aan" if on else "uit, alles gesloten"))
            self.save()

    def set_level(self, coin: str, kind: str, price: float | None) -> dict:
        """Stop-loss of take-profit van een open trade met de hand verschuiven (of met None weghalen)."""
        if kind not in ("sl", "tp"):
            raise ValueError("kind moet sl of tp zijn")
        with self.lock:
            m = self.markets.get(coin)
            t = m and m.ticket
            if not t or t["state"] not in ("open", "exiting"):
                raise ValueError(f"Geen open trade op {coin}")
            if price is not None:
                price, long = float(price), t["side"] == "long"
                below = (kind == "sl") == long  # long: SL onder de koers, TP erboven; short omgekeerd
                if (below and price >= m.px) or (not below and price <= m.px):
                    raise ValueError(f"{'Stop-loss' if kind == 'sl' else 'Take-profit'} moet {'onder' if below else 'boven'} "
                                     f"de huidige koers {m.px:,.6g} liggen")
            t[kind] = price
            self.log("closer", kind.upper(), f"{'stop-loss' if kind == 'sl' else 'take-profit'} met de hand "
                     + (f"verschoven naar {price:,.6g}" if price is not None else "weggehaald"), coin=coin)
            if t["paper"]:
                self.save()
            return {"ok": True, kind: price}

    def approve(self, order_id: int, yes: bool) -> dict:
        cfg = desk.live_config()
        with self.lock:
            p = self.pending
            if not p or p["id"] != order_id:
                raise ValueError("Deze order wacht niet meer op goedkeuring")
            self.pending = None
            if not yes:
                self.log("taker", "NEE", "order geweigerd door jou", coin=p["coin"])
                return {"ok": True}
            m = self.markets[p["coin"]]
            if p["win"] != m.win or time.time() > p["expires"]:
                self.log("taker", "NEE", "te laat goedgekeurd", coin=p["coin"])
                raise ValueError("Te laat: het venster of de bedenktijd is voorbij")
            bid, ask = m.best()
            p["limit"] = bid if p["side"] == "long" else ask  # limiet op de actuele beste prijs
            self.enter(p, cfg)
            return {"ok": m.ticket is not None}

    @staticmethod
    def shown_levels(m: Market) -> list[dict]:
        """Voor de grafiek: de 4 dichtste niveaus erboven en eronder die minstens 2x geraakt werden."""
        if not m.px:
            return []
        lv = [l for l in m.levels if l["n"] >= 2]
        up = sorted((l for l in lv if l["px"] > m.px), key=lambda l: l["px"])[:4]
        dn = sorted((l for l in lv if l["px"] < m.px), key=lambda l: -l["px"])[:4]
        return dn[::-1] + up

    def snapshot(self, since: int = 0) -> dict:
        st = desk.status()
        cfg = desk.config()
        with self.lock:
            markets = {c: {"on": self.coins[c], "px": m.px, "open": m.open0, "win": m.win * 1000, "ticket": m.ticket,
                           "model": {k: v for k, v in m.model.items() if k != "prior"},
                           "prior": m.model.get("prior", {}), "score": m.feat.get("score"),
                           "levels": self.shown_levels(m), "trends": m.trends, "bt": bt_bias(c)}
                       for c, m in self.markets.items()}
            return {
                "allowed": st["auto_allowed"], "live": self.on and st["auto_allowed"], "on": self.on, "confirm": self.confirm,
                "coins": self.coins, "window_min": self.window_min, "min_edge": self.edge_floor(cfg, False), "model": MODEL,
                "auto_scan": self.auto_scan, "scan_n": self.scan_n, "scan_at": int(self.scan_at * 1000),
                "lev": dict(zip(("max", "risk"), self.lev_limits(cfg, False)), live_max=desk.num(cfg, "DESK_MAX_LEV", 3)),
                "size": dict(zip(("kelly", "max_margin_pct"), self.size_limits())),
                "exits": {"mode": "auto", "hold": self.hold_max if self.hold_max is not None else 3},
                "learn": {"on": LEARN["on"], **desk_learn.summary(self.window)}, "windows": WINDOWS, "costs": costs(cfg),
                "halted": self.halted, "feed": self.feed_ok, "markets": markets, "pending": self.pending,
                "portfolio": self.portfolio(), "history": list(self.history)[-200:],
                "trace": [t for t in self.trace if t["id"] > since], "seq": self.seq,
            }


swarm = Swarm()
