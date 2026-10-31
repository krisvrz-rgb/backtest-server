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

from . import desk
from .data import DATA_DIR

APPROVE_SECONDS = 30
STATE_FILE = DATA_DIR / "desk-auto.json"
WS_URL = "wss://api.hyperliquid.xyz/ws"
WINDOWS = (15, 60, 240)
# Prijsmodel, instelbaar op /desk → Instellingen: gewichten van de Spotter-score en hoe snel Prior zijn β uit de data haalt.
MODEL_DEFAULT = {"beta0": 0.25, "k": 30, "w": {"flow30": .35, "flow10": .2, "book": .25, "bookDeep": .1, "mom": .1}}
MODEL: dict = json.loads(json.dumps(MODEL_DEFAULT))


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def costs(cfg: dict) -> dict:
    """Kosten per ronde in bp: maker bij instap, uitstap half maker / half taker gerekend."""
    maker, taker = desk.num(cfg, "DESK_MAKER_BPS", 1.5), desk.num(cfg, "DESK_FEE_BPS", 4.5)
    return {"maker": maker, "taker": taker, "round": maker + (maker + taker) / 2}


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
        if not self.px and cs:
            self.px = cs[-1]["c"]

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
        bhat = sy / ss if ss > 1e-9 else 0.0
        return {"beta": (n * bhat + k * beta0) / (n + k), "n": n, "hit": hit}

    def compute(self, f: dict, now: float, cfg: dict, window: int) -> dict:
        left = max(1.0, window - (now - self.win))
        sd = math.sqrt(max(self.var1, 1e-12) * left)
        pr = self.prior(window)
        p = _phi(pr["beta"] * f["score"])
        e_move = sd * math.sqrt(2 / math.pi) * 1e4
        be = .5 + costs(cfg)["round"] / (2 * max(e_move, .01))
        p_dir = max(p, 1 - p)
        return {"p": p, "pDir": p_dir, "side": "long" if p >= .5 else "short", "be": be,
                "edge": (p_dir - be) * 100, "sd": sd, "left": left, "prior": pr, "eMove": e_move}


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
        self.sl_sigma: float | None = saved.get("sl_sigma")
        self.tp_sigma: float | None = saved.get("tp_sigma")
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
                                          "min_edge": self.min_edge, "auto_scan": self.auto_scan, "scan_n": self.scan_n, "max_lev": self.max_lev, "risk_pct": self.risk_pct, "kelly_frac": self.kelly_frac, "sl_sigma": self.sl_sigma, "tp_sigma": self.tp_sigma,
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
                            "sl": t.get("sl"), "tp": t.get("tp")})
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
        if m.ticket:
            self.settle(m, cfg)
        if m.sample and m.px:
            y = math.log(m.px / m.sample["px"]) / max(1e-9, m.sample["sd"])
            a = m.samples(self.window)
            a.append({"s": round(m.sample["s"], 4), "y": round(y, 4), "t": m.win})
            m.prior_file(self.window).write_text(json.dumps(a[-2000:]))
            if self.coins[m.coin]:
                self.log("prior", "LEER", f"score {m.sample['s']:+.2f} → uitkomst {y:+.2f}σ", f"n={len(a)}", m.coin)
        m.win, m.open0, m.decided, m.sample = win, m.px, False, None
        if self.pending and self.pending["coin"] == m.coin:
            self.pending = None
            self.log("taker", "NEE", "goedkeuring verlopen: venster voorbij", coin=m.coin)

    def decide(self, m: Market, now: float, cfg: dict) -> None:
        f, mo, el, W = m.feat, m.model, now - m.win, self.window
        if not m.sample and el >= .15 * W:
            m.sample = {"s": f["score"], "px": m.px, "sd": mo["sd"]}
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
        if mo["edge"] < min_edge:
            self.log("edge", "PASS", f"{mo['side']} {mo['pDir'] * 100:.1f}¢ < break-even {mo['be'] * 100:.1f}¢ {min_edge:+g}¢",
                     f"{mo['edge']:+.1f}¢", m.coin)
            if el + every > .67 * W:
                m.decided = True
            return
        m.decided = True
        self.log("edge", "EDGE", f"{mo['side']} {mo['pDir'] * 100:.1f}¢ vs break-even {mo['be'] * 100:.1f}¢",
                 f"{mo['edge']:+.1f}¢", m.coin)
        sized = self.kelly(m, cfg, live)
        if not sized:
            return
        lev, usd = sized
        bid, ask = m.best()
        limit = bid if mo["side"] == "long" else ask
        order = {"id": int(now * 1000), "coin": m.coin, "side": mo["side"], "usd": usd, "lev": lev, "edge": mo["edge"],
                 "p": mo["pDir"], "be": mo["be"], "win": m.win, "expires": now + APPROVE_SECONDS, "paper": not live,
                 "limit": limit}
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
        return (self.sl_sigma if self.sl_sigma is not None else 2.0, self.tp_sigma if self.tp_sigma is not None else 2.0)

    def size_limits(self) -> tuple[float, float]:
        """Kelly-fractie (0,25 = kwart-Kelly) en max marge per trade in % van de portefeuille."""
        return (self.kelly_frac if self.kelly_frac is not None else 0.25,
                self.max_margin_pct if self.max_margin_pct is not None else 5.0)

    def lev_limits(self, cfg: dict, live: bool) -> tuple[float, float]:
        """Max hefboom en risico per trade. Op papier vrij in te stellen; live nooit boven DESK_MAX_LEV uit desk.env."""
        env_cap = desk.num(cfg, "DESK_MAX_LEV", 3)
        cap = self.max_lev if self.max_lev is not None else env_cap
        risk = self.risk_pct if self.risk_pct is not None else desk.num(cfg, "DESK_RISK_PCT", 10)
        return (min(cap, env_cap) if live else cap), risk

    def kelly(self, m: Market, cfg: dict, live: bool) -> tuple[int, float] | None:
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
        want = math.floor(risk / 100 / (2 * sd))
        lev = int(_clamp(want, 1, max(1, cap)))
        mo, (kf, mpct) = m.model, self.size_limits()
        mu = (2 * mo["pDir"] - 1) * sd * math.sqrt(2 / math.pi) - costs(cfg)["round"] / 1e4
        kelly_notional = max(0.0, kf * mu / (sd * sd)) * pf["equity"]
        cap_margin = min(pf["equity"] * mpct / 100, pf["free"] / 4)
        margin = min(kelly_notional / lev, cap_margin)
        if 0 < margin * lev < 10 <= cap_margin * lev:  # Hyperliquid wil minstens $10: kleinste order
            margin = 10 / lev
        usd = margin * lev
        if live:
            usd = min(usd, desk.num(cfg, "DESK_MAX_USD", 25), desk.num(cfg, "DESK_ORDER_USD", 20) * lev)
        if usd < 10:  # Hyperliquid weigert orders onder $10
            self.log("kelly", "TE KLEIN", f"vrije marge ${pf['free']:,.2f} te klein voor een order", coin=m.coin)
            return None
        self.log("kelly", "SIZE", f"σ venster {sd * 1e4:.0f}bp → hefboom {lev}x"
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
                self.filled_entry(m, r["px"])
            else:
                self.log("taker", "LIMIET", f"LIVE {order['side']} ${order['usd']:g} ({order['lev']}x) limiet {r['px']:,.6g}", coin=m.coin)
        except Exception as e:
            self.errors += 1
            self.log("taker", "FOUT", f"order mislukt ({self.errors}/3): {e}", coin=m.coin)
            if self.errors >= 3:
                self.on = False
                self.log("closer", "STOP", "3 mislukte orders op rij: autonome handel uit")
                self.save()

    def filled_entry(self, m: Market, px: float) -> None:
        t = m.ticket
        t.update(state="open", entry=px, filled_at=time.time())
        # stop-loss en take-profit in σ van de rest van het venster, vanaf de instap
        sl_k, tp_k = self.exit_limits()
        sd, sign = max(m.model.get("sd", 0.0), 1e-5), 1 if t["side"] == "long" else -1
        t["sl"] = px * (1 - sign * sl_k * sd) if sl_k else None
        t["tp"] = px * (1 + sign * tp_k * sd) if tp_k else None
        if t["paper"]:
            self.save()
        self.log("taker", "TAKE", f"{'papier' if t['paper'] else 'LIVE'} {t['side']} gevuld als maker @ {px:,.6g}",
                 f"{t['lev']}x ${t['usd']:g}", m.coin)

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
                self.filled_entry(m, t["limit"])
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
                        self.filled_entry(m, t["limit"])
                        return
                m.ticket = None
                self.log("taker", "VERLOPEN", f"limiet {t['limit']:,.6g} niet gevuld binnen {wait / 60:.0f} min, geen trade",
                         coin=m.coin)
            return
        if t["state"] in ("open", "exiting") and m.px:  # stop-loss of take-profit geraakt: meteen dicht met een marktorder
            long = t["side"] == "long"
            hit = ("SL" if t.get("sl") and (m.px <= t["sl"] if long else m.px >= t["sl"]) else
                   "TP" if t.get("tp") and (m.px >= t["tp"] if long else m.px <= t["tp"]) else None)
            if hit:
                t["exit_reason"], t["maker_exit"] = hit, False
                self.log("closer", hit, ("stop-loss" if hit == "SL" else "take-profit") + f" geraakt @ {m.px:,.6g} "
                         f"(niveau {t['sl' if hit == 'SL' else 'tp']:,.6g})", coin=m.coin)
                self.settle(m, cfg)
                return
        end = m.win + W
        if t["state"] == "open" and now >= end - wait:  # uitstap als maker proberen
            bid, ask = m.best()
            t["exit_limit"] = ask if t["side"] == "long" else bid
            t["state"] = "exiting"
            t["exit_started"] = now
            if not t["paper"]:
                try:
                    size = abs(desk.position_size(cfg, m.coin))
                    r = desk.place_maker(cfg, m.coin, t["side"] == "short", size, t["exit_limit"], reduce_only=True)
                    t["exit_oid"] = r["oid"]
                    if r["filled"]:
                        t["maker_exit"] = True
                except Exception as e:
                    self.log("closer", "FOUT", f"maker-uitstap mislukt: {e}", coin=m.coin)
            self.log("closer", "LIMIET", f"uitstap als maker @ {t['exit_limit']:,.6g}", coin=m.coin)
        elif t["state"] == "exiting" and not t["maker_exit"]:
            if t["paper"]:
                t["maker_exit"] = any(tt[0] >= t["exit_started"] and ((tt[3] >= t["exit_limit"]) if t["side"] == "long"
                                                                      else (tt[3] <= t["exit_limit"])) for tt in m.trades)
            elif t["exit_oid"] and int(now) % 5 == 0:
                try:
                    t["maker_exit"] = desk.order_filled(cfg, t["exit_oid"])
                except Exception:
                    pass
            if t["maker_exit"]:
                self.settle(m, cfg)

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
                    if t.get("exit_oid"):
                        try:
                            desk.cancel(cfg, m.coin, t["exit_oid"])
                        except Exception:
                            pass
                    desk.do_close(cfg, t["coin"], by="zwerm")
                bid, ask = m.best()
                exit_px, exit_fee = (bid if t["side"] == "long" else ask), c["taker"]
            sign = 1 if t["side"] == "long" else -1
            pnl = t["usd"] * sign * (exit_px - t["entry"]) / t["entry"] - t["usd"] * (c["maker"] + exit_fee) / 1e4
            if t["paper"]:
                self.paper["cash"] += pnl
                self.paper["curve"].append([int(time.time() * 1000), round(self.paper["cash"], 4)])
            self.history.append({"t": int(time.time() * 1000), "opened": int(t.get("filled_at", t["opened"]) * 1000),
                                 "coin": t["coin"], "side": t["side"], "entry": t["entry"], "exit": exit_px, "usd": t["usd"],
                                 "lev": t["lev"], "pnl": round(pnl, 4), "edge": round(t["edge"], 2), "paper": t["paper"],
                                 "maker_exit": bool(t.get("maker_exit")), "reason": t.get("exit_reason") or "venster"})
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
            sl_sigma: float | None = None, tp_sigma: float | None = None) -> None:
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
            if sl_sigma is not None or tp_sigma is not None:
                for v in (sl_sigma, tp_sigma):
                    if v is not None and not (float(v) == 0 or 0.5 <= float(v) <= 6):
                        raise ValueError("Stop-loss en take-profit: 0 (uit) of 0,5σ tot 6σ")
                if sl_sigma is not None:
                    self.sl_sigma = float(sl_sigma)
                if tp_sigma is not None:
                    self.tp_sigma = float(tp_sigma)
                sl, tp = self.exit_limits()
                self.log("closer", "SL/TP", f"stop-loss {f'{sl:g}σ' if sl else 'uit'} · take-profit {f'{tp:g}σ' if tp else 'uit'} "
                                            "(geldt voor nieuwe trades)")
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

    def snapshot(self, since: int = 0) -> dict:
        st = desk.status()
        cfg = desk.config()
        with self.lock:
            markets = {c: {"on": self.coins[c], "px": m.px, "open": m.open0, "win": m.win * 1000, "ticket": m.ticket,
                           "model": {k: v for k, v in m.model.items() if k != "prior"},
                           "prior": m.model.get("prior", {}), "score": m.feat.get("score")}
                       for c, m in self.markets.items()}
            return {
                "allowed": st["auto_allowed"], "live": self.on and st["auto_allowed"], "on": self.on, "confirm": self.confirm,
                "coins": self.coins, "window_min": self.window_min, "min_edge": self.edge_floor(cfg, False), "model": MODEL,
                "auto_scan": self.auto_scan, "scan_n": self.scan_n, "scan_at": int(self.scan_at * 1000),
                "lev": dict(zip(("max", "risk"), self.lev_limits(cfg, False)), live_max=desk.num(cfg, "DESK_MAX_LEV", 3)),
                "size": dict(zip(("kelly", "max_margin_pct"), self.size_limits())),
                "exits": dict(zip(("sl", "tp"), self.exit_limits())), "windows": WINDOWS, "costs": costs(cfg),
                "halted": self.halted, "feed": self.feed_ok, "markets": markets, "pending": self.pending,
                "portfolio": self.portfolio(), "history": list(self.history)[-200:],
                "trace": [t for t in self.trace if t["id"] > since], "seq": self.seq,
            }


swarm = Swarm()
