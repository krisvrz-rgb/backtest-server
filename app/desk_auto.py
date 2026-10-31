"""Autonome zwerm: dezelfde zes agents als op /desk, maar op de server, zodat hij doorhandelt als de pagina dicht is.

Draait alleen als data/desk.env live handel toestaat én DESK_AUTO=1 bevat. Vanaf /desk (met het desk-token) zet je
de zwerm aan/uit, per coin (BTC, ETH, SOL) aan/uit en bevestigen per order aan/uit. Elke aangezette coin heeft
zijn eigen venster en zijn eigen Prior; de grenzen gelden voor alle coins samen:
- grootte per order DESK_ORDER_USD, afgekapt op DESK_MAX_USD; per coin één positie tegelijk,
  altijd gesloten aan het einde van het venster van 5 minuten (reduce-only);
- daglimiet: zakt de accountwaarde meer dan DESK_DAILY_LOSS onder de stand van 00:00 UTC, dan sluit hij
  alles en handelt hij niet meer tot de volgende UTC-dag;
- na 3 mislukte orders op rij gaat hij uit.
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

WIN = 300
DECIDE_FROM, DECIDE_UNTIL = 45, 200
APPROVE_SECONDS = 20
STATE_FILE = DATA_DIR / "desk-auto.json"
WS_URL = "wss://api.hyperliquid.xyz/ws"


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Market:
    """Koersdata, venster en open ticket van één coin."""

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

    def prior_file(self):
        return DATA_DIR / f"desk-prior-{self.coin}.json"

    def samples(self) -> list:
        f = self.prior_file()
        return json.loads(f.read_text()) if f.exists() else []

    def warm_up(self) -> None:
        """σ per seconde uit de 1m-candles van de afgelopen 3 uur."""
        cs = desk.candles(self.coin, "1m", 180)
        rs = [math.log(cs[i]["c"] / cs[i - 1]["c"]) for i in range(1, len(cs))]
        self.var1 = sum(r * r for r in rs) / max(1, len(rs)) / 60
        if not self.px and cs:
            self.px = cs[-1]["c"]

    def features(self, now: float) -> dict:
        cut = now - 60
        while self.trades and self.trades[0][0] < cut:
            self.trades.popleft()

        def flow(sec: float) -> float:
            b = sum(sz for t, sz, buy in self.trades if t > now - sec and buy)
            s = sum(sz for t, sz, buy in self.trades if t > now - sec and not buy)
            return (b - s) / (b + s) if b + s else 0.0

        f = {"flow10": flow(10), "flow30": flow(30), "book": 0.0, "bookDeep": 0.0}
        if self.book:
            bids, asks = self.book

            def imb(n: int) -> float:
                b, a = sum(x[1] for x in bids[:n]), sum(x[1] for x in asks[:n])
                return (b - a) / (b + a) if b + a else 0.0

            f["book"], f["bookDeep"] = imb(5), imb(20)
        el = max(1.0, now - self.win)
        f["mom"] = math.log(self.px / self.open0) / math.sqrt(self.var1 * el) if self.open0 and self.var1 else 0.0
        f["score"] = _clamp(.35 * f["flow30"] + .2 * f["flow10"] + .25 * f["book"] + .1 * f["bookDeep"]
                            + .1 * _clamp(f["mom"] / 2, -1, 1), -1, 1)
        return f

    def prior(self) -> dict:
        a = self.samples()
        n, beta0, k = len(a), .25, 30
        sy = sum(s["s"] * s["y"] for s in a)
        ss = sum(s["s"] ** 2 for s in a)
        hit = sum(1 for s in a if (s["s"] > 0) == (s["y"] > 0)) / n if n else None
        bhat = sy / ss if ss > 1e-9 else 0.0
        return {"beta": (n * bhat + k * beta0) / (n + k), "n": n, "hit": hit}

    def compute(self, f: dict, now: float, cfg: dict) -> dict:
        left = max(1.0, WIN - (now - self.win))
        sd = math.sqrt(max(self.var1, 1e-12) * left)
        pr = self.prior()
        p = _phi(pr["beta"] * f["score"])
        e_move = sd * math.sqrt(2 / math.pi) * 1e4
        cost = 2 * desk.num(cfg, "DESK_FEE_BPS", 4.5) + 1
        be = .5 + cost / (2 * max(e_move, .01))
        p_dir = max(p, 1 - p)
        return {"p": p, "pDir": p_dir, "side": "long" if p >= .5 else "short", "be": be,
                "edge": (p_dir - be) * 100, "sd": sd, "left": left, "prior": pr}


class Swarm:
    def __init__(self) -> None:
        saved = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        self.on: bool = saved.get("on", False)
        self.confirm: bool = saved.get("confirm", True)
        self.coins: dict[str, bool] = {c: c == "BTC" for c in desk.COINS}
        self.coins.update(saved.get("coins", {}))
        self.resubscribe = False
        self.day: str = saved.get("day", "")
        self.day_start: float | None = saved.get("day_start")
        self.halted: str | None = saved.get("halted") if saved.get("day") == _today() else None
        self.lock = threading.RLock()
        self.trace: deque = deque(maxlen=300)
        self.seq = 0
        self.history: deque = deque(saved.get("history", []), maxlen=200)
        self.markets = {c: Market(c) for c in self.coins}
        self.pending: dict | None = None
        self.errors = 0
        self.feed_ok = False
        self.last_account = 0.0
        self.started = False

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"on": self.on, "confirm": self.confirm, "coins": self.coins, "day": self.day,
                                          "day_start": self.day_start, "halted": self.halted,
                                          "history": list(self.history)}))

    def log(self, agent: str, act: str, text: str, val: str = "", coin: str = "") -> None:
        self.seq += 1
        self.trace.append({"id": self.seq, "t": int(time.time() * 1000), "agent": agent, "act": act,
                           "text": f"{coin} · {text}" if coin else text, "val": val, "coin": coin})

    # ---------- marktdata: één websocket voor alle coins ----------
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
                    while not self.resubscribe:  # nieuwe coin erbij: opnieuw verbinden met de nieuwe lijst
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
                                        m.trades.append((t["time"] / 1000, float(t["sz"]), t["side"] == "B"))
                        elif ch == "l2Book" and data.get("coin") in self.markets:
                            self.markets[data["coin"]].book = [[(float(l["px"]), float(l["sz"])) for l in side]
                                                               for side in data["levels"]]
            except Exception as e:  # opnieuw verbinden
                self.feed_ok = False
                self.log("spotter", "FEED", f"verbinding met Hyperliquid weg: {e}")
                time.sleep(5)

    # ---------- de zwerm, per coin ----------
    def new_window(self, m: Market, now: float, cfg: dict | None) -> None:
        win = int(now // WIN) * WIN
        if win == m.win:
            return
        if m.ticket:
            self.settle(m, cfg)
        if m.sample and m.px:
            y = math.log(m.px / m.sample["px"]) / max(1e-9, m.sample["sd"])
            a = m.samples()
            a.append({"s": round(m.sample["s"], 4), "y": round(y, 4), "t": m.win})
            m.prior_file().write_text(json.dumps(a[-2000:]))
            if self.coins[m.coin]:
                self.log("prior", "LEER", f"score {m.sample['s']:+.2f} → uitkomst {y:+.2f}σ", f"n={len(a)}", m.coin)
        m.win, m.open0, m.decided, m.sample = win, m.px, False, None
        if self.pending and self.pending["coin"] == m.coin:
            self.pending = None
            self.log("taker", "NEE", "goedkeuring verlopen: venster voorbij", coin=m.coin)

    def decide(self, m: Market, now: float, cfg: dict) -> None:
        f, mo, el = m.feat, m.model, now - m.win
        if not m.sample and el >= DECIDE_FROM:
            m.sample = {"s": f["score"], "px": m.px, "sd": mo["sd"]}
        if not (self.on and self.coins[m.coin]) or self.halted or m.decided or m.ticket or self.pending:
            return
        if el < DECIDE_FROM or el > DECIDE_UNTIL or int(el) % 15:
            return
        min_edge = desk.num(cfg, "DESK_MIN_EDGE", 2)
        self.log("spotter", "SCAN", f"flow {f['flow30']:+.2f} · boek {f['book']:+.2f} · score {f['score']:+.2f}", coin=m.coin)
        if mo["edge"] < min_edge:
            self.log("edge", "PASS", f"{mo['side']} {mo['pDir'] * 100:.1f}¢ < break-even {mo['be'] * 100:.1f}¢ + {min_edge:g}¢",
                     f"{mo['edge']:+.1f}¢", m.coin)
            if el >= DECIDE_UNTIL - 15:
                m.decided = True
            return
        usd = min(desk.num(cfg, "DESK_ORDER_USD", 20), desk.num(cfg, "DESK_MAX_USD", 25))
        m.decided = True
        self.log("edge", "EDGE", f"{mo['side']} {mo['pDir'] * 100:.1f}¢ vs break-even {mo['be'] * 100:.1f}¢",
                 f"{mo['edge']:+.1f}¢", m.coin)
        order = {"id": int(now * 1000), "coin": m.coin, "side": mo["side"], "usd": usd, "edge": mo["edge"],
                 "p": mo["pDir"], "be": mo["be"], "win": m.win, "expires": now + APPROVE_SECONDS}
        if self.confirm:
            self.pending = order
            self.log("taker", "WACHT", f"{mo['side']} ${usd:g} wacht op jouw goedkeuring", "mens", m.coin)
        else:
            self.execute(order, cfg)

    def execute(self, order: dict, cfg: dict) -> None:
        m = self.markets[order["coin"]]
        try:
            res = desk.do_open(cfg, order["coin"], order["side"], order["usd"], by="zwerm")
            if not res["ok"]:
                raise RuntimeError(json.dumps(res["result"])[:200])
            self.errors = 0
            m.ticket = {**order, "entry": res["mid"], "size": res["size"], "opened": time.time()}
            self.log("taker", "TAKE", f"LIVE {order['side']} ${order['usd']:g} @ {res['mid']:,.2f}", coin=m.coin)
        except Exception as e:
            self.errors += 1
            self.log("taker", "FOUT", f"order mislukt ({self.errors}/3): {e}", coin=m.coin)
            if self.errors >= 3:
                self.on = False
                self.log("closer", "STOP", "3 mislukte orders op rij: autonome handel uit")
                self.save()

    def settle(self, m: Market, cfg: dict | None) -> None:
        t, m.ticket = m.ticket, None
        if not t:
            return
        try:
            if cfg is None:
                cfg = desk.live_config()
            desk.do_close(cfg, t["coin"], by="zwerm")
            sign = 1 if t["side"] == "long" else -1
            pnl = t["usd"] * sign * (m.px - t["entry"]) / t["entry"] - t["usd"] * 2 * desk.num(cfg, "DESK_FEE_BPS", 4.5) / 1e4
            self.history.append({"t": int(time.time() * 1000), "opened": int(t["opened"] * 1000), "coin": t["coin"], "side": t["side"], "entry": t["entry"],
                                 "exit": m.px, "usd": t["usd"], "pnl": round(pnl, 4), "edge": round(t["edge"], 2)})
            self.log("closer", "AFREK", f"{t['side']} {t['entry']:,.2f} → {m.px:,.2f} (geschat)", f"{pnl:+.2f}", m.coin)
            self.save()
        except Exception as e:
            self.log("closer", "FOUT", f"sluiten mislukt: {e}. Controleer Hyperliquid!", coin=m.coin)

    def risk(self, cfg: dict) -> None:
        """Daglimiet op verlies, gemeten op de accountwaarde, voor alle coins samen."""
        if time.time() - self.last_account < 30:
            return
        self.last_account = time.time()
        try:
            acc = desk.account([desk.dex_of(c) for c in self.coins])
        except Exception:
            return
        if not acc.get("connected"):
            return
        value = acc["value"]
        if self.day != _today():
            self.day, self.day_start, self.halted = _today(), value, None
            self.log("prior", "DAG", f"nieuwe handelsdag, startwaarde ${value:,.2f}")
            self.save()
        limit = desk.num(cfg, "DESK_DAILY_LOSS", 10)
        if self.day_start is not None and not self.halted and self.day_start - value > limit:
            self.halted = f"daglimiet geraakt: ${self.day_start - value:,.2f} verlies (limiet ${limit:g})"
            self.log("closer", "STOP", self.halted)
            self.pending = None
            for m in self.markets.values():
                if m.ticket:
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
        for m in self.markets.values():
            try:
                m.warm_up()
            except Exception as e:
                self.log("prior", "FOUT", f"opwarmen mislukt: {e}", coin=m.coin)
        cfg = desk.config()
        if self.on and desk.status()["auto_allowed"]:
            self.close_orphans(cfg)
        actief = ", ".join(c for c, v in self.coins.items() if v) or "geen coins"
        self.log("prior", "START", f"zwerm gestart ({actief}), {'aan' if self.on else 'uit'}, "
                                   f"bevestigen {'aan' if self.confirm else 'uit'}")
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
            cfg = desk.config() if st["auto_allowed"] else None
            if not st["auto_allowed"] and (self.on or any(m.ticket for m in self.markets.values())):
                self.log("closer", "STOP", "live/DESK_AUTO staat niet meer aan in desk.env")
                self.on = False
                for m in self.markets.values():
                    m.ticket = None
                self.save()
            if self.pending and now > self.pending["expires"]:
                self.log("taker", "NEE", "goedkeuring verlopen", coin=self.pending["coin"])
                self.pending = None
            if cfg:
                self.risk(cfg)
            for m in self.markets.values():
                if not m.px:
                    continue
                if m.last_px and m.var1:
                    r = math.log(m.px / m.last_px)
                    m.var1 = .985 * m.var1 + .015 * r * r
                m.last_px = m.px
                self.new_window(m, now, cfg)
                m.feat = m.features(now)
                m.model = m.compute(m.feat, now, cfg or desk.config())
                if cfg:
                    self.decide(m, now, cfg)

    # ---------- bediening vanaf /desk ----------
    def start(self) -> None:
        if not self.started:
            self.started = True
            threading.Thread(target=self.feed, daemon=True, name="desk-feed").start()
            threading.Thread(target=self.loop, daemon=True, name="desk-swarm").start()

    def set(self, on: bool | None = None, confirm: bool | None = None, coins: dict | None = None) -> None:
        cfg = desk.live_config()
        if not desk.status()["auto_allowed"]:
            raise PermissionError("DESK_AUTO=1 ontbreekt in data/desk.env")
        self.start()
        with self.lock:
            for coin, val in (coins or {}).items():
                coin = desk._coin(coin)
                if coin not in self.markets:  # nieuwe markt uit de scanner
                    m = Market(coin)
                    try:
                        m.warm_up()
                    except Exception as e:
                        self.log("prior", "FOUT", f"opwarmen mislukt: {e}", coin=coin)
                    self.markets[coin] = m
                    self.coins[coin] = False
                    self.resubscribe = True
                if bool(val) != self.coins[coin]:
                    self.coins[coin] = bool(val)
                    m = self.markets[coin]
                    if not val:
                        if self.pending and self.pending["coin"] == coin:
                            self.pending = None
                        if m.ticket:
                            self.settle(m, cfg)
                    self.log("prior", "COIN", "zwerm handelt nu " + ("wel" if val else "niet meer"), coin=coin)
            if confirm is not None and confirm != self.confirm:
                self.confirm = confirm
                self.log("taker", "MODE", "bevestigen per order " + ("aan" if confirm else "uit: de zwerm handelt zelf"))
            if on is not None and on != self.on:
                self.on = on
                if not on:
                    self.pending = None
                    for m in self.markets.values():
                        if m.ticket:
                            self.settle(m, cfg)
                    for c in list(self.coins):  # kill switch: alles dicht
                        try:
                            desk.do_close(cfg, c, by="kill")
                        except Exception as e:
                            self.log("closer", "FOUT", f"sluiten mislukt: {e}", coin=c)
                self.errors = 0
                self.log("closer", "AAN" if on else "UIT", "autonome handel " + ("aan" if on else "uit, alles gesloten"))
            self.save()

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
            self.execute(p, cfg)
            return {"ok": m.ticket is not None}

    def snapshot(self, since: int = 0) -> dict:
        st = desk.status()
        with self.lock:
            markets = {c: {"on": self.coins[c], "px": m.px, "open": m.open0, "ticket": m.ticket,
                           "model": {k: v for k, v in m.model.items() if k != "prior"},
                           "prior": m.model.get("prior", {}), "score": m.feat.get("score")}
                       for c, m in self.markets.items()}
            return {
                "allowed": st["auto_allowed"], "on": self.on, "confirm": self.confirm, "coins": self.coins,
                "halted": self.halted, "feed": self.feed_ok, "markets": markets, "pending": self.pending,
                "day_start": self.day_start, "history": list(self.history)[-50:],
                "trace": [t for t in self.trace if t["id"] > since], "seq": self.seq,
            }


swarm = Swarm()
