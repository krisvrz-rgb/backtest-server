"""Autonome zwerm: dezelfde zes agents als op /desk, maar op de server, zodat hij doorhandelt als de pagina dicht is.

Draait alleen als data/desk.env live handel toestaat én DESK_AUTO=1 bevat. Aan/uit, bevestigen per order
en de coin zet je vanaf /desk (met het desk-token). Harde grenzen:
- grootte per order DESK_ORDER_USD, afgekapt op DESK_MAX_USD;
- één positie tegelijk, altijd gesloten aan het einde van het venster van 5 minuten (reduce-only);
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


class Swarm:
    def __init__(self) -> None:
        saved = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        self.on: bool = saved.get("on", False)
        self.confirm: bool = saved.get("confirm", True)
        self.coin: str = saved.get("coin", "BTC")
        self.day: str = saved.get("day", "")
        self.day_start: float | None = saved.get("day_start")
        self.halted: str | None = saved.get("halted") if saved.get("day") == _today() else None
        self.lock = threading.RLock()
        self.trace: deque = deque(maxlen=300)
        self.seq = 0
        self.history: deque = deque(saved.get("history", []), maxlen=200)
        self.reset_market()
        self.ticket: dict | None = None
        self.pending: dict | None = None
        self.errors = 0
        self.feed_ok = False
        self.last_account = 0.0
        self.started = False

    # ---------- opslag ----------
    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"on": self.on, "confirm": self.confirm, "coin": self.coin, "day": self.day,
                                          "day_start": self.day_start, "halted": self.halted,
                                          "history": list(self.history)}))

    def prior_file(self):
        return DATA_DIR / f"desk-prior-{self.coin}.json"

    def samples(self) -> list:
        f = self.prior_file()
        return json.loads(f.read_text()) if f.exists() else []

    def reset_market(self) -> None:
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

    def log(self, agent: str, act: str, text: str, val: str = "") -> None:
        self.seq += 1
        self.trace.append({"id": self.seq, "t": int(time.time() * 1000), "agent": agent, "act": act, "text": text, "val": val})

    # ---------- marktdata ----------
    def feed(self) -> None:
        from websockets.sync.client import connect

        while True:
            coin = self.coin
            try:
                with connect(WS_URL, open_timeout=15, close_timeout=5) as ws:
                    for sub in ({"type": "trades", "coin": coin}, {"type": "l2Book", "coin": coin}):
                        ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
                    last_ping = time.time()
                    while self.coin == coin:
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
                                    if t["coin"] == coin:
                                        self.px = float(t["px"])
                                        self.trades.append((t["time"] / 1000, float(t["sz"]), t["side"] == "B"))
                        elif ch == "l2Book" and data.get("coin") == coin:
                            self.book = [[(float(l["px"]), float(l["sz"])) for l in side] for side in data["levels"]]
            except Exception as e:  # opnieuw verbinden
                self.feed_ok = False
                self.log("spotter", "FEED", f"verbinding met Hyperliquid weg: {e}")
                time.sleep(5)

    def warm_up(self) -> None:
        """σ per seconde uit de 1m-candles van de afgelopen 3 uur."""
        try:
            cs = desk.candles(self.coin, "1m", 180)
            rs = [math.log(cs[i]["c"] / cs[i - 1]["c"]) for i in range(1, len(cs))]
            self.var1 = sum(r * r for r in rs) / max(1, len(rs)) / 60
            if not self.px and cs:
                self.px = cs[-1]["c"]
        except Exception as e:
            self.log("prior", "FOUT", f"opwarmen mislukt: {e}")

    # ---------- de zwerm ----------
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

    def new_window(self, now: float, cfg: dict | None) -> None:
        win = int(now // WIN) * WIN
        if win == self.win:
            return
        if self.ticket:
            self.settle(cfg)
        if self.sample and self.px:
            y = math.log(self.px / self.sample["px"]) / max(1e-9, self.sample["sd"])
            a = self.samples()
            a.append({"s": round(self.sample["s"], 4), "y": round(y, 4), "t": self.win})
            self.prior_file().write_text(json.dumps(a[-2000:]))
            self.log("prior", "LEER", f"score {self.sample['s']:+.2f} → uitkomst {y:+.2f}σ", f"n={len(a)}")
        self.win, self.open0, self.decided, self.sample = win, self.px, False, None
        if self.pending:
            self.pending = None
            self.log("taker", "NEE", "goedkeuring verlopen: venster voorbij")

    def decide(self, f: dict, m: dict, now: float, cfg: dict) -> None:
        el = now - self.win
        if not self.sample and el >= DECIDE_FROM:
            self.sample = {"s": f["score"], "px": self.px, "sd": m["sd"]}
        if not self.on or self.halted or self.decided or self.ticket or self.pending:
            return
        if el < DECIDE_FROM or el > DECIDE_UNTIL or int(el) % 15:
            return
        min_edge = desk.num(cfg, "DESK_MIN_EDGE", 2)
        self.log("spotter", "SCAN", f"flow {f['flow30']:+.2f} · boek {f['book']:+.2f} · score {f['score']:+.2f}")
        if m["edge"] < min_edge:
            self.log("edge", "PASS", f"{m['side']} {m['pDir'] * 100:.1f}¢ < break-even {m['be'] * 100:.1f}¢ + {min_edge:g}¢",
                     f"{m['edge']:+.1f}¢")
            if el >= DECIDE_UNTIL - 15:
                self.decided = True
            return
        usd = min(desk.num(cfg, "DESK_ORDER_USD", 20), desk.num(cfg, "DESK_MAX_USD", 25))
        self.decided = True
        self.log("edge", "EDGE", f"{m['side']} {m['pDir'] * 100:.1f}¢ vs break-even {m['be'] * 100:.1f}¢", f"{m['edge']:+.1f}¢")
        order = {"id": int(now * 1000), "coin": self.coin, "side": m["side"], "usd": usd, "edge": m["edge"],
                 "p": m["pDir"], "win": self.win, "expires": now + APPROVE_SECONDS}
        if self.confirm:
            self.pending = order
            self.log("taker", "WACHT", f"{m['side']} {self.coin} ${usd:g} wacht op jouw goedkeuring", "mens")
        else:
            self.execute(order, cfg)

    def execute(self, order: dict, cfg: dict) -> None:
        try:
            res = desk.do_open(cfg, order["coin"], order["side"], order["usd"], by="zwerm")
            if not res["ok"]:
                raise RuntimeError(json.dumps(res["result"])[:200])
            self.errors = 0
            self.ticket = {**order, "entry": res["mid"], "size": res["size"], "opened": time.time()}
            self.log("taker", "TAKE", f"LIVE {order['side']} {order['coin']} ${order['usd']:g} @ {res['mid']:,.2f}")
        except Exception as e:
            self.errors += 1
            self.log("taker", "FOUT", f"order mislukt ({self.errors}/3): {e}")
            if self.errors >= 3:
                self.on = False
                self.log("closer", "STOP", "3 mislukte orders op rij: autonome handel uit")
                self.save()

    def settle(self, cfg: dict | None) -> None:
        t, self.ticket = self.ticket, None
        if not t:
            return
        try:
            if cfg is None:
                cfg = desk.live_config()
            desk.do_close(cfg, t["coin"], by="zwerm")
            sign = 1 if t["side"] == "long" else -1
            pnl = t["usd"] * sign * (self.px - t["entry"]) / t["entry"] - t["usd"] * 2 * desk.num(cfg, "DESK_FEE_BPS", 4.5) / 1e4
            self.history.append({"t": int(time.time() * 1000), "coin": t["coin"], "side": t["side"], "entry": t["entry"],
                                 "exit": self.px, "usd": t["usd"], "pnl": round(pnl, 4), "edge": round(t["edge"], 2)})
            self.log("closer", "AFREK", f"{t['side']} {t['entry']:,.2f} → {self.px:,.2f} (geschat)", f"{pnl:+.2f}")
            self.save()
        except Exception as e:
            self.log("closer", "FOUT", f"sluiten mislukt: {e}. Controleer Hyperliquid!")

    def risk(self, cfg: dict) -> None:
        """Daglimiet op verlies, gemeten op de accountwaarde."""
        if time.time() - self.last_account < 30:
            return
        self.last_account = time.time()
        try:
            acc = desk.account()
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
            if self.ticket:
                self.settle(cfg)
            self.save()

    def close_orphans(self, cfg: dict) -> None:
        """Na een herstart: open posities op het aparte sub-account horen bij de zwerm en gaan dicht."""
        try:
            acc = desk.account()
            for p in acc.get("positions", []):
                if p["coin"] in desk.COINS and p["size"]:
                    if cfg.get("HL_SUBACCOUNT"):
                        desk.do_close(cfg, p["coin"], by="zwerm-herstart")
                        self.log("closer", "HERSTEL", f"achtergebleven {p['coin']}-positie gesloten")
                    else:
                        self.log("closer", "LET OP", f"open {p['coin']}-positie op je hoofdaccount; die laat ik staan")
        except Exception as e:
            self.log("closer", "FOUT", f"posities controleren mislukt: {e}")

    def loop(self) -> None:
        self.warm_up()
        cfg = desk.config()
        if self.on and desk.status()["auto_allowed"]:
            self.close_orphans(cfg)
        self.log("prior", "START", f"zwerm gestart op {self.coin}, {'aan' if self.on else 'uit'}, "
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
            if not self.px:
                return
            if self.last_px and self.var1:
                r = math.log(self.px / self.last_px)
                self.var1 = .985 * self.var1 + .015 * r * r
            self.last_px = self.px
            st = desk.status()
            cfg = desk.config() if st["auto_allowed"] else None
            if not st["auto_allowed"] and (self.on or self.ticket):
                self.log("closer", "STOP", "live/DESK_AUTO staat niet meer aan in desk.env")
                self.on = False
                self.ticket = None
                self.save()
            self.new_window(now, cfg)
            if self.pending and now > self.pending["expires"]:
                self.log("taker", "NEE", "goedkeuring verlopen")
                self.pending = None
            self.feat = self.features(now)
            self.model = self.compute(self.feat, now, cfg or desk.config())
            if cfg:
                self.risk(cfg)
                self.decide(self.feat, self.model, now, cfg)

    # ---------- bediening vanaf /desk ----------
    def start(self) -> None:
        if not self.started:
            self.started = True
            threading.Thread(target=self.feed, daemon=True, name="desk-feed").start()
            threading.Thread(target=self.loop, daemon=True, name="desk-swarm").start()

    def set(self, on: bool | None = None, confirm: bool | None = None, coin: str | None = None) -> None:
        cfg = desk.live_config()
        if not desk.status()["auto_allowed"]:
            raise PermissionError("DESK_AUTO=1 ontbreekt in data/desk.env")
        self.start()
        with self.lock:
            if coin and coin.upper() != self.coin:
                coin = desk._coin(coin)
                if self.ticket:
                    self.settle(cfg)
                self.coin = coin
                self.reset_market()
                self.warm_up()
                self.log("prior", "COIN", f"zwerm handelt nu {coin}")
            if confirm is not None and confirm != self.confirm:
                self.confirm = confirm
                self.log("taker", "MODE", "bevestigen per order " + ("aan" if confirm else "uit: de zwerm handelt zelf"))
            if on is not None and on != self.on:
                self.on = on
                if not on:
                    self.pending = None
                    if self.ticket:
                        self.settle(cfg)
                    for c in desk.COINS:  # kill switch: alles dicht
                        try:
                            desk.do_close(cfg, c, by="kill")
                        except Exception as e:
                            self.log("closer", "FOUT", f"{c} sluiten mislukt: {e}")
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
                self.log("taker", "NEE", "order geweigerd door jou")
                return {"ok": True}
            if p["win"] != self.win or time.time() > p["expires"]:
                self.log("taker", "NEE", "te laat goedgekeurd")
                raise ValueError("Te laat: het venster of de bedenktijd is voorbij")
            self.execute(p, cfg)
            return {"ok": self.ticket is not None}

    def snapshot(self, since: int = 0) -> dict:
        st = desk.status()
        with self.lock:
            m, pr = self.model, self.model.get("prior", {})
            return {
                "allowed": st["auto_allowed"], "on": self.on, "confirm": self.confirm, "coin": self.coin,
                "halted": self.halted, "feed": self.feed_ok, "px": self.px, "window": self.win * 1000,
                "open": self.open0, "ticket": self.ticket, "pending": self.pending,
                "model": {k: v for k, v in m.items() if k != "prior"}, "prior": pr, "score": self.feat.get("score"),
                "day_start": self.day_start, "history": list(self.history)[-50:],
                "trace": [t for t in self.trace if t["id"] > since], "seq": self.seq,
            }


swarm = Swarm()
