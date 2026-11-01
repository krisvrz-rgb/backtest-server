"""Swarm Desk terug in de tijd: dezelfde beslisregels als de live zwerm, op 2 jaar 1m-candles per munt.

Wat wel en niet kan (eerlijk):
- Hyperliquid geeft maar ~5000 candles per interval terug (1m = 3,5 dag). Daarom 1m-candles van Binance USDT-perps
  (data.binance.vision, maandbestanden). Die hebben het volume van agressieve kopers (taker buy), zodat de koopdruk
  te benaderen is. Binance-flow is niet de Hyperliquid-flow, maar dezelfde munt beweegt op beide beurzen samen.
- Koopdruk 3 min ('flow30') = taker-kopers min -verkopers over de laatste 3 candles van 1 minuut.
- Koopdruk 30 s ('flow10') = idem over de laatste 1 minuut (30 s bestaat niet in 1m-data).
- Orderboek top 5 / top 20: historisch niet beschikbaar. Weggelaten (waarde 0), dus de score is kleiner;
  Prior leert β per munt toch uit de vensters, dus de schaal regelt zichzelf.
- Momentum, RSI 14 en MACD (5m), steun/weerstand (draaipunten 5m, 2 dagen, elk uur), EMA 20/50/200, trendlijnen,
  SL/TP-plan, ruimte-check, Kelly-grootte, hefboom × vertrouwen × trefzekerheid × staat van dienst, trailing stop,
  doorlopen (verlengen), omdraaien: zoals in desk_auto.py, op het ritme van 1 minuut.
- Niet meegenomen: de stem van de 165 strategieën en de dag-backtests (elk +1¢ drempel), de goedkeuringsknop.
- Maker-instap: gevuld als de koers in de minuten erna ONDER de limiet (long) handelt, niet alleen raakt.
  Stop-loss: marktorder met halve spread erbij. Als SL en TP in dezelfde minuut liggen: SL eerst (voorzichtig).
- Kosten: Hyperliquid maker 1,5 bp, taker 4,5 bp; funding per uur uit Hyperliquid (fundingHistory), anders Binance.

Om overfitting te vermijden: elke instelling draait causaal (alleen verleden), en de keuze per munt gebeurt
walk-forward: kies op alles vóór een kwartaal, meet op dat kwartaal, schuif door. Een nieuwe instelling wordt
alleen VOORGESTELD als ze out-of-sample beter is dan wat de desk nu doet; niets wordt automatisch toegepast.
"""
from __future__ import annotations

import io
import json
import math
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone

import numpy as np

from .data import DATA_DIR

CACHE = DATA_DIR / "desk-bt"
RESULTS = CACHE / "results.json"
BINANCE = "https://data.binance.vision/data/futures/um"
HL_INFO = "https://api.hyperliquid.xyz/info"
DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK", "SUI", "LTC", "DOT", "NEAR",
                 "APT", "ARB", "OP", "WIF", "kPEPE", "HYPE", "ENA"]
WINDOWS = (15, 60, 240)
EDGES = (0.0, 2.0, 4.0)
MODELS = ("vast", "geleerd")
TRAILS = (True, False)
BASELINE = {"window": 60, "min_edge": 2.0, "model": "geleerd", "trail": True}  # wat de desk nu standaard doet
W_DEFAULT = {"flow30": .35, "flow10": .2, "book": .25, "bookDeep": .1, "mom": .1}
START_EQ = 1000.0
MIN_TRADES = 20      # minder trades in de leerperiode: telt niet als bewezen
DAY = 86_400_000


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def bn_symbol(coin: str) -> str:
    """Hyperliquid-naam → Binance USDT-perp. kPEPE = 1000 PEPE op beide beurzen."""
    return ("1000" + coin[1:] if coin.startswith("k") and coin[1:].isupper() else coin).upper() + "USDT"


# ---------- data ----------
def _client():
    import httpx
    return httpx.Client(follow_redirects=True)


def _get(url: str, client) -> bytes | None:
    for attempt in range(4):
        try:
            r = client.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * 2 ** attempt)
    return None


def _parse(raw: bytes) -> np.ndarray:
    """Binance kline-CSV (met of zonder kopregel) → kolommen t, o, h, l, c, v, taker-buy-v."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        txt = z.read(z.namelist()[0]).decode()
    rows = []
    for line in txt.splitlines():
        p = line.split(",")
        if len(p) < 10 or not p[0][:1].isdigit():
            continue
        rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[9])))
    a = np.array(rows, dtype=float) if rows else np.zeros((0, 7))
    if len(a) and a[0, 0] > 1e14:  # sommige recente bestanden in microseconden
        a[:, 0] /= 1000
    return a


def _months(n: int) -> list[str]:
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(f"{y:04d}-{m:02d}")
    return out[::-1]


def load_klines(coin: str, months: int = 24, client=None) -> np.ndarray:
    """1m-candles van de afgelopen `months` volle maanden + de dagen van deze maand. Gecachet per maand/dag."""
    sym = bn_symbol(coin)
    d = CACHE / sym
    d.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or _client()
    parts = []
    try:
        for mo in _months(months):
            f = d / f"{mo}.npy"
            if not f.exists():
                raw = _get(f"{BINANCE}/monthly/klines/{sym}/1m/{sym}-1m-{mo}.zip", client)
                np.save(f, _parse(raw) if raw else np.zeros((0, 7)))
            parts.append(np.load(f))
        now = datetime.now(timezone.utc)
        for day in range(1, now.day):
            ds = f"{now.year:04d}-{now.month:02d}-{day:02d}"
            f = d / f"{ds}.npy"
            if not f.exists():
                raw = _get(f"{BINANCE}/daily/klines/{sym}/1m/{sym}-1m-{ds}.zip", client)
                if raw is None:
                    continue  # nog niet gepubliceerd: volgende keer opnieuw proberen
                np.save(f, _parse(raw))
            parts.append(np.load(f))
    finally:
        if own:
            client.close()
    a = np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.zeros((0, 7))
    if len(a):
        a = a[np.argsort(a[:, 0], kind="stable")]
        a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
    return a


def load_funding(coin: str, start_ms: int, client=None) -> dict[int, float]:
    """Funding per uur {uur-index: rente} uit Hyperliquid; ontbreekt die, Binance (8-uurs rente / 8 per uur)."""
    f = CACHE / f"funding-{coin}.json"
    have: dict[int, float] = {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}
    own = client is None
    client = client or _client()
    try:
        t = max([start_ms] + [(h + 1) * 3_600_000 for h in have])
        end = int(time.time() * 1000)
        while t < end - 3_600_000:
            try:
                r = client.post(HL_INFO, json={"type": "fundingHistory", "coin": coin, "startTime": t}, timeout=30)
                r.raise_for_status()
                rows = r.json()
            except Exception:
                break
            if not rows:
                break
            for x in rows:
                have[int(x["time"] // 3_600_000)] = float(x["fundingRate"])
            nt = int(rows[-1]["time"]) + 1
            if nt <= t:
                break
            t = nt
            time.sleep(.25)  # Hyperliquid geeft 429 bij te veel vragen
        if not have or min(have) * 3_600_000 > start_ms + 7 * DAY:  # gat aan het begin: Binance ervoor
            sym = bn_symbol(coin)
            first = min(have) if have else 10 ** 12
            for mo in _months(25):
                raw = _get(f"{BINANCE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{mo}.zip", client)
                if not raw:
                    continue
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    txt = z.read(z.namelist()[0]).decode()
                for line in txt.splitlines():
                    p = line.split(",")
                    if len(p) >= 3 and p[0][:1].isdigit():
                        h0 = int(float(p[0]) // 3_600_000)
                        for k in range(8):  # 8-uurs rente uitsmeren over de 8 uur ervoor
                            hk = h0 - k
                            if hk < first and hk not in have:
                                have[hk] = float(p[2]) / 8
    finally:
        if own:
            client.close()
    f.write_text(json.dumps({str(k): v for k, v in sorted(have.items())}))
    return have


# ---------- voorbereiden: alles wat niet van de instellingen afhangt ----------
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    a, out, e = 2 / (n + 1), np.empty(len(x)), x[0]
    for i, v in enumerate(x):
        e += a * (v - e)
        out[i] = e
    out[:n] = np.nan
    return out


def _rsi(cl: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(cl, prepend=cl[0])
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    out = np.full(len(cl), np.nan)
    if len(cl) < n + 2:
        return out
    u, w = up[1:n + 1].mean(), dn[1:n + 1].mean()
    for i in range(n + 1, len(cl)):
        u, w = (u * (n - 1) + up[i]) / n, (w * (n - 1) + dn[i]) / n
        out[i] = 100.0 if w == 0 else 100 - 100 / (1 + u / w)
    return out


def _macd_hist(cl: np.ndarray) -> np.ndarray:
    e12, e26 = np.empty(len(cl)), np.empty(len(cl))
    a, b = cl[0], cl[0]
    for i, c in enumerate(cl):
        a += 2 / 13 * (c - a)
        b += 2 / 27 * (c - b)
        e12[i], e26[i] = a, b
    m = e12 - e26
    sig = np.empty(len(cl))
    s = m[0]
    for i, v in enumerate(m):
        s = s + 2 / 10 * (v - s) if i else v
        sig[i] = s
    out = m - sig
    out[:40] = np.nan
    return out


def prepare(a: np.ndarray, funding: dict[int, float] | None = None) -> dict:
    """Ruwe 1m-candles → raster van volle minuten (vanaf middernacht UTC) met alle kenmerken die niet van de
    instellingen afhangen. Ontbrekende minuten: vorige slotkoers, geen volume."""
    t0 = int(a[0, 0] // DAY * DAY) + DAY  # eerste volle dag
    a = a[a[:, 0] >= t0]
    n = int((a[-1, 0] - t0) // 60_000) + 1
    idx = ((a[:, 0] - t0) // 60_000).astype(int)
    c = np.full(n, np.nan)
    c[idx] = a[:, 4]
    filled = ~np.isnan(c)
    last = np.maximum.accumulate(np.where(filled, np.arange(n), 0))
    c = c[last]
    o, h, lo = c.copy(), c.copy(), c.copy()
    o[idx], h[idx], lo[idx] = a[:, 1], a[:, 2], a[:, 3]
    v, tb = np.zeros(n), np.zeros(n)
    v[idx], tb[idx] = a[:, 5], a[:, 6]
    sell = v - tb
    f1 = np.where(v > 0, (tb - sell) / np.where(v > 0, v, 1), 0.0)
    v3 = np.convolve(v, np.ones(3))[:n]
    tb3 = np.convolve(tb, np.ones(3))[:n]
    f3 = np.where(v3 > 0, (2 * tb3 - v3) / np.where(v3 > 0, v3, 1), 0.0)
    # 5m-candles
    n5 = n // 5
    c5 = c[4:n5 * 5:5]
    h5 = h[:n5 * 5].reshape(n5, 5).max(1)
    l5 = lo[:n5 * 5].reshape(n5, 5).min(1)
    r5 = np.diff(np.log(c5), prepend=np.log(c5[0]))
    ema = {k: _ema(c5, k) for k in (20, 50, 200)}
    rsi5 = _rsi(c5)
    macd5 = _macd_hist(c5)
    # draaipunten op 5m (k = 6): bevestigd 6 candles later
    k = 6
    from numpy.lib.stride_tricks import sliding_window_view as swv
    hi_piv = np.zeros(n5, bool)
    lo_piv = np.zeros(n5, bool)
    if n5 > 2 * k + 1:
        hi_piv[k:n5 - k] = h5[k:n5 - k] >= swv(h5, 2 * k + 1).max(1)
        lo_piv[k:n5 - k] = l5[k:n5 - k] <= swv(l5, 2 * k + 1).min(1)
    piv_hi, piv_lo = np.flatnonzero(hi_piv), np.flatnonzero(lo_piv)
    rng = h5 - l5
    # per uur (zoals level_refresher): niveaus, trendlijnen en σ uit de 5m-candles van de 2 dagen ervoor
    nh = n // 60 + 1
    hours = []
    span = 2 * 24 * 12
    for H in range(nh):
        e = min(H * 12, n5)  # 5m-candles die af zijn bij het begin van dit uur
        s = max(0, e - span)
        if e - s < 2 * k + 5:
            hours.append(None)
            continue
        rr = r5[s + 1:e]
        var1 = float(np.mean(rr * rr)) / 300 if len(rr) else 0.0
        tol = max(float(np.median(rng[s:e])) * .8, float(c5[e - 1]) * 2e-4)
        ph = piv_hi[(piv_hi >= s + k) & (piv_hi < e - k)]
        pl = piv_lo[(piv_lo >= s + k) & (piv_lo < e - k)]
        pts = sorted([(float(h5[j]), j) for j in ph] + [(float(l5[j]), j) for j in pl])
        lv: list[list[float]] = []
        for px, _ in pts:
            if lv and px - lv[-1][2] <= 1.5 * tol:
                lv[-1][0] += px
                lv[-1][1] += 1
            else:
                lv.append([px, 1, px])
        levels = [(s_ / nn, int(nn)) for s_, nn, _ in lv]
        tr = []
        if len(pl) >= 2 and l5[pl[-1]] > l5[pl[-2]]:
            tr.append((int(pl[-2]), float(l5[pl[-2]]), int(pl[-1]), float(l5[pl[-1]])))
        if len(ph) >= 2 and h5[ph[-1]] < h5[ph[-2]]:
            tr.append((int(ph[-2]), float(h5[ph[-2]]), int(ph[-1]), float(h5[ph[-1]])))
        hours.append({"var1": var1, "levels": levels, "trends": tr})
    fund = np.zeros(nh + 1)
    if funding:
        h0 = t0 // 3_600_000
        for hh, rate in funding.items():
            j = hh - h0
            if 0 <= j < nh:
                fund[j] = rate
    return {"t0": t0, "n": n, "o": o.tolist(), "h": h.tolist(), "l": lo.tolist(), "c": c.tolist(),
            "f1": f1.tolist(), "f3": f3.tolist(), "c5": c5.tolist(),
            "ema": {k_: np.nan_to_num(v_, nan=0.0).tolist() for k_, v_ in ema.items()},
            "rsi5": np.nan_to_num(rsi5, nan=50.0).tolist(), "macd5": np.nan_to_num(macd5, nan=0.0).tolist(),
            "hours": hours, "fundcum": np.concatenate([[0.0], np.cumsum(fund)]).tolist(), "has_funding": bool(funding)}


# ---------- geleerd model: logistische regressie, walk-forward per munt ----------
class Learner:
    """Zoals desk_learn: L2-logistische regressie op de kenmerken bij 15% van elk venster. Elke 30 dagen opnieuw
    getraind op alles daarvoor, en alleen gebruikt als ze op de nieuwste 30% beter voorspelt dan het vaste model."""

    def __init__(self) -> None:
        self.rows: list[tuple[list[float], float, float]] = []
        self.w: np.ndarray | None = None
        self.next_train = 0

    def add(self, f: list[float], y: float, p_base: float) -> None:
        self.rows.append((f, 1.0 if y > 0 else 0.0, p_base))

    def maybe_train(self, t_ms: int) -> None:
        if t_ms < self.next_train:
            return
        self.next_train = t_ms + 30 * DAY
        from .desk_learn import MARGIN, MIN_ROWS, _fit, _ll
        if len(self.rows) < MIN_ROWS:
            return
        X = np.array([r[0] + [1.0] for r in self.rows])
        y = np.array([r[1] for r in self.rows])
        b = np.array([r[2] for r in self.rows])
        cut = int(len(X) * .7)
        w = _fit(X[:cut], y[:cut])
        pte = 1 / (1 + np.exp(-(X[cut:] @ w)))
        ll_l, ll_b = _ll(pte, y[cut:]), _ll(b[cut:], y[cut:])
        acc_l = float(np.mean((pte > .5) == (y[cut:] > .5)))
        acc_b = float(np.mean((b[cut:] > .5) == (y[cut:] > .5)))
        self.w = _fit(X, y) if (ll_l < ll_b - MARGIN and acc_l >= acc_b) else None

    def predict(self, f: list[float]) -> float | None:
        if self.w is None:
            return None
        z = float(np.dot(self.w[:-1], f) + self.w[-1])
        return 1 / (1 + math.exp(-_clamp(z, -30, 30)))




# ---------- simulatie van één munt met één instelling ----------
class Sim:
    """Loopt minuut per minuut door de candles met de regels van desk_auto.Swarm: decide, kelly, manage,
    try_hold en settle. Eén munt, eigen papierportefeuille van $1000."""

    def __init__(self, P: dict, window: int = 60, min_edge: float = 2.0, model: str = "geleerd", trail: bool = True,
                 hold_max: int = 3, maker_bps: float = 1.5, taker_bps: float = 4.5, half_spread_bps: float = 1.0,
                 max_lev: float = 10, risk_pct: float = 10, kelly_frac: float = .25, max_margin_pct: float = 5,
                 beta0: float = .25, k_prior: float = 30, trail_pct: float = 50, weights: dict | None = None,
                 wname: str | None = None) -> None:
        self.P, self.W, self.min_edge, self.trail_on, self.hold_max = P, window, min_edge, trail, hold_max
        self.maker, self.taker, self.hs = maker_bps, taker_bps, half_spread_bps / 1e4
        self.max_lev, self.risk_pct, self.kf, self.mpct = max_lev, risk_pct, kelly_frac, max_margin_pct
        self.beta0, self.k, self.trail_pct, self.w = beta0, k_prior, trail_pct, weights or W_DEFAULT
        self.eq = START_EQ
        self.trades: list[dict] = []
        self.samples: deque = deque()
        self.sy = self.ss = 0.0
        self.hits = 0
        self.maker_hist: deque = deque(maxlen=50)
        self.share = .5
        self.learner = Learner() if model == "geleerd" else None
        self.lead = max(1, int(min(.25 * window, 10)))
        self.nwait = max(1, int(min(.1 * window * 60, 300) // 60))
        self.step = max(1, window // 40)

    # --- model, zoals Market.features / prior / compute ---
    def rcost(self) -> float:
        return self.maker + self.share * self.maker + (1 - self.share) * self.taker

    def feats(self, j: int, a: int, hr: dict) -> dict:
        P, w = self.P, self.w
        C = P["c"]
        var1, open0 = hr["var1"], C[a - 1]
        el = max(1.0, (j - a + 1) * 60)
        mom = math.log(C[j] / open0) / math.sqrt(var1 * el) if open0 and var1 else 0.0
        i5 = (j + 1) // 5 - 1
        s5 = C[j] * math.sqrt(var1 * 300)
        f = {"flow30": P["f3"][j], "flow10": P["f1"][j], "mom": mom,
             "rsi": (P["rsi5"][i5] - 50) / 50 if i5 >= 0 else 0.0,
             "macd": _clamp(P["macd5"][i5] / s5 / 2, -1, 1) if i5 >= 0 and s5 else 0.0}
        e50 = P["ema"][50][i5] if i5 >= 0 else 0.0  # trend: afstand tot EMA 50 (5m) in σ van 4 uur
        f["trend"] = _clamp((C[j] / e50 - 1) / math.sqrt(var1 * 14400) / 2, -1, 1) if e50 and var1 else 0.0
        f["score"] = _clamp(w.get("flow30", 0) * f["flow30"] + w.get("flow10", 0) * f["flow10"]
                            + w.get("mom", 0) * _clamp(mom / 2, -1, 1) + w.get("rsi", 0) * f["rsi"]
                            + w.get("macd", 0) * f["macd"] + w.get("trend", 0) * f["trend"], -1, 1)
        return f

    @staticmethod
    def fvec(f: dict) -> list[float]:
        return [f["flow30"], f["flow10"], 0.0, 0.0, _clamp(f["mom"] / 2, -1, 1), f["rsi"], f["macd"], 0.0]

    def beta(self) -> tuple[float, int, float | None]:
        n = len(self.samples)
        bhat = _clamp(self.sy / self.ss, -1, 1) if self.ss > 1e-9 else 0.0
        return _clamp((n * bhat + self.k * self.beta0) / (n + self.k), -1, 1), n, (self.hits / n if n else None)

    def compute(self, f: dict, left_s: float, hr: dict) -> dict:
        sd = math.sqrt(max(hr["var1"], 1e-12) * max(1.0, left_s))
        b, n, hit = self.beta()
        p = _phi(b * f["score"])
        if self.learner is not None:
            pl = self.learner.predict(self.fvec(f))
            if pl is not None:
                p = pl
        e_move = sd * math.sqrt(2 / math.pi) * 1e4
        be = .5 + self.rcost() / (2 * max(e_move, .01))
        pd = max(p, 1 - p)
        return {"p": p, "pDir": pd, "side": 1 if p >= .5 else -1, "be": be, "edge": (pd - be) * 100, "sd": sd, "n": n, "hit": hit}

    # --- structuur: steun/weerstand, EMA's, trendlijnen ---
    def levels(self, j: int, hr: dict) -> list:
        i5 = (j + 1) // 5 - 1
        out = list(hr["levels"])
        if i5 >= 0:
            for k_ in (20, 50, 200):
                v = self.P["ema"][k_][i5]
                if v:
                    out.append((v, 2))
        return out

    @staticmethod
    def wall(px: float, lv: list, side: int):
        if side > 0:
            ab = [x for x in lv if x[0] > px]
            return min(ab) if ab else None
        bl = [x for x in lv if x[0] < px]
        return max(bl) if bl else None

    def plan(self, j: int, side: int, px: float, sd: float, hr: dict) -> dict:
        """Zoals Swarm.exit_plan: SL net voorbij steun/weerstand/trendlijn binnen 0,4..2σ, anders 1σ;
        TP net voor het volgende niveau binnen 0,5..3σ, anders 1,5σ."""
        sd = max(sd, 1e-5)
        cand = self.levels(j, hr)
        x5 = (j + 1) / 5
        for a5, p1, b5, p2 in hr["trends"]:
            cand.append((p1 + (p2 - p1) * (x5 - a5) / max(1, b5 - a5), 2))
        d = [(side * (x[0] - px) / px / sd, x[0]) for x in cand]
        back = [x for x in d if -2.0 <= x[0] <= -.4]
        front = [x for x in d if .5 <= x[0] <= 3.0]
        sl = max(back)[1] * (1 - side * .15 * sd) if back else px * (1 - side * sd)
        tp = min(front)[1] * (1 - side * .05 * sd) if front else px * (1 + side * 1.5 * sd)
        sl_d, tp_d = abs(px - sl) / px, abs(tp - px) / px
        return {"sl": sl, "tp": tp, "sl_d": sl_d, "rr": tp_d / max(sl_d, 1e-9)}

    # --- grootte, zoals Swarm.trust / kelly ---
    def trust(self) -> float:
        n = len(self.trades)
        t = .3 + .7 * min(1.0, n / 200)
        if n >= 20 and sum(x["pnl"] for x in self.trades[-50:]) < 0:
            t *= .5
        return _clamp(t, .1, 1.0)

    def size(self, mo: dict, sl_d: float) -> tuple[int, float] | None:
        sd, eq = max(mo["sd"], 1e-5), self.eq
        conf = _clamp(mo["edge"] / 8, .25, 1.0)
        hit = mo["hit"] or .5
        skill = .25 + .75 * max(0.0, (hit - .5) * 2) * mo["n"] / (mo["n"] + 75)
        want_risk = math.floor(self.risk_pct / 100 / max(sl_d, 1e-6))
        lev = int(_clamp(max(1, math.floor(want_risk * conf * skill * self.trust())), 1, max(1, self.max_lev)))
        mu = (2 * mo["pDir"] - 1) * sd * math.sqrt(2 / math.pi) - self.rcost() / 1e4
        kn = max(0.0, self.kf * mu / (sd * sd)) * eq
        cap_margin = min(eq * self.mpct / 100, eq / 4)
        margin = min(kn / lev, cap_margin)
        if 0 < margin * lev < 10 <= cap_margin * lev:  # Hyperliquid: minstens $10
            margin = 10 / lev
        usd = min(margin * lev, eq * .5)
        return (lev, usd) if usd >= 10 else None

    # --- afrekenen, zoals Swarm.settle (+ funding) ---
    def close(self, tk: dict, j: int, px: float, maker: bool, reason: str) -> None:
        side, usd, e = tk["side"], tk["usd"], tk["entry"]
        fees = usd * (self.maker + (self.maker if maker else self.taker)) / 1e4
        FC = self.P["fundcum"]
        hf, hx = (tk["j_fill"] + 1) // 60, (j + 1) // 60
        fund = side * usd * (FC[min(hx, len(FC) - 1)] - FC[min(hf, len(FC) - 1)]) if hx > hf else 0.0
        gross = usd * side * (px - e) / e
        pnl = gross - fees - fund
        eq0 = self.eq
        self.eq += pnl
        self.maker_hist.append(maker)
        if len(self.maker_hist) >= 10:
            self.share = sum(self.maker_hist) / len(self.maker_hist)
        t0 = self.P["t0"]
        self.trades.append({"t": t0 + (tk["j_fill"] + 1) * 60_000, "tx": t0 + (j + 1) * 60_000, "side": side,
                            "entry": e, "exit": px, "usd": round(usd, 2), "lev": tk["lev"], "pnl": pnl, "gross": gross,
                            "fees": fees, "funding": fund, "ret": pnl / eq0, "bp": pnl / usd * 1e4, "reason": reason,
                            "maker": maker, "edge": round(tk["edge"], 2), "holds": tk["holds"]})

    def exit_px(self, tk: dict, px: float) -> float:
        return px * (1 - tk["side"] * self.hs)  # markt: de halve spread kwijt

    # --- een open ticket per minuut, zoals Swarm.manage ---
    def manage(self, tk: dict, j: int) -> dict | None:
        P = self.P
        O, H, L, C = P["o"], P["h"], P["l"], P["c"]
        side, W = tk["side"], self.W
        if tk["state"] == "entering":
            if j > tk["dec"] + self.nwait:
                return None  # niet gevuld binnen 10% van het venster: geen trade
            if (L[j] < tk["limit"]) if side > 0 else (H[j] > tk["limit"]):
                hr = P["hours"][(j + 1) // 60] or P["hours"][(tk["dec"] + 1) // 60]
                pl = self.plan(j, side, tk["limit"], tk["sd"], hr)
                tk.update(state="open", entry=tk["limit"], j_fill=j, sl=pl["sl"], tp=pl["tp"], best=tk["limit"],
                          trailed=False, against=False)
                if (L[j] <= tk["sl"]) if side > 0 else (H[j] >= tk["sl"]):  # zelfde minuut door de stop
                    self.close(tk, j, self.exit_px(tk, tk["sl"]), False, "SL")
                    return None
            return tk
        # in de minuut: stop-loss eerst (voorzichtig), dan take-profit (limiet = maker), dan de uitstap-limiet
        if (L[j] <= tk["sl"]) if side > 0 else (H[j] >= tk["sl"]):
            px = min(tk["sl"], O[j]) if side > 0 else max(tk["sl"], O[j])
            self.close(tk, j, self.exit_px(tk, px), False, "trailing" if tk["trailed"] else "SL")
            return None
        if (H[j] > tk["tp"]) if side > 0 else (L[j] < tk["tp"]):
            self.close(tk, j, tk["tp"], True, "TP")
            return None
        if tk["state"] == "exiting" and ((H[j] > tk["q"]) if side > 0 else (L[j] < tk["q"])):
            self.close(tk, j, tk["q"], True, tk["reason"])
            return None
        # aan het einde van de minuut
        e = tk["entry"]
        tk["best"] = max(tk["best"], H[j]) if side > 0 else min(tk["best"], L[j])
        if self.trail_on:
            gain = side * (tk["best"] - e) / e
            if gain * 1e4 >= 2 * self.rcost():
                new = e + (tk["best"] - e) * self.trail_pct / 100
                if side * (new - tk["sl"]) > 0:
                    tk["sl"], tk["trailed"] = new, True
        if tk["state"] == "exiting":
            if j >= tk["deadline"]:
                self.close(tk, j, self.exit_px(tk, C[j]), False, tk["reason"])
                return None
            tk["q"] = C[j]  # bijstellen naar de nieuwe beste prijs
            return tk
        a = (j // W) * W
        end = a + W
        hr = P["hours"][(j + 1) // 60]
        if hr is None:
            return tk
        if j - tk["j_fill"] >= 1:  # het model draait om: twee minuten na elkaar ≥ 2¢ de andere kant op
            mo = self.compute(self.feats(j, a, hr), (end - j - 1) * 60, hr)
            against = mo["side"] != side and mo["edge"] >= 2
            if against and tk["against"]:
                tk.update(state="exiting", q=C[j], deadline=j + 1, reason="omgedraaid")
                return tk
            tk["against"] = against
        if end - (j + 1) <= self.lead and tk.get("hold_win") != end:
            if self.try_hold(tk, j, a, hr):
                tk["hold_win"], tk["holds"] = end, tk["holds"] + 1
            else:
                tk.update(state="exiting", q=C[j], deadline=end - 1, reason="venster")
        return tk

    def try_hold(self, tk: dict, j: int, a: int, hr: dict) -> bool:
        """Zoals Swarm.try_hold: zou hij nu opnieuw dezelfde kant op instappen voor een vol venster?"""
        if tk["holds"] >= self.hold_max:
            return False
        mo = self.compute(self.feats(j, a, hr), self.W * 60, hr)
        if mo["side"] != tk["side"] or mo["edge"] < self.min_edge:
            return False
        wl = self.wall(self.P["c"][j], self.levels(j, hr), tk["side"])
        return wl is None or abs(wl[0] - self.P["c"][j]) / self.P["c"][j] / max(mo["sd"], 1e-5) >= .3

    def learn_window(self, sample: dict, px_end: float, t_ms: int) -> None:
        """Einde venster: LEER-venster voor Prior (β) en voor het geleerde model."""
        y = math.log(px_end / sample["px"]) / max(1e-9, sample["sd"])
        hit = (sample["s"] > 0) == (y > 0)
        self.samples.append((sample["s"], y, hit))
        self.sy += sample["s"] * y
        self.ss += sample["s"] ** 2
        self.hits += hit
        if len(self.samples) > 2000:
            s0, y0, h0 = self.samples.popleft()
            self.sy -= s0 * y0
            self.ss -= s0 * s0
            self.hits -= h0
        if self.learner is not None:
            self.learner.add(sample["fv"], y, sample["p"])
            self.learner.maybe_train(t_ms)

    def run(self) -> dict:
        P, W = self.P, self.W
        C, HRS, N, t0 = P["c"], P["hours"], P["n"], P["t0"]
        lo_el, hi_el = .15 * W, .67 * W
        tk: dict | None = None
        sample = None
        for wi in range(1, N // W):
            a, end = wi * W, wi * W + W
            if sample:
                self.learn_window(sample, C[a - 1], t0 + a * 60_000)
            if tk and tk.get("hold_win") != a:  # venster voorbij en niet verlengd
                if tk["state"] != "entering":
                    self.close(tk, a - 1, self.exit_px(tk, C[a - 1]), False, tk.get("reason") or "venster")
                tk = None
            decided, sample, first = False, None, None
            for j in range(a, end):
                el = j - a + 1
                if tk is None and sample is not None and (decided or el > hi_el):
                    break
                hr = HRS[(j + 1) // 60]
                if hr is None:
                    break
                if tk is not None:
                    tk = self.manage(tk, j)
                if sample is None and el >= lo_el:  # LEER-venster: ook als er een trade open staat
                    f = self.feats(j, a, hr)
                    mo = self.compute(f, (W - el) * 60, hr)
                    sample = {"s": f["score"], "px": C[j], "sd": mo["sd"], "fv": self.fvec(f), "p": mo["p"]}
                if tk is not None or decided or el < lo_el or el > hi_el:
                    continue
                first = el if first is None else first
                if (el - first) % self.step:
                    continue
                f = self.feats(j, a, hr)
                mo = self.compute(f, (W - el) * 60, hr)
                last_eval = el + self.step > hi_el
                if mo["edge"] < self.min_edge:
                    decided = last_eval
                    continue
                px, side, sd = C[j], mo["side"], max(mo["sd"], 1e-5)
                wl = self.wall(px, self.levels(j, hr), side)
                if wl is not None and abs(wl[0] - px) / px / sd < .3:  # geen ruimte tot steun/weerstand
                    decided = last_eval
                    continue
                if self.plan(j, side, px, sd, hr)["rr"] < .7:
                    decided = last_eval
                    continue
                decided = True
                pl = self.plan(j, side, px, sd, hr)
                sz = self.size(mo, pl["sl_d"])
                if sz:
                    tk = {"state": "entering", "side": side, "limit": px, "lev": sz[0], "usd": sz[1],
                          "edge": mo["edge"], "sd": sd, "dec": j, "holds": 0}
        if tk and tk["state"] != "entering":
            self.close(tk, N - 1, C[N - 1], False, "einde data")
        return {"trades": self.trades, "eq": self.eq, "hit": self.hits / len(self.samples) if self.samples else None,
                "n_windows": len(self.samples), "learned": self.learner is not None and self.learner.w is not None,
                "beta": self.beta()[0]}


# ---------- beoordelen: statistiek, walk-forward en voorstel per munt ----------
def config_key(c: dict) -> str:
    k = f"{c['window']}m·{c['min_edge']:+g}¢·{c['model']}·{'trail' if c['trail'] else 'vast-SL'}"
    return k + f"·{c['wname']}" if c.get("wname") else k


# Gewichten van de score om te doorzoeken. Het teken hoeft niet: Prior leert β per munt, ook negatief
# (tegendraads). Het orderboek ontbreekt historisch, dus dat zit er niet in.
WEIGHT_SETS = {
    "desk": {"flow30": .35, "flow10": .2, "mom": .1},
    "koopdruk3m": {"flow30": 1.0},
    "koopdruk1m": {"flow10": 1.0},
    "koopdruk": {"flow30": .6, "flow10": .4},
    "momentum": {"mom": 1.0},
    "rsi": {"rsi": 1.0},
    "macd": {"macd": 1.0},
    "trend": {"trend": 1.0},
    "mom+trend": {"mom": .5, "trend": .5},
    "rsi+macd": {"rsi": .5, "macd": .5},
    "flow+mom": {"flow30": .5, "mom": .5},
    "flow−rsi": {"flow30": .5, "rsi": -.5},  # koopdruk mee, overkocht tegen
    "trend−rsi": {"trend": .5, "rsi": -.5},  # trend volgen, maar instappen op een terugval
    "alles": {"flow30": .2, "flow10": .1, "mom": .2, "rsi": .15, "macd": .15, "trend": .2},
}


def weight_grid() -> list[dict]:
    """Elke gewichtenset op 60m en 240m (15m verliest op kosten), drempel +2¢, vast model, vaste SL."""
    return [{"window": w, "min_edge": 2.0, "model": "vast", "trail": False, "wname": n, "weights": WEIGHT_SETS[n]}
            for w in (60, 240) for n in WEIGHT_SETS]


def grid() -> list[dict]:
    return [{"window": w, "min_edge": e, "model": m, "trail": t} for w in WINDOWS for e in EDGES for m in MODELS for t in TRAILS]


def stats(trades: list[dict], t_from: int = 0, t_to: int = 1 << 62) -> dict:
    tr = [x for x in trades if t_from <= x["t"] < t_to]
    n = len(tr)
    if not n:
        return {"n": 0, "t": None, "ret": 0.0, "win": None, "bp": None, "pf": None, "dd": 0.0, "fees": 0.0, "funding": 0.0, "gross": 0.0}
    eq, peak, dd = 1.0, 1.0, 0.0
    for x in tr:
        eq *= 1 + x["ret"]
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
    gw = sum(x["pnl"] for x in tr if x["pnl"] > 0)
    gl = -sum(x["pnl"] for x in tr if x["pnl"] < 0)
    usd = sum(x["usd"] for x in tr)
    bps = [x["bp"] for x in tr]
    mean = sum(bps) / n
    sdv = math.sqrt(sum((b - mean) ** 2 for b in bps) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "t": round(mean / (sdv / math.sqrt(n)), 2) if sdv else None, "ret": round((eq - 1) * 100, 3), "win": round(sum(1 for x in tr if x["pnl"] > 0) / n * 100, 1),
            "bp": round(sum(x["pnl"] for x in tr) / usd * 1e4, 2) if usd else None,
            "gross_bp": round(sum(x["gross"] for x in tr) / usd * 1e4, 2) if usd else None,
            "pf": round(gw / gl, 2) if gl else None, "dd": round(dd * 100, 2),
            "fees": round(sum(x["fees"] for x in tr), 2), "funding": round(sum(x["funding"] for x in tr), 2),
            "gross": round(sum(x["gross"] for x in tr), 2), "maker": round(sum(1 for x in tr if x["maker"]) / n * 100, 1),
            "lev": round(sum(x["lev"] for x in tr) / n, 1), "usd": round(usd / n, 1),
            "reasons": {k: sum(1 for x in tr if x["reason"] == k) for k in sorted({x["reason"] for x in tr})}}


def quarters(t_start: int, t_end: int) -> list[tuple[int, int]]:
    d = datetime.fromtimestamp(t_start / 1000, timezone.utc)
    y, q = d.year, (d.month - 1) // 3
    out = []
    while True:
        a = int(datetime(y, q * 3 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        q += 1
        if q == 4:
            y, q = y + 1, 0
        b = int(datetime(y, q * 3 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        if a >= t_end:
            break
        out.append((max(a, t_start), min(b, t_end)))
    return out


def pick(runs: dict[str, list[dict]], t_to: int, t_from: int = 0) -> tuple[str | None, float]:
    """Beste instelling op [t_from, t_to): hoogste rendement met minstens MIN_TRADES trades. Niets positief: None = niet handelen."""
    best, best_ret = None, 0.0
    for k, tr in runs.items():
        s = stats(tr, t_from, t_to)
        if s["n"] >= MIN_TRADES and s["ret"] > best_ret:
            best, best_ret = k, s["ret"]
    return best, best_ret


def evaluate(coin: str, runs: dict[str, list[dict]], cfgs: dict[str, dict], t_start: int, t_end: int,
             warmup_days: int = 180) -> dict:
    """Walk-forward: per kwartaal kiezen op alles daarvoor (minstens `warmup_days`), meten op dat kwartaal."""
    base_k = config_key(BASELINE)
    qs = [q for q in quarters(t_start, t_end) if q[0] >= t_start + warmup_days * DAY]
    oos_from = qs[0][0] if qs else t_end
    chain: list[dict] = []
    steps = []
    for a, b in qs:
        k, r_is = pick(runs, a)
        tr = [x for x in runs[k] if a <= x["t"] < b] if k else []
        chain += tr
        steps.append({"from": a, "to": b, "pick": k, "is_ret": r_is, "oos": stats(tr), "base": stats(runs[base_k], a, b)})
    wf, base = stats(chain), stats(runs[base_k], oos_from, t_end)
    final_k, final_ret = pick(runs, t_end)
    table = []
    for k, tr in runs.items():
        table.append({"key": k, **cfgs[k], "all": stats(tr, t_start, t_end), "is": stats(tr, t_start, oos_from),
                      "oos": stats(tr, oos_from, t_end)})
    per_window = {}
    for w in WINDOWS:  # beste per venster, eerlijk gekozen: alleen op de periode vóór de out-of-sample-periode
        sub = {k: v for k, v in runs.items() if cfgs[k]["window"] == w}
        if not sub:
            continue
        k, _ = pick(sub, oos_from)
        per_window[str(w)] = {"pick": k, "oos": stats(sub[k], oos_from, t_end) if k else stats([]),
                         "base_like": stats(runs.get(config_key({**BASELINE, "window": w}), []), oos_from, t_end)}
    # streng: veel instellingen proberen geeft altijd wel een geluksvogel. Daarom ook statistisch bewijs:
    # t ≥ 2 op de bp per trade van de walk-forward-keten (≈ 95% zeker dat het gemiddelde boven 0 ligt)
    propose = (bool(final_k) and wf["n"] >= MIN_TRADES and wf["ret"] > 0 and wf["ret"] > base["ret"]
               and (wf.get("t") or 0) >= 2)
    if propose:
        c = cfgs[final_k]
        advice = (f"Voorstel: venster {c['window']}m, drempel {c['min_edge']:+g}¢, model {c['model']}, "
                  + (f"gewichten '{c['wname']}' {WEIGHT_SETS.get(c['wname'])}, " if c.get("wname") else "") +
                  f"{'trailing stop aan' if c['trail'] else 'trailing stop uit'}. Walk-forward {wf['ret']:+.2f}% "
                  f"tegen {base['ret']:+.2f}% voor de desk zoals nu, op dezelfde out-of-sample-periode.")
        verdict = "voorstel"
    elif wf["ret"] > max(0, base["ret"]) and wf["n"] >= MIN_TRADES:
        advice = (f"Twijfel: walk-forward {wf['ret']:+.2f}% op {wf['n']} trades is beter dan de desk zoals nu "
                  f"({base['ret']:+.2f}%), maar niet overtuigend (t = {wf.get('t')}, nodig ≥ 2). Kan geluk zijn: op papier volgen.")
        verdict = "twijfel"
    elif base["ret"] < 0 and wf["ret"] <= 0:
        advice = (f"Uitzetten: de desk zoals nu verloor {base['ret']:+.2f}% out-of-sample en geen instelling die vooraf "
                  f"gekozen kon worden deed het beter ({wf['ret']:+.2f}%).")
        verdict = "uit"
    else:
        advice = (f"Laten zoals het is: walk-forward {wf['ret']:+.2f}% is niet beter dan de desk zoals nu "
                  f"({base['ret']:+.2f}%).")
        verdict = "zo laten"
    # dagcurves voor de grafiek: desk zoals nu en de walk-forward-keuze, vanaf de out-of-sample-start
    def curve(tr: list[dict]) -> list[list[float]]:
        eq, out, last = 1.0, [[oos_from, 0.0]], None
        for x in sorted(tr, key=lambda x: x["tx"]):
            if x["t"] < oos_from:
                continue
            eq *= 1 + x["ret"]
            d = x["tx"] // DAY * DAY
            if out and out[-1][0] == d:
                out[-1][1] = round((eq - 1) * 100, 3)
            else:
                out.append([d, round((eq - 1) * 100, 3)])
            last = d
        return out
    return {"coin": coin, "from": t_start, "to": t_end, "oos_from": oos_from, "baseline": base_k, "wf": wf, "base": base,
            "steps": steps, "final": final_k, "final_is_ret": final_ret, "propose": propose, "verdict": verdict,
            "advice": advice, "table": table, "per_window": per_window,
            "curve_base": curve(runs[base_k]), "curve_wf": curve(chain)}


# ---------- draaien: per munt downloaden, voorbereiden, alle instellingen simuleren ----------
def run_coin(coin: str, months: int = 24, cfgs: list[dict] | None = None, kind: str = "basis") -> dict:
    t = time.time()
    a = load_klines(coin, months)
    if len(a) < 60 * 24 * 30:
        return {"coin": coin, "error": f"te weinig 1m-data op Binance ({len(a)} candles)"}
    try:
        fund = load_funding(coin, int(a[0, 0]))
    except Exception:
        fund = {}
    P = prepare(a, fund)
    t_start, t_end = P["t0"], P["t0"] + P["n"] * 60_000
    runs, keys, extra = {}, {}, {}
    for c in cfgs or (weight_grid() if kind == "gewichten" else grid()):
        k = config_key(c)
        r = Sim(P, **c).run()
        runs[k], keys[k] = r["trades"], {x: v for x, v in c.items() if x != "weights"}
        extra[k] = {"hit": r["hit"], "beta": r["beta"], "learned": r["learned"], "windows": r["n_windows"]}
    if config_key(BASELINE) not in runs:
        r = Sim(P, **BASELINE).run()
        runs[config_key(BASELINE)], keys[config_key(BASELINE)] = r["trades"], dict(BASELINE)
    ev = evaluate(coin, runs, keys, t_start, t_end)
    for row in ev["table"]:
        row.update(extra.get(row["key"], {}))
    ev.update(candles=len(a), funding_hours=len(fund), funding_source="Hyperliquid + Binance" if fund else "geen",
              seconds=round(time.time() - t, 1), symbol=bn_symbol(coin))
    # trades van de desk zoals nu en van het voorstel apart bewaren (voor wie wil narekenen)
    keep = {k: runs[k] for k in {ev["baseline"], ev["final"]} if k}
    (CACHE / f"trades-{coin}{'' if kind == 'basis' else '-' + kind}.json").write_text(json.dumps(keep))
    return ev


_EMPTY = {"base_like": {"ret": 0.0}}
STATUS: dict = {"running": False, "done": 0, "total": 0, "coin": None, "started": None, "error": None, "log": []}


def _summary(coins: dict) -> dict:
    ok = [c for c in coins.values() if "error" not in c]
    return {"coins": len(ok), "propose": [c["coin"] for c in ok if c["verdict"] == "voorstel"],
            "off": [c["coin"] for c in ok if c["verdict"] == "uit"],
            "keep": [c["coin"] for c in ok if c["verdict"] == "zo laten"],
            "doubt": [c["coin"] for c in ok if c["verdict"] == "twijfel"],
            "wf_avg": round(sum(c["wf"]["ret"] for c in ok) / len(ok), 2) if ok else None,
            "base_avg": round(sum(c["base"]["ret"] for c in ok) / len(ok), 2) if ok else None,
            "per_window": {w: {"avg": round(sum(c["per_window"].get(str(w), _EMPTY)["base_like"]["ret"] for c in ok) / len(ok), 2) if ok else None,
                               "pos": sum(1 for c in ok if c["per_window"].get(str(w), _EMPTY)["base_like"]["ret"] > 0)} for w in WINDOWS}}


def results_file(kind: str = "basis"):
    return RESULTS if kind == "basis" else CACHE / f"results-{kind}.json"


def load_results(kind: str = "basis") -> dict:
    f = results_file(kind)
    return json.loads(f.read_text()) if f.exists() else {}


def run_all(coins: list[str] | None = None, months: int = 24, workers: int = 2, kind: str = "basis") -> dict:
    """Alle munten, `workers` tegelijk (elk in een eigen proces). Tussenresultaten worden meteen bewaard."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    coins = coins or DEFAULT_COINS
    CACHE.mkdir(parents=True, exist_ok=True)
    res = load_results(kind)
    res.setdefault("coins", {})
    res.update(started=int(time.time() * 1000), months=months, kind=kind,
               grid=[config_key(c) for c in (weight_grid() if kind == "gewichten" else grid())],
               weight_sets=WEIGHT_SETS if kind == "gewichten" else None,
               baseline=config_key(BASELINE), notes=NOTES)
    STATUS.update(running=True, done=0, total=len(coins), started=time.time(), error=None, log=[])
    try:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(run_coin, c, months, None, kind): c for c in coins}
            for f in as_completed(futs):
                c = futs[f]
                try:
                    res["coins"][c] = f.result()
                except Exception as e:
                    res["coins"][c] = {"coin": c, "error": str(e)[:300]}
                STATUS["done"] += 1
                STATUS["log"].append(f"{c}: " + (res["coins"][c].get("error") or res["coins"][c]["advice"]))
                res["summary"], res["finished"] = _summary(res["coins"]), int(time.time() * 1000)
                results_file(kind).write_text(json.dumps(res))
    except Exception as e:
        STATUS["error"] = str(e)
    finally:
        STATUS["running"] = False
    return res


def start(coins: list[str] | None = None, months: int = 24, workers: int = 2, kind: str = "basis") -> dict:
    if STATUS["running"]:
        return {"ok": False, "error": "er loopt al een backtest"}
    threading.Thread(target=run_all, args=(coins, months, workers, kind), daemon=True).start()
    return {"ok": True}


NOTES = [
    "Data: 1m-candles van Binance USDT-perps (Hyperliquid geeft maar ~3,5 dag 1m-historie). Kosten en funding van Hyperliquid.",
    "Koopdruk 3 min en 30 s: benaderd met taker-kopers min -verkopers over de laatste 3 en 1 minuut (Binance).",
    "Orderboek top 5 en top 20: historisch niet beschikbaar, weggelaten. Dat waren samen 35% van de score.",
    "Niet meegenomen: stem van de 165 strategieën en de dag-backtests (elk +1¢ drempel tegen hun richting).",
    "Maker-instap alleen gevuld als de koers door de limiet handelt; stop-loss als marktorder met halve spread (1 bp).",
    "SL en TP in dezelfde minuut: SL eerst. Elke munt een eigen papierportefeuille van $1000, één positie tegelijk.",
    "Walk-forward: elk kwartaal de instelling kiezen op alles daarvoor (min. 6 maanden, min. 20 trades), meten op dat kwartaal.",
    "Voorstel pas als de walk-forward ook statistisch overtuigt (t ≥ 2 op de winst per trade); beter maar niet overtuigend = 'twijfel'.",
]


if __name__ == "__main__":  # python -m app.desk_backtest [--gewichten] BTC ETH
    import sys
    args = sys.argv[1:]
    kind = "gewichten" if "--gewichten" in args else "basis"
    out = run_all([a for a in args if not a.startswith("--")] or None, kind=kind)
    print(json.dumps(out.get("summary"), indent=1))
