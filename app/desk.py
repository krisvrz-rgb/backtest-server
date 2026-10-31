"""Swarm Desk: koersdata en (optioneel) echte orders op Hyperliquid.

Marktdata (candles, account) komt van de openbare Hyperliquid info-API. Echte orders gaan alleen door als
data/desk.env (of omgevingsvariabelen) een API-wallet bevat, DESK_LIVE=1 staat, het verzoek het juiste
DESK_TOKEN meestuurt en de grootte onder DESK_MAX_USD blijft. De private key verlaat de server nooit.

data/desk.env, één sleutel=waarde per regel:
    HL_SECRET_KEY=0x...        private key van een Hyperliquid API-wallet (kan handelen, niet opnemen)
    HL_ACCOUNT_ADDRESS=0x...   adres van je hoofdaccount (dat de API-wallet heeft geautoriseerd)
    HL_SUBACCOUNT=0x...        optioneel: handel op dit sub-account, los van je eigen trades
    DESK_TOKEN=iets-geheims    moet de pagina meesturen bij elke order
    DESK_LIVE=1                zonder deze regel blijft alles papier
    DESK_MAX_USD=25            maximale positiegrootte per order in dollar
    DESK_AUTO=1                optioneel: de zwerm handelt zelfstandig op de server (zie desk_auto.py)
    DESK_ORDER_USD=20          grootte per autonome order (afgekapt op DESK_MAX_USD)
    DESK_DAILY_LOSS=10         autonome handel stopt voor de rest van de UTC-dag na zoveel verlies
    DESK_FEE_BPS=4.5           kosten per kant waarmee de zwerm rekent
    DESK_MIN_EDGE=2            minimale edge in centen boven break-even
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .data import DATA_DIR

HL_INFO = "https://api.hyperliquid.xyz/info"
ENV_FILE = DATA_DIR / "desk.env"
LOG_FILE = DATA_DIR / "desk-orders.jsonl"
COINS = ("BTC", "ETH", "SOL")   # standaardlijst; via de scanner kan elke Hyperliquid-perp erbij
_lock = threading.Lock()


def config() -> dict:
    cfg: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    for k in list(cfg) + ["HL_SECRET_KEY", "HL_ACCOUNT_ADDRESS", "HL_SUBACCOUNT", "DESK_TOKEN", "DESK_LIVE", "DESK_MAX_USD",
                          "DESK_AUTO", "DESK_ORDER_USD", "DESK_DAILY_LOSS", "DESK_FEE_BPS", "DESK_MIN_EDGE"]:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def num(cfg: dict, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except ValueError:
        return default


def _max_usd(cfg: dict) -> float:
    return num(cfg, "DESK_MAX_USD", 25.0)


def trading_address(cfg: dict) -> str | None:
    """Het adres waarop gehandeld wordt: het sub-account als dat is ingesteld, anders het hoofdaccount."""
    return cfg.get("HL_SUBACCOUNT") or cfg.get("HL_ACCOUNT_ADDRESS")


def status() -> dict:
    cfg = config()
    has_key = bool(cfg.get("HL_SECRET_KEY") and cfg.get("HL_ACCOUNT_ADDRESS"))
    try:
        import hyperliquid  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    live = has_key and sdk and cfg.get("DESK_LIVE") == "1" and bool(cfg.get("DESK_TOKEN"))
    reasons = []
    if not has_key:
        reasons.append("geen HL_SECRET_KEY/HL_ACCOUNT_ADDRESS in data/desk.env")
    if not sdk:
        reasons.append("pakket hyperliquid-python-sdk niet geïnstalleerd")
    if not cfg.get("DESK_TOKEN"):
        reasons.append("geen DESK_TOKEN")
    if cfg.get("DESK_LIVE") != "1":
        reasons.append("DESK_LIVE staat niet op 1")
    addr = trading_address(cfg) or ""
    return {"live": live, "max_usd": _max_usd(cfg), "reasons": reasons, "subaccount": bool(cfg.get("HL_SUBACCOUNT")),
            "auto_allowed": live and cfg.get("DESK_AUTO") == "1",
            "account": f"{addr[:6]}…{addr[-4:]}" if addr else None}


def _info(payload: dict) -> object:
    resp = httpx.post(HL_INFO, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def candles(coin: str, interval: str = "1m", minutes: int = 180) -> list[dict]:
    coin = _coin(coin)
    end = int(time.time() * 1000)
    rows = _info({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
                                                    "startTime": end - minutes * 60_000, "endTime": end}})
    return [{"t": r["t"], "o": float(r["o"]), "h": float(r["h"]), "l": float(r["l"]),
             "c": float(r["c"]), "v": float(r["v"])} for r in rows]


def account(dexes: list[str] | None = None) -> dict:
    """Accountwaarde en posities, opgeteld over de hoofdbeurs en de HIP-3-beurzen (aandelen enz.) die in gebruik zijn."""
    cfg = config()
    addr = trading_address(cfg)
    if not addr:
        return {"connected": False}
    value, withdrawable, positions = 0.0, 0.0, []
    for dex in [""] + sorted(set(d for d in (dexes or []) if d)):
        state = _info({"type": "clearinghouseState", "user": addr, **({"dex": dex} if dex else {})})
        for p in state.get("assetPositions", []):
            pos = p.get("position", {})
            positions.append({"coin": pos.get("coin"), "size": float(pos.get("szi", 0)),
                              "entry": float(pos.get("entryPx") or 0), "pnl": float(pos.get("unrealizedPnl") or 0)})
        value += float(state.get("marginSummary", {}).get("accountValue", 0))
        withdrawable += float(state.get("withdrawable", 0))
    return {"connected": True, "value": value, "withdrawable": withdrawable, "positions": positions}


# ---------- universum: alle perps van Hyperliquid, ook de HIP-3-beurzen met tokenized aandelen ----------
_universe: dict = {"t": 0, "assets": {}}


def dex_of(coin: str) -> str:
    return coin.split(":", 1)[0] if ":" in coin else ""


def universe(max_age: float = 600) -> dict:
    """{coin: {...}} met volume, prijs, spread en funding per markt. Gecachet."""
    if time.time() - _universe["t"] < max_age and _universe["assets"]:
        return _universe["assets"]
    assets: dict = {}
    dexes = [""]
    try:
        dexes += [d["name"] for d in _info({"type": "perpDexs"}) if d and d.get("name")]
    except Exception:
        pass
    for dex in dexes:
        try:
            meta, ctxs = _info({"type": "metaAndAssetCtxs", **({"dex": dex} if dex else {})})
        except Exception:
            continue
        for a, c in zip(meta["universe"], ctxs):
            if a.get("isDelisted"):
                continue
            name = a["name"]
            mark = float(c.get("markPx") or c.get("midPx") or 0)
            if mark <= 0:
                continue
            imp = c.get("impactPxs") or []
            spread = (float(imp[1]) - float(imp[0])) / mark * 1e4 if len(imp) == 2 else None
            prev = float(c.get("prevDayPx") or mark)
            assets[name] = {"coin": name, "dex": dex or "hyperliquid", "szDecimals": a["szDecimals"],
                            "maxLeverage": a.get("maxLeverage"), "px": mark,
                            "volume": float(c.get("dayNtlVlm") or 0), "oi": float(c.get("openInterest") or 0) * mark,
                            "funding": float(c.get("funding") or 0), "change": (mark / prev - 1) * 100 if prev else 0,
                            "spread": spread}
    if assets:
        _universe.update(t=time.time(), assets=assets)
    return _universe["assets"]


_scan_cache: dict = {"t": 0, "rows": []}


def scan_cached(max_age: float = 300) -> dict:
    if time.time() - _scan_cache["t"] > max_age or not _scan_cache["rows"]:
        _scan_cache.update(t=time.time(), rows=scan())
    return {"updated": int(_scan_cache["t"] * 1000), "rows": _scan_cache["rows"]}


def scan(top: int = 25, min_volume: float = 2e6) -> list[dict]:
    """Voorselectie voor de zwerm: genoeg volume, kleine spread en vooral veel beweging per 5 minuten
    tegenover de kosten (σ5m / kosten per ronde). Hoe hoger 'ratio', hoe minder de fees de edge opeten."""
    cfg = config()
    cost = 2 * num(cfg, "DESK_FEE_BPS", 4.5) + 1
    cands = [a for a in universe().values() if a["volume"] >= min_volume and (a["spread"] is None or a["spread"] < 8)]
    cands.sort(key=lambda a: a["volume"], reverse=True)
    out = []
    for a in cands[:60]:  # alleen de 60 grootste markten krijgen een volatiliteitsmeting
        try:
            cs = candles(a["coin"], "5m", 12 * 60)
        except Exception:
            continue
        rs = [math.log(cs[i]["c"] / cs[i - 1]["c"]) for i in range(1, len(cs)) if cs[i - 1]["c"] > 0]
        if len(rs) < 24:
            continue
        sigma = math.sqrt(sum(r * r for r in rs) / len(rs)) * 1e4
        tot_cost = cost + (a["spread"] or 0)
        out.append({**a, "sigma5m": sigma, "cost": tot_cost, "ratio": sigma / tot_cost,
                    "stock": a["dex"] != "hyperliquid"})
    out.sort(key=lambda a: a["ratio"], reverse=True)
    return out[:top]


def _coin(coin: str) -> str:
    """Normaliseer een coinnaam en controleer dat Hyperliquid hem verhandelt (HIP-3: 'dex:TICKER')."""
    coin = coin.strip()
    dex, _, name = coin.rpartition(":")
    coin = f"{dex.lower()}:{name.upper()}" if dex else name.upper()
    if coin in COINS:
        return coin
    if coin not in universe():
        raise ValueError(f"{coin} is geen markt op Hyperliquid")
    return coin


def _exchange(cfg: dict):
    import eth_account
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants

    wallet = eth_account.Account.from_key(cfg["HL_SECRET_KEY"])
    dexes = [""] + sorted({a["dex"] for a in universe().values() if a["dex"] != "hyperliquid"})
    return Exchange(wallet, constants.MAINNET_API_URL, account_address=cfg["HL_ACCOUNT_ADDRESS"],
                    vault_address=cfg.get("HL_SUBACCOUNT") or None, perp_dexs=dexes)


def live_config() -> dict:
    st = status()
    if not st["live"]:
        raise PermissionError("Live handel staat uit: " + "; ".join(st["reasons"]))
    return config()


def check_token(token: str | None) -> dict:
    cfg = live_config()
    if not token or token != cfg.get("DESK_TOKEN"):
        raise PermissionError("Verkeerd of ontbrekend desk-token")
    return cfg


def _log(entry: dict) -> None:
    entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def open_position(coin: str, side: str, usd: float, token: str | None, slippage: float = 0.003, by: str = "mens") -> dict:
    return do_open(check_token(token), coin, side, usd, slippage, by)


def close_position(coin: str, token: str | None, by: str = "mens") -> dict:
    return do_close(check_token(token), coin, by)


def do_open(cfg: dict, coin: str, side: str, usd: float, slippage: float = 0.003, by: str = "zwerm") -> dict:
    """Marktorder (IOC met maximale slippage). Grootte in dollar, afgekapt op DESK_MAX_USD."""
    coin = _coin(coin)
    if side not in ("long", "short"):
        raise ValueError("side moet long of short zijn")
    if usd <= 0 or usd > _max_usd(cfg):
        raise ValueError(f"Grootte moet tussen 0 en {_max_usd(cfg):.0f} dollar liggen")
    dex = dex_of(coin)
    meta = _info({"type": "meta", **({"dex": dex} if dex else {})})
    decimals = next(a["szDecimals"] for a in meta["universe"] if a["name"] == coin)
    mid = float(_info({"type": "allMids", **({"dex": dex} if dex else {})})[coin])
    size = round(usd / mid, decimals)
    if size <= 0:
        raise ValueError("Grootte te klein voor de minimale stap van Hyperliquid")
    with _lock:
        is_buy = side == "long"
        res = _exchange(cfg).order(coin, is_buy, size, _limit_px(coin, is_buy, slippage, mid), {"limit": {"tif": "Ioc"}})
    _log({"action": "open", "by": by, "coin": coin, "side": side, "usd": usd, "size": size, "mid": mid, "result": res})
    return {"ok": res.get("status") == "ok", "size": size, "mid": mid, "result": res}


def _limit_px(coin: str, is_buy: bool, slippage: float, mid: float | None = None) -> float:
    """Grensprijs voor een IOC-'marktorder': mid ± slippage, afgerond zoals Hyperliquid eist
    (max. 5 significante cijfers en max. 6 - szDecimals decimalen). Werkt ook voor HIP-3-markten."""
    dex = dex_of(coin)
    if mid is None:
        mid = float(_info({"type": "allMids", **({"dex": dex} if dex else {})})[coin])
    px = mid * (1 + slippage if is_buy else 1 - slippage)
    decimals = 6 - universe().get(coin, {}).get("szDecimals", 0)
    return round(float(f"{px:.5g}"), max(0, decimals))


def do_close(cfg: dict, coin: str, by: str = "zwerm") -> dict:
    """Sluit de volledige positie in deze coin (reduce-only)."""
    coin = _coin(coin)
    # Positie zelf opzoeken (ook op HIP-3-beurzen) en met een reduce-only IOC-order sluiten.
    dex = dex_of(coin)
    state = _info({"type": "clearinghouseState", "user": trading_address(cfg), **({"dex": dex} if dex else {})})
    size = next((float(p["position"]["szi"]) for p in state.get("assetPositions", []) if p["position"]["coin"] == coin), 0.0)
    if not size:
        return {"ok": True, "result": None}
    with _lock:
        ex = _exchange(cfg)
        is_buy = size < 0
        px = _limit_px(coin, is_buy, 0.01)
        res = ex.order(coin, is_buy, abs(size), px, {"limit": {"tif": "Ioc"}}, reduce_only=True)
    _log({"action": "close", "by": by, "coin": coin, "size": size, "result": res})
    return {"ok": res.get("status") == "ok", "result": res}
