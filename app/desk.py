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
COINS = ("BTC", "ETH")
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


def account() -> dict:
    cfg = config()
    addr = trading_address(cfg)
    if not addr:
        return {"connected": False}
    state = _info({"type": "clearinghouseState", "user": addr})
    positions = []
    for p in state.get("assetPositions", []):
        pos = p.get("position", {})
        positions.append({"coin": pos.get("coin"), "size": float(pos.get("szi", 0)),
                          "entry": float(pos.get("entryPx") or 0), "pnl": float(pos.get("unrealizedPnl") or 0)})
    summary = state.get("marginSummary", {})
    return {"connected": True, "value": float(summary.get("accountValue", 0)),
            "withdrawable": float(state.get("withdrawable", 0)), "positions": positions}


def _coin(coin: str) -> str:
    coin = coin.upper()
    if coin not in COINS:
        raise ValueError(f"Alleen {', '.join(COINS)} toegestaan")
    return coin


def _exchange(cfg: dict):
    import eth_account
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants

    wallet = eth_account.Account.from_key(cfg["HL_SECRET_KEY"])
    return Exchange(wallet, constants.MAINNET_API_URL, account_address=cfg["HL_ACCOUNT_ADDRESS"],
                    vault_address=cfg.get("HL_SUBACCOUNT") or None)


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
    meta = _info({"type": "meta"})
    decimals = next(a["szDecimals"] for a in meta["universe"] if a["name"] == coin)
    mid = float(_info({"type": "allMids"})[coin])
    size = round(usd / mid, decimals)
    if size <= 0:
        raise ValueError("Grootte te klein voor de minimale stap van Hyperliquid")
    with _lock:
        res = _exchange(cfg).market_open(coin, side == "long", size, None, slippage)
    _log({"action": "open", "by": by, "coin": coin, "side": side, "usd": usd, "size": size, "mid": mid, "result": res})
    return {"ok": res.get("status") == "ok", "size": size, "mid": mid, "result": res}


def do_close(cfg: dict, coin: str, by: str = "zwerm") -> dict:
    """Sluit de volledige positie in deze coin (reduce-only)."""
    coin = _coin(coin)
    with _lock:
        res = _exchange(cfg).market_close(coin)
    _log({"action": "close", "by": by, "coin": coin, "result": res})
    if res is None:  # geen open positie
        return {"ok": True, "result": None}
    return {"ok": res.get("status") == "ok", "result": res}
