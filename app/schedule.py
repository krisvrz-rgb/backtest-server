"""Automatisch vernieuwen: de server start zelf een scan op vaste dagen en een vast uur.

De instellingen staan in data/schedule.json en zijn aan te passen op de overzichtspagina. Een
achtergrondthread kijkt elke minuut of het tijd is; er is geen externe planner nodig. Tijden zijn de
lokale tijd van de computer waarop de server draait.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any

from .data import DATA_DIR
from .watchlist import SCANNER, default_periods, list_name, load_watchlist

SCHEDULE_FILE = DATA_DIR / "schedule.json"
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "time": "22:30",          # na het slot van Wall Street (22:00 Belgische tijd)
    "days": [0, 1, 2, 3, 4],  # maandag t/m vrijdag (0 = maandag)
    "lists": [],              # leeg = alle lijsten
    "last_run": None,
}


def load_schedule() -> dict[str, Any]:
    if SCHEDULE_FILE.exists():
        try:
            return DEFAULTS | json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return dict(DEFAULTS)


def save_schedule(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = DEFAULTS | {k: v for k, v in cfg.items() if k in DEFAULTS}
    DATA_DIR.mkdir(exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    return cfg


def due(cfg: dict[str, Any], now: datetime) -> bool:
    """Is er vandaag een scan gepland die nog niet gelopen heeft en waarvan het uur voorbij is?"""
    if not cfg.get("enabled") or now.weekday() not in cfg.get("days", []):
        return False
    try:
        hh, mm = (int(x) for x in cfg.get("time", "22:30").split(":"))
    except ValueError:
        return False
    if (now.hour, now.minute) < (hh, mm):
        return False
    return (cfg.get("last_run") or "")[:10] != now.date().isoformat()


def symbols_for(lists: list[str]) -> list[str]:
    return list(dict.fromkeys(s["yahoo"] for g in load_watchlist()
                              if not lists or list_name(g) in lists
                              for s in g["symbols"] if s.get("yahoo")))


def run_now(cfg: dict[str, Any] | None = None) -> bool:
    cfg = cfg or load_schedule()
    syms = symbols_for(cfg.get("lists") or [])
    if not syms or not SCANNER.start(syms, default_periods()):
        return False
    cfg["last_run"] = datetime.now().isoformat(timespec="seconds")
    save_schedule(cfg)
    return True


def next_run(cfg: dict[str, Any], now: datetime | None = None) -> str | None:
    """Wanneer de volgende automatische scan start (voor op de pagina)."""
    if not cfg.get("enabled") or not cfg.get("days"):
        return None
    now = now or datetime.now()
    hh, mm = (int(x) for x in cfg.get("time", "22:30").split(":"))
    for add in range(0, 8):
        day = (now.toordinal() + add)
        d = datetime.fromordinal(day).replace(hour=hh, minute=mm)
        if d.weekday() not in cfg["days"]:
            continue
        if add == 0 and ((now.hour, now.minute) >= (hh, mm)) and not due(cfg, now):
            continue
        return d.isoformat(timespec="minutes")
    return None


def _loop() -> None:
    while True:
        try:
            cfg = load_schedule()
            if due(cfg, datetime.now()) and not SCANNER.status()["running"]:
                run_now(cfg)
        except Exception:  # noqa: BLE001 - de planner mag nooit stoppen
            pass
        time.sleep(60)


_started = False


def start_scheduler() -> None:
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_loop, daemon=True, name="auto-scan").start()
