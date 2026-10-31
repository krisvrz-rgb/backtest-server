"""Dagelijkse coach: je eigen lokale taalmodel (op CrushMini) leest de afgesloten trades en hun uitleg
en schrijft een kort rapport in het Nederlands. Geen Claude, geen cloud: alleen een OpenAI-compatibele
server op deze machine (Ollama, LM Studio, llama.cpp of iets gelijkaardigs).

Het taalmodel leert de gewichten niet; dat doet desk_learn met cijfers. De coach legt uit en stelt voor.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime

import httpx

from . import desk_learn
from .data import DATA_DIR

CONFIG_FILE = DATA_DIR / "desk-coach.json"
REPORT_FILE = DATA_DIR / "desk-coach-reports.json"
CANDIDATES = ("http://127.0.0.1:11434", "http://127.0.0.1:1234", "http://127.0.0.1:8080", "http://127.0.0.1:8000")
_lock = threading.Lock()
STATUS = {"running": False, "error": None}


def config() -> dict:
    cfg = {"url": None, "model": None, "hour": 22, "enabled": True}  # leeg: bij de eerste run zelf zoeken (Ollama op 11434)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    return cfg


def save_config(**kw) -> dict:
    cfg = config()
    cfg.update({k: v for k, v in kw.items() if v is not None})
    CONFIG_FILE.write_text(json.dumps(cfg))
    return cfg


def reports() -> list[dict]:
    return json.loads(REPORT_FILE.read_text()) if REPORT_FILE.exists() else []


def detect() -> dict | None:
    """Zoek een lokale OpenAI-compatibele server en kies het grootste model (de 35B)."""
    for base in CANDIDATES:
        try:
            r = httpx.get(base + "/v1/models", timeout=3)
            if r.status_code != 200:
                continue
            ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        except Exception:
            continue
        if not ids:
            continue

        def size(i: str) -> float:  # "qwen3:35b", "…-35B-…" → 35
            m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", i.replace("-", " ").replace(":", " ").replace("_", " "))
            return float(m.group(1)) if m else 0.0
        ids.sort(key=size, reverse=True)
        return {"url": base, "model": ids[0], "models": ids}
    return None


def _prompt(hs: list[dict], learn: dict) -> str:
    lines = []
    for h in hs[-60:]:
        w, e = h.get("why") or {}, h.get("exit_why") or {}
        parts = ", ".join(f"{n} {c:+.2f}" for n, c, _ in (w.get("parts") or [])[:3])
        lines.append(f"- {datetime.fromtimestamp(h['t'] / 1000):%H:%M} {h['coin']} {h['side']} {h['lev']}x ${h['usd']:.0f}: "
                     f"pnl {h['pnl']:+.2f}, dicht door {e.get('reason') or h.get('reason')}, "
                     f"kans bij instap {w.get('p', 0) * 100:.0f}% vs break-even {w.get('be', 0) * 100:.0f}%, "
                     f"edge {w.get('edge', 0):+.1f}c, kenmerken: {parts or '?'}, "
                     f"beweging {e.get('move', 0):+.2f} sigma, model bij uitstap {e.get('conf', 0) * 100:.0f}%, "
                     f"verlengd {h.get('holds', 0)}x")
    n = len(hs)
    wins = sum(1 for h in hs if h["pnl"] > 0)
    tot = sum(h["pnl"] for h in hs)
    lg = "; ".join(f"{g}: n={i.get('n')}, geleerd {i.get('ll_learned')} vs vast {i.get('ll_base')}, in gebruik={i.get('use')}"
                   for g, i in (learn.get("groups") or {}).items()) or "nog te weinig data"
    return (
        "Je bent de coach van een handelsbot (de 'zwerm') die op Hyperliquid perps handelt in vensters van "
        f"{learn.get('window', 900) // 60} minuten. Hij stapt in als maker, met stop-loss/take-profit, en sluit op het einde "
        "van het venster tenzij het model dezelfde kant op blijft wijzen. Kenmerken: koopdruk 3 min, koopdruk 30 s, "
        "orderboek top 5, orderboek top 20, momentum. Kosten ongeveer 4,5 bp per ronde.\n\n"
        f"Vandaag: {n} trades, {wins} gewonnen, totaal {tot:+.2f} dollar.\n"
        f"Leermodel (log-loss, lager is beter): {lg}\n\nTrades:\n" + "\n".join(lines) + "\n\n"
        "Schrijf in het Nederlands, kort en concreet (max 200 woorden), zonder em-dashes:\n"
        "1. Wat ging goed en wat misleidde de zwerm bij de verliezers (welk kenmerk, welke markt, welk uur)?\n"
        "2. Eén of twee concrete instellingen om aan te passen (edge-drempel, SL/TP in sigma, venster, doorlopen, "
        "markten uitzetten), met waarom.\n3. Wat je morgen moet nakijken.\n"
        "Verzin geen cijfers die hierboven niet staan. Is er te weinig data, zeg dat dan eerlijk."
    )


def _chat(base: str, model: str, prompt: str) -> str:
    """Ollama eerst (native /api/chat, denken uit, model blijft geladen zoals de AI-server het wil),
    anders een gewone OpenAI-compatibele server."""
    msgs = [{"role": "user", "content": prompt}]
    r = httpx.post(base + "/api/chat", timeout=900, json={
        "model": model, "messages": msgs, "stream": False, "think": False, "keep_alive": -1,
        "options": {"temperature": 0.3, "num_ctx": 32768, "num_predict": 1200}})
    if r.status_code == 200 and "message" in r.json():
        return r.json()["message"].get("content", "")
    r = httpx.post(base + "/v1/chat/completions", timeout=900, json={
        "model": model, "messages": msgs, "temperature": 0.3, "max_tokens": 1200, "reasoning_effort": "none"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def run(force: bool = False) -> dict:
    """Rapport over de laatste 24 uur. Kan minuten duren bij een 35B-model."""
    if not _lock.acquire(blocking=False):
        raise RuntimeError("De coach is al bezig")
    STATUS.update(running=True, error=None)
    try:
        cfg = config()
        if not cfg.get("url") or not cfg.get("model"):
            found = detect()
            if not found:
                raise RuntimeError("Geen lokaal taalmodel gevonden op " + ", ".join(CANDIDATES)
                                   + ". Vul de URL en het model in bij Coach-instellingen.")
            cfg = save_config(url=found["url"], model=found["model"])
        hs = desk_learn.trades(int((time.time() - 86400) * 1000))
        learn = next(iter(desk_learn.STATE.values()), {}) if desk_learn.STATE else {}
        t0 = time.time()
        text = _chat(cfg["url"].rstrip("/"), cfg["model"], _prompt(hs, learn))
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # denkstappen van redeneermodellen weglaten
        rep = {"at": int(time.time() * 1000), "model": cfg["model"], "n_trades": len(hs),
               "pnl": round(sum(h["pnl"] for h in hs), 4), "seconds": round(time.time() - t0, 1), "text": text}
        all_ = reports() + [rep]
        REPORT_FILE.write_text(json.dumps(all_[-30:]))
        return rep
    except Exception as e:
        STATUS["error"] = str(e)
        raise
    finally:
        STATUS["running"] = False
        _lock.release()


def scheduler() -> None:
    """Elke dag één keer, op het ingestelde uur (lokale tijd van CrushMini)."""
    while True:
        time.sleep(60)
        cfg = config()
        if not cfg.get("enabled"):
            continue
        now = datetime.now()
        last = reports()[-1]["at"] / 1000 if reports() else 0
        if now.hour == int(cfg.get("hour", 22)) and datetime.fromtimestamp(last).date() != now.date():
            try:
                run()
            except Exception:
                pass


def start_background_run() -> None:
    threading.Thread(target=lambda: _safe_run(), daemon=True, name="desk-coach-run").start()


def _safe_run() -> None:
    try:
        run(force=True)
    except Exception:
        pass
