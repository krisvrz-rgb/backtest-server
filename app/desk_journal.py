"""Dagjournaal van de zwerm: cijfers per handelsdag, welke regels toen golden, en het rapport van de coach.

Een handelsdag loopt van 22.00 tot 22.00 (lokale tijd van CrushMini) en draagt de datum van het einde: om 22.00
sluit de dag af en schrijft je eigen taalmodel wat er geleerd is. De cijfers rekent de server zelf uit; het
taalmodel krijgt ze aangereikt en verzint niets. Elke dag blijft bewaard, zodat je oudere dagen kunt teruglezen.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

from . import desk_learn
from .data import DATA_DIR

RULES_FILE = DATA_DIR / "desk-rules.jsonl"
JOURNAL_FILE = DATA_DIR / "desk-journal.json"
DAY_END_HOUR = 22

# wat de coach mag voorstellen, met de grenzen die de zwerm zelf ook afdwingt
PROPOSABLE = {
    "min_edge": ("edge-drempel in ¢ boven break-even", -3, 10),
    "window_min": ("venster in minuten (15, 60 of 240)", 15, 240),
    "max_lev": ("max hefboom", 1, 20),
    "max_margin_pct": ("max marge per trade in % van de portefeuille", 0.5, 25),
    "hold_max": ("hoeveel vensters een trade mag doorlopen", 0, 12),
    "risk_pct": ("risico per trade in % van de marge", 1, 50),
    "tp_rr": ("take-profit in × de stopafstand (0 = net voor het volgende niveau)", 0, 6),
    "trail_pct": ("trailing stop: % van de winst vasthouden", 10, 90),
}


def day_bounds(day: str | None = None) -> tuple[int, int, str]:
    """Begin en einde (ms) van een handelsdag. Zonder dag: de lopende (eindigt vandaag of morgen om 22.00)."""
    now = datetime.now()
    if day:
        end = datetime.fromisoformat(day).replace(hour=DAY_END_HOUR, minute=0, second=0, microsecond=0)
    else:
        end = now.replace(hour=DAY_END_HOUR, minute=0, second=0, microsecond=0)
        if now >= end:
            end += timedelta(days=1)
    start = end - timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000), end.date().isoformat()


# ---------- regels ----------
def log_rules(rules: dict, why: str = "") -> None:
    """Instellingen bewaren telkens ze veranderen (en bij het opstarten), met tijdstip."""
    last = _rules()[-1:] or [None]
    if last[0] and last[0]["rules"] == rules:
        return
    with open(RULES_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": int(time.time() * 1000), "why": why, "rules": rules}) + "\n")


def _rules() -> list[dict]:
    if not RULES_FILE.exists():
        return []
    out = []
    for line in RULES_FILE.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def rules_between(t0: int, t1: int) -> list[dict]:
    """De regels die golden bij het begin van de periode, plus elke wijziging erin."""
    rs = _rules()
    before = [r for r in rs if r["t"] < t0][-1:]
    return before + [r for r in rs if t0 <= r["t"] < t1]


# ---------- cijfers ----------
def _trades(t0: int, t1: int, extra: list[dict] | None = None) -> list[dict]:
    seen, out = set(), []
    for h in list(desk_learn.trades(t0)) + list(extra or []):
        k = (h.get("t"), h.get("coin"))
        if k in seen or h.get("pnl") is None or not t0 <= h.get("t", 0) < t1:
            continue
        seen.add(k)
        out.append(h)
    return sorted(out, key=lambda h: h["t"])


def stats(t0: int, t1: int, extra: list[dict] | None = None) -> dict:
    hs = _trades(t0, t1, extra)
    n = len(hs)
    pnl = sum(h["pnl"] for h in hs)
    wins = [h for h in hs if h["pnl"] > 0]
    loss = [h for h in hs if h["pnl"] <= 0]
    fees = sum((h.get("exit_why") or {}).get("fees") or 0 for h in hs)

    def group(key) -> list[dict]:
        g: dict = {}
        for h in hs:
            k = key(h)
            a = g.setdefault(k, {"k": k, "n": 0, "w": 0, "pnl": 0.0})
            a["n"] += 1
            a["w"] += h["pnl"] > 0
            a["pnl"] = round(a["pnl"] + h["pnl"], 4)
        return sorted(g.values(), key=lambda a: a["pnl"], reverse=True)

    def edge_bucket(h) -> str:
        e = h.get("edge", 0)
        return "<0¢" if e < 0 else "0-2¢" if e < 2 else "2-5¢" if e < 5 else "≥5¢"

    coins = group(lambda h: h["coin"])
    rows = [r for r in desk_learn._read(desk_learn.LEARN_FILE) if t0 <= r.get("t", 0) * 1000 < t1 and r.get("y")]
    hit = sum(1 for r in rows if (r["p"] > .5) == (r["y"] > 0)) / len(rows) if rows else None
    return {
        "n": n, "wins": len(wins), "winrate": round(len(wins) / n, 3) if n else None, "pnl": round(pnl, 4),
        "fees": round(fees, 4), "gross": round(pnl + fees, 4),
        "avg_win": round(sum(h["pnl"] for h in wins) / len(wins), 4) if wins else None,
        "avg_loss": round(sum(h["pnl"] for h in loss) / len(loss), 4) if loss else None,
        "maker_exit": round(sum(1 for h in hs if h.get("maker_exit")) / n, 3) if n else None,
        "avg_edge": round(sum(h.get("edge", 0) for h in hs) / n, 2) if n else None,
        "avg_lev": round(sum(h.get("lev", 0) for h in hs) / n, 2) if n else None,
        "best": max(hs, key=lambda h: h["pnl"], default=None) and _short(max(hs, key=lambda h: h["pnl"])),
        "worst": min(hs, key=lambda h: h["pnl"], default=None) and _short(min(hs, key=lambda h: h["pnl"])),
        "by_reason": group(lambda h: (h.get("exit_why") or {}).get("reason") or h.get("reason") or "venster"),
        "by_side": group(lambda h: h["side"]),
        "by_edge": group(edge_bucket),
        "by_group": group(lambda h: "aandelen" if ":" in h["coin"] else "crypto"),
        "top": coins[:3], "bottom": coins[-3:][::-1] if len(coins) > 3 else [],
        "windows": len(rows), "model_hit": round(hit, 3) if hit is not None else None,
    }


def _short(h: dict) -> dict:
    return {"coin": h["coin"], "side": h["side"], "pnl": round(h["pnl"], 4), "t": h["t"],
            "reason": (h.get("exit_why") or {}).get("reason") or h.get("reason") or "venster"}


# ---------- journaal ----------
def journal() -> list[dict]:
    return json.loads(JOURNAL_FILE.read_text(encoding="utf-8")) if JOURNAL_FILE.exists() else []


def save_entry(entry: dict) -> None:
    js = [j for j in journal() if j["day"] != entry["day"]] + [entry]
    js.sort(key=lambda j: j["day"])
    JOURNAL_FILE.write_text(json.dumps(js), encoding="utf-8")


def entry(day: str) -> dict | None:
    return next((j for j in journal() if j["day"] == day), None)


def parse_report(text: str) -> dict:
    """Het rapport in vaste stukken: GELEERD, CONCLUSIES, TE BEKIJKEN, VOORSTEL. Het voorstel als instelling = waarde."""
    import re
    heads = {"GELEERD": "learned", "CONCLUSIES": "conclusions", "TE BEKIJKEN": "review", "VOORSTEL": "proposal"}
    out = {v: "" for v in heads.values()}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^\s*[#*\d.\s]*\**\s*(GELEERD|CONCLUSIES|TE BEKIJKEN|VOORSTEL)\s*\**\s*:?\s*\**\s*(.*)$", line, re.I)
        if m:
            cur = heads[m.group(1).upper()]
            line = m.group(2)
        if cur and line.strip():
            out[cur] += line.strip() + "\n"
    out = {k: v.strip() for k, v in out.items()}
    prop = None
    m = re.search(r"\b(" + "|".join(PROPOSABLE) + r")\s*=\s*(-?\d+(?:[.,]\d+)?)", out["proposal"])
    if m:
        key, val = m.group(1), float(m.group(2).replace(",", "."))
        lo, hi = PROPOSABLE[key][1], PROPOSABLE[key][2]
        if lo <= val <= hi and not (key == "window_min" and val not in (15, 60, 240)):
            prop = {"key": key, "value": val, "label": PROPOSABLE[key][0]}
    out["setting"] = prop
    if not any(out[k] for k in ("learned", "conclusions", "review", "proposal")):
        out["conclusions"] = text.strip()  # het model hield zich niet aan de koppen: alles tonen
    return out
