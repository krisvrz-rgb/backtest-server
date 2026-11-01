import math
import random
import time

from app import desk, desk_rustig


def _fake_info(payload):
    if payload["type"] == "candleSnapshot":
        c = payload["req"]["coin"]
        rnd = random.Random(hash(c) % 1000)
        h0, h1 = payload["req"]["startTime"] // 3_600_000, payload["req"]["endTime"] // 3_600_000
        drift = (rnd.random() - .5) * 0.002
        p, out = 100.0, []
        for h in range(h0, h1 + 1):
            p *= math.exp(rnd.gauss(drift, .006))
            out.append({"t": h * 3_600_000, "o": p, "h": p, "l": p, "c": str(p), "v": "1"})
        return out
    if payload["type"] == "fundingHistory":
        t = payload["startTime"] // 3_600_000 * 3_600_000 + 3_600_000
        rows = []
        while t <= payload["endTime"] and len(rows) < 500:
            rows.append({"time": t, "fundingRate": "0.0000125"})
            t += 3_600_000
        return rows
    raise ValueError(payload)


def _history(n=120):
    now = int(time.time() * 1000)
    rnd = random.Random(1)
    out = []
    for i in range(n):
        t = now - (n - i) * 3_600_000
        held = rnd.choice([0.5, 2, 3, 10, 40, 90])
        out.append({"t": t, "opened": t - int(held * 60000), "coin": "BTC", "side": "long", "usd": 100, "pnl": rnd.gauss(-.05, .5),
                    "paper": True, "reason": rnd.choice(["SL", "TP", "venster"]), "why": {"emove": rnd.uniform(3, 30), "cost": 6},
                    "exit_why": {"fees": .06}})
    return out


def test_compare(monkeypatch):
    monkeypatch.setattr(desk, "_info", _fake_info)
    monkeypatch.setattr(desk_rustig.time, "sleep", lambda s: None)
    desk_rustig._COINS.clear()
    hist = _history()
    r = desk_rustig.compare(hist, {"start": 1000, "curve": [[hist[0]["opened"], 1000]]}, 4.5, force=True)
    assert r["ok"], r
    own = r["own"]
    assert own["total"]["n"] == 120 and sum(b["n"] for b in own["buckets"]) == 120
    assert own["under5"] == sum(1 for h in hist if (h["t"] - h["opened"]) / 60000 < 5)
    assert len(own["edge_filter"]) == len(desk_rustig.EDGE_STEPS)
    hi, hr = r["hist"], r["hourly"]
    assert hi["curve"][0][1] <= 1000 and hi["closed"] < hr["closed"]  # elk uur herbalanceren = veel meer rondes
    assert hr["fees"] > hi["fees"]
    assert abs(r["desk"]["end"] - (1000 + sum(h["pnl"] for h in hist))) < .05
    assert len(r["grid"]) == 7 and r["grid"][-1]["hourly"]
    assert hi["funding"] != 0 and r["funding_ok"]


def test_incomplete_universe_waits(monkeypatch):
    """Krijgt één munt geen candles (429), dan geen vergelijking met een half universum; de volgende vraag vult aan."""
    calls = {"n": 0}

    def flaky(payload):
        if payload["type"] == "candleSnapshot" and payload["req"]["coin"] == "SOL" and calls["n"] < 4:
            calls["n"] += 1
            raise RuntimeError("429 Too Many Requests")
        return _fake_info(payload)
    monkeypatch.setattr(desk, "_info", flaky)
    monkeypatch.setattr(desk_rustig.time, "sleep", lambda s: None)
    desk_rustig._COINS.clear()
    hist = _history(20)
    r = desk_rustig.compare(hist, {"start": 1000}, 4.5)
    assert not r["ok"] and "SOL" in r["missing"]
    n_before = len(desk_rustig._COINS)
    r = desk_rustig.compare(hist, {"start": 1000}, 4.5)
    assert r["ok"] and len(r["coins"]) == len(desk_rustig.UNIVERSE) and len(desk_rustig._COINS) > n_before
