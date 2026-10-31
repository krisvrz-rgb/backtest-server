import math
import random

from app import desk, desk_strats


def _candles(drift):
    def fake(coin, iv, minutes):
        rnd = random.Random(3)
        p, out = 100.0, []
        for i in range(1500):
            o = p
            p *= math.exp(rnd.gauss(drift, .004) + .003 * math.sin(i / 60))
            out.append({"t": i * 900000, "o": o, "h": max(o, p) * 1.001, "l": min(o, p) * .999, "c": p, "v": 10})
        return out
    return fake


def test_select_and_vote(tmp_path, monkeypatch):
    monkeypatch.setattr(desk_strats, "STATE_FILE", tmp_path / "s.json")
    monkeypatch.setattr(desk_strats, "STATE", {})
    monkeypatch.setattr(desk, "candles", _candles(0.0))
    r = desk_strats.select("BTC", 900)
    assert r["tested"] > 100 and r["bars"] == 1499
    assert len(r["picked"]) <= desk_strats.TOP
    for x in r["picked"]:
        assert x["ret_in"] > 0 and x["ret_out"] > 0 and x["ret_out"] > x["bh_out"] and x["trades_out"] >= desk_strats.MIN_TRADES
    if r["picked"]:
        assert -1 <= r["vote"]["v"] <= 1
    assert desk_strats.get("BTC", 900) is r
    assert (tmp_path / "s.json").exists()
