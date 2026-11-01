"""Desk-backtest: de zwermregels op synthetische 1m-candles (geen netwerk)."""
import io
import zipfile

import numpy as np

from app import desk_backtest as bt


def _candles(days: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = 1_700_000_000_000 // 86_400_000 * 86_400_000 + np.arange(n) * 60_000
    r = rng.normal(0, 8e-4, n)
    c = 30000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    v = rng.uniform(50, 150, n)
    tb = v * np.clip(.5 + r / 8e-4 * .05 + rng.normal(0, .05, n), 0, 1)
    return np.c_[t, o, h, lo, c, v, tb]


def test_symbol_map():
    assert bt.bn_symbol("BTC") == "BTCUSDT"
    assert bt.bn_symbol("kPEPE") == "1000PEPEUSDT"


def test_parse_with_and_without_header():
    for head in ("", "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("x.csv", head + "1700000000000,1,2,0.5,1.5,10,1700000059999,15,3,6,9,0\n")
        a = bt._parse(buf.getvalue())
        assert a.shape == (1, 7) and a[0, 6] == 6 and a[0, 4] == 1.5


def test_simulate_and_walk_forward():
    P = bt.prepare(_candles(240), {})
    assert P["n"] % 1440 == 0 and P["hours"][100] is not None
    runs, cfgs = {}, {}
    for c in [{"window": 60, "min_edge": e, "model": m, "trail": True} for e in (0.0, 2.0) for m in ("vast", "geleerd")]:
        r = bt.Sim(P, **c).run()
        k = bt.config_key(c)
        runs[k], cfgs[k] = r["trades"], c
        assert 0.4 < r["hit"] < 0.6  # willekeurige wandeling: geen voorspellende kracht
        for x in r["trades"]:
            assert x["tx"] >= x["t"] and x["usd"] >= 10 and x["fees"] > 0
    ev = bt.evaluate("SYN", runs, cfgs, P["t0"], P["t0"] + P["n"] * 60_000, warmup_days=60)
    assert ev["verdict"] in ("voorstel", "uit", "zo laten") and ev["steps"]
    for s in ev["steps"]:  # de keuze per kwartaal mag alleen het verleden gebruiken
        assert s["pick"] is None or s["pick"] in runs


def test_costs_eat_random_edge():
    """Op ruis zonder voorspelbaarheid mag de zwerm na kosten gemiddeld geen geld verdienen."""
    P = bt.prepare(_candles(120, seed=7), {})
    tr = bt.Sim(P, window=15, min_edge=0.0, model="vast").run()["trades"]
    if tr:
        assert sum(x["pnl"] for x in tr) - sum(x["gross"] for x in tr) < 0  # kosten zijn altijd negatief


def test_weight_search_needs_proof():
    """Veel gewichtensets op ruis: walk-forward mag geen 'voorstel' opleveren (t ≥ 2 nodig)."""
    P = bt.prepare(_candles(240, seed=3), {})
    runs, cfgs = {}, {}
    for c in bt.weight_grid() + [dict(bt.BASELINE)]:
        k = bt.config_key(c)
        runs[k], cfgs[k] = bt.Sim(P, **c).run()["trades"], {x: v for x, v in c.items() if x != "weights"}
    ev = bt.evaluate("SYN", runs, cfgs, P["t0"], P["t0"] + P["n"] * 60_000, warmup_days=60)
    assert ev["verdict"] != "voorstel"
    assert any("·trend" in k for k in runs)
