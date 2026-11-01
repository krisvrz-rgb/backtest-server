"""Signalenlab op synthetische uurdata (geen netwerk): een geplant signaal moet gevonden worden, ruis niet."""
import numpy as np
import pandas as pd

from app import desk_signals as ds


def _hr(n_days=730, seed=0, plant=0.0):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    x = rng.normal(0, 1, n)                      # een 'koopdruk' die het volgende uur voorspelt
    r = rng.normal(0, 0.006, n)
    r[1:] += plant * 0.006 * x[:-1]
    c = 100 * np.exp(np.cumsum(r))
    v = np.full(n, 1000.0)
    tb = v * np.clip(.5 + .1 * x, 0, 1)
    return pd.DataFrame({"c": c, "v": v, "tb": tb, "fund": 0.0}, index=idx)


def test_finds_planted_signal_and_not_noise():
    out = {}
    for name, plant in (("echt", 1.0), ("ruis", 0.0)):
        hr = _hr(plant=plant, seed=1)
        sig = ds.signals(hr, pd.DataFrame(), None)
        out[name] = ds.test_signal("koopdruk 1u", sig["koopdruk 1u"], hr)["1"]
    assert out["echt"]["bp"] > 0 and out["echt"]["t"] > 3 and out["echt"]["ic"] > .2
    assert abs(out["ruis"]["ic"]) < .05 and out["ruis"]["bp"] < 0  # ruis: kosten blijven over


def test_pool_verdicts():
    coins = {}
    for i in range(5):
        hr = _hr(plant=1.0, seed=10 + i)
        sig = ds.signals(hr, pd.DataFrame(), None)
        coins[f"C{i}"] = {"signals": {"koopdruk 1u": ds.test_signal("x", sig["koopdruk 1u"], hr),
                                      "uur van de dag": ds.test_hour_of_day(hr)}}
    p = ds.pool(coins)
    assert p["koopdruk 1u"]["1"]["verdict"] == "voorspelt, ook na kosten"
    assert p["uur van de dag"]["1"]["verdict"] != "voorspelt, ook na kosten"
