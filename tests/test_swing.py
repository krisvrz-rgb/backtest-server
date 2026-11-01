import time

import numpy as np
import pandas as pd
import pytest

from app import desk, swing, swing_desk
from app.data import synthetic


def _mean_reverting(seed=1, n=2300):
    """Dagkoersen met wat terugkeer naar het gemiddelde, zodat sommige strategieën echt iets vinden."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-30", periods=n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.0004 - 0.25 * r[i - 1] + rng.normal(0, 0.016)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def test_analyze_structure_and_filters():
    a = swing.analyze(_mean_reverting())
    assert a["tested"] > 100 and a["passed"] <= a["swing_candidates"]
    assert len(a["top"]) <= swing.TOP
    for r in a["top"]:
        assert r["n"] >= swing.MIN_TRADES and r["mean"] > 0 and r["alpha"] > 0
        assert swing.SWING_BARS[0] <= r["avg_bars"] <= swing.SWING_BARS[1]
        assert swing.TP_RANGE[0] <= r["tp_atr"] <= swing.TP_RANGE[1]
        assert sum(r["exits"].values()) == r["n"]
    assert [r["score"] for r in a["top"]] == sorted((r["score"] for r in a["top"]), reverse=True)
    assert len(a["wf"]["years"]) == swing.WF_YEARS
    assert set(a["signals"]) and all(isinstance(v, bool) for v in a["signals"].values())


def test_too_short_history():
    p = synthetic("X", "2024-01-01", "2026-10-30")
    with pytest.raises(ValueError):
        swing.analyze(p)


def test_overlay_stop_and_target():
    idx = pd.bdate_range("2026-01-01", periods=6)
    p = pd.DataFrame({"open": [100, 100, 99, 97, 96, 95], "high": [101, 101, 100, 98, 97, 96],
                      "low": [99, 99, 95, 96, 95, 94], "close": [100, 99, 97, 96, 95, 95]}, index=idx)
    t = {"date": idx[1], "i0": 1, "i1": 5, "entry": 100.0, "raw": -0.05, "atr": 0.02}
    o = swing.overlay(p, t, 2.0, 3.0, 0.0)
    assert o["why"] == "sl" and abs(o["ret"] - (-0.04)) < 1e-9 and o["exit"] == idx[2]
    p2 = p.copy()
    p2.loc[idx[3], "high"] = 107
    p2.loc[idx[2], "low"] = 98
    p2.loc[idx[3], "low"] = 97
    o2 = swing.overlay(p2, t, 2.0, 3.0, 0.001)
    assert o2["why"] == "tp" and abs(o2["ret"] - (0.06 - 0.002)) < 1e-9


def _analysis(wf_n=12, wf_pf=1.6, run_up_atr=0.2, open_=True, good_years=2):
    pick = {"strategy": "rsi_dip", "category": "Terugkeer", "description": "", "score": 2.5, "n": 40, "win": 0.6,
            "mean": 0.012, "pf": 1.9, "alpha": 0.006, "cons": 0.8, "var": 0.0016, "tp_atr": 3.0, "avg_bars": 7.0,
            "exits": {"sl": 10, "tp": 15, "signaal": 15}, "last12": {"n": 8, "sum": 0.05}, "open": open_,
            "bars_since_buy": 1, "last_buy": "2026-10-29", "run_up": 0.004, "run_up_atr": run_up_atr}
    years = [{"year_start": f"202{k}-10-30", "strategy": "rsi_dip", "n": 4, "sum": s} for k, s in ((3, .05), (4, -.01), (5, .08))]
    wf = {"n": wf_n, "win": 0.58, "mean": 0.01, "pf": wf_pf, "sum": 0.12, "var": 0.0016, "years": years[:good_years + 1]
          if good_years < 2 else years}
    return {"top": [pick], "pick": pick if open_ else None, "wf": wf, "atr_pct": 0.025, "agree": 1, "signals": {"rsi_dip": open_}}


CFG = {"kelly": 0.25, "risk_pct": 2.0, "margin_pct": 10.0, "max_lev": 3, "min_volume": 250_000}
MKT = {"px": 200.0, "dex": "xyz", "volume": 5e6, "spread": 3, "maxLeverage": 10}


def test_proposal_levels_and_sizing():
    p = swing.proposal("xyz:TEST", _analysis(), MKT, 0.0000125, 1000.0, CFG)
    assert p["status"] == "voorstel", p["why"]
    s = p["size"]
    assert 1 <= s["lev"] <= 3
    assert s["margin"] <= 100.0 + 1e-6                       # marge ≤ 10% van 1000
    assert s["notional"] * p["sl_dist"] <= 20.0 + 1e-6       # verlies bij stop ≤ 2%
    assert p["sl"] < 200 < p["tp"] and abs(p["sl"] - 200 * (1 - 0.05)) < 1e-9
    if s["liq_dist"]:
        assert s["liq_dist"] >= 1.5 * p["sl_dist"]
    assert p["funding_frac"] == pytest.approx(0.0000125 * 24 * 7 * 7 / 5)


@pytest.mark.parametrize("kw,status", [({"wf_n": 3}, "te weinig bewijs"), ({"wf_pf": 1.05}, "te weinig bewijs"),
                                       ({"run_up_atr": 1.5}, "loopt al"), ({"good_years": 1}, "te weinig bewijs"), ({"open_": False}, "geen signaal")])
def test_proposal_gates(kw, status):
    assert swing.proposal("xyz:TEST", _analysis(**kw), MKT, 0.0, 1000.0, CFG)["status"] == status


def test_funding_can_kill_the_trade():
    p = swing.proposal("xyz:TEST", _analysis(), MKT, 0.0001, 1000.0, CFG)  # 0,01%/u ≈ 2,4%/dag
    assert p["status"] == "te duur"


def test_illiquid_market():
    p = swing.proposal("xyz:TEST", _analysis(), MKT | {"volume": 1000}, 0.0, 1000.0, CFG)
    assert p["status"] == "te illiquide"


def test_yahoo_mapping():
    assert swing_desk.yahoo_for("xyz:NVDA") == "NVDA"
    assert swing_desk.yahoo_for("xyz:XYZ100") == "^NDX"
    assert swing_desk.yahoo_for("xyz:GOLD") == "GC=F"
    assert swing_desk.yahoo_for("xyz:EUR") is None
    uni = {"xyz:NVDA": {"coin": "xyz:NVDA", "dex": "xyz", "volume": 5e6}, "flx:NVDA": {"coin": "flx:NVDA", "dex": "flx", "volume": 9e6},
           "BTC": {"coin": "BTC", "dex": "hyperliquid", "volume": 1e9}}
    g = swing_desk.stock_markets(uni)
    assert list(g) == ["NVDA"] and g["NVDA"]["market"]["coin"] == "flx:NVDA" and g["NVDA"]["alts"] == ["xyz:NVDA"]


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(swing_desk, "STATE_FILE", tmp_path / "s.json")
    monkeypatch.setattr(swing_desk, "ANALYSIS_FILE", tmp_path / "a.json")
    monkeypatch.setattr(swing_desk, "TRADES_FILE", tmp_path / "t.jsonl")
    prices = _mean_reverting()
    last = float(prices["close"].iloc[-1])
    uni = {"xyz:TEST": {"coin": "xyz:TEST", "dex": "xyz", "px": last, "volume": 5e6, "spread": 3, "funding": 0.00001,
                        "maxLeverage": 10},
           "xyz:WRONG": {"coin": "xyz:WRONG", "dex": "xyz", "px": last * 3, "volume": 1e6, "spread": 3, "funding": 0.0,
                         "maxLeverage": 10}}
    monkeypatch.setattr(desk, "universe", lambda max_age=600: uni)
    monkeypatch.setattr(swing_desk, "funding_avg", lambda coin, days=7: 0.00001)
    b = swing_desk.SwingBook(loader=lambda y: prices)
    b.uni = uni
    return b


def test_refresh_and_price_check(book):
    book.refresh()
    rows = book.analysis["rows"]
    assert set(rows) == {"TEST", "WRONG"}
    assert "analysis" in rows["TEST"]
    assert "koppeling onzeker" in rows["WRONG"]["error"]
    props = book.proposals(book.uni)
    assert {p["coin"] for p in props} == {"xyz:TEST", "xyz:WRONG"}
    assert book.progress["running"] is False and book.progress["done"] == 2


def test_paper_trade_with_funding_and_stop(book):
    p = swing.proposal("xyz:TEST", _analysis(), book.uni["xyz:TEST"], 0.00001, 1000.0, book.cfg)
    p["yahoo"] = "TEST"
    pos = book.open(p)
    assert book.cash < 1000 and len(book.positions) == 1
    with pytest.raises(ValueError):
        book.open(p)                                       # niet twee keer hetzelfde aandeel
    pos["fund_t"] -= 24 * 3600                             # een dag later
    uni = {"xyz:TEST": book.uni["xyz:TEST"] | {"px": pos["sl"] * 0.999}}
    book.tick(uni)
    assert not book.positions and len(book.history) == 1
    h = book.history[0]
    assert h["reason"] == "sl" and h["pnl"] < 0 and h["funding"] > 0
    assert h["pnl"] == pytest.approx(h["notional"] * (h["exit"] / h["entry"] - 1) - h["fees"] - h["funding"])
    assert book.cash == pytest.approx(1000 + h["pnl"])
    assert swing_desk.TRADES_FILE.exists()


def test_signal_exit_after_refresh(book):
    p = swing.proposal("xyz:TEST", _analysis(), book.uni["xyz:TEST"], 0.0, 1000.0, book.cfg)
    p["yahoo"] = "TEST"
    book.open(p)
    book.analysis = {"at": int(time.time() * 1000), "rows": {"TEST": {"coin": "xyz:TEST", "analysis": _analysis(open_=False)}}}
    book.cfg["auto"] = False
    book.after_refresh(book.uni)
    assert not book.positions and book.history[-1]["reason"] == "signaal uit"


def test_compare_keeps_books_apart(book):
    v = book.view()
    c = swing_desk.compare(v, {"start": 1000, "equity": 1010, "curve": [[0, 1000], [1, 1010]]},
                           [{"t": 1, "opened": 0, "pnl": 10.0, "paper": True}, {"t": 2, "pnl": 99.0, "paper": False}])
    assert c["scalp"]["n"] == 1 and c["scalp"]["pnl"] == 10.0
    assert c["swing"]["n"] == 0


def test_due_schedule(book):
    from datetime import datetime
    book.analysis = {}
    assert book.due()
    book.analysis = {"at": int(time.time() * 1000) - 5 * 3600_000, "date": "2026-10-29"}
    assert book.due(datetime(2026, 10, 30, 22, 45))      # vrijdag na 22.40
    assert not book.due(datetime(2026, 10, 30, 21, 0))   # nog voor de slot
    assert not book.due(datetime(2026, 10, 31, 23, 0))   # zaterdag
