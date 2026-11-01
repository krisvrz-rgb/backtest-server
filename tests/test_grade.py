import pytest

from app import grade
from app.data import synthetic
from app.grade_check import summarize, ticker_observations


def test_weight_needs_both_periods_and_profit():
    assert grade.weight(0.30, 0.20, 0.10, 0.05) > 0           # verslaat B&H in beide, winst in testjaar
    assert grade.weight(0.05, 0.20, 0.10, 0.05) == 0          # leerperiode slechter dan B&H
    assert grade.weight(0.30, 0.02, 0.10, 0.05) == 0          # testjaar slechter dan B&H
    assert grade.weight(0.30, -0.01, 0.10, -0.20) == 0        # minder verlies dan B&H is geen winst
    assert grade.weight(5.0, 5.0, 0.0, 0.0) == pytest.approx(0.5 + 0.5 * 0.5)  # begrensd


def test_score_and_letters():
    assert grade.score([], [])["grade"] == "?"  # niets bewezen: geen oordeel
    none_open = grade.score([0.3, 0.3], [False, False])
    assert none_open["score"] == 0 and none_open["grade"] == "F"
    all_open = grade.score([0.4] * 6, [True] * 6)
    assert all_open["grade"] == "A+" and all_open["vote"] == 1
    half = grade.score([0.4] * 6, [True, True, True, False, False, False])
    assert half["vote"] == 0.5 and half["score"] < all_open["score"]
    weak = grade.score([0.01], [True])
    assert weak["score"] < 35  # één zwakke strategie geeft nooit een goede grade
    assert [grade.letter(x) for x in (85, 70, 55, 40, 25, 5)] == ["A+", "A", "B", "C", "D", "F"]


def _result():
    return {
        "oos": {"bh_train": 0.10, "bh_test": 0.05, "train": {"start": "2022-11-01", "end": "2024-10-31"},
                "test": {"start": "2024-11-01", "end": "2025-10-31"}},
        "oos_returns": {"a": [0.40, 0.30], "b": [0.30, 0.20], "c": [0.0, 0.50], "d": [0.5, None]},
        "open_now": {"a": {"date": "2025-10-20", "bars": 8, "entry": 10.0, "trade_return": 0.03, "win_rate": 0.6,
                           "trades": 20},
                     "c": {"date": "2025-10-30", "bars": 1}},
        "returns": {"2j": {"a": 0.5, "b": 0.4}},
    }


def test_grade_result_detail():
    g = grade.grade_result(_result(), detail=True)
    assert g["working"] == 2 and g["open_working"] == 1 and g["open_total"] == 2
    assert [x["strategy"] for x in g["working_list"]] == ["a", "b"]  # open eerst
    assert g["working_list"][0]["open"] and g["working_list"][0]["since"] == "2025-10-20"
    assert [x["strategy"] for x in g["open_other"]] == ["c"]  # open maar niet bewezen: telt niet mee
    assert 0 < g["vote"] < 1 and g["grade"] in {"A+", "A", "B", "C", "D", "F", "?"}
    assert grade.grade_result({"oos": {"error": "kort"}}) is None
    bulk = grade.grades({"X": _result(), "Y": {}})
    assert list(bulk) == ["X"] and "working_list" not in bulk["X"]


def test_walk_forward_has_no_edge_on_random_walks():
    """Op toevalskoersen mag de grade niets voorspellen; doet ze dat wel, dan kijkt de check vooruit."""
    per = {s: ticker_observations(synthetic(s, "2014-01-01", "2024-12-31")) for s in ("R1", "R2", "R3", "R4", "R5", "R6")}
    assert all(len(v) > 40 for v in per.values())  # één meetpunt per maand
    s = summarize(per)
    assert s["observations"] == sum(len(v) for v in per.values())
    assert sum(b["n"] for b in s["by_grade"]) == s["observations"]
    assert s["ic"] is None or abs(s["ic"]) < 0.2
    assert s["verdict"]


def test_grade_api():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app import grade_api
    from app.main import app

    grade_api.scan_cached = lambda: {"results": {"AAPL": _result()}}
    grade_api.load_watchlist = lambda: [{"name": "W", "symbols": [{"tv": "NASDAQ:AAPL", "yahoo": "AAPL"}]}]
    c = TestClient(app)
    r = c.get("/api/grades")
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    assert r.json()["grades"]["AAPL"]["grade"] and r.json()["aliases"]["NASDAQ:AAPL"] == "AAPL"
    d = c.get("/api/grade/NASDAQ:AAPL").json()
    assert d["symbol"] == "AAPL" and d["working_list"] and d["formula"]
    assert c.get("/api/grade/ONBEKEND").status_code == 404
    assert c.get("/api/grade/check").status_code == 200
