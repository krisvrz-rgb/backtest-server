"""Trading Desk: ticketproces, sizing, goedkeuring en papieren uitvoering, met nep-marktdata (geen netwerk)."""
import pytest

from app import trading_desk as td

PX = {"ETH": 3000.0}


def fake_info(payload):
    t = payload["type"]
    if t == "metaAndAssetCtxs":
        return [{"universe": [{"name": "ETH", "szDecimals": 4, "maxLeverage": 25}]},
                [{"markPx": str(PX["ETH"]), "midPx": str(PX["ETH"]), "oraclePx": str(PX["ETH"]), "funding": "0.0000125",
                  "openInterest": "500000", "dayNtlVlm": "900000000"}]]
    if t == "l2Book":
        p = PX["ETH"]
        return {"levels": [[{"px": str(p - 0.1), "sz": "500", "n": 9}], [{"px": str(p + 0.1), "sz": "500", "n": 9}]]}
    if t == "candleSnapshot":
        return [{"h": PX["ETH"] + 15, "l": PX["ETH"] - 15, "c": PX["ETH"]} for _ in range(48)]
    if t == "allMids":
        return {"ETH": str(PX["ETH"])}
    raise AssertionError(t)


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    PX["ETH"] = 3000.0
    td.reset_for_tests(tmp_path / "state.json")
    monkeypatch.setattr(td, "_info", fake_info)
    monkeypatch.setattr(td, "research_hook", None)
    monkeypatch.setattr(td, "review_llm", None)
    yield


def test_pass_ticket_sizes_from_stressed_distance():
    p = td.propose("ETH", "long", stop=2900, tp_r=2)
    assert p["status"] == "awaiting" and p["risk"]["verdict"] == "PASS"
    t = p["ticket"]
    # $1000 × 0,5% = $5 risico; gestrest > 100 per ETH, dus grootte < 0,05 en het risico nooit boven $5
    assert t["size"] < 0.05 and t["risk_usd"] <= 5.0 + 1e-9
    assert t["tp"] == pytest.approx(3200) and "approve " + p["id"] in t["text"]


def test_wrong_side_stop_and_ceiling_rejects():
    p = td.propose("ETH", "long", stop=3100)
    assert p["status"] == "rejected" and "verkeerde kant" in p["risk"]["gate"]
    with pytest.raises(ValueError):
        td.set_limits({"risk_per_trade": 3})  # boven het plafond van 2%


def test_only_exact_phrase_approves_and_executes_on_paper():
    p = td.propose("ETH", "long", stop=2900, tp_r=2)
    for bad in ("ja", "ok " + p["id"], "approve", "approve HG-19990101-01"):
        with pytest.raises((ValueError, KeyError)):
            td.approve(bad)
    assert td.find(p["id"])["status"] == "awaiting"
    p = td.approve("approve " + p["id"])
    assert p["status"] == "live" and p["position"]["size"] == p["ticket"]["size"]
    with pytest.raises(ValueError):
        td.approve("approve " + p["id"])  # één keer per goedkeuring


def test_expired_ticket_cannot_be_approved(monkeypatch):
    p = td.propose("ETH", "long", stop=2900)
    now = td._now()
    monkeypatch.setattr(td, "_now", lambda: now + 31 * 60)
    with pytest.raises(ValueError):
        td.approve("approve " + p["id"])
    assert td.find(p["id"])["status"] == "void"


def test_stop_hit_closes_near_minus_one_r_and_reviews():
    p = td.approve("approve " + td.propose("ETH", "long", stop=2900, tp_r=2)["id"])
    td.tick({"ETH": 2899.0})
    p = td.find(p["id"])
    assert p["status"] == "closed" and p["position"]["why"] == "stop-loss"
    assert -1.2 < p["position"]["r"] < -0.8 and p["review"]["facts"]
    s = td.summary()
    assert s["all"]["n"] == 1 and s["all"]["losses"] == 1 and s["equity"] < 1000


def test_take_profit_and_close_ticket():
    a = td.approve("approve " + td.propose("ETH", "long", stop=2900, tp_r=2)["id"])
    td.tick({"ETH": 3201.0})
    assert td.find(a["id"])["position"]["why"] == "take-profit" and td.find(a["id"])["position"]["pnl"] > 0
    b = td.approve("approve " + td.propose("ETH", "short", stop=3100)["id"])
    td.request_close(b["id"])
    b = td.approve("approve " + b["id"] + "-B")
    assert b["status"] == "closed" and "ticket" in b["position"]["why"]


def test_limits_on_positions_and_duplicates():
    td.approve("approve " + td.propose("ETH", "long", stop=2900)["id"])
    p = td.propose("ETH", "long", stop=2900)
    assert p["status"] == "rejected" and "al een positie" in p["risk"]["gate"]


def test_resting_limit_fills_as_maker():
    p = td.approve("approve " + td.propose("ETH", "long", stop=2900, entry=2990)["id"])
    assert p["status"] == "resting"
    td.tick({"ETH": 2995.0})
    assert td.find(p["id"])["status"] == "resting"
    td.tick({"ETH": 2989.0})
    p = td.find(p["id"])
    assert p["status"] == "live" and p["position"]["entry"] == 2990
