from datetime import datetime

from app import desk_journal, desk_learn


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(desk_journal, "RULES_FILE", tmp_path / "rules.jsonl")
    monkeypatch.setattr(desk_journal, "JOURNAL_FILE", tmp_path / "journal.json")
    monkeypatch.setattr(desk_learn, "TRADES_FILE", tmp_path / "trades.jsonl")
    monkeypatch.setattr(desk_learn, "LEARN_FILE", tmp_path / "learn.jsonl")


def test_day_bounds_run_22_to_22():
    t0, t1, day = desk_journal.day_bounds("2026-11-01")
    assert day == "2026-11-01"
    assert datetime.fromtimestamp(t1 / 1000) == datetime(2026, 11, 1, 22)
    assert datetime.fromtimestamp(t0 / 1000) == datetime(2026, 10, 31, 22)


def test_stats_and_rules(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    t0, t1, _ = desk_journal.day_bounds("2026-11-01")
    for i, (coin, pnl, why) in enumerate([("BTC", .5, "TP"), ("ETH", -.2, "SL"), ("xyz:TSLA", .1, "trailing"), ("BTC", -.1, "venster")]):
        desk_learn.record_trade({"t": t0 + 1000 * (i + 1), "coin": coin, "side": "long", "pnl": pnl, "edge": i, "lev": 2,
                                 "maker_exit": why == "TP", "exit_why": {"reason": why, "fees": .01}})
    desk_learn.record_trade({"t": t1 + 5, "coin": "BTC", "side": "long", "pnl": 9})  # volgende dag: telt niet mee
    st = desk_journal.stats(t0, t1)
    assert st["n"] == 4 and st["wins"] == 2 and abs(st["pnl"] - .3) < 1e-9 and abs(st["fees"] - .04) < 1e-9
    assert st["top"][0]["k"] == "BTC" or st["top"][0]["k"] == "xyz:TSLA"
    assert {g["k"] for g in st["by_reason"]} == {"TP", "SL", "trailing", "venster"}
    assert st["maker_exit"] == .25
    desk_journal.log_rules({"window_min": 15}, "opgestart")
    desk_journal.log_rules({"window_min": 15}, "zelfde")  # geen dubbele regel
    assert len(desk_journal._rules()) == 1


def test_parse_report_and_proposal():
    r = desk_journal.parse_report("GELEERD: TP werkt.\nCONCLUSIES:\n- SL te krap\nTE BEKIJKEN: HYPE 's nachts\n"
                                  "VOORSTEL: min_edge = 1 | te veel trades onder break-even")
    assert r["learned"] == "TP werkt." and "SL te krap" in r["conclusions"] and r["review"].startswith("HYPE")
    assert r["setting"] == {"key": "min_edge", "value": 1.0, "label": desk_journal.PROPOSABLE["min_edge"][0]}
    assert desk_journal.parse_report("VOORSTEL: window_min = 30 | x")["setting"] is None  # geen geldig venster
    assert desk_journal.parse_report("vrije tekst")["conclusions"] == "vrije tekst"


def test_journal_keeps_every_day(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    desk_journal.save_entry({"day": "2026-11-01", "at": 1})
    desk_journal.save_entry({"day": "2026-10-31", "at": 1})
    desk_journal.save_entry({"day": "2026-11-01", "at": 2})
    assert [j["day"] for j in desk_journal.journal()] == ["2026-10-31", "2026-11-01"]
    assert desk_journal.entry("2026-11-01")["at"] == 2
