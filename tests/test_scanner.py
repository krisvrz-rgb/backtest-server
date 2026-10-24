from datetime import datetime

from fastapi.testclient import TestClient

from app import live_scanner, tickers
from app.main import app

SYMBOLS = {s: {"exchange": "NASDAQ", "name": s + " Inc"} for s in ["CLRO", "YMAT", "VEEA", "MRNA", "TGEN", "AM", "ET"]}


def test_parse_nasdaq_files():
    nasdaq = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
              "YMAT|J-Star Holding|S|N|N|100|N|N\nZZZT|Test|S|Y|N|100|N|N\nFile Creation Time: 1024202600:00|||||||\n")
    other = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
             "VLO|Valero Energy|N|VLO|N|100|N|VLO\nBRK.A|Berkshire|N|BRK.A|N|1|N|BRK.A\n")
    assert tickers._parse(nasdaq, "nasdaq") == {"YMAT": {"exchange": "NASDAQ", "name": "J-Star Holding"}}
    assert tickers._parse(other, "other") == {"VLO": {"exchange": "NYSE", "name": "Valero Energy"}}


def test_extract_tickers_filters_noise():
    texts = ["Momo HOD", "CLRO", "YMAT+11.2%/2Min", "TGEN 2.37", "09:47:52 AM", "$VEEA", "CLR0", "MRNA 199.24", "ET"]
    assert tickers.extract_tickers(texts, SYMBOLS) == ["CLRO", "YMAT", "TGEN", "VEEA", "MRNA"]
    assert "HOD" not in tickers.extract_tickers(texts, None)


def test_board_counts_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(live_scanner, "SCAN_DIR", tmp_path)
    b = live_scanner.TickerBoard("2026-10-23")
    t = datetime(2026, 10, 23, 9, 47, tzinfo=live_scanner.NY)
    assert b.add(["CLRO", "YMAT"], SYMBOLS, t) == ["CLRO", "YMAT"]
    assert b.add(["CLRO"], SYMBOLS, t.replace(minute=50)) == []
    b.save()
    again = live_scanner.TickerBoard("2026-10-23")
    assert again.tickers["CLRO"]["count"] == 2 and again.rows(min_count=2)[0]["symbol"] == "CLRO"


def test_market_hours():
    assert live_scanner.market_open(datetime(2026, 10, 23, 9, 30))
    assert not live_scanner.market_open(datetime(2026, 10, 23, 21, 0))
    assert not live_scanner.market_open(datetime(2026, 10, 24, 10, 0))  # zaterdag


def test_scanner_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(live_scanner, "SCAN_DIR", tmp_path)
    sc = live_scanner.scanner
    monkeypatch.setattr(sc, "_ocr", lambda img: ["CLRO 2.76", "YMAT", "Gap Up"])
    monkeypatch.setattr(live_scanner, "load_symbols", lambda: SYMBOLS)
    sc.board = live_scanner.TickerBoard(live_scanner.now_ny().date().isoformat())
    res = sc.process_image(object())
    assert res["tickers"] == ["CLRO", "YMAT"]

    client = TestClient(app)
    body = client.get("/scanner/tickers").json()
    assert {r["symbol"] for r in body["tickers"]} == {"CLRO", "YMAT"}
    assert client.get("/scanner/status").json()["running"] is False
    page = client.get("/scanner")
    assert page.status_code == 200 and "Live scanner" in page.text
