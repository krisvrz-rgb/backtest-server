"""Live tickers van een YouTube-livestream lezen: frame pakken, OCR, tickers bijhouden per handelsdag.

Standaard de dagelijkse stream van Zendoo (https://www.youtube.com/@zendoo/live): yt-dlp zoekt elke
dag zelf de actuele livestream op. Alleen actief tijdens de Amerikaanse handelsdag inclusief pre- en
aftermarket (04:00-20:00 New York tijd, ma-vr).

Instellen met omgevingsvariabelen:
  SCANNER_URL        kanaal- of video-URL (standaard https://www.youtube.com/@zendoo/live)
  SCANNER_INTERVAL   seconden tussen frames (standaard 10)
  SCANNER_AUTOSTART  1 = meteen starten met de server (standaard), 0 = handmatig via /scanner/start
  SCANNER_SCALE      vergroting voor de OCR, helpt bij kleine tekst (standaard 1.5)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .data import DATA_DIR
from .tickers import extract_tickers, load_symbols

log = logging.getLogger("scanner")
NY = ZoneInfo("America/New_York")
SCAN_DIR = DATA_DIR / "scanner"
DEFAULT_URL = "https://www.youtube.com/@zendoo/live"


def now_ny() -> datetime:
    return datetime.now(NY)


def market_open(t: datetime) -> bool:
    return t.weekday() < 5 and 4 <= t.hour < 20


class OCR:
    """RapidOCR (ONNX, geen losse installatie nodig). Ondersteunt het nieuwe en het oude pakket."""

    def __init__(self, scale: float = 1.5):
        self.scale = scale
        try:
            from rapidocr import RapidOCR
            self._new = True
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
            self._new = False
        self.engine = RapidOCR()

    def __call__(self, img) -> list[str]:
        import cv2

        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        if self._new:
            res = self.engine(img)
            return list(res.txts or [])
        res, _ = self.engine(img)
        return [r[1] for r in res or []]


def resolve_stream(url: str) -> dict:
    """Directe video-URL (HLS) van de actuele livestream via yt-dlp."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "format": "best[height<=1080]/best", "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist":
        info = next(iter(info.get("entries") or []), None) or {}
    if not info.get("url"):
        raise RuntimeError("Geen stream-URL gevonden")
    return {"url": info["url"], "title": info.get("title"), "id": info.get("id"),
            "is_live": bool(info.get("is_live")), "webpage_url": info.get("webpage_url")}


def grab_frame(stream_url: str):
    import cv2

    cap = cv2.VideoCapture(stream_url)
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class TickerBoard:
    """Tickers van één handelsdag: hoe vaak, wanneer voor het eerst en laatst gezien. Bewaard als JSON."""

    def __init__(self, day: str):
        self.day = day
        self.tickers: dict[str, dict] = {}
        self.path = SCAN_DIR / f"{day}.json"
        if self.path.exists():
            self.tickers = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, found: list[str], symbols: dict | None, ts: datetime) -> list[str]:
        stamp = ts.isoformat(timespec="seconds")
        new = []
        for sym in found:
            row = self.tickers.get(sym)
            if row is None:
                info = (symbols or {}).get(sym, {})
                row = self.tickers[sym] = {"symbol": sym, "exchange": info.get("exchange", ""),
                                           "name": info.get("name", ""), "count": 0, "first_seen": stamp}
                new.append(sym)
            row["count"] += 1
            row["last_seen"] = stamp
        return new

    def save(self):
        SCAN_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.tickers, indent=1), encoding="utf-8")

    def rows(self, min_count: int = 1) -> list[dict]:
        rows = [r for r in self.tickers.values() if r["count"] >= min_count]
        return sorted(rows, key=lambda r: r["last_seen"], reverse=True)


class LiveScanner:
    def __init__(self, url: str | None = None, interval: float | None = None, scale: float | None = None):
        self.url = url or os.getenv("SCANNER_URL", DEFAULT_URL)
        self.interval = interval or float(os.getenv("SCANNER_INTERVAL", "10"))
        self.scale = scale or float(os.getenv("SCANNER_SCALE", "1.5"))
        self.lock = threading.Lock()
        self.board = TickerBoard(now_ny().date().isoformat())
        self.status: dict = {"running": False, "state": "gestopt", "error": None, "frames": 0,
                             "last_frame": None, "stream": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ocr: OCR | None = None
        self._stream: dict | None = None

    # ---- verwerking ----
    def ocr(self) -> OCR:
        if self._ocr is None:
            self._ocr = OCR(self.scale)
        return self._ocr

    def process_image(self, img, ts: datetime | None = None) -> dict:
        """OCR op één beeld en de gevonden tickers op het bord zetten."""
        ts = ts or now_ny()
        texts = self.ocr()(img)
        symbols = load_symbols()
        found = extract_tickers(texts, symbols)
        with self.lock:
            day = ts.date().isoformat()
            if day != self.board.day:
                self.board = TickerBoard(day)
            new = self.board.add(found, symbols, ts)
            self.board.save()
        return {"tickers": found, "new": new, "text": texts}

    # ---- achtergrondlus ----
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="live-scanner", daemon=True)
        self._thread.start()
        self.status["running"] = True

    def stop(self):
        self._stop.set()
        self.status.update(running=False, state="gestopt")

    def _set(self, state: str, error: str | None = None):
        self.status.update(state=state, error=error)

    def _loop(self):
        while not self._stop.is_set():
            wait = self.interval
            try:
                if not market_open(now_ny()):
                    self._stream = None
                    self._set("wacht op handelsdag (04:00-20:00 New York)")
                    wait = 60
                else:
                    self._tick()
            except Exception as e:  # noqa: BLE001 - lus moet blijven draaien
                log.exception("scanner")
                self._stream = None
                self._set("fout, probeer opnieuw", f"{type(e).__name__}: {e}")
                wait = max(30, self.interval)
            self._stop.wait(wait)

    def _tick(self):
        if self._stream is None:
            self._set("livestream zoeken")
            self._stream = resolve_stream(self.url)
            self.status["stream"] = {k: self._stream[k] for k in ("title", "id", "is_live", "webpage_url")}
        frame = grab_frame(self._stream["url"])
        if frame is None:
            self._stream = None  # URL verlopen of stream gestopt: volgende keer opnieuw zoeken
            raise RuntimeError("Geen beeld uit de stream")
        res = self.process_image(frame)
        self.status.update(frames=self.status["frames"] + 1, last_frame=now_ny().isoformat(timespec="seconds"),
                           last_tickers=res["tickers"])
        self._set("live")


scanner = LiveScanner()


def days() -> list[str]:
    return sorted((p.stem for p in SCAN_DIR.glob("*.json")), reverse=True)


def board_for(day: str | None) -> TickerBoard:
    if not day or day == scanner.board.day:
        return scanner.board
    return TickerBoard(day)
