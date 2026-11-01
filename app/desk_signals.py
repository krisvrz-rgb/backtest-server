"""Signalenlab: voorspelt een kandidaat-signaal de koers van de volgende 1, 4 of 24 uur beter dan toeval, ook na kosten?

Geen desk-logica: elk signaal krijgt een eigen, eenvoudige test, zodat duidelijk wordt WAT voorspelt voordat het in de
zwerm gaat. Per munt, per uur (UTC), 2 jaar:
- het signaal op uur h gebruikt alleen data tot en met h (z-score tegenover de 30 dagen ervoor);
- uitkomst = log-rendement van h tot h+k (k = 1, 4, 24 uur);
- handel: |z| ≥ 1 → instappen in de richting die het verleden aanwees, k uur houden, niet overlappend,
  kosten 6 bp per ronde (maker in + taker uit op Hyperliquid) plus funding over de houdtijd;
- walk-forward per kwartaal: de richting (meelopen of tegendraads) komt uit alle kwartalen ervoor (min. 6 maanden).

Kandidaten: momentum/omkeer, koopdruk (taker), funding, open interest, long/short-verhouding van de top-traders,
taker-verhouding (Binance-metrics), BTC als leider voor de altcoins, en het uur van de dag.
"""
from __future__ import annotations

import io
import json
import math
import time
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import desk_backtest as dbt

CACHE = dbt.CACHE
RESULTS = CACHE / "results-signalen.json"
HORIZONS = (1, 4, 24)
COST_BP = 6.0
Z_IN = 1.0
HOUR = 3_600_000


# ---------- data ----------
def load_metrics(coin: str, days: int = 740, client=None) -> pd.DataFrame:
    """Binance-metrics (5 min): open interest, long/short-verhouding top-traders, taker-verhouding. Gecachet per dag."""
    sym = dbt.bn_symbol(coin)
    d = CACHE / sym / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or dbt._client()
    frames = []
    today = datetime.now(timezone.utc).date()
    days_l = [(today - timedelta(days=i)).isoformat() for i in range(days, 0, -1)]

    def fetch(ds: str) -> None:
        f = d / f"{ds}.csv"
        if f.exists():
            return
        raw = dbt._get(f"{dbt.BINANCE}/daily/metrics/{sym}/{sym}-metrics-{ds}.zip", client)
        if raw is None:
            if ds < (today - timedelta(days=3)).isoformat():  # recente dagen kunnen nog komen
                f.write_text("")
            return
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            f.write_text(z.read(z.namelist()[0]).decode())

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:  # ~730 kleine bestanden per munt
            list(ex.map(fetch, days_l))
        for ds in days_l:
            f = d / f"{ds}.csv"
            if f.exists() and f.stat().st_size:
                frames.append(f.read_text())
    finally:
        if own:
            client.close()
    if not frames:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(io.StringIO(t)) for t in frames], ignore_index=True)
    df["t"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.set_index("t").sort_index()
    cols = {"sum_open_interest_value": "oi", "count_toptrader_long_short_ratio": "lsr_top",
            "count_long_short_ratio": "lsr_all", "sum_taker_long_short_vol_ratio": "taker_ratio"}
    return df[[c for c in cols if c in df]].rename(columns=cols).apply(pd.to_numeric, errors="coerce")


def hourly(coin: str, months: int = 24) -> pd.DataFrame:
    """1m-candles → uurtabel: slot, volume, taker-koopvolume, funding per uur."""
    a = dbt.load_klines(coin, months)
    if not len(a):
        return pd.DataFrame()
    df = pd.DataFrame(a, columns=["t", "o", "h", "l", "c", "v", "tb"])
    df["t"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("t")
    hr = pd.DataFrame({"c": df["c"].resample("1h", label="right", closed="left").last(),
                       "v": df["v"].resample("1h", label="right", closed="left").sum(),
                       "tb": df["tb"].resample("1h", label="right", closed="left").sum()}).dropna(subset=["c"])
    try:
        fund = dbt.load_funding(coin, int(a[0, 0]))
    except Exception:
        fund = {}
    if fund:
        fs = pd.Series({pd.Timestamp(h * HOUR, unit="ms", tz="UTC"): v for h, v in fund.items()}).sort_index()
        hr["fund"] = fs.reindex(hr.index).fillna(0.0)
    else:
        hr["fund"] = 0.0
    return hr


# ---------- signalen ----------
def _z(x: pd.Series, days: int = 30) -> pd.Series:
    w = days * 24
    m = x.rolling(w, min_periods=7 * 24).mean()
    s = x.rolling(w, min_periods=7 * 24).std()
    return ((x - m) / s.replace(0, np.nan)).clip(-5, 5)


def signals(hr: pd.DataFrame, met: pd.DataFrame, btc: pd.DataFrame | None) -> pd.DataFrame:
    lc = np.log(hr["c"])
    r1, r4, r24 = lc.diff(1), lc.diff(4), lc.diff(24)
    sig = pd.DataFrame(index=hr.index)
    sig["momentum 1u"] = _z(r1)
    sig["momentum 4u"] = _z(r4)
    sig["momentum 24u"] = _z(r24)
    imb = (2 * hr["tb"] - hr["v"]) / hr["v"].replace(0, np.nan)
    sig["koopdruk 1u"] = _z(imb)
    sig["koopdruk 24u"] = _z(imb.rolling(24).mean())
    sig["funding 8u"] = _z(hr["fund"].rolling(8).mean())
    if len(met):
        m = met.resample("1h", label="right", closed="left").last().reindex(hr.index).ffill(limit=3)
        if "oi" in m:
            oi24 = np.log(m["oi"]).diff(24)
            sig["open interest 24u"] = _z(oi24)
            sig["OI × prijs 24u"] = _z(oi24 * np.sign(r24))  # nieuwe posities in de richting van de beweging
        if "lsr_top" in m:
            sig["long/short top-traders"] = _z(m["lsr_top"])
        if "lsr_all" in m:
            sig["long/short iedereen"] = _z(m["lsr_all"])
        if "taker_ratio" in m:
            sig["taker-verhouding 1u"] = _z(np.log(m["taker_ratio"]))
    if btc is not None:
        b = np.log(btc["c"].reindex(hr.index).ffill())
        sig["BTC 1u ervoor"] = _z(b.diff(1))
        sig["BTC loopt voor 4u"] = _z(b.diff(4)) - _z(r4)  # BTC steeg meer dan deze munt: inhaalbeweging?
    return sig


# ---------- toetsen ----------
def _quarters(idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    q = pd.period_range(idx[0], idx[-1], freq="Q")
    return [(p.start_time.tz_localize("UTC"), (p.end_time + pd.Timedelta(1, "ns")).tz_localize("UTC")) for p in q]


def _trades(z: np.ndarray, fwd: np.ndarray, fund: np.ndarray, k: int, direction: np.ndarray) -> list[tuple[int, float, float]]:
    """Niet-overlappende trades: (uurindex, netto bp, bruto bp). direction per uur: +1 meelopen, −1 tegendraads, 0 niet."""
    out, i, n = [], 0, len(z)
    while i < n - k:
        if direction[i] and not math.isnan(z[i]) and abs(z[i]) >= Z_IN and not math.isnan(fwd[i]):
            side = direction[i] * (1 if z[i] > 0 else -1)
            gross = side * fwd[i] * 1e4
            f = side * fund[i + 1:i + k + 1].sum() * 1e4
            out.append((i, gross - COST_BP - f, gross))
            i += k
        else:
            i += 1
    return out


def _tstat(x: list[float]) -> float | None:
    if len(x) < 3:
        return None
    a = np.array(x)
    s = a.std(ddof=1)
    return round(float(a.mean() / (s / math.sqrt(len(a)))), 2) if s else None


def test_signal(name: str, s: pd.Series, hr: pd.DataFrame, warmup_days: int = 180) -> dict:
    """Walk-forward per kwartaal: richting uit het verleden (teken van de correlatie), toetsen op het kwartaal erna."""
    lc = np.log(hr["c"].values)
    fund = hr["fund"].values
    z = s.values
    idx = hr.index
    t0 = idx[0] + pd.Timedelta(days=warmup_days)
    out = {}
    for k in HORIZONS:
        fwd = np.full(len(lc), np.nan)
        fwd[:-k] = lc[k:] - lc[:-k]
        direction = np.zeros(len(z))
        ic_oos_x, ic_oos_y = [], []
        for a, b in _quarters(idx):
            if a < t0:
                continue
            past = (idx < a) & ~np.isnan(z) & ~np.isnan(fwd)
            # niet-overlappend in het verleden: elk k-de uur
            pi = np.flatnonzero(past)[::k]
            if len(pi) < 200:
                continue
            c = np.corrcoef(z[pi], fwd[pi])[0, 1]
            d = 1 if c > 0 else -1
            cur = (idx >= a) & (idx < b)
            direction[cur] = d
            ci = np.flatnonzero(cur & ~np.isnan(z) & ~np.isnan(fwd))[::k]
            ic_oos_x += list(d * z[ci])
            ic_oos_y += list(fwd[ci])
        tr = _trades(z, fwd, fund, k, direction)
        net = [x[1] for x in tr]
        ic = float(np.corrcoef(ic_oos_x, ic_oos_y)[0, 1]) if len(ic_oos_x) > 30 else None
        out[str(k)] = {"n": len(tr), "bp": round(float(np.mean(net)), 2) if net else None,
                       "gross_bp": round(float(np.mean([x[2] for x in tr])), 2) if tr else None,
                       "t": _tstat(net), "hit": round(sum(1 for x in tr if x[2] > 0) / len(tr) * 100, 1) if tr else None,
                       "ic": round(ic, 4) if ic is not None else None, "ic_n": len(ic_oos_x),
                       "ic_t": round(ic * math.sqrt(len(ic_oos_x)), 2) if ic is not None else None,
                       "trades": [[int(idx[i].value // 1_000_000), round(b_, 2)] for i, b_, _ in tr]}
    return out


def test_hour_of_day(hr: pd.DataFrame, warmup_days: int = 180) -> dict:
    """Uur van de dag: per uur (UTC) het gemiddelde rendement van het verleden; handel alleen uren waar dat
    gemiddelde de kosten overtreft, in die richting."""
    lc = np.log(hr["c"].values)
    fund = hr["fund"].values
    idx = hr.index
    hod = idx.hour.values
    t0 = idx[0] + pd.Timedelta(days=warmup_days)
    out = {}
    for k in (1, 4):
        fwd = np.full(len(lc), np.nan)
        fwd[:-k] = lc[k:] - lc[:-k]
        z = np.zeros(len(lc))
        for a, b in _quarters(idx):
            if a < t0:
                continue
            past = (idx < a) & ~np.isnan(fwd)
            cur = np.flatnonzero((idx >= a) & (idx < b))
            means = {h: np.nanmean(fwd[past & (hod == h)]) * 1e4 for h in range(24)}
            for i in cur:
                m = means[hod[i]]
                z[i] = (1 if m > 0 else -1) * Z_IN * 2 if abs(m) > COST_BP else 0.0
        direction = (z != 0).astype(int)
        tr = _trades(z, fwd, fund, k, direction)
        net = [x[1] for x in tr]
        out[str(k)] = {"n": len(tr), "bp": round(float(np.mean(net)), 2) if net else None,
                       "gross_bp": round(float(np.mean([x[2] for x in tr])), 2) if tr else None,
                       "t": _tstat(net), "hit": round(sum(1 for x in tr if x[2] > 0) / len(tr) * 100, 1) if tr else None,
                       "ic": None, "ic_n": 0, "ic_t": None,
                       "trades": [[int(idx[i].value // 1_000_000), round(b_, 2)] for i, b_, _ in tr]}
    return out


def run_coin(coin: str, months: int = 24, btc: pd.DataFrame | None = None) -> dict:
    t = time.time()
    hr = hourly(coin, months)
    if len(hr) < 24 * 200:
        return {"coin": coin, "error": "te weinig data"}
    try:
        met = load_metrics(coin, days=months * 31)
    except Exception as e:
        met = pd.DataFrame()
    sig = signals(hr, met, None if coin == "BTC" else btc)
    res = {name: test_signal(name, sig[name], hr) for name in sig.columns}
    res["uur van de dag"] = test_hour_of_day(hr)
    return {"coin": coin, "from": int(hr.index[0].value // 1_000_000), "to": int(hr.index[-1].value // 1_000_000),
            "hours": len(hr), "metrics": len(met), "signals": res, "seconds": round(time.time() - t, 1)}


def pool(coins: dict) -> dict:
    """Alle munten samen per signaal en horizon (out-of-sample). Strenge lat omdat er veel signalen getest worden."""
    names = sorted({s for c in coins.values() if "signals" in c for s in c["signals"]})
    out = {}
    for s in names:
        out[s] = {}
        for k in HORIZONS:
            rows = [c["signals"][s].get(str(k)) for c in coins.values() if "signals" in c and s in c["signals"]]
            rows = [r for r in rows if r]
            if not rows:
                continue
            net = [b for r in rows for _, b in r["trades"]]
            ics = [(r["ic"], r["ic_n"]) for r in rows if r["ic"] is not None]
            n_ic = sum(n for _, n in ics)
            ic = sum(i * n for i, n in ics) / n_ic if n_ic else None
            t = _tstat(net)
            mean = float(np.mean(net)) if net else None
            pos = sum(1 for r in rows if (r["bp"] or 0) > 0 and r["n"] >= 10)
            if mean is not None and mean > 0 and (t or 0) >= 3 and pos >= .6 * len(rows):
                verdict = "voorspelt, ook na kosten"
            elif ic is not None and ic * math.sqrt(n_ic) >= 3:
                verdict = "voorspelt, maar te weinig voor de kosten"
            else:
                verdict = "niets"
            out[s][str(k)] = {"coins": len(rows), "n": len(net), "bp": round(mean, 2) if mean is not None else None,
                              "gross_bp": round(float(np.mean([r["gross_bp"] for r in rows if r["gross_bp"] is not None])), 2) if rows else None,
                              "t": t, "pos": pos, "ic": round(ic, 4) if ic is not None else None,
                              "ic_t": round(ic * math.sqrt(n_ic), 2) if ic is not None else None, "verdict": verdict}
    return out


def run_all(coins: list[str] | None = None, months: int = 24) -> dict:
    coins = coins or dbt.DEFAULT_COINS
    btc = hourly("BTC", months)
    res = {"started": int(time.time() * 1000), "months": months, "coins": {},
           "notes": ["Elk signaal: z-score tegenover de 30 dagen ervoor, alleen data tot en met dat uur.",
                     f"Handel bij |z| ≥ {Z_IN:g}, richting gekozen op de kwartalen ervoor (walk-forward, min. 6 maanden), "
                     f"k uur houden, niet overlappend, {COST_BP:g} bp kosten per ronde + funding.",
                     "Verdict over alle munten samen: 'voorspelt, ook na kosten' = gemiddeld > 0 bp, t ≥ 3 en ≥ 60% van de munten in de plus. "
                     "t ≥ 3 in plaats van 2 omdat er veel signalen en horizonnen getest worden.",
                     "Open interest en long/short komen uit de Binance-metrics (5 min); funding uit Hyperliquid."]}
    for c in coins:
        try:
            res["coins"][c] = run_coin(c, months, btc)
        except Exception as e:
            res["coins"][c] = {"coin": c, "error": str(e)[:300]}
        print(c, res["coins"][c].get("error") or f"{res['coins'][c]['seconds']} s", flush=True)
    res["pool"] = pool(res["coins"])
    res["finished"] = int(time.time() * 1000)
    slim = json.loads(json.dumps(res))
    for c in slim["coins"].values():  # trades niet in het resultaatbestand: te groot
        for s in c.get("signals", {}).values():
            for h in s.values():
                h.pop("trades", None)
    RESULTS.write_text(json.dumps(slim))
    return slim


if __name__ == "__main__":  # python -m app.desk_signals [COINS]
    import sys
    out = run_all(sys.argv[1:] or None)
    for s, hs in out["pool"].items():
        print(s, {k: (v["bp"], v["t"], v["ic"], v["verdict"]) for k, v in hs.items()})
