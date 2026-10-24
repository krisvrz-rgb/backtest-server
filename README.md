# Backtest Server

Python + FastAPI server die handelsstrategieën op aandelen backtest, met export naar
TradingView Pine Script v6 zodat dezelfde strategie ook in de Strategy Tester zichtbaar is.

## Starten

```bash
pip install -r requirements.txt
python run.py
```

De server draait standaard op poort **9000** (poort 8000 is al de portfolio server). Andere poort:
`BACKTEST_PORT=9100 python run.py`. Open daarna http://localhost:9000 voor de startpagina (backtest draaien, grafiek, trades en Pine Script kopiëren) of http://localhost:9000/docs voor de API.

## Endpoints

| Methode | Pad | Wat |
|---|---|---|
| GET | `/strategies` | Alle strategieën (120) met categorie en standaardparameters |
| GET | `/categories` | Categorieën en aantal strategieën per categorie |
| POST | `/rank` | Alle strategieën op één ticker testen, gerangschikt op rendement (of `sort_by`) |
| POST | `/backtest` | Eén backtest: metrics, trades en equity curve |
| POST | `/compare` | Meerdere symbolen x strategieën, gesorteerd op Sharpe |
| POST | `/pine` | Pine Script v6 voor een strategie, klaar voor de Pine Editor |
| GET/POST | `/data`, `/data/{symbol}` | Koersdata (CSV) bekijken of uploaden |

Voorbeeld:

```bash
curl -X POST localhost:9000/backtest -H 'Content-Type: application/json' -d '{
  "symbol": "AAPL", "strategy": "sma_crossover", "params": {"fast": 50, "slow": 200},
  "source": "yahoo", "start": "2015-01-01", "fee_bps": 5 }'
```

## Databronnen (`source`)

- `csv` (standaard): `data/<SYMBOOL>.csv`. Exporteer koersdata uit TradingView
  (grafiek > Export chart data) en upload met `POST /data/{symbol}`.
- `yahoo`: haalt data op via yfinance.
- `ibkr`: dagkoersen (dividend-gecorrigeerd) via TWS of IB Gateway op je eigen pc. Zet in TWS
  *API > Settings > Enable ActiveX and Socket Clients* aan. Instellen met `IBKR_HOST`, `IBKR_PORT`
  (7497 paper, 7496 live, 4002/4001 Gateway) en `IBKR_CLIENT_ID`. Symbool `AAPL` of `ASML:AEB:EUR`.
- `hyperliquid`: dagcandles via de openbare Hyperliquid API, symbool = coin (bv. `BTC`, `ETH`).
- `synthetic`: willekeurige testdata, handig zonder internet.

## Metrics

Totaal rendement, CAGR, volatiliteit, Sharpe, Sortino, max drawdown, Calmar, aantal trades,
win rate, gemiddeld rendement per trade, exposure. Steeds ook voor buy & hold als benchmark.

## Uitvoering (gelijk aan TradingView)

Signaal op de slotkoers van bar t, order gevuld op de opening van bar t+1, 100% van het kapitaal,
commissie in procent. De Pine-export gebruikt exact dezelfde instellingen, zodat de resultaten
van de server en de Strategy Tester vergelijkbaar zijn (kleine verschillen door afronding of data
blijven mogelijk).

## Strategieën

120 strategieën in zes categorieën: trend volgen, mean reversion, momentum, breakout, volatiliteit en
volume (plus buy & hold als benchmark). Elke strategie is een instap- en een uitstapregel, long-only,
met instelbare parameters. Dezelfde regels worden in Python uitgerekend én naar Pine Script vertaald.

Voor de startdatum laadt de server ~1,5 jaar extra koersdata om indicatoren op te warmen, net zoals
TradingView de volledige historie gebruikt maar pas vanaf de startdatum handelt.

### Eigen strategie toevoegen

Voeg in `app/strategies/library.py` één `add(...)`-regel toe:

```python
add("mijn_strategie", "trend", "Uitleg in één zin.", {"fast": 10, "slow": 30},
    lambda p: (EMA(CLOSE, p.fast) > SMA(CLOSE, p.slow),   # instappen
               EMA(CLOSE, p.fast) < SMA(CLOSE, p.slow)))  # uitstappen
```

Beschikbare bouwstenen staan in `app/strategies/dsl.py`: SMA, EMA, WMA, HMA, VWMA, RSI, MACD, Stoch,
CCI, CMO, TSI, ROC, MOM, Williams %R, MFI, OBV, ATR, DMI/ADX, Supertrend, Parabolic SAR, Bollinger,
Keltner, Donchian, `crosses_above`/`crosses_below`, `expr[n]` (n bars terug) en `&`, `|`, `~`.

## Tests

```bash
pytest -q
```
