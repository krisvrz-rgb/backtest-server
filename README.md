# Backtest Server

Python + FastAPI server die handelsstrategieën op aandelen backtest, met export naar
TradingView Pine Script v6 zodat dezelfde strategie ook in de Strategy Tester zichtbaar is.

## Starten

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open daarna http://localhost:8000/docs voor de interactieve API.

## Endpoints

| Methode | Pad | Wat |
|---|---|---|
| GET | `/strategies` | Beschikbare strategieën en hun standaardparameters |
| POST | `/backtest` | Eén backtest: metrics, trades en equity curve |
| POST | `/compare` | Meerdere symbolen x strategieën, gesorteerd op Sharpe |
| POST | `/pine` | Pine Script v6 voor een strategie, klaar voor de Pine Editor |
| GET/POST | `/data`, `/data/{symbol}` | Koersdata (CSV) bekijken of uploaden |

Voorbeeld:

```bash
curl -X POST localhost:8000/backtest -H 'Content-Type: application/json' -d '{
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

## Eigen strategie toevoegen

1. Maak in `app/strategies/` een subclass van `Strategy` met `name`, `params` en een
   `signals(prices)` die per dag 1 (long), 0 (cash) of -1 (short) teruggeeft.
2. Registreer hem in `app/strategies/__init__.py`.
3. Voeg in `app/pine.py` een Pine-template toe onder dezelfde naam.

## Tests

```bash
pytest -q
```
