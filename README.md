# Backtest Server

Python + FastAPI server die handelsstrategieën op aandelen backtest, met export naar
TradingView Pine Script v6 zodat dezelfde strategie ook in de Strategy Tester zichtbaar is.

## Starten

```bash
pip install -r requirements.txt
python run.py
```

De server draait standaard op poort **9000** (poort 8000 is al de portfolio server). Andere poort:
`BACKTEST_PORT=9100 python run.py`. Open daarna http://localhost:9000 voor het overzicht van alle tickers (plaats van buy & hold, winstgevende strategieën en open buys voor 2015, 2020 en 2024 tot nu, met een knop Vernieuwen), http://localhost:9000/tool voor één backtest (grafiek met de indicatoren van de strategie, trades, ranglijst, optimaliseren, automatische keuze en Pine Script) of http://localhost:9000/docs voor de API.

## Endpoints

| Methode | Pad | Wat |
|---|---|---|
| GET | `/` | Overzicht: per ticker de drie testperiodes naast elkaar |
| GET | `/tool` | Backtestpagina (één ticker) |
| POST | `/adaptive` | Automatische strategiekeuze per markttoestand, getest met walk-forward |
| GET | `/strategies` | Alle strategieën (166) met categorie en standaardparameters |
| GET | `/categories` | Categorieën en aantal strategieën per categorie |
| POST | `/rank` | Alle strategieën op één ticker testen, gerangschikt op rendement (of `sort_by`) |
| POST | `/backtest` | Eén backtest: metrics, trades en equity curve |
| POST | `/compare` | Meerdere symbolen x strategieën, gesorteerd op Sharpe |
| POST | `/optimize` | Instellingen van één strategie fine-tunen: zoeken op 70% van de periode, controleren op de laatste 30% |
| GET | `/watchlist` | Watchlist-pagina: alle tickers met backtestcijfers per groep |
| GET | `/api/watchlist` | Watchlist met groepen en laatste scanresultaten |
| POST | `/api/watchlist/import` | TradingView-export importeren (###SECTIE = groep) |
| POST | `/api/watchlist/import-tradingview` | Alle TradingView-watchlists overnemen als lijsten (uit `TV_WATCHLIST_FILE`, anders de meegeleverde momentopname) |
| POST | `/api/watchlist/import-portfolio` | Groepen van de portfolio server overnemen (lijst "Portfolio server") |
| POST | `/api/watchlist/scan` | Scanner starten: alle strategieën op elke ticker, per testperiode (standaard 2015, 2020, 2024 tot nu) |
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

165 strategieën in negen categorieën: vasthouden met trendfilter, trend volgen, combinaties, kalender,
mean reversion, momentum, breakout, volatiliteit en volume (plus buy & hold als benchmark). Daarbij de bekende
regels uit de literatuur (Connors R3, %b, Double 7s, cumulatieve RSI, Williams VIX Fix, TTM Squeeze, Elder Impulse
en Triple Screen, KST, Aroon, Vortex, Ultimate Oscillator, Coppock, Holy Grail, maandwissel- en Halloween-effect)
en combinaties: trendfilter + instap, meervoudige bevestiging, stemmingen tussen 8 of 12 indicatoren en een
strategie die per markttoestand wisselt. "Vasthouden met trendfilter" blijft zo goed als
altijd belegd en stapt alleen uit in een duidelijke neerwaartse trend: de realistische manier om een sterk
stijgend aandeel bij te houden. Elke strategie is een instap- en een uitstapregel, long-only,
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

## Swarm Desk (`/desk`)

Een live dashboard waarop zes agents (Spotter, Prior, Edge, Kelly, Taker, Closer) per venster van 5 minuten
beslissen of ze BTC, ETH of SOL long of short gaan op Hyperliquid. Koersen, trades en orderboek komen rechtstreeks
van de openbare Hyperliquid-websocket; zonder verbinding valt de pagina terug op een gesimuleerde koers.

- **Spotter** combineert koopdruk (trades), orderboek en momentum tot een score tussen -1 en 1.
- **Prior** leert uit elk afgesloten venster hoeveel die score echt voorspelt (β, met krimp naar 0,25).
- **Edge** vergelijkt de winkans met de break-even na kosten (standaard taker 4,5 bp per kant + 1 bp slippage).
- **Kelly** bepaalt de grootte (kwart-Kelly, afgetopt), **Taker** voert uit, **Closer** rekent af bij sluiting.

Standaard is alles papier (saldo per coin in de browser). Echt geld loopt via de **zwerm op de server**
(`app/desk_auto.py`): die handelt door als de pagina dicht is.

1. `pip install -r requirements.txt` in de venv (hyperliquid-python-sdk en websockets).
2. Stort USDC op Hyperliquid en maak via More → API een API-wallet aan (kan handelen, niet opnemen).
3. Zet in `data/desk.env`:

   ```
   HL_SECRET_KEY=0x...          private key van de API-wallet
   HL_ACCOUNT_ADDRESS=0x...     je hoofdaccount
   HL_SUBACCOUNT=0x...          optioneel: handel op een apart sub-account
   DESK_TOKEN=iets-geheims
   DESK_LIVE=1
   DESK_AUTO=1
   DESK_MAX_USD=25              maximum per order
   DESK_ORDER_USD=20            grootte per order
   DESK_DAILY_LOSS=10           stopt voor de rest van de UTC-dag na zoveel verlies
   DESK_FEE_BPS=4.5
   DESK_MIN_EDGE=2
   ```
4. Herstart de server, vul op `/desk` → Instellingen het token in en zet met **Zwerm live** de zwerm aan.

Met de coinknoppen naast **Zwerm live** kies je op welke coins (BTC, ETH, SOL) de zwerm tegelijk handelt; de limieten gelden voor alle coins samen.
Met **Bevestigen: AAN** wacht elke nieuwe positie 20 seconden op jouw klik; met **UIT** handelt de zwerm zelf.
Posities gaan altijd dicht aan het einde van het venster (reduce-only). Na 3 mislukte orders op rij of bij het
raken van de daglimiet stopt hij. De kill switch zet de zwerm uit en sluit BTC, ETH en SOL.

**Privacy.** Hyperliquid vraagt geen KYC, maar alles staat openbaar on-chain onder je adres. Wil je de
zwerm los van jezelf: gebruik een nieuwe wallet die je nergens anders voor gebruikt, of minstens een sub-account.
Wie stort vanaf een exchange met KYC is via die storting te herleiden. Winst blijft gewoon aangifteplichtig.
Alle live orders komen in `data/desk-orders.jsonl`.

## Tests

```bash
pytest -q
```
