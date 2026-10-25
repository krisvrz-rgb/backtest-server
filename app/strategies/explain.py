"""Uitleg in gewone taal: wat een indicator toont en wat een strategie doet.

De teksten zijn bedoeld voor wie geen technische analyse kent. Ze worden getoond onder de grafiek
(per indicatorvenster) en bij de keuze van een strategie.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

# Sleutel = eerste woord van de titel van een indicatorvenster of het label van een lijn op de koersgrafiek.
INDICATORS: dict[str, str] = {
    "RSI": "RSI (Relative Strength Index) meet hoe sterk de koers de laatste dagen steeg tegenover daalde, op een schaal van 0 tot 100. "
           "Boven 70 is het aandeel 'overgekocht' (snel en veel gestegen), onder 30 'oversold' (snel en veel gedaald). "
           "Boven 50 hebben de stijgingen de bovenhand.",
    "MACD": "MACD is het verschil tussen een snel en een traag voortschrijdend gemiddelde van de koers. Boven 0 zit de korte trend "
            "boven de lange (opwaarts). De signaallijn is een gemiddelde van de MACD zelf: kruist de MACD erboven, dan wint de "
            "koers aan vaart; kruist hij eronder, dan verliest hij vaart.",
    "ROC": "ROC (Rate of Change) is hoeveel procent de koers veranderde tegenover een aantal dagen geleden. Boven 0 = hoger dan toen.",
    "Momentum": "Momentum is het verschil tussen de koers nu en een aantal dagen geleden. Boven 0 = de koers staat hoger dan toen.",
    "Volume": "Volume is het aantal verhandelde aandelen per dag. Een stijging met hoog volume (boven het gemiddelde) is "
              "overtuigender dan een stijging waar weinig handel achter zit.",
    "ADX": "ADX meet hoe sterk een trend is, niet de richting. Boven 25 is er een duidelijke trend, onder 20 beweegt de koers "
           "zijwaarts. +DI boven −DI betekent dat de opwaartse kracht groter is dan de neerwaartse.",
    "Stochastic": "De Stochastic toont waar de slotkoers staat binnen de hoogste en laagste koers van de laatste dagen (0–100). "
                  "Boven 80 sluit de koers dicht bij de top (overgekocht), onder 20 dicht bij de bodem (oversold).",
    "StochRSI": "Stochastic RSI past de Stochastic toe op de RSI: waar staat de RSI binnen zijn eigen bereik van de laatste dagen (0–1 of 0–100). "
                "Reageert sneller dan de gewone RSI en geeft dus meer signalen, ook meer valse.",
    "Stemmen": "Het aantal deelregels (bijvoorbeeld koers boven gemiddelde, RSI boven 50, MACD positief) dat op dit moment 'ja' zegt. "
               "Hoe meer stemmen, hoe meer indicatoren het eens zijn dat de trend opwaarts is.",
    "MFI": "MFI (Money Flow Index) is een RSI die ook het volume meetelt: stroomt er geld in of uit het aandeel (0–100). "
           "Boven 80 overgekocht, onder 20 oversold.",
    "ATR": "ATR (Average True Range) is de gemiddelde dagelijkse beweging van de koers in dollars of euro's. Een hoge ATR betekent "
           "een beweeglijk aandeel; strategieën gebruiken het om stops op een passende afstand te zetten.",
    "OBV": "OBV (On-Balance Volume) telt het volume op op stijgende dagen en trekt het af op dalende dagen. Stijgt de OBV, dan "
           "gebeurt er meer handel op stijgende dagen: kopers hebben de bovenhand.",
    "Williams": "Williams %R toont waar de koers staat binnen het bereik van de laatste dagen, op een schaal van −100 tot 0. "
                "Boven −20 is overgekocht, onder −80 oversold.",
    "CCI": "CCI (Commodity Channel Index) meet hoe ver de koers afwijkt van zijn gemiddelde, vergeleken met hoe ver hij er normaal "
           "van afwijkt. Rond 0 = normaal. Boven +100 staat de koers ongewoon hoog (sterk, of overgekocht), onder −100 ongewoon "
           "laag (zwak, of oversold). Een mean-reversion-strategie koopt onder −100 in de hoop dat de koers terugveert naar het gemiddelde.",
    "CMO": "CMO (Chande Momentum Oscillator) zet de som van de stijgingen tegenover die van de dalingen, van −100 tot +100. "
           "Boven +50 sterk gestegen (overgekocht), onder −50 sterk gedaald (oversold).",
    "TSI": "TSI (True Strength Index) is een dubbel afgevlakt momentum: boven 0 overheerst de stijging. Kruist hij boven zijn "
           "signaallijn, dan neemt het opwaartse momentum toe.",
    "Force": "Force Index = koersverandering × volume. Positief betekent dat kopers met overtuiging (veel volume) de koers omhoog duwen.",
    "AO": "Awesome Oscillator is het verschil tussen een gemiddelde van 5 en van 34 dagen van het midden van de dagkaars. "
          "Boven 0 zit de korte termijn boven de lange: opwaarts momentum.",
    "%b": "%b toont waar de koers staat tegenover de Bollinger Bands: 0 = op de onderste band, 1 = op de bovenste, 0,5 = in het midden. "
          "Onder 0 staat de koers ongewoon laag.",
    "VIX": "Williams VIX Fix bootst de 'angstindex' na voor één aandeel: hoe ver de koers onder de recente top zakte. Een piek betekent "
           "paniek of een scherpe daling, vaak in de buurt van een bodem.",
    "Squeeze-momentum": "Het squeeze-momentum toont de richting waarin de koers losbreekt nadat de beweeglijkheid een tijd laag was "
                        "(de Bollinger Bands lagen binnen de Keltner Channels). Boven 0 en stijgend = uitbraak naar boven.",
    "KST": "KST (Know Sure Thing) combineert vier ROC's over verschillende periodes tot één momentumlijn. Boven de signaallijn = momentum neemt toe.",
    "Aroon": "Aroon meet hoe recent de hoogste (Aroon up) en laagste koers (Aroon down) van de periode was, 0–100. "
             "Aroon up boven Aroon down = recente toppen, dus een opwaartse trend.",
    "CMF": "CMF (Chaikin Money Flow) meet of de koers dicht bij de top of de bodem van de dag sluit, gewogen met volume. "
           "Boven 0 = koopdruk, onder 0 = verkoopdruk.",
    "Chaikin": "De Chaikin Oscillator is het momentum van de accumulatie/distributie-lijn (geld in of uit het aandeel). Boven 0 = er komt geld in.",
    "Vortex": "De Vortex-indicator vergelijkt opwaartse en neerwaartse bewegingen. VI+ boven VI− = opwaartse trend.",
    "Ultimate": "De Ultimate Oscillator combineert koopdruk over 7, 14 en 28 dagen (0–100). Onder 30 oversold, boven 70 overgekocht.",
    "TRIX": "TRIX is de procentuele verandering van een driedubbel afgevlakt gemiddelde: filtert ruis weg. Boven 0 of boven de signaallijn = opwaarts.",
    "Coppock": "De Coppock Curve is een traag momentum voor de lange termijn. Draait hij onder 0 weer omhoog, dan is dat vaak het begin van een herstel.",
    "Hulpreeks": "Een tussenberekening die de strategie gebruikt voor haar koop- of verkoopregel.",
    # Lijnen op de koersgrafiek
    "SMA": "SMA is het gewone gemiddelde van de slotkoers over zoveel dagen. Koers erboven = opwaartse trend op die termijn. "
           "De SMA 200 is de bekendste lijn voor de lange trend.",
    "EMA": "EMA is een gemiddelde van de slotkoers dat recente dagen zwaarder laat wegen, dus sneller reageert dan een SMA.",
    "WMA": "WMA is een gewogen gemiddelde: recente dagen tellen zwaarder, lineair aflopend.",
    "HMA": "HMA (Hull) is een snel gemiddelde dat weinig achterloopt op de koers.",
    "VWMA": "VWMA is een gemiddelde waarin dagen met veel volume zwaarder wegen.",
    "Hoogste": "De hoogste koers van de laatste zoveel dagen. Breekt de koers erboven, dan is dat een uitbraak naar een nieuwe top.",
    "Laagste": "De laagste koers van de laatste zoveel dagen. Zakt de koers eronder, dan is dat een nieuwe bodem (vaak het verkoopsignaal).",
    "BB": "Bollinger Bands liggen 2 standaardafwijkingen boven en onder het gemiddelde van 20 dagen. Ongeveer 95% van de tijd "
          "zit de koers ertussen; een slot buiten de banden is ongewoon.",
    "Midden": "De middenlijn van een kanaal (het gemiddelde waarrond de banden liggen).",
    "Keltner": "Keltner Channels liggen een aantal keer de ATR (gemiddelde dagbeweging) boven en onder een EMA. Een slot erbuiten is een sterke beweging.",
    "Supertrend": "Supertrend is een volglijn op een ATR-afstand van de koers. Koers erboven = opwaartse trend; zakt de koers eronder, dan draait de trend.",
    "SAR": "Parabolic SAR zijn punten die de koers volgen en steeds dichterbij komen. Koers erboven = opwaartse trend; raakt de koers ze, dan draait het.",
    "Regressie": "Een rechte lijn die zo goed mogelijk door de koersen van de laatste dagen loopt; de helling toont de richting.",
}

CATEGORY_INFO: dict[str, dict[str, str]] = {
    "trend_hold": {"how": "Blijft in het aandeel zolang de lange trend goed is en stapt pas uit bij een duidelijke breuk.",
                   "suits": "Aandelen die lang stijgen: je houdt het grootste deel van de stijging vast.",
                   "weak": "Stapt laat uit en laat in een daling eerst een stuk winst liggen; kan bij een snelle V-vormige dip net op de bodem verkopen."},
    "trend": {"how": "Koopt als de koers in een opwaartse trend komt en verkoopt als die trend breekt.",
              "suits": "Aandelen met lange, duidelijke trends.",
              "weak": "In een zijwaartse markt koopt ze vaak net te laat en verkoopt ze te vroeg (veel kleine verliezen, 'whipsaws')."},
    "combo": {"how": "Combineert meerdere voorwaarden (trend, momentum, volume) en koopt pas als die het eens zijn.",
              "suits": "Wie minder maar betere signalen wil.",
              "weak": "Meer regels = meer kans dat de strategie toevallig goed past op het verleden; komt soms laat in een trend."},
    "seasonal": {"how": "Handelt op vaste momenten van de kalender (bv. rond de maandwissel of in bepaalde maanden).",
                 "suits": "Brede indexen en grote aandelen waar zulke kalenderpatronen historisch bestaan.",
                 "weak": "Kijkt niet naar de koers zelf: een crash in een 'goede' periode wordt volledig meegenomen."},
    "mean_reversion": {"how": "Koopt na een ongewoon sterke daling, in de verwachting dat de koers terugveert naar het gemiddelde, "
                              "en verkoopt na het herstel.",
                       "suits": "Aandelen die zijwaarts of in een stijgende trend schommelen, met regelmatige dipjes.",
                       "weak": "In een echte daling koopt ze steeds opnieuw 'goedkoop' terwijl het aandeel blijft zakken; "
                               "houdt de positie soms lang vast tot het herstel komt."},
    "momentum": {"how": "Koopt als de koers aan kracht wint en verkoopt als het momentum wegvalt.",
                 "suits": "Aandelen in een stijgende fase met duidelijke versnellingen.",
                 "weak": "Koopt na een stijging, dus vaak dicht bij een kortetermijntop; valse signalen in een zijwaartse markt."},
    "breakout": {"how": "Koopt als de koers boven een recente top of een kanaal uitbreekt en verkoopt bij een breuk naar beneden.",
                 "suits": "Aandelen die na een rustige periode plots gaan lopen.",
                 "weak": "Veel uitbraken mislukken en vallen terug: kleine verliezen tot er één grote winnaar komt."},
    "volatility": {"how": "Gebruikt de beweeglijkheid van de koers (banden, ATR) om te kopen bij ongewone bewegingen of na rustige periodes.",
                   "suits": "Aandelen waar rustige periodes afwisselen met sterke bewegingen.",
                   "weak": "Een sterke beweging kan net zo goed naar beneden doorzetten; gevoelig voor nieuwsgedreven sprongen."},
    "volume": {"how": "Kijkt of er geld in of uit het aandeel stroomt (volume, geldstroom) en koopt bij koopdruk.",
               "suits": "Liquide aandelen waar grote spelers hun sporen nalaten in het volume.",
               "weak": "Volume is vaak rumoerig (cijfers, indexwijzigingen) en geeft dan valse signalen."},
    "benchmark": {"how": "Koopt op de eerste dag en houdt vast tot het einde: de maatstaf waar elke strategie tegen moet opboksen.",
                  "suits": "Aandelen die op lange termijn stijgen.",
                  "weak": "Geen bescherming bij een daling: je draagt elke crash volledig."},
}


def indicator_info(label: str) -> str | None:
    key = label.split(" ")[0]
    return INDICATORS.get(key) or INDICATORS.get(key.strip("(%"))


@lru_cache(maxsize=1)
def _used_indicators() -> dict[str, list[str]]:
    """Welke indicatoren elke strategie tekent (één keer berekend op een synthetische koersreeks)."""
    import numpy as np
    import pandas as pd

    from . import STRATEGIES
    from .plot import indicators

    idx = pd.bdate_range("2020-01-01", periods=420)
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(idx))))
    df = pd.DataFrame({"open": close * (1 + rng.normal(0, .005, len(idx))), "close": close,
                       "high": close * 1.01, "low": close * 0.99, "volume": rng.integers(1e5, 1e6, len(idx)).astype(float)},
                      index=idx)
    out: dict[str, list[str]] = {}
    for name, cls in STRATEGIES.items():
        keys: list[str] = []
        try:
            entry, exit_ = cls().rules()
            d = indicators(entry, exit_, df, idx[-50:])
            labels = [p["title"] for p in d["panes"]] + [o["label"] for o in d["overlays"]]
        except Exception:  # noqa: BLE001 - strategie zonder regels (bv. buy & hold)
            labels = []
        for lab in labels:
            k = lab.split(" ")[0]
            if indicator_info(lab) and k not in keys:
                keys.append(k)
        out[name] = keys
    return out


def explain_strategy(name: str, category: str) -> dict[str, Any]:
    info = CATEGORY_INFO.get(category, {})
    used = _used_indicators().get(name, [])
    return {"how": info.get("how"), "suits": info.get("suits"), "weak": info.get("weak"),
            "indicators": [{"name": k, "text": INDICATORS[k]} for k in used if k in INDICATORS]}
