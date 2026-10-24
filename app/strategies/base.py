from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd


class Strategy(ABC):
    """Een strategie geeft per dag een gewenste positie terug: 1 = long, 0 = cash, -1 = short.

    Het signaal van dag t wordt pas op dag t+1 uitgevoerd (geen look-ahead bias).
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(self.params)
        if unknown:
            raise ValueError(f"Onbekende parameters voor {self.name}: {sorted(unknown)}")
        self.p = {**self.params, **kwargs}

    @abstractmethod
    def signals(self, prices: pd.DataFrame) -> pd.Series: ...
