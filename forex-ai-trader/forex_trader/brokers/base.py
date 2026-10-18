from __future__ import annotations

from typing import Protocol

import pandas as pd

from ..models import Account, ClosedTrade, Position, Quote


class Broker(Protocol):
    """Contratto minimo che ogni broker (simulato o reale) deve rispettare."""

    def get_candles(self, instrument: str, count: int) -> pd.DataFrame:
        """Candele H1 complete, indicizzate per tempo UTC, colonne open/high/low/close[/volume]."""
        ...

    def get_quote(self, instrument: str) -> Quote: ...

    def get_account(self) -> Account: ...

    def open_positions(self) -> list[Position]: ...

    def place_market_order(
        self, instrument: str, units: int, stop_loss: float, take_profit: float
    ) -> str: ...

    def modify_stop(self, position_id: str, stop_loss: float) -> None: ...

    def close_position(self, position_id: str) -> None: ...

    def closed_trade(self, position_id: str) -> ClosedTrade | None: ...

    def conversion_rate(self, from_ccy: str, to_ccy: str) -> float:
        """Quante unità di to_ccy vale 1 unità di from_ccy."""
        ...
