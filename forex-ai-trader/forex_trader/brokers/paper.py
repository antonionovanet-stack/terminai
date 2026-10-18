"""Broker simulato su dati storici, usato per backtest.

Realismo: gli ordini si eseguono all'apertura della barra successiva (niente look-ahead), con spread
bid/ask; se in una barra vengono toccati sia stop sia target si assume prima lo stop (ipotesi prudente).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import pandas as pd

from ..fx import pip_size, split_instrument
from ..models import Account, ClosedTrade, Position, Quote


@dataclass
class _Pending:
    id: str
    instrument: str
    units: int
    stop_loss: float
    take_profit: float


class PaperBroker:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        initial_balance: float = 10_000.0,
        account_currency: str = "USD",
        spread_pips: float | dict[str, float] = 1.0,
        start_index: int = 0,
    ) -> None:
        lengths = {len(df) for df in data.values()}
        if len(lengths) != 1:
            raise ValueError("Tutti gli strumenti devono avere lo stesso numero di barre allineate")
        self.data = data
        self.n_bars = lengths.pop()
        self.i = start_index
        self.balance = initial_balance
        self.currency = account_currency
        self._spread = spread_pips
        self._ids = itertools.count(1)
        self._pending: list[_Pending] = []
        self._positions: dict[str, Position] = {}
        self.closed: dict[str, ClosedTrade] = {}

    # --- tempo simulato ---
    def can_advance(self) -> bool:
        return self.i + 1 < self.n_bars

    def advance(self) -> list[ClosedTrade]:
        self.i += 1
        for order in self._pending:
            self._fill(order)
        self._pending.clear()
        closed: list[ClosedTrade] = []
        for pos in list(self._positions.values()):
            trade = self._check_exits(pos)
            if trade:
                closed.append(trade)
        return closed

    @property
    def now(self) -> pd.Timestamp:
        return next(iter(self.data.values())).index[self.i]

    # --- prezzi ---
    def _half_spread(self, instrument: str) -> float:
        pips = self._spread[instrument] if isinstance(self._spread, dict) else self._spread
        return pips * pip_size(instrument) / 2

    def _bar(self, instrument: str) -> pd.Series:
        return self.data[instrument].iloc[self.i]

    def get_candles(self, instrument: str, count: int) -> pd.DataFrame:
        df = self.data[instrument]
        return df.iloc[max(0, self.i - count + 1) : self.i + 1]

    def get_quote(self, instrument: str) -> Quote:
        mid = float(self._bar(instrument)["close"])
        hs = self._half_spread(instrument)
        return Quote(instrument, mid - hs, mid + hs, self.now.to_pydatetime())

    def conversion_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        direct, inverse = f"{from_ccy}_{to_ccy}", f"{to_ccy}_{from_ccy}"
        if direct in self.data:
            return float(self._bar(direct)["close"])
        if inverse in self.data:
            return 1.0 / float(self._bar(inverse)["close"])
        raise ValueError(f"Nessun tasso di conversione {from_ccy}->{to_ccy} nei dati del backtest")

    # --- conto ---
    def _pl(self, pos: Position, exit_price: float) -> float:
        _, quote_ccy = split_instrument(pos.instrument)
        return pos.units * (exit_price - pos.entry_price) * self.conversion_rate(quote_ccy, self.currency)

    def _exit_price_now(self, pos: Position) -> float:
        q = self.get_quote(pos.instrument)
        return q.bid if pos.is_long else q.ask

    def get_account(self) -> Account:
        unrealized = sum(self._pl(p, self._exit_price_now(p)) for p in self._positions.values())
        return Account(balance=self.balance, equity=self.balance + unrealized, currency=self.currency)

    def open_positions(self) -> list[Position]:
        return [replace(p, unrealized_pl=self._pl(p, self._exit_price_now(p))) for p in self._positions.values()]

    # --- ordini ---
    def place_market_order(self, instrument: str, units: int, stop_loss: float, take_profit: float) -> str:
        oid = str(next(self._ids))
        self._pending.append(_Pending(oid, instrument, units, stop_loss, take_profit))
        return oid

    def _fill(self, order: _Pending) -> None:
        open_mid = float(self._bar(order.instrument)["open"])
        hs = self._half_spread(order.instrument)
        price = open_mid + hs if order.units > 0 else open_mid - hs
        self._positions[order.id] = Position(
            id=order.id, instrument=order.instrument, units=order.units, entry_price=price,
            stop_loss=order.stop_loss, take_profit=order.take_profit, open_time=self.now.to_pydatetime(),
        )

    def _check_exits(self, pos: Position) -> ClosedTrade | None:
        bar = self._bar(pos.instrument)
        hs = self._half_spread(pos.instrument)
        o, h, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if pos.is_long:
            # Una posizione long esce al bid.
            bid_open, bid_low, bid_high = o - hs, low - hs, h - hs
            if pos.stop_loss is not None and bid_low <= pos.stop_loss:
                return self._close(pos, min(pos.stop_loss, bid_open), "stop_loss")
            if pos.take_profit is not None and bid_high >= pos.take_profit:
                return self._close(pos, max(pos.take_profit, bid_open), "take_profit")
        else:
            ask_open, ask_low, ask_high = o + hs, low + hs, h + hs
            if pos.stop_loss is not None and ask_high >= pos.stop_loss:
                return self._close(pos, max(pos.stop_loss, ask_open), "stop_loss")
            if pos.take_profit is not None and ask_low <= pos.take_profit:
                return self._close(pos, min(pos.take_profit, ask_open), "take_profit")
        return None

    def _close(self, pos: Position, price: float, reason: str) -> ClosedTrade:
        pl = self._pl(pos, price)
        self.balance += pl
        del self._positions[pos.id]
        trade = ClosedTrade(
            id=pos.id, instrument=pos.instrument, units=pos.units, entry_price=pos.entry_price,
            exit_price=price, open_time=pos.open_time, close_time=self.now.to_pydatetime(),
            realized_pl=pl, reason=reason,
        )
        self.closed[pos.id] = trade
        return trade

    def modify_stop(self, position_id: str, stop_loss: float) -> None:
        pos = self._positions[position_id]
        self._positions[position_id] = replace(pos, stop_loss=stop_loss)

    def close_position(self, position_id: str) -> None:
        pos = self._positions[position_id]
        self._close(pos, self._exit_price_now(pos), "manual")

    def close_all(self) -> list[ClosedTrade]:
        self._pending.clear()
        out = []
        for pos in list(self._positions.values()):
            out.append(self._close(pos, self._exit_price_now(pos), "end_of_test"))
        return out

    def closed_trade(self, position_id: str) -> ClosedTrade | None:
        return self.closed.get(position_id)
