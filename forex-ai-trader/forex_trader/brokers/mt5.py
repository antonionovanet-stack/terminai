"""Connettore MetaTrader 5 tramite il pacchetto ufficiale `MetaTrader5`.

Il pacchetto dialoga con un terminale MT5 in esecuzione sullo stesso PC (solo Windows): il programma va
avviato sul computer dove MT5 è aperto, con il pulsante "Algo Trading" attivo.

Differenze rispetto al resto del sistema, gestite qui:
- MT5 ragiona in lotti: le unità calcolate dal gestore del rischio vengono convertite in lotti
  arrotondando per difetto al passo del simbolo (mai più rischio del previsto).
- I simboli hanno spesso un suffisso del broker (es. EURUSD.m): configurabile con symbol_suffix o
  symbol_map.
- Il bot gestisce solo le posizioni con il proprio magic number, così non tocca i trade manuali.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any

import pandas as pd

from ..fx import split_instrument
from ..models import Account, ClosedTrade, Position, Quote

log = logging.getLogger(__name__)

LIVE_OPT_IN_ENV = "FOREX_ALLOW_LIVE"
DEFAULT_MAGIC = 20261018

# Valori documentati da MetaQuotes, usati se il modulo non espone la costante.
_RETCODE_OK = {10008, 10009, 10010}  # PLACED, DONE, DONE_PARTIAL
_ACCOUNT_REAL = 2
_DEAL_ENTRY_IN = 0
_DEAL_REASON_SL = 4
_DEAL_REASON_TP = 5
_FILLING_FOK_FLAG = 1
_FILLING_IOC_FLAG = 2


class MT5Error(RuntimeError):
    pass


def _load_module() -> ModuleType:
    try:
        import MetaTrader5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MT5Error(
            "Pacchetto MetaTrader5 non installato. Funziona solo su Windows con il terminale MT5: "
            "pip install MetaTrader5"
        ) from exc
    return MetaTrader5


class MT5Broker:
    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        live: bool = False,
        symbol_suffix: str = "",
        symbol_map: dict[str, str] | None = None,
        magic: int = DEFAULT_MAGIC,
        deviation_points: int = 20,
        server_utc_offset_hours: float = 0.0,
        mt5_module: ModuleType | Any | None = None,
    ) -> None:
        self.mt5 = mt5_module or _load_module()
        self.suffix = symbol_suffix
        self.symbol_map = dict(symbol_map or {})
        self._reverse_map = {v: k for k, v in self.symbol_map.items()}
        self.magic = magic
        self.deviation = deviation_points
        self.utc_offset = timedelta(hours=server_utc_offset_hours)

        kwargs: dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs.update(login=int(login), password=password or "", server=server or "")
        if not self.mt5.initialize(**kwargs):
            raise MT5Error(f"Connessione a MetaTrader 5 fallita: {self.mt5.last_error()}")

        info = self.mt5.account_info()
        if info is None:
            raise MT5Error(f"Nessun conto connesso nel terminale MT5: {self.mt5.last_error()}")
        is_real = info.trade_mode == getattr(self.mt5, "ACCOUNT_TRADE_MODE_REAL", _ACCOUNT_REAL)
        if is_real and not (live and os.environ.get(LIVE_OPT_IN_ENV) == "yes"):
            self.mt5.shutdown()
            raise MT5Error(
                f"Il conto {info.login} è REALE: per operarci servono --live e {LIVE_OPT_IN_ENV}=yes."
            )
        terminal = self.mt5.terminal_info()
        if terminal is not None and not terminal.trade_allowed:
            log.warning("Algo Trading disattivato nel terminale MT5: gli ordini verranno rifiutati.")
        log.info("Connesso a MT5: conto %s (%s) su %s, %s", info.login,
                 "REALE" if is_real else "demo", info.server, info.currency)

    @classmethod
    def from_env(cls, live: bool = False, **options: Any) -> "MT5Broker":
        login = os.environ.get("MT5_LOGIN")
        return cls(
            login=int(login) if login else None,
            password=os.environ.get("MT5_PASSWORD"),
            server=os.environ.get("MT5_SERVER"),
            path=os.environ.get("MT5_PATH") or options.pop("path", None),
            live=live,
            **options,
        )

    def shutdown(self) -> None:
        self.mt5.shutdown()

    # --- simboli ---
    def symbol(self, instrument: str) -> str:
        if instrument in self.symbol_map:
            return self.symbol_map[instrument]
        base, quote = split_instrument(instrument)
        return f"{base}{quote}{self.suffix}"

    def instrument(self, symbol: str) -> str | None:
        if symbol in self._reverse_map:
            return self._reverse_map[symbol]
        core = symbol[: len(symbol) - len(self.suffix)] if self.suffix and symbol.endswith(self.suffix) else symbol
        if len(core) == 6 and core.isalpha():
            return f"{core[:3].upper()}_{core[3:].upper()}"
        return None

    def _info(self, symbol: str) -> Any:
        if not self.mt5.symbol_select(symbol, True):
            raise MT5Error(f"Simbolo {symbol} non disponibile: {self.mt5.last_error()}")
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"Informazioni sul simbolo {symbol} non disponibili")
        return info

    def _tick(self, symbol: str) -> Any:
        self._info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            raise MT5Error(f"Nessun prezzo per {symbol} (mercato chiuso?)")
        return tick

    def _server_time(self, epoch: float) -> datetime:
        return datetime.fromtimestamp(epoch, tz=timezone.utc) - self.utc_offset

    # --- dati ---
    def get_candles(self, instrument: str, count: int) -> pd.DataFrame:
        symbol = self.symbol(instrument)
        self._info(symbol)
        # start_pos=1 salta la barra in formazione: si ragiona solo su candele chiuse.
        rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_H1, 1, count)
        if rates is None or len(rates) == 0:
            raise MT5Error(f"Nessuna candela per {symbol}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) - self.utc_offset
        df = df.rename(columns={"tick_volume": "volume"}).set_index("time").sort_index()
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def get_quote(self, instrument: str) -> Quote:
        tick = self._tick(self.symbol(instrument))
        return Quote(instrument, float(tick.bid), float(tick.ask), self._server_time(tick.time))

    def conversion_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        try:
            tick = self._tick(self.symbol(f"{from_ccy}_{to_ccy}"))
            return (tick.bid + tick.ask) / 2
        except MT5Error:
            tick = self._tick(self.symbol(f"{to_ccy}_{from_ccy}"))
            return 2 / (tick.bid + tick.ask)

    # --- conto ---
    def get_account(self) -> Account:
        info = self.mt5.account_info()
        if info is None:
            raise MT5Error(f"Conto non disponibile: {self.mt5.last_error()}")
        return Account(balance=float(info.balance), equity=float(info.equity), currency=info.currency)

    def _own_positions(self, **query: Any) -> list[Any]:
        raw = self.mt5.positions_get(**query)
        return [p for p in (raw or ()) if p.magic == self.magic]

    def _to_position(self, p: Any) -> Position | None:
        instrument = self.instrument(p.symbol)
        if instrument is None:
            return None
        contract = float(self._info(p.symbol).trade_contract_size)
        sign = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
        return Position(
            id=str(p.ticket), instrument=instrument, units=sign * p.volume * contract,
            entry_price=float(p.price_open), stop_loss=float(p.sl) or None, take_profit=float(p.tp) or None,
            open_time=self._server_time(p.time), unrealized_pl=float(p.profit),
        )

    def open_positions(self) -> list[Position]:
        out = (self._to_position(p) for p in self._own_positions())
        return [p for p in out if p is not None]

    # --- ordini ---
    def units_to_lots(self, symbol: str, units: float) -> float:
        info = self._info(symbol)
        step = float(info.volume_step)
        lots = math.floor(abs(units) / float(info.trade_contract_size) / step + 1e-9) * step
        lots = round(min(lots, float(info.volume_max)), max(0, -int(math.floor(math.log10(step)))))
        if lots < float(info.volume_min):
            raise MT5Error(
                f"{abs(units):.0f} unità = {lots} lotti, sotto il minimo {info.volume_min} di {symbol}"
            )
        return lots

    def _filling(self, info: Any) -> int:
        mode = int(getattr(info, "filling_mode", 0))
        if mode & _FILLING_FOK_FLAG:
            return self.mt5.ORDER_FILLING_FOK
        if mode & _FILLING_IOC_FLAG:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, request: dict[str, Any]) -> Any:
        result = self.mt5.order_send(request)
        if result is None:
            raise MT5Error(f"order_send fallito: {self.mt5.last_error()}")
        if result.retcode not in _RETCODE_OK:
            raise MT5Error(f"Ordine rifiutato da MT5 (retcode {result.retcode}): {result.comment}")
        return result

    def place_market_order(self, instrument: str, units: int, stop_loss: float, take_profit: float) -> str:
        symbol = self.symbol(instrument)
        info = self._info(symbol)
        tick = self._tick(symbol)
        is_buy = units > 0
        result = self._send({
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.units_to_lots(symbol, units),
            "type": self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL,
            "price": float(tick.ask if is_buy else tick.bid),
            "sl": round(stop_loss, info.digits),
            "tp": round(take_profit, info.digits),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "forex-ai-trader",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        })
        # In MT5 il ticket della posizione coincide con quello dell'ordine che l'ha aperta.
        return str(result.order)

    def _position(self, position_id: str) -> Any:
        found = self._own_positions(ticket=int(position_id))
        if not found:
            raise MT5Error(f"Posizione {position_id} non trovata")
        return found[0]

    def modify_stop(self, position_id: str, stop_loss: float) -> None:
        p = self._position(position_id)
        digits = self._info(p.symbol).digits
        self._send({
            "action": self.mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol,
            "sl": round(stop_loss, digits), "tp": p.tp, "magic": self.magic,
        })

    def close_position(self, position_id: str) -> None:
        p = self._position(position_id)
        info = self._info(p.symbol)
        tick = self._tick(p.symbol)
        is_long = p.type == self.mt5.POSITION_TYPE_BUY
        self._send({
            "action": self.mt5.TRADE_ACTION_DEAL, "position": p.ticket, "symbol": p.symbol,
            "volume": p.volume, "type": self.mt5.ORDER_TYPE_SELL if is_long else self.mt5.ORDER_TYPE_BUY,
            "price": float(tick.bid if is_long else tick.ask), "deviation": self.deviation,
            "magic": self.magic, "comment": "forex-ai-trader close",
            "type_time": self.mt5.ORDER_TIME_GTC, "type_filling": self._filling(info),
        })

    def closed_trade(self, position_id: str) -> ClosedTrade | None:
        if self.mt5.positions_get(ticket=int(position_id)):
            return None
        deals = self.mt5.history_deals_get(position=int(position_id))
        if not deals:
            return None
        entry_in = getattr(self.mt5, "DEAL_ENTRY_IN", _DEAL_ENTRY_IN)
        opens = [d for d in deals if d.entry == entry_in]
        closes = [d for d in deals if d.entry != entry_in]
        if not opens or not closes:
            return None
        first, last = opens[0], closes[-1]
        instrument = self.instrument(first.symbol) or first.symbol
        contract = float(self._info(first.symbol).trade_contract_size)
        sign = 1 if first.type == self.mt5.DEAL_TYPE_BUY else -1
        reason = {
            getattr(self.mt5, "DEAL_REASON_SL", _DEAL_REASON_SL): "stop_loss",
            getattr(self.mt5, "DEAL_REASON_TP", _DEAL_REASON_TP): "take_profit",
        }.get(last.reason, "closed")
        return ClosedTrade(
            id=position_id, instrument=instrument, units=sign * first.volume * contract,
            entry_price=float(first.price), exit_price=float(last.price),
            open_time=self._server_time(first.time), close_time=self._server_time(last.time),
            realized_pl=float(sum(d.profit + d.commission + d.swap for d in deals)), reason=reason,
        )
