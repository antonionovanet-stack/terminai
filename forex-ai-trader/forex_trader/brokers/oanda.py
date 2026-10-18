"""Connettore OANDA v20 REST. Usa il conto demo (practice) se non abilitato esplicitamente il conto reale."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests

from ..fx import price_precision
from ..models import Account, ClosedTrade, Position, Quote

PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
LIVE_URL = "https://api-fxtrade.oanda.com/v3"
LIVE_OPT_IN_ENV = "FOREX_ALLOW_LIVE"


class OandaError(RuntimeError):
    pass


def _ts(value: str) -> datetime:
    return pd.Timestamp(value).tz_convert("UTC").to_pydatetime()


class OandaBroker:
    def __init__(
        self,
        token: str,
        account_id: str,
        live: bool = False,
        granularity: str = "H1",
        session: requests.Session | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        if live and os.environ.get(LIVE_OPT_IN_ENV) != "yes":
            raise OandaError(
                f"Trading con denaro reale bloccato: imposta {LIVE_OPT_IN_ENV}=yes per confermare consapevolmente."
            )
        self.base_url = LIVE_URL if live else PRACTICE_URL
        self.account_id = account_id
        self.granularity = granularity
        self.timeout_s = timeout_s
        self.http = session or requests.Session()
        self.http.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
             "Accept-Datetime-Format": "RFC3339"}
        )

    @classmethod
    def from_env(cls, live: bool = False) -> "OandaBroker":
        token = os.environ.get("OANDA_API_TOKEN")
        account = os.environ.get("OANDA_ACCOUNT_ID")
        if not token or not account:
            raise OandaError("Imposta OANDA_API_TOKEN e OANDA_ACCOUNT_ID")
        return cls(token, account, live=live)

    def _req(self, method: str, path: str, **kwargs: object) -> dict:
        resp = self.http.request(method, f"{self.base_url}{path}", timeout=self.timeout_s, **kwargs)
        if resp.status_code >= 400:
            raise OandaError(f"OANDA {method} {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def _fmt(self, instrument: str, price: float) -> str:
        return f"{price:.{price_precision(instrument)}f}"

    # --- dati ---
    def get_candles(self, instrument: str, count: int) -> pd.DataFrame:
        data = self._req(
            "GET", f"/instruments/{instrument}/candles",
            params={"granularity": self.granularity, "count": min(count, 5000), "price": "M"},
        )
        rows = [
            {"time": pd.Timestamp(c["time"]), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
             "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": int(c["volume"])}
            for c in data.get("candles", []) if c.get("complete")
        ]
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).set_index("time").sort_index()

    def get_quote(self, instrument: str) -> Quote:
        data = self._req("GET", f"/accounts/{self.account_id}/pricing", params={"instruments": instrument})
        p = data["prices"][0]
        return Quote(instrument, float(p["bids"][0]["price"]), float(p["asks"][0]["price"]), _ts(p["time"]))

    def conversion_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        try:
            return self.get_quote(f"{from_ccy}_{to_ccy}").mid
        except OandaError:
            return 1.0 / self.get_quote(f"{to_ccy}_{from_ccy}").mid

    # --- conto ---
    def get_account(self) -> Account:
        acc = self._req("GET", f"/accounts/{self.account_id}/summary")["account"]
        return Account(balance=float(acc["balance"]), equity=float(acc["NAV"]), currency=acc["currency"])

    def open_positions(self) -> list[Position]:
        trades = self._req("GET", f"/accounts/{self.account_id}/openTrades")["trades"]
        return [
            Position(
                id=t["id"], instrument=t["instrument"], units=float(t["currentUnits"]),
                entry_price=float(t["price"]),
                stop_loss=float(t["stopLossOrder"]["price"]) if "stopLossOrder" in t else None,
                take_profit=float(t["takeProfitOrder"]["price"]) if "takeProfitOrder" in t else None,
                open_time=_ts(t["openTime"]), unrealized_pl=float(t.get("unrealizedPL", 0.0)),
            )
            for t in trades
        ]

    # --- ordini ---
    def place_market_order(self, instrument: str, units: int, stop_loss: float, take_profit: float) -> str:
        body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": self._fmt(instrument, stop_loss)},
                "takeProfitOnFill": {"price": self._fmt(instrument, take_profit)},
            }
        }
        data = self._req("POST", f"/accounts/{self.account_id}/orders", json=body)
        if "orderFillTransaction" not in data:
            reason = data.get("orderCancelTransaction", {}).get("reason", "sconosciuto")
            raise OandaError(f"Ordine non eseguito: {reason}")
        fill = data["orderFillTransaction"]
        opened = fill.get("tradeOpened") or {}
        return str(opened.get("tradeID", fill["id"]))

    def modify_stop(self, position_id: str, stop_loss: float) -> None:
        trade = self._req("GET", f"/accounts/{self.account_id}/trades/{position_id}")["trade"]
        self._req(
            "PUT", f"/accounts/{self.account_id}/trades/{position_id}/orders",
            json={"stopLoss": {"price": self._fmt(trade["instrument"], stop_loss), "timeInForce": "GTC"}},
        )

    def close_position(self, position_id: str) -> None:
        self._req("PUT", f"/accounts/{self.account_id}/trades/{position_id}/close")

    def closed_trade(self, position_id: str) -> ClosedTrade | None:
        t = self._req("GET", f"/accounts/{self.account_id}/trades/{position_id}")["trade"]
        if t.get("state") != "CLOSED":
            return None
        reason = "closed"
        if t.get("stopLossOrder", {}).get("state") == "FILLED":
            reason = "stop_loss"
        elif t.get("takeProfitOrder", {}).get("state") == "FILLED":
            reason = "take_profit"
        return ClosedTrade(
            id=t["id"], instrument=t["instrument"], units=float(t["initialUnits"]),
            entry_price=float(t["price"]), exit_price=float(t.get("averageClosePrice", t["price"])),
            open_time=_ts(t["openTime"]), close_time=_ts(t.get("closeTime", t["openTime"])),
            realized_pl=float(t.get("realizedPL", 0.0)), reason=reason,
        )
