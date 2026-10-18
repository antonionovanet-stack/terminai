from datetime import datetime, timezone
from types import SimpleNamespace as NS

import numpy as np
import pytest

from forex_trader.brokers.mt5 import MT5Broker, MT5Error

T0 = int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp())


class FakeMT5:
    TIMEFRAME_H1 = 16385
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    DEAL_TYPE_BUY, DEAL_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
    DEAL_ENTRY_IN, DEAL_ENTRY_OUT = 0, 1
    DEAL_REASON_SL, DEAL_REASON_TP = 4, 5

    def __init__(self, trade_mode=0, retcode=10009):
        self.trade_mode, self.retcode = trade_mode, retcode
        self.requests, self.init_kwargs, self.closed = [], None, False
        self.positions, self.deals = [], {}

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        return True

    def shutdown(self):
        self.closed = True

    def last_error(self):
        return (1, "fake")

    def account_info(self):
        return NS(login=123, server="Demo-Server", currency="USD", balance=10_000.0, equity=10_050.0,
                  trade_mode=self.trade_mode)

    def terminal_info(self):
        return NS(trade_allowed=True)

    def symbol_select(self, symbol, enable):
        return symbol in ("EURUSD.m", "USDJPY.m")

    def symbol_info(self, symbol):
        digits = 3 if "JPY" in symbol else 5
        return NS(digits=digits, trade_contract_size=100_000.0, volume_min=0.01, volume_max=50.0,
                  volume_step=0.01, filling_mode=2)

    def symbol_info_tick(self, symbol):
        return NS(bid=150.00, ask=150.02, time=T0) if "JPY" in symbol else NS(bid=1.10000, ask=1.10010, time=T0)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                 ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
        rows = [(T0 + 3600 * i, 1.1, 1.101, 1.099, 1.1005, 100, 1, 0) for i in range(count)]
        self.last_rates_args = (symbol, timeframe, start, count)
        return np.array(rows, dtype=dtype)

    def positions_get(self, **query):
        if "ticket" in query:
            return tuple(p for p in self.positions if p.ticket == query["ticket"])
        return tuple(self.positions)

    def history_deals_get(self, position):
        return self.deals.get(position, ())

    def order_send(self, request):
        self.requests.append(request)
        return NS(retcode=self.retcode, order=555, comment="ok" if self.retcode == 10009 else "no money")


def _broker(fake=None, **kw):
    return MT5Broker(symbol_suffix=".m", mt5_module=fake or FakeMT5(), **kw)


def test_symbol_mapping_with_suffix_and_override():
    b = _broker(symbol_map={"GBP_USD": "GBPUSDpro"})
    assert b.symbol("EUR_USD") == "EURUSD.m"
    assert b.symbol("GBP_USD") == "GBPUSDpro"
    assert b.instrument("EURUSD.m") == "EUR_USD"
    assert b.instrument("GBPUSDpro") == "GBP_USD"
    assert b.instrument("XAUUSD_spot.m") is None


def test_login_credentials_forwarded():
    fake = FakeMT5()
    MT5Broker(login="42", password="pw", server="Broker-Demo", mt5_module=fake)
    assert fake.init_kwargs == {"login": 42, "password": "pw", "server": "Broker-Demo"}


def test_real_account_blocked_without_double_opt_in(monkeypatch):
    monkeypatch.delenv("FOREX_ALLOW_LIVE", raising=False)
    fake = FakeMT5(trade_mode=2)
    with pytest.raises(MT5Error, match="REALE"):
        _broker(fake, live=True)
    assert fake.closed
    monkeypatch.setenv("FOREX_ALLOW_LIVE", "yes")
    with pytest.raises(MT5Error):
        _broker(FakeMT5(trade_mode=2), live=False)
    _broker(FakeMT5(trade_mode=2), live=True)


def test_candles_skip_forming_bar_and_rename_volume():
    fake = FakeMT5()
    df = _broker(fake).get_candles("EUR_USD", 300)
    assert fake.last_rates_args == ("EURUSD.m", FakeMT5.TIMEFRAME_H1, 1, 300)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC" and len(df) == 300


def test_units_to_lots_rounds_down_and_enforces_minimum():
    b = _broker()
    assert b.units_to_lots("EURUSD.m", 33_333) == 0.33
    assert b.units_to_lots("EURUSD.m", -129_999) == 1.29
    with pytest.raises(MT5Error, match="minimo"):
        b.units_to_lots("EURUSD.m", 900)


def test_market_order_request_shape():
    fake = FakeMT5()
    oid = _broker(fake).place_market_order("EUR_USD", -33_333, stop_loss=1.103004, take_profit=1.094)
    req = fake.requests[-1]
    assert oid == "555"
    assert req["type"] == FakeMT5.ORDER_TYPE_SELL and req["price"] == 1.10000
    assert req["volume"] == 0.33 and req["sl"] == 1.103 and req["tp"] == 1.094
    assert req["magic"] == 20261018 and req["type_filling"] == FakeMT5.ORDER_FILLING_IOC


def test_rejected_order_raises():
    with pytest.raises(MT5Error, match="10019"):
        _broker(FakeMT5(retcode=10019)).place_market_order("EUR_USD", 50_000, 1.09, 1.12)


def test_only_own_positions_are_managed():
    fake = FakeMT5()
    mine = NS(ticket=7, magic=20261018, symbol="EURUSD.m", type=0, volume=0.5, price_open=1.1,
              sl=1.095, tp=1.11, time=T0, profit=12.5)
    manual = NS(ticket=8, magic=0, symbol="EURUSD.m", type=1, volume=1.0, price_open=1.1,
                sl=0.0, tp=0.0, time=T0, profit=-3.0)
    fake.positions = [mine, manual]
    positions = _broker(fake).open_positions()
    assert [p.id for p in positions] == ["7"]
    assert positions[0].units == 50_000 and positions[0].stop_loss == 1.095


def test_modify_stop_and_close_requests():
    fake = FakeMT5()
    fake.positions = [NS(ticket=7, magic=20261018, symbol="EURUSD.m", type=0, volume=0.5, price_open=1.1,
                         sl=1.095, tp=1.11, time=T0, profit=0.0)]
    b = _broker(fake)
    b.modify_stop("7", 1.1000004)
    assert fake.requests[-1] == {"action": 6, "position": 7, "symbol": "EURUSD.m", "sl": 1.1,
                                 "tp": 1.11, "magic": 20261018}
    b.close_position("7")
    close = fake.requests[-1]
    assert close["type"] == FakeMT5.ORDER_TYPE_SELL and close["position"] == 7 and close["volume"] == 0.5


def test_closed_trade_from_deal_history():
    fake = FakeMT5()
    fake.deals[7] = (
        NS(entry=0, type=0, symbol="EURUSD.m", volume=0.5, price=1.1, time=T0, profit=0.0,
           commission=-3.5, swap=0.0, reason=3),
        NS(entry=1, type=1, symbol="EURUSD.m", volume=0.5, price=1.095, time=T0 + 7200, profit=-250.0,
           commission=-3.5, swap=-0.4, reason=4),
    )
    t = _broker(fake).closed_trade("7")
    assert t.reason == "stop_loss" and t.units == 50_000
    assert t.realized_pl == pytest.approx(-257.4)
    assert t.exit_price == 1.095


def test_conversion_rate_uses_inverse_pair():
    assert _broker().conversion_rate("JPY", "USD") == pytest.approx(1 / 150.01)
