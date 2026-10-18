import pandas as pd

from forex_trader.brokers.paper import PaperBroker


def _df(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_fill_next_open_and_take_profit():
    df = _df([(1.10, 1.101, 1.099, 1.100), (1.1005, 1.1010, 1.1000, 1.1008), (1.1008, 1.1050, 1.1005, 1.1040)])
    b = PaperBroker({"EUR_USD": df}, spread_pips=1.0)
    b.place_market_order("EUR_USD", 10_000, stop_loss=1.0980, take_profit=1.1030)
    assert b.open_positions() == []
    b.advance()
    pos = b.open_positions()[0]
    assert abs(pos.entry_price - 1.10055) < 1e-9
    closed = b.advance()
    assert closed[0].reason == "take_profit"
    assert abs(b.balance - (10_000 + 10_000 * (1.1030 - 1.10055))) < 1e-6


def test_stop_wins_when_both_touched_and_gap_fills_worse():
    df = _df([(1.10, 1.101, 1.099, 1.100), (1.1000, 1.1001, 1.0999, 1.1000), (1.0950, 1.1100, 1.0900, 1.1000)])
    b = PaperBroker({"EUR_USD": df}, spread_pips=0.0)
    b.place_market_order("EUR_USD", 1000, stop_loss=1.0980, take_profit=1.1050)
    b.advance()
    closed = b.advance()
    assert closed[0].reason == "stop_loss"
    assert closed[0].exit_price == 1.0950  # gap oltre lo stop: esecuzione all'apertura


def test_short_position_and_conversion_for_jpy():
    df = _df([(150.0, 150.1, 149.9, 150.0)] * 2 + [(150.0, 150.0, 148.0, 148.5)])
    b = PaperBroker({"USD_JPY": df}, spread_pips=0.0)
    b.place_market_order("USD_JPY", -1000, stop_loss=151.0, take_profit=149.0)
    b.advance()
    closed = b.advance()
    assert closed[0].reason == "take_profit"
    assert abs(closed[0].realized_pl - 1000 * 1.0 / 148.5) < 1e-6
