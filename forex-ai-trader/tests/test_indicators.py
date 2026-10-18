import numpy as np
import pandas as pd

from forex_trader import indicators as ind


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(1.0, 2.0, 100))
    assert ind.rsi(up).iloc[-1] == 100.0
    down = pd.Series(np.linspace(2.0, 1.0, 100))
    assert ind.rsi(down).iloc[-1] < 1.0


def test_atr_matches_constant_range():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    df = pd.DataFrame({"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0}, index=idx)
    assert abs(ind.atr(df).iloc[-1] - 0.02) < 1e-9


def test_adx_high_in_steady_trend(candles):
    from tests.conftest import make_trend

    out = ind.adx(make_trend())
    assert out["adx"].iloc[-1] > 25
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]


def test_swing_levels_ordering(candles):
    supports, resistances = ind.swing_levels(candles)
    assert supports and resistances
    assert all(isinstance(x, float) for x in supports + resistances)


def test_macd_hist_is_line_minus_signal(candles):
    m = ind.macd(candles["close"]).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])
