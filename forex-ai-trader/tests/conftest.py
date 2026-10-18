import pandas as pd
import pytest

from forex_trader.data import synthetic_candles


@pytest.fixture(scope="session")
def candles() -> pd.DataFrame:
    return synthetic_candles(bars=900, seed=3)


def make_trend(bars: int = 600, step: float = 0.0004, start: float = 1.10) -> pd.DataFrame:
    """Trend rialzista pulito con piccole oscillazioni (pullback regolari)."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=bars, freq="h", tz="UTC")
    wave = 0.0015 * np.sin(np.arange(bars) / 6)
    close = start + step * np.arange(bars) + wave
    open_ = np.concatenate(([start], close[:-1]))
    high = np.maximum(open_, close) + 0.0003
    low = np.minimum(open_, close) - 0.0003
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
