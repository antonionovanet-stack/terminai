"""Indicatori tecnici vettoriali. Tutti accettano Series/DataFrame pandas con colonne open/high/low/close."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), period)
    avg_loss = _wilder(-delta.clip(upper=0), period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # Nessuna perdita nella finestra: RSI al massimo, non indefinito.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr, index=df.index)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _wilder(true_range(df), period)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def adx(df: pd.DataFrame, period: int = 14, atr_: pd.Series | None = None) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    if atr_ is None:
        atr_ = atr(df, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr_
    minus_di = 100 * _wilder(minus_dm, period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di})


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + k * std, "lower": mid - k * std, "width": (2 * k * std) / mid}
    )


def swing_levels(df: pd.DataFrame, wing: int = 3, max_levels: int = 3) -> tuple[list[float], list[float]]:
    """Massimi/minimi di swing confermati (frattali), dal più recente. Ritorna (supporti, resistenze)."""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    resistances: list[float] = []
    supports: list[float] = []
    for i in range(len(df) - wing - 1, wing - 1, -1):
        window_h = highs[i - wing : i + wing + 1]
        window_l = lows[i - wing : i + wing + 1]
        if len(resistances) < max_levels and highs[i] == window_h.max():
            resistances.append(float(highs[i]))
        if len(supports) < max_levels and lows[i] == window_l.min():
            supports.append(float(lows[i]))
        if len(resistances) >= max_levels and len(supports) >= max_levels:
            break
    return supports, resistances
