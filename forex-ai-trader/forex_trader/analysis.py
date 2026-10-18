"""Lettura del mercato: trend multi-timeframe, regime, livelli chiave."""

from __future__ import annotations

import math

import pandas as pd

from . import indicators as ind
from .fx import pip_size
from .models import MarketSnapshot, Regime, TimeframeView, Trend

HTF_RULE = "4h"
MIN_LTF_BARS = 220
MIN_HTF_BARS = 60
ATR_RANK_WINDOW = 250


def resample(df: pd.DataFrame, rule: str = HTF_RULE) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])


def _trend(close: float, fast: float, slow: float, slope: float) -> Trend:
    if close > fast > slow and slope > 0:
        return "up"
    if close < fast < slow and slope < 0:
        return "down"
    return "flat"


def _view(df: pd.DataFrame, timeframe: str, slow_period: int) -> TimeframeView | None:
    close = df["close"]
    e20, e50, eslow = ind.ema(close, 20), ind.ema(close, 50), ind.ema(close, slow_period)
    atr_s = ind.atr(df)
    adx_df = ind.adx(df, atr_=atr_s)
    macd_df = ind.macd(close)
    bb = ind.bollinger(close)
    rsi_s = ind.rsi(close)

    last = df.index[-1]
    atr_now = float(atr_s.iloc[-1])
    values = [e20.iloc[-1], e50.iloc[-1], eslow.iloc[-1], atr_now, adx_df["adx"].iloc[-1],
              rsi_s.iloc[-1], macd_df["hist"].iloc[-2], bb["mid"].iloc[-1]]
    if any(pd.isna(v) for v in values) or atr_now <= 0:
        return None

    slope = float((e50.iloc[-1] - e50.iloc[-6]) / atr_now) if len(e50) > 6 else 0.0
    recent_atr = atr_s.dropna().iloc[-ATR_RANK_WINDOW:]
    atr_pct = float((recent_atr <= atr_now).mean())
    fast, slow = float(e20.iloc[-1]), float(eslow.iloc[-1])
    c = float(close.loc[last])
    return TimeframeView(
        timeframe=timeframe,
        close=c,
        trend=_trend(c, fast, slow, slope),
        ema20=fast,
        ema50=float(e50.iloc[-1]),
        ema200=float(eslow.iloc[-1]) if slow_period == 200 else None,
        ema50_slope_atr=round(slope, 3),
        adx=float(adx_df["adx"].iloc[-1]),
        plus_di=float(adx_df["plus_di"].iloc[-1]),
        minus_di=float(adx_df["minus_di"].iloc[-1]),
        rsi=float(rsi_s.iloc[-1]),
        macd_hist=float(macd_df["hist"].iloc[-1]),
        macd_hist_prev=float(macd_df["hist"].iloc[-2]),
        atr=atr_now,
        atr_percentile=round(atr_pct, 3),
        bb_upper=float(bb["upper"].iloc[-1]),
        bb_mid=float(bb["mid"].iloc[-1]),
        bb_lower=float(bb["lower"].iloc[-1]),
    )


def classify_regime(ltf: TimeframeView) -> Regime:
    if ltf.atr_percentile >= 0.97:
        return "volatile"
    if ltf.adx >= 25:
        return "trending"
    if ltf.adx < 20:
        return "ranging"
    return "transitional"


def analyze(candles: pd.DataFrame, instrument: str, ltf_name: str = "H1") -> MarketSnapshot | None:
    """Analizza candele del timeframe operativo (H1) e del superiore (H4, ricampionato).

    Ritorna None se lo storico non basta per indicatori affidabili.
    """
    if len(candles) < MIN_LTF_BARS:
        return None
    htf_df = resample(candles)
    if len(htf_df) < MIN_HTF_BARS:
        return None
    # Il timeframe operativo usa EMA200; l'H4 usa EMA50 come lenta per ridurre il warm-up richiesto.
    ltf = _view(candles, ltf_name, slow_period=200)
    htf = _view(htf_df, "H4", slow_period=50)
    if ltf is None or htf is None:
        return None

    supports, resistances = ind.swing_levels(candles.iloc[-200:])
    tail = candles.iloc[-24:]
    digits = max(2, int(-math.log10(pip_size(instrument))) + 1)
    recent = [
        {"t": ts.isoformat(), "o": round(r.open, digits), "h": round(r.high, digits),
         "l": round(r.low, digits), "c": round(r.close, digits)}
        for ts, r in tail.iterrows()
    ]
    return MarketSnapshot(
        instrument=instrument,
        time=candles.index[-1].to_pydatetime(),
        price=ltf.close,
        pip_size=pip_size(instrument),
        ltf=ltf,
        htf=htf,
        regime=classify_regime(ltf),
        supports=supports,
        resistances=resistances,
        recent_candles=recent,
    )
