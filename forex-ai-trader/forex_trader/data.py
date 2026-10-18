"""Caricamento dati storici (CSV) e generatore sintetico a regimi per test offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "time": "time", "date": "time", "datetime": "time", "timestamp": "time",
    "open": "open", "o": "open", "high": "high", "h": "high",
    "low": "low", "l": "low", "close": "close", "c": "close", "volume": "volume", "v": "volume",
}


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV con colonne time/open/high/low/close[/volume] (nomi tolleranti). Timestamp interpretati in UTC."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: COLUMN_ALIASES.get(c.strip().lower(), c) for c in df.columns})
    missing = {"time", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df.set_index("time").sort_index()[cols].astype(float)


def synthetic_candles(
    bars: int = 4000,
    start_price: float = 1.10,
    seed: int = 7,
    hourly_vol: float = 0.0012,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Serie H1 sintetica con alternanza di trend e fasi laterali e volatilità a cluster."""
    rng = np.random.default_rng(seed)
    log_p = np.log(start_price)
    anchor = log_p
    closes, opens, highs, lows = [], [], [], []
    vol = hourly_vol
    regime_left, drift, ranging = 0, 0.0, False
    for _ in range(bars):
        if regime_left <= 0:
            regime_left = int(rng.integers(150, 600))
            ranging = rng.random() < 0.4
            drift = 0.0 if ranging else rng.choice([-1, 1]) * rng.uniform(0.08, 0.25) * hourly_vol
            anchor = log_p
        regime_left -= 1
        vol = 0.97 * vol + 0.03 * hourly_vol * rng.uniform(0.5, 1.8)
        pull = -0.03 * (log_p - anchor) if ranging else 0.0
        ret = drift + pull + vol * rng.standard_normal()
        o = log_p
        c = log_p + ret
        wick = np.abs(rng.standard_normal(2)) * vol * 0.6
        opens.append(np.exp(o))
        closes.append(np.exp(c))
        highs.append(np.exp(max(o, c) + wick[0]))
        lows.append(np.exp(min(o, c) - wick[1]))
        log_p = c
    index = pd.date_range(start, periods=bars, freq="h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=index)
