"""Tipi condivisi tra analisi, ragionamento, rischio ed esecuzione."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

Action = Literal["BUY", "SELL", "HOLD"]
Trend = Literal["up", "down", "flat"]
Regime = Literal["trending", "ranging", "volatile", "transitional"]


@dataclass(frozen=True)
class TimeframeView:
    timeframe: str
    close: float
    trend: Trend
    ema20: float
    ema50: float
    ema200: float | None
    ema50_slope_atr: float
    adx: float
    plus_di: float
    minus_di: float
    rsi: float
    macd_hist: float
    macd_hist_prev: float
    atr: float
    atr_percentile: float
    bb_upper: float
    bb_mid: float
    bb_lower: float


@dataclass(frozen=True)
class MarketSnapshot:
    instrument: str
    time: datetime
    price: float
    pip_size: float
    ltf: TimeframeView
    htf: TimeframeView
    regime: Regime
    supports: list[float]
    resistances: list[float]
    recent_candles: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["time"] = self.time.isoformat()
        return data


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    stop_atr_mult: float
    take_profit_rr: float
    thesis: str
    supporting_factors: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    invalidation: str = ""
    source: str = "rules"
    score: float = 0.0
    regime: str = ""


@dataclass(frozen=True)
class Quote:
    instrument: str
    bid: float
    ask: float
    time: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class Account:
    balance: float
    equity: float
    currency: str


@dataclass(frozen=True)
class Position:
    id: str
    instrument: str
    units: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    open_time: datetime
    unrealized_pl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.units > 0


@dataclass(frozen=True)
class ClosedTrade:
    id: str
    instrument: str
    units: float
    entry_price: float
    exit_price: float
    open_time: datetime
    close_time: datetime
    realized_pl: float
    reason: str


@dataclass(frozen=True)
class OrderPlan:
    instrument: str
    units: int
    entry_ref: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    stop_pips: float


@dataclass(frozen=True)
class Rejection:
    reasons: list[str]
