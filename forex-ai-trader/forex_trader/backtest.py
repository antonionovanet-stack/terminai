"""Backtest bar-by-bar con lo stesso motore usato dal vivo (stessa analisi, stesso rischio)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .brokers.paper import PaperBroker
from .analysis import analyze
from .engine import Analyzer, EngineConfig, TradingEngine
from .journal import Journal
from .models import ClosedTrade
from .reasoning import Reasoner
from .risk import RiskConfig, RiskManager


@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    trades: list[ClosedTrade]
    equity_curve: pd.Series
    halted_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def compute_metrics(initial: float, trades: list[ClosedTrade], equity: pd.Series) -> dict[str, float]:
    pls = [t.realized_pl for t in trades]
    wins = [p for p in pls if p > 0]
    losses = [p for p in pls if p <= 0]
    gross_loss = -sum(losses)
    peak = equity.cummax()
    dd = ((peak - equity) / peak).max() if len(equity) else 0.0
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252 * 24)) if len(rets) > 1 and rets.std() > 0 else 0.0
    final = float(equity.iloc[-1]) if len(equity) else initial
    return {
        "rendimento_totale_pct": round((final / initial - 1) * 100, 2),
        "num_trade": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "profit_factor": round(sum(wins) / gross_loss, 2) if gross_loss > 0 else float("inf") if wins else 0.0,
        "media_trade": round(sum(pls) / len(pls), 2) if pls else 0.0,
        "max_drawdown_pct": round(float(dd) * 100, 2),
        "sharpe_annualizzato": round(sharpe, 2),
    }


def run_backtest(
    data: dict[str, pd.DataFrame],
    reasoner: Reasoner,
    risk_config: RiskConfig | None = None,
    initial_balance: float = 10_000.0,
    account_currency: str = "USD",
    spread_pips: float | dict[str, float] = 1.0,
    warmup_bars: int = 300,
    journal: Journal | None = None,
    engine_config: EngineConfig | None = None,
    instruments: list[str] | None = None,
    analyzer: Analyzer = analyze,
) -> BacktestResult:
    broker = PaperBroker(data, initial_balance, account_currency, spread_pips, start_index=warmup_bars)
    risk = RiskManager(risk_config)
    engine = TradingEngine(
        broker, reasoner, risk, instruments or list(data), journal or Journal(), engine_config or EngineConfig(),
        analyzer,
    )
    times, values = [], []
    while broker.can_advance():
        engine.step()
        broker.advance()
        times.append(broker.now)
        values.append(broker.get_account().equity)
    broker.close_all()
    engine.sync_closed()
    if values:
        values[-1] = broker.balance
    trades = sorted(broker.closed.values(), key=lambda t: t.close_time)
    equity = pd.Series(values, index=pd.DatetimeIndex(times), dtype=float)
    return BacktestResult(
        initial_balance=initial_balance,
        final_balance=broker.balance,
        trades=trades,
        equity_curve=equity,
        halted_reason=risk.state.halted_reason,
        metrics=compute_metrics(initial_balance, trades, equity),
    )
