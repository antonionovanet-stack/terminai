"""Ciclo operativo: osserva -> analizza -> ragiona -> controlla il rischio -> esegue -> registra."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd

from .analysis import analyze
from .brokers.base import Broker
from .fx import round_price, split_instrument
from .journal import Journal
from .models import ClosedTrade, MarketSnapshot, OrderPlan, Position
from .reasoning import Reasoner, ReasoningContext
from .risk import RiskManager

log = logging.getLogger(__name__)

Analyzer = Callable[[pd.DataFrame, str], MarketSnapshot | None]


@dataclass(frozen=True)
class EngineConfig:
    history_bars: int = 500
    break_even_at_r: float | None = 1.0
    dry_run: bool = False


class TradingEngine:
    def __init__(
        self,
        broker: Broker,
        reasoner: Reasoner,
        risk: RiskManager,
        instruments: list[str],
        journal: Journal | None = None,
        config: EngineConfig | None = None,
        analyzer: Analyzer = analyze,
    ) -> None:
        self.broker = broker
        self.analyzer = analyzer
        self.reasoner = reasoner
        self.risk = risk
        self.instruments = instruments
        self.journal = journal or Journal()
        self.cfg = config or EngineConfig()
        self.trade_history: deque[ClosedTrade] = deque(maxlen=50)
        self._known: dict[str, Position] = {}
        self._initial_risk: dict[str, float] = {}

    def sync_closed(self, at: datetime | None = None) -> list[ClosedTrade]:
        current = {p.id: p for p in self.broker.open_positions()}
        closed: list[ClosedTrade] = []
        for pid in set(self._known) - set(current):
            trade = self.broker.closed_trade(pid)
            if trade:
                closed.append(trade)
                self.trade_history.append(trade)
                self.journal.record("trade_closed", at=at, instrument=trade.instrument, trade=trade)
            self._initial_risk.pop(pid, None)
        self._known = current
        return closed

    def _manage(self, pos: Position, at: datetime) -> None:
        if self.cfg.break_even_at_r is None or pos.stop_loss is None:
            return
        risk = self._initial_risk.setdefault(pos.id, abs(pos.entry_price - pos.stop_loss))
        if risk <= 0:
            return
        q = self.broker.get_quote(pos.instrument)
        price = q.bid if pos.is_long else q.ask
        progress = (price - pos.entry_price) / risk if pos.is_long else (pos.entry_price - price) / risk
        already = pos.stop_loss >= pos.entry_price if pos.is_long else pos.stop_loss <= pos.entry_price
        if progress >= self.cfg.break_even_at_r and not already:
            new_stop = round_price(pos.instrument, pos.entry_price)
            if not self.cfg.dry_run:
                self.broker.modify_stop(pos.id, new_stop)
            self.journal.record("stop_to_break_even", at=at, instrument=pos.instrument,
                                position_id=pos.id, new_stop=new_stop, progress_r=round(progress, 2))

    def step(self) -> list[OrderPlan]:
        account = self.broker.get_account()
        placed: list[OrderPlan] = []
        for inst in self.instruments:
            candles = self.broker.get_candles(inst, self.cfg.history_bars)
            snapshot = self.analyzer(candles, inst)
            if snapshot is None:
                continue
            at = snapshot.time
            self.risk.update_equity(account.equity, at)
            self.sync_closed(at)
            positions = list(self._known.values())
            for pos in positions:
                if pos.instrument == inst:
                    self._manage(pos, at)
            if any(p.instrument == inst for p in positions):
                continue

            ctx = ReasoningContext(
                account=account,
                open_positions=positions,
                recent_trades=[t for t in self.trade_history if t.instrument == inst][-5:],
                recent_decisions=[
                    {k: e.get(k) for k in ("at", "action", "confidence", "thesis")}
                    for e in self.journal.recent("decision", inst, limit=3)
                ],
            )
            decision = self.reasoner.decide(snapshot, ctx)
            self.journal.record(
                "decision", at=at, instrument=inst, action=decision.action, confidence=decision.confidence,
                thesis=decision.thesis, factors=decision.supporting_factors, risks=decision.risks,
                invalidation=decision.invalidation, source=decision.source, score=decision.score,
                regime=decision.regime, price=snapshot.price,
            )
            if decision.action == "HOLD":
                continue

            quote = self.broker.get_quote(inst)
            _, quote_ccy = split_instrument(inst)
            conv = self.broker.conversion_rate(quote_ccy, account.currency)
            verdict = self.risk.evaluate(decision, snapshot, quote, account, positions, conv)
            if not isinstance(verdict, OrderPlan):
                self.journal.record("risk_rejected", at=at, instrument=inst, reasons=verdict.reasons)
                log.debug("%s %s rifiutato dal rischio: %s", inst, decision.action, "; ".join(verdict.reasons))
                continue

            if self.cfg.dry_run:
                self.journal.record("order_dry_run", at=at, instrument=inst, plan=verdict)
                placed.append(verdict)
                continue
            try:
                oid = self.broker.place_market_order(inst, verdict.units, verdict.stop_loss, verdict.take_profit)
            except Exception as exc:  # noqa: BLE001 - un errore del broker non deve fermare il ciclo
                self.journal.record("order_error", at=at, instrument=inst, error=str(exc))
                log.error("Ordine %s fallito: %s", inst, exc)
                continue
            self._initial_risk[oid] = abs(verdict.entry_ref - verdict.stop_loss)
            self.journal.record("order_placed", at=at, instrument=inst, order_id=oid, plan=verdict)
            log.info("%s: %+d unità, SL %s TP %s (rischio %.2f %s)", inst, verdict.units,
                     verdict.stop_loss, verdict.take_profit, verdict.risk_amount, account.currency)
            placed.append(verdict)
            self._known = {p.id: p for p in self.broker.open_positions()}
            # Alcuni broker (es. il simulatore) eseguono alla barra successiva: conta subito l'ordine
            # nei limiti di esposizione degli strumenti seguenti.
            self._known.setdefault(oid, Position(
                id=oid, instrument=inst, units=verdict.units, entry_price=verdict.entry_ref,
                stop_loss=verdict.stop_loss, take_profit=verdict.take_profit, open_time=at,
            ))
        return placed
