"""Interfaccia comune dei ragionatori e contesto passato a ogni decisione."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import Account, ClosedTrade, Decision, MarketSnapshot, Position


@dataclass(frozen=True)
class ReasoningContext:
    account: Account
    open_positions: list[Position] = field(default_factory=list)
    recent_trades: list[ClosedTrade] = field(default_factory=list)
    recent_decisions: list[dict[str, object]] = field(default_factory=list)


class Reasoner(Protocol):
    def decide(self, snapshot: MarketSnapshot, context: ReasoningContext) -> Decision: ...
