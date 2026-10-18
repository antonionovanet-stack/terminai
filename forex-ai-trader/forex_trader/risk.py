"""Gestione del rischio: dimensionamento, limiti di perdita, kill switch.

Nessuna decisione del ragionatore (regole o LLM) può aggirare questi controlli.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

from .fx import round_price, split_instrument, to_pips
from .models import Account, Decision, MarketSnapshot, OrderPlan, Position, Quote, Rejection


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_open_trades: int = 3
    max_trades_per_currency: int = 2
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    min_confidence: float = 0.55
    min_reward_risk: float = 1.2
    max_spread_pips: float = 3.0
    max_spread_stop_fraction: float = 0.15
    max_leverage: float = 10.0
    min_stop_atr: float = 0.8
    max_stop_atr: float = 4.0
    min_units: int = 1


@dataclass
class RiskState:
    day: date | None = None
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted_reason: str | None = None
    notes: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()
        self.state = RiskState()

    def update_equity(self, equity: float, now: datetime) -> None:
        st = self.state
        if st.day != now.date():
            st.day = now.date()
            st.day_start_equity = equity
            if st.halted_reason and st.halted_reason.startswith("daily"):
                st.halted_reason = None
        st.peak_equity = max(st.peak_equity, equity)
        if st.peak_equity and equity <= st.peak_equity * (1 - self.cfg.max_drawdown):
            # Kill switch permanente: richiede intervento umano (riavvio) per riprendere.
            st.halted_reason = (
                f"drawdown massimo superato ({1 - equity / st.peak_equity:.1%} dal picco)"
            )
        elif st.day_start_equity and equity <= st.day_start_equity * (1 - self.cfg.max_daily_loss):
            st.halted_reason = st.halted_reason or (
                f"daily: perdita giornaliera oltre il {self.cfg.max_daily_loss:.0%}"
            )

    def evaluate(
        self,
        decision: Decision,
        snapshot: MarketSnapshot,
        quote: Quote,
        account: Account,
        open_positions: list[Position],
        quote_to_account: float,
    ) -> OrderPlan | Rejection:
        cfg = self.cfg
        reasons: list[str] = []
        inst = snapshot.instrument

        if decision.action == "HOLD":
            return Rejection(["decisione HOLD"])
        if self.state.halted_reason:
            reasons.append(f"trading sospeso: {self.state.halted_reason}")
        if decision.confidence < cfg.min_confidence:
            reasons.append(f"confidenza {decision.confidence:.2f} < minimo {cfg.min_confidence:.2f}")
        if decision.take_profit_rr < cfg.min_reward_risk:
            reasons.append(f"R/R {decision.take_profit_rr:.2f} < minimo {cfg.min_reward_risk:.2f}")
        if len(open_positions) >= cfg.max_open_trades:
            reasons.append(f"già {len(open_positions)} posizioni aperte (max {cfg.max_open_trades})")
        if any(p.instrument == inst for p in open_positions):
            reasons.append(f"posizione già aperta su {inst}")

        base, quote_ccy = split_instrument(inst)
        for ccy in (base, quote_ccy):
            exposure = sum(1 for p in open_positions if ccy in split_instrument(p.instrument))
            if exposure >= cfg.max_trades_per_currency:
                reasons.append(f"esposizione eccessiva su {ccy} ({exposure} posizioni)")

        spread_pips = to_pips(inst, quote.spread)
        if spread_pips > cfg.max_spread_pips:
            reasons.append(f"spread {spread_pips:.1f} pip > max {cfg.max_spread_pips}")

        stop_mult = min(max(decision.stop_atr_mult, cfg.min_stop_atr), cfg.max_stop_atr)
        stop_dist = snapshot.ltf.atr * stop_mult
        if stop_dist <= 0:
            reasons.append("ATR non valido")
        elif quote.spread > stop_dist * cfg.max_spread_stop_fraction:
            reasons.append("spread troppo ampio rispetto alla distanza dello stop")
        if reasons:
            return Rejection(reasons)

        is_buy = decision.action == "BUY"
        entry = quote.ask if is_buy else quote.bid
        stop = entry - stop_dist if is_buy else entry + stop_dist
        target = entry + stop_dist * decision.take_profit_rr if is_buy else entry - stop_dist * decision.take_profit_rr

        risk_amount = account.equity * cfg.risk_per_trade
        loss_per_unit = stop_dist * quote_to_account
        units = math.floor(risk_amount / loss_per_unit)
        # Tetto di leva: valore nozionale in valuta del conto <= equity * leva massima.
        notional_per_unit = entry * quote_to_account
        max_units = math.floor(account.equity * cfg.max_leverage / notional_per_unit)
        units = min(units, max_units)
        if units < cfg.min_units:
            return Rejection([f"dimensione calcolata troppo piccola ({units} unità)"])

        return OrderPlan(
            instrument=inst,
            units=units if is_buy else -units,
            entry_ref=entry,
            stop_loss=round_price(inst, stop),
            take_profit=round_price(inst, target),
            risk_amount=round(units * loss_per_unit, 2),
            stop_pips=round(to_pips(inst, stop_dist), 1),
        )
