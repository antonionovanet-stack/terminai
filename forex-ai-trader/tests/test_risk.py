from dataclasses import replace
from datetime import datetime, timedelta, timezone

from forex_trader.analysis import analyze
from forex_trader.models import Account, Decision, OrderPlan, Position, Quote, Rejection
from forex_trader.risk import RiskConfig, RiskManager
from tests.conftest import make_trend

NOW = datetime(2024, 3, 1, 12, tzinfo=timezone.utc)
ACCOUNT = Account(10_000.0, 10_000.0, "USD")


def _decision(action="BUY", conf=0.8, stop=1.5, rr=2.0):
    return Decision(action=action, confidence=conf, stop_atr_mult=stop, take_profit_rr=rr, thesis="t")


def _snap(atr=0.0020):
    s = analyze(make_trend(), "EUR_USD")
    return replace(s, ltf=replace(s.ltf, atr=atr))


def _quote(mid=1.2000, spread=0.0001):
    return Quote("EUR_USD", mid - spread / 2, mid + spread / 2, NOW)


def test_position_sized_to_risk_one_percent():
    plan = RiskManager().evaluate(_decision(), _snap(), _quote(), ACCOUNT, [], 1.0)
    assert isinstance(plan, OrderPlan)
    # Stop a 1.5 ATR = 0.0030 -> 100 USD / 0.0030 = 33.333 unità
    assert plan.units == 33_333
    assert abs(plan.risk_amount - 100.0) < 0.5
    assert plan.stop_loss < plan.entry_ref < plan.take_profit
    assert abs((plan.take_profit - plan.entry_ref) - 2 * (plan.entry_ref - plan.stop_loss)) < 1e-4


def test_sell_levels_mirrored():
    plan = RiskManager().evaluate(_decision("SELL"), _snap(), _quote(), ACCOUNT, [], 1.0)
    assert plan.units < 0
    assert plan.take_profit < plan.entry_ref < plan.stop_loss


def test_leverage_cap_limits_units():
    rm = RiskManager(RiskConfig(max_leverage=2.0))
    plan = rm.evaluate(_decision(), _snap(atr=0.0010), _quote(), ACCOUNT, [], 1.0)
    # Senza tetto sarebbero ~66.666 unità (80k nozionali): la leva 2x le limita a 20k.
    assert isinstance(plan, OrderPlan)
    assert plan.units * plan.entry_ref <= 20_000 + 1
    assert plan.risk_amount < 100


def test_jpy_quote_conversion():
    s = analyze(make_trend(start=150.0, step=0.04), "USD_JPY")
    s = replace(s, ltf=replace(s.ltf, atr=0.20))
    q = Quote("USD_JPY", 149.99, 150.01, NOW)
    plan = RiskManager().evaluate(_decision(), s, q, ACCOUNT, [], 1 / 150.0)
    # Perdita allo stop = unità * 0.30 JPY / 150 = ~100 USD
    assert abs(plan.units * 0.30 / 150 - 100) < 1


def test_rejections_low_confidence_wide_spread_and_duplicate():
    rm = RiskManager()
    pos = Position("1", "EUR_USD", 1000, 1.2, 1.19, 1.22, NOW)
    r = rm.evaluate(_decision(conf=0.2), _snap(), _quote(spread=0.0010), ACCOUNT, [pos], 1.0)
    assert isinstance(r, Rejection)
    text = " ".join(r.reasons)
    assert "confidenza" in text and "spread" in text and "già aperta" in text


def test_daily_loss_halts_and_resets_next_day():
    rm = RiskManager(RiskConfig(max_daily_loss=0.03, max_drawdown=0.5))
    rm.update_equity(10_000, NOW)
    rm.update_equity(9_600, NOW + timedelta(hours=1))
    assert rm.state.halted_reason and rm.state.halted_reason.startswith("daily")
    assert isinstance(rm.evaluate(_decision(), _snap(), _quote(), ACCOUNT, [], 1.0), Rejection)
    rm.update_equity(9_600, NOW + timedelta(days=1))
    assert rm.state.halted_reason is None


def test_max_drawdown_kill_switch_is_sticky():
    rm = RiskManager(RiskConfig(max_drawdown=0.10, max_daily_loss=0.5))
    rm.update_equity(10_000, NOW)
    rm.update_equity(8_900, NOW + timedelta(days=1))
    rm.update_equity(9_500, NOW + timedelta(days=2))
    assert "drawdown" in rm.state.halted_reason
