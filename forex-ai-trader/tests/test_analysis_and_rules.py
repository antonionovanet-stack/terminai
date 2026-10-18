from dataclasses import replace

from forex_trader.analysis import analyze
from forex_trader.models import Account
from forex_trader.reasoning import ReasoningContext
from forex_trader.strategy import RuleBasedReasoner
from tests.conftest import make_trend

CTX = ReasoningContext(account=Account(10_000, 10_000, "USD"))


def test_analyze_requires_history(candles):
    assert analyze(candles.iloc[:100], "EUR_USD") is None
    snap = analyze(candles, "EUR_USD")
    assert snap is not None
    assert snap.htf.timeframe == "H4"
    assert len(snap.recent_candles) == 24


def test_uptrend_detected_and_rules_never_sell():
    snap = analyze(make_trend(), "EUR_USD")
    assert snap.htf.trend == "up"
    assert snap.ltf.trend == "up"
    assert RuleBasedReasoner().decide(snap, CTX).action in ("BUY", "HOLD")


def test_volatile_regime_forces_hold():
    snap = analyze(make_trend(), "EUR_USD")
    volatile = replace(snap, regime="volatile")
    d = RuleBasedReasoner().decide(volatile, CTX)
    assert d.action == "HOLD"
    assert "Volatilità" in d.thesis


def test_buy_in_pullback_of_uptrend():
    snap = analyze(make_trend(), "EUR_USD")
    ltf = replace(snap.ltf, close=snap.ltf.ema20, rsi=55.0, macd_hist=0.0002, macd_hist_prev=0.0001)
    d = RuleBasedReasoner().decide(replace(snap, ltf=ltf, regime="trending"), CTX)
    assert d.action == "BUY"
    assert d.score > 0
    assert d.supporting_factors
