from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx2

from forex_trader.analysis import analyze
from forex_trader.llm_reasoner import ClaudeReasoner, HybridReasoner, TradeDecision
from forex_trader.models import Account, Decision
from forex_trader.reasoning import ReasoningContext
from tests.conftest import make_trend

CTX = ReasoningContext(account=Account(10_000, 10_000, "USD"))


class FakeMessages:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def _response(action="BUY", conf=0.7, stop_reason="end_turn", stop=1.8, rr=2.5):
    parsed = TradeDecision(
        action=action, confidence=conf, market_regime="trending", thesis="Trend rialzista intatto",
        supporting_factors=["H4 up"], risks=["dati macro"], invalidation="chiusura sotto EMA50",
        stop_loss_atr_multiple=stop, take_profit_rr=rr,
    )
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed, model="claude-opus-5-5")


def _snap():
    return analyze(make_trend(), "EUR_USD")


def test_claude_decision_mapped_and_clamped():
    msgs = FakeMessages(_response(conf=1.7, stop=12.0))
    d = ClaudeReasoner(client=fake_client(msgs)).decide(_snap(), CTX)
    assert d.action == "BUY" and d.confidence == 1.0 and d.stop_atr_mult == 5.0
    assert d.source == "claude:claude-opus-5-5"
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["fallbacks"] == "default"
    assert call["output_format"] is TradeDecision
    assert "EUR_USD" in call["messages"][0]["content"]


def test_refusal_falls_back_to_rules():
    resp = SimpleNamespace(stop_reason="refusal", parsed_output=None, model="claude-opus-5-5")
    d = ClaudeReasoner(client=fake_client(FakeMessages(resp))).decide(_snap(), CTX)
    assert d.source.startswith("rules(fallback")


def test_connection_error_falls_back_to_rules():
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    d = ClaudeReasoner(client=fake_client(FakeMessages(error=err))).decide(_snap(), CTX)
    assert d.source.startswith("rules(fallback: connessione")


class FixedRules:
    def __init__(self, score):
        self.score = score

    def decide(self, snapshot, context=None):
        action = "BUY" if self.score >= 3.5 else "SELL" if self.score <= -3.5 else "HOLD"
        return Decision(action=action, confidence=0.6, stop_atr_mult=1.5, take_profit_rr=2.0,
                        thesis="regole", score=self.score)


def test_hybrid_skips_llm_without_setup():
    msgs = FakeMessages(_response())
    h = HybridReasoner(ClaudeReasoner(client=fake_client(msgs)), FixedRules(0.5))
    assert h.decide(_snap(), CTX).action == "HOLD"
    assert msgs.calls == []


def test_hybrid_consensus_blocks_disagreement():
    h = HybridReasoner(ClaudeReasoner(client=fake_client(FakeMessages(_response("SELL")))), FixedRules(4.0))
    d = h.decide(_snap(), CTX)
    assert d.action == "HOLD" and "Disaccordo" in d.thesis


def test_hybrid_consensus_agreement_uses_claude_parameters():
    h = HybridReasoner(ClaudeReasoner(client=fake_client(FakeMessages(_response("BUY", rr=2.5)))), FixedRules(4.0))
    d = h.decide(_snap(), CTX)
    assert d.action == "BUY" and d.take_profit_rr == 2.5 and d.source.startswith("hybrid(")


def test_hybrid_respects_llm_call_budget():
    msgs = FakeMessages(_response())
    h = HybridReasoner(ClaudeReasoner(client=fake_client(msgs)), FixedRules(4.0), max_llm_calls=1)
    h.decide(_snap(), CTX)
    d = h.decide(_snap(), CTX)
    assert len(msgs.calls) == 1 and d.thesis == "regole"


def test_volatile_regime_never_calls_llm():
    msgs = FakeMessages(_response())
    h = HybridReasoner(ClaudeReasoner(client=fake_client(msgs)), FixedRules(5.0))
    h.decide(replace(_snap(), regime="volatile"), CTX)
    assert msgs.calls == []
