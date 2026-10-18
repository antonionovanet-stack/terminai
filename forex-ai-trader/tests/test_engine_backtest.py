from forex_trader.backtest import run_backtest
from forex_trader.brokers.paper import PaperBroker
from forex_trader.data import synthetic_candles
from forex_trader.engine import EngineConfig, TradingEngine
from forex_trader.journal import Journal
from forex_trader.models import Decision
from forex_trader.risk import RiskConfig, RiskManager
from forex_trader.strategy import RuleBasedReasoner


class AlwaysBuy:
    def decide(self, snapshot, context):
        return Decision(action="BUY", confidence=0.9, stop_atr_mult=1.5, take_profit_rr=2.0, thesis="test")


def test_backtest_runs_and_is_consistent():
    df = synthetic_candles(bars=900, seed=11)
    res = run_backtest({"EUR_USD": df}, RuleBasedReasoner())
    assert res.metrics["num_trade"] == len(res.trades)
    assert abs(res.final_balance - (10_000 + sum(t.realized_pl for t in res.trades))) < 1e-6
    # Una sola posizione per strumento alla volta.
    spans = sorted((t.open_time, t.close_time) for t in res.trades)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_losses_per_trade_bounded_by_risk_budget():
    df = synthetic_candles(bars=900, seed=5)
    res = run_backtest({"EUR_USD": df}, AlwaysBuy(), RiskConfig(risk_per_trade=0.01, max_drawdown=0.9, max_daily_loss=0.9))
    worst = min(t.realized_pl for t in res.trades)
    # 1% di rischio +/- gap e spread: mai perdite molto oltre il budget.
    assert worst > -10_000 * 0.01 * 1.6


def test_dry_run_places_no_orders_and_journals():
    df = synthetic_candles(bars=600, seed=2)
    broker = PaperBroker({"EUR_USD": df}, start_index=550)
    journal = Journal()
    engine = TradingEngine(broker, AlwaysBuy(), RiskManager(), ["EUR_USD"], journal, EngineConfig(dry_run=True))
    plans = engine.step()
    broker.advance()
    assert plans and broker.open_positions() == []
    kinds = [e["kind"] for e in journal.entries]
    assert kinds == ["decision", "order_dry_run"]
