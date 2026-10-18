import random
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from forex_trader.analysis import analyze
from forex_trader.autoevolve import is_due
from forex_trader.backtest import run_backtest
from forex_trader.config import AppConfig, load_config
from forex_trader.data import synthetic_candles
from forex_trader.evolution import (
    GENOME, EvolutionConfig, SnapshotCache, StrategyRegistry, crossover, evolve, mutate,
    params_from_dict, params_key, params_to_dict, random_params,
)
from forex_trader.strategy import RuleBasedReasoner, RuleParams
from forex_trader.strategy_proposer import ClaudeStrategyProposer, Proposal, ProposalSet, ProposedParams

FAST = EvolutionConfig(population=6, generations=2, folds=2, workers=1, seed=3, min_trades_per_segment=3,
                       min_holdout_trades=2)


@pytest.fixture(scope="module")
def data():
    return {"EUR_USD": synthetic_candles(bars=1300, seed=9)}


@pytest.fixture(scope="module")
def cache(data):
    return SnapshotCache.build(data, history_bars=500, workers=1)


def _in_bounds(p: RuleParams) -> bool:
    d = params_to_dict(p)
    return all(g.is_bool or g.low <= d[g.name] <= g.high for g in GENOME)


def test_params_from_dict_clamps_and_enforces_constraints():
    p = params_from_dict({"min_score": 99, "w_level": -5, "unknown": 1, "rsi_healthy_low": 54,
                          "rsi_healthy_high": 56, "pullback_atr": 1.4, "overextended_atr": 1.0,
                          "trend_enabled": False, "range_enabled": False})
    assert p.min_score == 6.0 and p.w_level == 0.0
    assert p.overextended_atr > p.pullback_atr
    assert p.trend_enabled  # almeno un setup resta attivo
    assert params_from_dict(params_to_dict(RuleParams())) == RuleParams()


def test_genetic_operators_stay_in_bounds():
    rng = random.Random(1)
    base = RuleParams()
    for _ in range(200):
        a, b = random_params(rng), mutate(base, rng)
        child = crossover(a, b, rng)
        assert _in_bounds(a) and _in_bounds(b) and _in_bounds(child)
        da, db, dc = params_to_dict(a), params_to_dict(b), params_to_dict(child)
        assert all(dc[g.name] in (da[g.name], db[g.name]) or not g.is_bool for g in GENOME)
    assert mutate(base, rng) != base


def test_cached_backtest_identical_to_normal_backtest(data, cache):
    params = RuleParams(min_score=3.0)
    normal = run_backtest(data, RuleBasedReasoner(params), warmup_bars=500)
    fast = run_backtest(data, RuleBasedReasoner(params), warmup_bars=500, analyzer=cache.analyzer)
    assert len(normal.trades) > 0
    assert [(t.open_time, t.realized_pl) for t in normal.trades] == [(t.open_time, t.realized_pl) for t in fast.trades]


def test_cached_snapshot_matches_live_analysis(data, cache):
    df = data["EUR_USD"]
    live = analyze(df.iloc[700 - 500 + 1 : 701], "EUR_USD")
    cached = cache.analyzer(df.iloc[:701], "EUR_USD")
    assert replace(live, recent_candles=[]) == cached


def test_evolve_records_lineage_and_champion(tmp_path, data, cache):
    reg = StrategyRegistry(tmp_path / "reg.json")
    report = evolve(data, reg, FAST, cache=cache, progress=lambda _: None)
    saved = StrategyRegistry(tmp_path / "reg.json")
    assert saved.champion_id in saved.data["strategies"]
    assert saved.data["runs"][-1]["candidate"] == report.candidate_id
    assert len(report.history) == FAST.generations
    if report.promoted:
        assert saved.champion_id == report.candidate_id
        assert saved.data["strategies"][report.candidate_id]["parent"] == report.champion_id
    else:
        assert saved.champion_id == report.champion_id


def test_no_promote_keeps_champion(tmp_path, data, cache):
    reg = StrategyRegistry(tmp_path / "reg.json")
    report = evolve(data, reg, FAST, cache=cache, allow_promotion=False, progress=lambda _: None)
    assert not report.promoted
    assert StrategyRegistry(tmp_path / "reg.json").champion_id == params_key(RuleParams())


def test_proposer_candidates_are_evaluated(tmp_path, data, cache):
    calls = []
    special = replace(RuleParams(), min_score=2.2, trend_rr=3.1)

    def proposer(champion, champion_eval, leaderboard, n):
        calls.append((champion, n, len(leaderboard)))
        return [(special, "target più lontani")]

    cfg = replace(FAST, llm_proposals=1, generations=2)
    evolve(data, StrategyRegistry(tmp_path / "r.json"), cfg, cache=cache, proposer=proposer,
           progress=lambda _: None)
    assert calls and calls[0][1] == 1


def test_champion_survives_restart(tmp_path):
    reg = StrategyRegistry(tmp_path / "r.json")
    p = replace(RuleParams(), min_score=4.2)
    sid = reg.add(p, "mutation", None, {"train": {}})
    reg.promote(sid, "test")
    reg.save()
    assert StrategyRegistry(tmp_path / "r.json").champion_params() == p


def test_auto_evolution_only_on_weekend_and_weekly(tmp_path):
    cfg = AppConfig()
    cfg.evolution.registry_path = str(tmp_path / "r.json")
    reg = StrategyRegistry(cfg.evolution.registry_path)
    saturday = datetime(2026, 10, 17, 10, tzinfo=timezone.utc)
    wednesday = datetime(2026, 10, 14, 10, tzinfo=timezone.utc)
    assert is_due(cfg, reg, saturday) and not is_due(cfg, reg, wednesday)
    reg.data["runs"].append({"at": saturday.isoformat()})
    assert not is_due(cfg, reg, saturday.replace(day=18))
    assert is_due(cfg, reg, datetime(2026, 10, 24, 10, tzinfo=timezone.utc))
    cfg.evolution.auto = False
    assert not is_due(cfg, reg, saturday)


def test_claude_proposer_maps_and_clamps():
    values = {g.name: (True if g.is_bool else g.high + 10) for g in GENOME}
    parsed = ProposalSet(diagnosis="troppi falsi segnali",
                         proposals=[Proposal(hypothesis="più selettivo", params=ProposedParams(**values))])
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=parsed)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    from forex_trader.evolution import Evaluation
    out = ClaudeStrategyProposer(client=client)(RuleParams(), Evaluation(0.1, 10, 1.0, 5.0), [], 3)
    assert len(out) == 1 and out[0][1] == "più selettivo"
    assert _in_bounds(out[0][0]) and out[0][0].min_score == 6.0
    assert captured["output_format"] is ProposalSet and captured["model"] == "claude-opus-5-5"


def test_claude_proposer_refusal_returns_empty():
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        parse=lambda **_: SimpleNamespace(stop_reason="refusal", parsed_output=None))))
    from forex_trader.evolution import Evaluation
    assert ClaudeStrategyProposer(client=client)(RuleParams(), Evaluation(0, 0, 0, 0), [], 2) == []


def test_evolution_config_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[evolution]\nauto = false\nevery_days = 14\n[evolution.search]\npopulation = 30\n")
    cfg = load_config(p)
    assert cfg.evolution.auto is False and cfg.evolution.every_days == 14
    assert cfg.evolution.search.population == 30
