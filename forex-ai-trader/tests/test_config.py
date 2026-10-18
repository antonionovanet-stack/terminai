import pytest

from forex_trader.config import load_config


def test_load_example_config():
    cfg = load_config("config.example.toml")
    assert cfg.risk.risk_per_trade == 0.01
    assert cfg.claude.model == "claude-opus-5-5"
    assert "EUR_USD" in cfg.instruments
    assert cfg.broker == "mt5" and cfg.mt5.magic == 20261018


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[risk]\nrisk_per_trad = 0.5\n")
    with pytest.raises(ValueError, match="sconosciute"):
        load_config(p)


def test_without_claude_credentials_falls_back_to_rules(monkeypatch, tmp_path):
    from forex_trader import cli
    from forex_trader.config import AppConfig
    from forex_trader.strategy import RuleBasedReasoner

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert isinstance(cli.build_reasoner(AppConfig(), "hybrid"), RuleBasedReasoner)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert not isinstance(cli.build_reasoner(AppConfig(), "hybrid"), RuleBasedReasoner)
