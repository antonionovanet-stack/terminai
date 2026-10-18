import pytest

from forex_trader.config import load_config


def test_load_example_config():
    cfg = load_config("config.example.toml")
    assert cfg.risk.risk_per_trade == 0.01
    assert cfg.claude.model == "claude-opus-5-5"
    assert "EUR_USD" in cfg.instruments


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[risk]\nrisk_per_trad = 0.5\n")
    with pytest.raises(ValueError, match="sconosciute"):
        load_config(p)
