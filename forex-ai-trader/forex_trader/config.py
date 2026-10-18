"""Configurazione da file TOML (vedi config.example.toml). Ogni sezione è opzionale."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .engine import EngineConfig
from .llm_reasoner import ClaudeSettings
from .risk import RiskConfig
from .strategy import RuleParams

T = TypeVar("T")


@dataclass
class ReasonerConfig:
    kind: str = "hybrid"  # rules | claude | hybrid
    hybrid_mode: str = "consensus"  # consensus | llm_final
    prefilter_score: float = 2.0
    max_llm_calls: int | None = None


@dataclass
class MT5Config:
    symbol_suffix: str = ""  # es. ".m" se il broker chiama i simboli EURUSD.m
    symbol_map: dict[str, str] = field(default_factory=dict)  # es. {"EUR_USD" = "EURUSDpro"}
    magic: int = 20261018
    deviation_points: int = 20
    server_utc_offset_hours: float = 0.0
    path: str | None = None  # percorso di terminal64.exe, se serve


@dataclass
class AppConfig:
    broker: str = "mt5"  # mt5 | oanda
    instruments: list[str] = field(default_factory=lambda: ["EUR_USD", "GBP_USD", "USD_JPY"])
    poll_seconds: int = 60
    journal_path: str = "journal/decisions.jsonl"
    risk: RiskConfig = field(default_factory=RiskConfig)
    rules: RuleParams = field(default_factory=RuleParams)
    claude: ClaudeSettings = field(default_factory=ClaudeSettings)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    mt5: MT5Config = field(default_factory=MT5Config)


def _build(cls: type[T], values: dict[str, Any]) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Chiavi sconosciute per {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sections = {
        "risk": RiskConfig, "rules": RuleParams, "claude": ClaudeSettings,
        "reasoner": ReasonerConfig, "engine": EngineConfig, "mt5": MT5Config,
    }
    kwargs: dict[str, Any] = {k: _build(cls, raw.pop(k)) for k, cls in sections.items() if k in raw}
    general = raw.pop("general", {})
    if raw:
        raise ValueError(f"Sezioni sconosciute: {sorted(raw)}")
    return _build(AppConfig, {**general, **kwargs})
