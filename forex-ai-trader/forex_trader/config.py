"""Configurazione da file TOML (vedi config.example.toml). Ogni sezione è opzionale."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .engine import EngineConfig
from .evolution import EvolutionConfig
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
class EvolutionSettings:
    auto: bool = True  # durante `run`, evolve da solo nel weekend (mercato chiuso)
    every_days: int = 7
    use_champion: bool = True  # opera con la strategia campione del registro, se esiste
    registry_path: str = "strategies/registry.json"
    cache_dir: str = ".cache"
    history_bars: int = 5000  # circa 10 mesi di candele H1 per strumento
    search: EvolutionConfig = field(default_factory=EvolutionConfig)


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
    evolution: EvolutionSettings = field(default_factory=EvolutionSettings)


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
    if "evolution" in raw:
        evo = dict(raw.pop("evolution"))
        search = _build(EvolutionConfig, evo.pop("search", {}))
        kwargs["evolution"] = _build(EvolutionSettings, {**evo, "search": search})
    general = raw.pop("general", {})
    if raw:
        raise ValueError(f"Sezioni sconosciute: {sorted(raw)}")
    return _build(AppConfig, {**general, **kwargs})
