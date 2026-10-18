"""Evoluzione automatica durante l'operatività: una volta a settimana, a mercato chiuso."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from .brokers.base import Broker
from .config import AppConfig
from .evolution import EvolutionReport, StrategyRegistry, evolve
from .strategy import RuleParams
from .strategy_proposer import ClaudeStrategyProposer

log = logging.getLogger(__name__)

WEEKEND = (5, 6)


def resolve_rules(cfg: AppConfig) -> RuleParams:
    """Strategia con cui operare: il campione evoluto, se esiste ed è abilitato, altrimenti la config."""
    if not cfg.evolution.use_champion:
        return cfg.rules
    registry = StrategyRegistry(cfg.evolution.registry_path)
    if registry.champion_id:
        log.info("Strategia campione dal registro: %s", registry.champion_id)
    return registry.champion_params(default=cfg.rules)


def is_due(cfg: AppConfig, registry: StrategyRegistry, now: datetime) -> bool:
    if not cfg.evolution.auto or now.weekday() not in WEEKEND:
        return False
    last = registry.last_run_at()
    return last is None or now - last >= timedelta(days=cfg.evolution.every_days) - timedelta(hours=1)


def collect_history(broker: Broker, instruments: list[str], bars: int) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for inst in instruments:
        try:
            df = broker.get_candles(inst, bars)
        except Exception as exc:  # noqa: BLE001 - uno strumento senza dati non blocca gli altri
            log.warning("Storico di %s non disponibile per l'evoluzione: %s", inst, exc)
            continue
        if len(df) >= 1500:
            data[inst] = df
        else:
            log.warning("Storico di %s troppo corto (%d barre): escluso dall'evoluzione", inst, len(df))
    return data


def run_evolution(
    cfg: AppConfig, data: dict[str, pd.DataFrame], allow_promotion: bool = True, disk_cache: bool = True
) -> EvolutionReport:
    evo = cfg.evolution
    proposer = ClaudeStrategyProposer(cfg.claude) if evo.search.llm_proposals > 0 else None
    return evolve(
        data, StrategyRegistry(evo.registry_path), evo.search, cfg.risk, cfg.engine,
        proposer=proposer, cache_dir=evo.cache_dir if disk_cache else None, allow_promotion=allow_promotion, base=cfg.rules,
    )


def maybe_evolve(broker: Broker, cfg: AppConfig, instruments: list[str], now: datetime) -> RuleParams | None:
    """Esegue l'evoluzione se è il momento. Ritorna i nuovi parametri se è stato promosso un campione."""
    registry = StrategyRegistry(cfg.evolution.registry_path)
    if not is_due(cfg, registry, now):
        return None
    log.info("Avvio dell'evoluzione settimanale delle strategie")
    data = collect_history(broker, instruments, cfg.evolution.history_bars)
    if not data:
        log.warning("Nessuno storico utilizzabile: evoluzione rimandata")
        return None
    # Lo storico dal vivo cambia ogni settimana: inutile accumulare cache su disco.
    report = run_evolution(cfg, data, disk_cache=False)
    log.info("Evoluzione completata: %s", report.reason)
    return report.candidate_params if report.promoted else None
