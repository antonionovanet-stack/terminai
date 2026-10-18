"""Auto-evoluzione delle strategie di ingresso.

Come funziona:
1. Il "DNA" di una strategia è l'insieme di pesi e soglie di RuleParams (GENOME descrive i limiti).
2. Si parte dalla strategia campione attuale e si genera una popolazione di varianti: mutazioni del
   campione, combinazioni tra le migliori, individui casuali e, se abilitato, proposte ragionate di
   Claude che analizza i risultati e formula ipotesi.
3. Ogni candidato viene valutato con un backtest walk-forward su più segmenti consecutivi del periodo di
   addestramento: premia la costanza tra periodi diversi, non il colpo di fortuna in uno solo.
4. Il migliore viene confrontato con il campione su un periodo finale MAI visto durante l'evoluzione
   (holdout). Diventa il nuovo campione solo se è chiaramente migliore lì, con abbastanza trade e un
   drawdown accettabile. Altrimenti resta il campione attuale.

I limiti di rischio (RiskConfig) non fanno parte del DNA: l'evoluzione può cambiare quando e come si
entra a mercato, mai quanto si rischia.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .analysis import analyze
from .backtest import BacktestResult, run_backtest
from .engine import EngineConfig
from .journal import Journal
from .models import MarketSnapshot
from .risk import RiskConfig
from .strategy import RuleBasedReasoner, RuleParams

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- DNA


@dataclass(frozen=True)
class Gene:
    name: str
    low: float
    high: float
    description: str
    is_bool: bool = False


GENOME: tuple[Gene, ...] = (
    Gene("min_score", 2.0, 6.0, "punteggio minimo per entrare (più alto = meno trade, più selettivi)"),
    Gene("trend_enabled", 0, 1, "abilita gli ingressi a favore di trend", is_bool=True),
    Gene("range_enabled", 0, 1, "abilita gli ingressi di ritorno alla media in laterale", is_bool=True),
    Gene("trend_stop_atr", 1.0, 3.0, "stop dei trade in trend, in multipli di ATR H1"),
    Gene("trend_rr", 1.3, 4.0, "rapporto rendimento/rischio del target in trend"),
    Gene("range_stop_atr", 0.8, 2.5, "stop dei trade in laterale, in ATR"),
    Gene("range_rr", 1.2, 3.0, "rapporto rendimento/rischio del target in laterale"),
    Gene("w_htf_trend", 0.0, 3.0, "peso del trend H4"),
    Gene("w_ltf_aligned", 0.0, 3.0, "peso dell'allineamento H1 con H4"),
    Gene("w_ltf_counter", 0.0, 3.0, "penalità se H1 è in controtendenza"),
    Gene("pullback_atr", 0.2, 1.5, "distanza massima dalla EMA20 (ATR) per considerarlo un pullback"),
    Gene("w_pullback", 0.0, 3.0, "peso dell'ingresso in pullback"),
    Gene("overextended_atr", 1.0, 4.0, "distanza dalla EMA20 (ATR) oltre cui il prezzo è troppo esteso"),
    Gene("w_overextended", 0.0, 3.0, "penalità per prezzo troppo esteso"),
    Gene("w_momentum", 0.0, 3.0, "peso del momentum MACD a favore"),
    Gene("adx_strong", 18.0, 40.0, "ADX oltre cui il trend è considerato forte"),
    Gene("w_adx", 0.0, 2.0, "peso del trend forte"),
    Gene("rsi_exhaustion", 65.0, 85.0, "RSI oltre cui il trend è a rischio esaurimento"),
    Gene("w_rsi_exhaustion", 0.0, 3.0, "penalità per RSI in eccesso"),
    Gene("rsi_healthy_low", 35.0, 55.0, "estremo basso della zona RSI sana (per i long)"),
    Gene("rsi_healthy_high", 55.0, 75.0, "estremo alto della zona RSI sana (per i long)"),
    Gene("w_rsi_healthy", 0.0, 2.0, "peso dell'RSI in zona sana"),
    Gene("range_rsi_extreme", 15.0, 40.0, "RSI di ipervenduto per il ritorno alla media (simmetrico)"),
    Gene("w_range_signal", 1.0, 5.0, "peso del segnale di ritorno alla media"),
    Gene("w_range_against_htf", 0.0, 3.0, "penalità se il ritorno alla media va contro H4"),
    Gene("w_level", 0.0, 2.0, "peso della vicinanza a supporto/resistenza"),
)
GENE_BY_NAME = {g.name: g for g in GENOME}


def params_to_dict(p: RuleParams) -> dict[str, Any]:
    return asdict(p)


def params_from_dict(values: dict[str, Any]) -> RuleParams:
    """Costruisce RuleParams validi: ignora chiavi sconosciute, rispetta i limiti e i vincoli."""
    known = {f.name for f in fields(RuleParams)}
    clean: dict[str, Any] = {}
    for k, v in values.items():
        if k not in known:
            continue
        gene = GENE_BY_NAME.get(k)
        if gene is None:
            clean[k] = v
        elif gene.is_bool:
            clean[k] = bool(v)
        else:
            clean[k] = round(min(gene.high, max(gene.low, float(v))), 3)
    p = replace(RuleParams(), **clean)
    if p.rsi_healthy_low >= p.rsi_healthy_high:
        p = replace(p, rsi_healthy_high=min(75.0, p.rsi_healthy_low + 5.0))
    if p.overextended_atr <= p.pullback_atr:
        p = replace(p, overextended_atr=p.pullback_atr + 0.5)
    if not p.trend_enabled and not p.range_enabled:
        p = replace(p, trend_enabled=True)
    return p


def params_key(p: RuleParams) -> str:
    blob = json.dumps(params_to_dict(p), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def random_params(rng: random.Random) -> RuleParams:
    values: dict[str, Any] = {}
    for g in GENOME:
        values[g.name] = rng.random() < 0.8 if g.is_bool else rng.uniform(g.low, g.high)
    return params_from_dict(values)


def mutate(p: RuleParams, rng: random.Random, rate: float = 0.3, scale: float = 0.15) -> RuleParams:
    values = params_to_dict(p)
    changed = False
    for g in GENOME:
        if rng.random() >= rate:
            continue
        changed = True
        if g.is_bool:
            values[g.name] = not values[g.name]
        else:
            values[g.name] = values[g.name] + rng.gauss(0.0, scale * (g.high - g.low))
    if not changed:
        g = rng.choice([g for g in GENOME if not g.is_bool])
        values[g.name] = values[g.name] + rng.gauss(0.0, scale * (g.high - g.low))
    return params_from_dict(values)


def crossover(a: RuleParams, b: RuleParams, rng: random.Random) -> RuleParams:
    da, db = params_to_dict(a), params_to_dict(b)
    return params_from_dict({g.name: (da if rng.random() < 0.5 else db)[g.name] for g in GENOME})


# --------------------------------------------------------------------------- analisi in cache


class SnapshotCache:
    """Analisi di mercato precalcolate per ogni barra.

    L'analisi non dipende dalla strategia, quindi si calcola una sola volta e ogni candidato viene
    valutato riusandola: è ciò che rende praticabile valutare centinaia di varianti.
    """

    def __init__(self, snapshots: dict[tuple[str, pd.Timestamp], MarketSnapshot]) -> None:
        self.snapshots = snapshots

    def analyzer(self, candles: pd.DataFrame, instrument: str) -> MarketSnapshot | None:
        if candles.empty:
            return None
        return self.snapshots.get((instrument, candles.index[-1]))

    @staticmethod
    def _key(instrument: str, df: pd.DataFrame, history_bars: int) -> str:
        h = hashlib.sha1()
        h.update(f"{instrument}|{history_bars}|{len(df)}".encode())
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        return h.hexdigest()[:16]

    @classmethod
    def build(
        cls,
        data: dict[str, pd.DataFrame],
        history_bars: int = 500,
        workers: int = 1,
        cache_dir: str | Path | None = None,
    ) -> "SnapshotCache":
        snapshots: dict[tuple[str, pd.Timestamp], MarketSnapshot] = {}
        for inst, df in data.items():
            path = Path(cache_dir) / f"snapshots-{cls._key(inst, df, history_bars)}.pkl" if cache_dir else None
            if path and path.exists():
                with path.open("rb") as fh:
                    part = pickle.load(fh)
            else:
                part = _compute_snapshots(inst, df, history_bars, workers)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as fh:
                        pickle.dump(part, fh)
            snapshots.update(part)
        return cls(snapshots)


def _snapshots_for(inst: str, df: pd.DataFrame, history_bars: int, indices: list[int]) -> dict:
    out = {}
    for i in indices:
        snap = analyze(df.iloc[max(0, i - history_bars + 1) : i + 1], inst)
        if snap is not None:
            # Le candele recenti servono solo a Claude: toglierle riduce molto la memoria.
            out[(inst, df.index[i])] = replace(snap, recent_candles=[])
    return out


def _compute_snapshots(inst: str, df: pd.DataFrame, history_bars: int, workers: int) -> dict:
    indices = list(range(len(df)))
    if workers <= 1:
        return _snapshots_for(inst, df, history_bars, indices)
    chunks = [list(c) for c in np.array_split(indices, workers * 4) if len(c)]
    out: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_snapshots_for, inst, df.iloc[max(0, c[0] - history_bars + 1) : c[-1] + 1],
                               history_bars, [i - max(0, c[0] - history_bars + 1) for i in c])
                   for c in chunks]
        for fut in futures:
            out.update(fut.result())
    return out


# --------------------------------------------------------------------------- valutazione


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 20
    generations: int = 10
    elite: int = 2
    tournament: int = 3
    mutation_rate: float = 0.3
    mutation_scale: float = 0.15
    folds: int = 3
    holdout_fraction: float = 0.3
    warmup_bars: int = 300
    min_trades_per_segment: int = 8
    min_holdout_trades: int = 10
    promote_margin: float = 0.25
    max_holdout_drawdown_pct: float = 15.0
    spread_pips: float = 1.0
    workers: int = 0  # 0 = automatico (CPU - 1, max 4)
    llm_proposals: int = 0
    seed: int | None = None


@dataclass(frozen=True)
class SegmentScore:
    instrument: str
    start: str
    end: str
    fitness: float
    trades: int
    return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    trades: int
    return_pct: float
    max_drawdown_pct: float
    segments: list[SegmentScore] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {"fitness": round(self.fitness, 3), "trades": self.trades,
                "return_pct": round(self.return_pct, 2), "max_drawdown_pct": round(self.max_drawdown_pct, 2)}


def segment_fitness(result: BacktestResult, min_trades: int) -> float:
    """Robustezza statistica del vantaggio: t-statistic dei rendimenti per trade, penalizzata dal drawdown.

    Favorisce un vantaggio costante su molti trade rispetto a pochi colpi grossi.
    """
    n = len(result.trades)
    if n < min_trades:
        return -3.0 + 2.0 * n / max(min_trades, 1)
    r = np.array([t.realized_pl / result.initial_balance * 100 for t in result.trades])
    std = r.std(ddof=1)
    t_stat = float(r.mean() / std * math.sqrt(n)) if std > 0 else (3.0 if r.mean() > 0 else -3.0)
    dd = result.metrics.get("max_drawdown_pct", 0.0)
    return t_stat - 0.3 * max(0.0, dd - 10.0)


def split_segments(
    data: dict[str, pd.DataFrame], folds: int, holdout_fraction: float, warmup_bars: int
) -> tuple[list[tuple[str, pd.DataFrame]], list[tuple[str, pd.DataFrame]]]:
    train: list[tuple[str, pd.DataFrame]] = []
    holdout: list[tuple[str, pd.DataFrame]] = []
    for inst, df in data.items():
        usable = df.iloc[warmup_bars:]
        if len(usable) < 200:
            raise ValueError(f"Storico troppo corto per {inst}: servono almeno {warmup_bars + 200} barre")
        cut = int(len(usable) * (1 - holdout_fraction))
        bounds = np.linspace(0, cut, folds + 1, dtype=int)
        train.extend((inst, usable.iloc[a:b]) for a, b in zip(bounds[:-1], bounds[1:]))
        holdout.append((inst, usable.iloc[cut:]))
    return train, holdout


def evaluate(
    params: RuleParams,
    segments: list[tuple[str, pd.DataFrame]],
    cache: SnapshotCache,
    risk: RiskConfig,
    engine: EngineConfig,
    cfg: EvolutionConfig,
    min_trades: int,
) -> Evaluation:
    scores: list[SegmentScore] = []
    for inst, seg in segments:
        res = run_backtest(
            {inst: seg}, RuleBasedReasoner(params), risk, spread_pips=cfg.spread_pips, warmup_bars=0,
            journal=Journal(memory=20), engine_config=engine, analyzer=cache.analyzer,
        )
        m = res.metrics
        pf = m["profit_factor"]
        scores.append(SegmentScore(
            instrument=inst, start=f"{seg.index[0]:%Y-%m-%d}", end=f"{seg.index[-1]:%Y-%m-%d}",
            fitness=round(segment_fitness(res, min_trades), 4), trades=m["num_trade"],
            return_pct=m["rendimento_totale_pct"], max_drawdown_pct=m["max_drawdown_pct"],
            profit_factor=pf if math.isfinite(pf) else 99.0, win_rate_pct=m["win_rate_pct"],
        ))
    fit = np.array([s.fitness for s in scores])
    # Media meno dispersione: una strategia brillante in un periodo e disastrosa in un altro perde.
    fitness = float(fit.mean() - 0.5 * fit.std()) if len(fit) > 1 else float(fit.mean())
    return Evaluation(
        fitness=fitness,
        trades=sum(s.trades for s in scores),
        return_pct=float(np.mean([s.return_pct for s in scores])),
        max_drawdown_pct=float(max(s.max_drawdown_pct for s in scores)),
        segments=scores,
    )


_WORKER: dict[str, Any] = {}


def _init_worker(segments, cache, risk, engine, cfg) -> None:
    logging.getLogger("forex_trader.engine").setLevel(logging.WARNING)
    _WORKER.update(segments=segments, cache=cache, risk=risk, engine=engine, cfg=cfg)


def _eval_in_worker(params: RuleParams) -> Evaluation:
    w = _WORKER
    return evaluate(params, w["segments"], w["cache"], w["risk"], w["engine"], w["cfg"],
                    w["cfg"].min_trades_per_segment)


# --------------------------------------------------------------------------- registro


class StrategyRegistry:
    """Storico delle strategie: campione attuale, genealogia, metriche e promozioni (file JSON)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"champion": None, "strategies": {}, "promotions": [], "runs": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def champion_id(self) -> str | None:
        return self.data.get("champion")

    def champion_params(self, default: RuleParams | None = None) -> RuleParams:
        cid = self.champion_id
        if cid and cid in self.data["strategies"]:
            return params_from_dict(self.data["strategies"][cid]["params"])
        return default or RuleParams()

    def last_run_at(self) -> datetime | None:
        runs = self.data.get("runs") or []
        return datetime.fromisoformat(runs[-1]["at"]) if runs else None

    def add(self, params: RuleParams, origin: str, parent: str | None, metrics: dict[str, Any],
            rationale: str = "") -> str:
        sid = params_key(params)
        entry = self.data["strategies"].setdefault(sid, {
            "params": params_to_dict(params), "origin": origin, "parent": parent,
            "created": datetime.now(timezone.utc).isoformat(), "rationale": rationale, "evaluations": [],
        })
        entry["evaluations"].append({"at": datetime.now(timezone.utc).isoformat(), **metrics})
        return sid

    def promote(self, sid: str, reason: str) -> None:
        self.data["promotions"].append({
            "at": datetime.now(timezone.utc).isoformat(), "from": self.champion_id, "to": sid, "reason": reason,
        })
        self.data["champion"] = sid

    def record_run(self, summary: dict[str, Any]) -> None:
        self.data.setdefault("runs", []).append({"at": datetime.now(timezone.utc).isoformat(), **summary})


# --------------------------------------------------------------------------- evoluzione


@dataclass
class EvolutionReport:
    champion_id: str
    candidate_id: str
    promoted: bool
    reason: str
    champion_train: Evaluation
    candidate_train: Evaluation
    champion_holdout: Evaluation
    candidate_holdout: Evaluation
    candidate_params: RuleParams
    history: list[dict[str, float]] = field(default_factory=list)


Proposer = Callable[[RuleParams, Evaluation, list[tuple[RuleParams, Evaluation]], int], list[tuple[RuleParams, str]]]


def _tournament(pop: list[tuple[RuleParams, Evaluation]], k: int, rng: random.Random) -> RuleParams:
    contenders = rng.sample(pop, min(k, len(pop)))
    return max(contenders, key=lambda pe: pe[1].fitness)[0]


def evolve(
    data: dict[str, pd.DataFrame],
    registry: StrategyRegistry,
    cfg: EvolutionConfig | None = None,
    risk: RiskConfig | None = None,
    engine: EngineConfig | None = None,
    proposer: Proposer | None = None,
    cache: SnapshotCache | None = None,
    cache_dir: str | Path | None = None,
    allow_promotion: bool = True,
    base: RuleParams | None = None,
    progress: Callable[[str], None] = log.info,
) -> EvolutionReport:
    """Una sessione di evoluzione. `base` è la strategia di partenza se il registro non ha un campione."""
    cfg = cfg or EvolutionConfig()
    risk = risk or RiskConfig()
    engine = replace(engine or EngineConfig(), dry_run=False)
    rng = random.Random(cfg.seed)
    if cfg.workers <= 0:
        cfg = replace(cfg, workers=default_workers())

    progress("Precalcolo delle analisi di mercato...")
    cache = cache or SnapshotCache.build(data, engine.history_bars, cfg.workers, cache_dir)
    train, holdout = split_segments(data, cfg.folds, cfg.holdout_fraction, cfg.warmup_bars)

    champion = registry.champion_params(default=base)
    champion_id = params_key(champion)
    origins: dict[str, tuple[str, str | None, str]] = {champion_id: ("champion", None, "")}
    evaluated: dict[str, tuple[RuleParams, Evaluation]] = {}

    # Gli ordini simulati dei backtest non interessano chi segue l'evoluzione.
    engine_log = logging.getLogger("forex_trader.engine")
    previous_level = engine_log.level
    engine_log.setLevel(logging.WARNING)

    pool = None
    if cfg.workers > 1:
        pool = ProcessPoolExecutor(max_workers=cfg.workers, initializer=_init_worker,
                                   initargs=(train, cache, risk, engine, cfg))

    def score(batch: list[RuleParams]) -> None:
        todo = list({params_key(p): p for p in batch if params_key(p) not in evaluated}.values())
        if pool is not None:
            results = list(pool.map(_eval_in_worker, todo))
        else:
            results = [evaluate(p, train, cache, risk, engine, cfg, cfg.min_trades_per_segment) for p in todo]
        for p, ev in zip(todo, results):
            evaluated[params_key(p)] = (p, ev)

    def ranked() -> list[tuple[RuleParams, Evaluation]]:
        return sorted(evaluated.values(), key=lambda pe: pe[1].fitness, reverse=True)

    def ask_claude(n: int) -> list[RuleParams]:
        if proposer is None or n <= 0:
            return []
        champ_eval = evaluated[champion_id][1]
        try:
            proposals = proposer(champion, champ_eval, ranked()[:5], n)
        except Exception as exc:  # noqa: BLE001 - le proposte sono un extra, non devono fermare l'evoluzione
            log.warning("Proposte di Claude non disponibili: %s", exc)
            return []
        out = []
        for p, why in proposals:
            origins.setdefault(params_key(p), ("claude", champion_id, why))
            out.append(p)
        progress(f"Claude ha proposto {len(out)} varianti ragionate")
        return out

    try:
        score([champion])
        mutants = [mutate(champion, rng, cfg.mutation_rate, cfg.mutation_scale) for _ in range(cfg.population // 2)]
        randoms = [random_params(rng) for _ in range(cfg.population - 1 - len(mutants))]
        for p in mutants:
            origins.setdefault(params_key(p), ("mutation", champion_id, ""))
        for p in randoms:
            origins.setdefault(params_key(p), ("random", None, ""))
        population = [champion, *mutants, *randoms, *ask_claude(cfg.llm_proposals)]

        history = []
        for gen in range(cfg.generations):
            score(population)
            ranking = ranked()
            best_p, best_e = ranking[0]
            history.append({"generation": gen + 1, "best_fitness": round(best_e.fitness, 3),
                            "champion_fitness": round(evaluated[champion_id][1].fitness, 3),
                            "evaluated": len(evaluated)})
            progress(f"Generazione {gen + 1}/{cfg.generations}: migliore {best_e.fitness:+.3f} "
                     f"({best_e.trades} trade, {best_e.return_pct:+.1f}%), "
                     f"campione {evaluated[champion_id][1].fitness:+.3f}")
            if gen == cfg.generations - 1:
                break
            current = [(p, evaluated[params_key(p)][1]) for p in population]
            nxt = [p for p, _ in ranking[: cfg.elite]]
            while len(nxt) < cfg.population:
                a, b = _tournament(current, cfg.tournament, rng), _tournament(current, cfg.tournament, rng)
                child = mutate(crossover(a, b, rng), rng, cfg.mutation_rate, cfg.mutation_scale)
                origins.setdefault(params_key(child), ("offspring", params_key(a), ""))
                nxt.append(child)
            if cfg.llm_proposals and gen == cfg.generations // 2 - 1:
                nxt += ask_claude(cfg.llm_proposals)
            population = nxt
        candidate, cand_train = ranked()[0]
        cand_id = params_key(candidate)
        champ_train = evaluated[champion_id][1]

        progress("Verifica su dati mai visti (holdout)...")
        min_ho = max(1, cfg.min_holdout_trades // max(len(holdout), 1))
        champ_ho = evaluate(champion, holdout, cache, risk, engine, cfg, min_ho)
        cand_ho = (champ_ho if cand_id == champion_id
                   else evaluate(candidate, holdout, cache, risk, engine, cfg, min_ho))
    finally:
        engine_log.setLevel(previous_level)
        if pool is not None:
            pool.shutdown()

    if cand_id == champion_id:
        promoted, reason = False, "il campione attuale resta la strategia migliore anche in addestramento"
    elif cand_ho.trades < cfg.min_holdout_trades:
        promoted, reason = False, f"troppo pochi trade sui dati mai visti ({cand_ho.trades} < {cfg.min_holdout_trades})"
    elif cand_ho.max_drawdown_pct > cfg.max_holdout_drawdown_pct:
        promoted, reason = False, f"drawdown sui dati mai visti troppo alto ({cand_ho.max_drawdown_pct:.1f}%)"
    elif cand_ho.fitness <= 0:
        promoted, reason = False, f"nessun vantaggio statistico sui dati mai visti (fitness {cand_ho.fitness:+.3f})"
    elif cand_ho.return_pct <= 0:
        promoted, reason = False, f"non guadagna sui dati mai visti ({cand_ho.return_pct:+.2f}%)"
    elif cand_ho.fitness < champ_ho.fitness + cfg.promote_margin:
        promoted, reason = False, (f"miglioramento sui dati mai visti insufficiente "
                                   f"({cand_ho.fitness:+.3f} vs {champ_ho.fitness:+.3f}, margine {cfg.promote_margin})")
    else:
        promoted, reason = True, (f"migliore sui dati mai visti: fitness {cand_ho.fitness:+.3f} vs "
                                  f"{champ_ho.fitness:+.3f}, rendimento {cand_ho.return_pct:+.2f}% "
                                  f"vs {champ_ho.return_pct:+.2f}%")
    promoted = promoted and allow_promotion

    data_range = {inst: [f"{df.index[0]:%Y-%m-%d}", f"{df.index[-1]:%Y-%m-%d}"] for inst, df in data.items()}
    registry.add(champion, "champion", None,
                 metrics={"train": champ_train.summary(), "holdout": champ_ho.summary(), "data": data_range})
    if cand_id != champion_id:
        origin, _, why = origins.get(cand_id, ("evolution", None, ""))
        # Genitore nel registro = il campione da cui è partita la sessione (gli intermedi non si salvano).
        registry.add(candidate, origin, champion_id,
                     metrics={"train": cand_train.summary(), "holdout": cand_ho.summary(), "data": data_range},
                     rationale=why)
    if promoted:
        registry.promote(cand_id, reason)
    elif registry.champion_id is None:
        registry.promote(champion_id, "strategia iniziale")
    registry.record_run({"candidate": cand_id, "promoted": promoted, "reason": reason,
                         "evaluated": len(evaluated), "data": data_range})
    registry.save()

    return EvolutionReport(
        champion_id=champion_id, candidate_id=cand_id, promoted=promoted, reason=reason,
        champion_train=champ_train, candidate_train=cand_train, champion_holdout=champ_ho,
        candidate_holdout=cand_ho, candidate_params=candidate, history=history,
    )


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))
