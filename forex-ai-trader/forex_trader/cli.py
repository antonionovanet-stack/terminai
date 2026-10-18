"""Riga di comando: backtest, analisi singola, evoluzione delle strategie e trading continuo."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

from .analysis import analyze
from .autoevolve import maybe_evolve, resolve_rules, run_evolution
from .backtest import run_backtest
from .brokers.mt5 import MT5Error
from .brokers.oanda import OandaError
from .config import AppConfig, load_config
from .data import load_csv, synthetic_candles
from .engine import TradingEngine
from .evolution import GENOME, StrategyRegistry, params_to_dict
from .fx import round_price
from .journal import Journal
from .llm_reasoner import ClaudeReasoner, HybridReasoner, claude_credentials_available
from .reasoning import Reasoner, ReasoningContext
from .risk import RiskManager
from .strategy import RuleBasedReasoner

log = logging.getLogger("forex_trader")

RISK_BANNER = """\
ATTENZIONE: il trading sul Forex con leva comporta un rischio elevato di perdita del capitale.
Questo software è uno strumento sperimentale, non una consulenza finanziaria e non garantisce profitti.
Prova sempre a lungo in backtest e su conto demo prima di usare denaro reale."""


def build_reasoner(cfg: AppConfig, kind: str | None = None) -> Reasoner:
    kind = kind or cfg.reasoner.kind
    rules = RuleBasedReasoner(cfg.rules)
    if kind != "rules" and not claude_credentials_available():
        log.warning("Nessuna credenziale Claude (ANTHROPIC_API_KEY): uso solo il ragionatore a regole")
        kind = "rules"
    if kind == "rules":
        return rules
    llm = ClaudeReasoner(cfg.claude, fallback=rules)
    if kind == "claude":
        return llm
    if kind == "hybrid":
        mode = cfg.reasoner.hybrid_mode
        if mode not in ("consensus", "llm_final"):
            raise SystemExit(f"hybrid_mode non valido: {mode}")
        return HybridReasoner(llm, rules, mode, cfg.reasoner.prefilter_score, cfg.reasoner.max_llm_calls)
    raise SystemExit(f"Ragionatore sconosciuto: {kind}")


def cmd_backtest(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.csv:
        df = load_csv(args.csv)
    else:
        df = synthetic_candles(bars=args.bars, seed=args.seed, start_price=args.start_price)
    if args.reasoner != "rules" and cfg.reasoner.max_llm_calls is None:
        cfg.reasoner.max_llm_calls = args.max_llm_calls
    reasoner = build_reasoner(cfg, args.reasoner)
    journal = Journal(args.journal) if args.journal else Journal()
    result = run_backtest(
        {args.instrument: df}, reasoner, cfg.risk, initial_balance=args.balance,
        spread_pips=args.spread, journal=journal, engine_config=cfg.engine,
    )
    print(f"Backtest {args.instrument} su {len(df)} barre H1 ({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})")
    print(f"Saldo: {result.initial_balance:,.2f} -> {result.final_balance:,.2f}")
    for k, v in result.metrics.items():
        print(f"  {k:24s} {v}")
    if result.halted_reason:
        print(f"  trading sospeso: {result.halted_reason}")
    if isinstance(reasoner, (ClaudeReasoner, HybridReasoner)):
        llm = reasoner if isinstance(reasoner, ClaudeReasoner) else reasoner.llm
        print(f"  chiamate a Claude: {llm.calls}")
    return 0


def _live_broker(args: argparse.Namespace, cfg: AppConfig):
    kind = args.broker or cfg.broker
    if kind == "mt5":
        from .brokers.mt5 import MT5Broker

        opts = asdict(cfg.mt5)
        return MT5Broker.from_env(live=args.live, **opts)
    if kind == "oanda":
        from .brokers.oanda import OandaBroker

        return OandaBroker.from_env(live=args.live)
    raise SystemExit(f"Broker sconosciuto: {kind}")


def cmd_analyze(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.synthetic:
        candles = synthetic_candles(bars=800, seed=args.seed)
        from .models import Account

        account = Account(10_000.0, 10_000.0, "USD")
    else:
        broker = _live_broker(args, cfg)
        candles = broker.get_candles(args.instrument, cfg.engine.history_bars)
        account = broker.get_account()
    snapshot = analyze(candles, args.instrument)
    if snapshot is None:
        print("Storico insufficiente per l'analisi.")
        return 1
    reasoner = build_reasoner(cfg, args.reasoner)
    decision = reasoner.decide(snapshot, ReasoningContext(account=account))
    ltf, htf = snapshot.ltf, snapshot.htf
    rp = lambda x: round_price(snapshot.instrument, x)  # noqa: E731
    print(f"{snapshot.instrument} @ {rp(snapshot.price)} ({snapshot.time:%Y-%m-%d %H:%M} UTC)")
    print(f"  Regime: {snapshot.regime} | Trend H4: {htf.trend} | Trend H1: {ltf.trend}")
    print(f"  ADX {ltf.adx:.1f}  RSI {ltf.rsi:.1f}  ATR {ltf.atr:.5f} (pct {ltf.atr_percentile:.0%})")
    print(f"  Supporti {[rp(x) for x in snapshot.supports]}  Resistenze {[rp(x) for x in snapshot.resistances]}")
    print(f"\nDecisione [{decision.source}]: {decision.action} (confidenza {decision.confidence:.2f})")
    print(f"  Tesi: {decision.thesis}")
    for f in decision.supporting_factors:
        print(f"  + {f}")
    for r in decision.risks:
        print(f"  - {r}")
    if decision.invalidation:
        print(f"  Invalidazione: {decision.invalidation}")
    if args.json:
        print(json.dumps({"snapshot": snapshot.to_dict(), "decision": asdict(decision)}, indent=2, default=str))
    return 0


def cmd_run(args: argparse.Namespace, cfg: AppConfig) -> int:
    print(RISK_BANNER, file=sys.stderr)
    broker = _live_broker(args, cfg)
    engine_cfg = replace(cfg.engine, dry_run=args.dry_run)
    instruments = args.instruments or cfg.instruments
    engine = TradingEngine(
        broker, build_reasoner(cfg, args.reasoner), RiskManager(cfg.risk), instruments,
        Journal(cfg.journal_path), engine_cfg,
    )
    mode = "REALE" if args.live else "DEMO"
    log.info("Avvio su conto %s, strumenti %s, dry_run=%s", mode, instruments, args.dry_run)
    last_hour = None
    while True:
        now = datetime.now(timezone.utc)
        try:
            evolved = maybe_evolve(broker, cfg, instruments, now)
        except Exception:  # noqa: BLE001 - un'evoluzione fallita non deve fermare il trading
            log.exception("Evoluzione fallita: continuo con la strategia attuale")
            evolved = None
        if evolved is not None:
            cfg.rules = evolved
            engine.reasoner = build_reasoner(cfg, args.reasoner)
            log.info("Nuova strategia campione in uso da ora")
        # Si ragiona una volta per candela H1 chiusa, con qualche secondo di margine per la chiusura.
        if now.hour != last_hour and (now.minute > 0 or now.second >= 5):
            try:
                engine.step()
            except Exception:  # noqa: BLE001 - il ciclo deve sopravvivere a errori transitori
                log.exception("Errore nel ciclo di trading")
            last_hour = now.hour
            if args.once:
                return 0
        time.sleep(min(cfg.poll_seconds, 30))


def _evolution_data(args: argparse.Namespace, cfg: AppConfig) -> dict:
    if args.csv:
        data = {}
        for item in args.csv:
            inst, sep, path = item.partition("=")
            if not sep:
                raise SystemExit(f"Formato --csv atteso STRUMENTO=percorso, ricevuto {item!r}")
            data[inst.upper()] = load_csv(path)
        return data
    instruments = args.instruments or cfg.instruments
    if args.synthetic:
        starts = {"JPY": 150.0}
        return {
            inst: synthetic_candles(bars=args.bars, seed=(args.seed or 7) + i,
                                    start_price=starts.get(inst[-3:], 1.10))
            for i, inst in enumerate(instruments)
        }
    from .autoevolve import collect_history

    return collect_history(_live_broker(args, cfg), instruments, cfg.evolution.history_bars)


def _fmt_eval(label: str, ev) -> str:
    return (f"  {label:22s} fitness {ev.fitness:+.3f} | {ev.trades:4d} trade | "
            f"rendimento medio {ev.return_pct:+6.2f}% | drawdown max {ev.max_drawdown_pct:5.2f}%")


def cmd_evolve(args: argparse.Namespace, cfg: AppConfig) -> int:
    search = cfg.evolution.search
    overrides = {k: v for k, v in {
        "generations": args.generations, "population": args.population, "seed": args.seed,
        "llm_proposals": args.claude, "workers": args.workers,
    }.items() if v is not None}
    cfg.evolution.search = replace(search, **overrides)
    if args.registry:
        cfg.evolution.registry_path = args.registry
    before = StrategyRegistry(cfg.evolution.registry_path).champion_params(default=cfg.rules)
    data = _evolution_data(args, cfg)
    if not data:
        print("Nessun dato disponibile per l'evoluzione.")
        return 1
    for inst, df in data.items():
        print(f"{inst}: {len(df)} barre H1 ({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})")
    report = run_evolution(cfg, data, allow_promotion=not args.no_promote)

    print("\nAddestramento (walk-forward):")
    print(_fmt_eval(f"campione {report.champion_id}", report.champion_train))
    print(_fmt_eval(f"candidato {report.candidate_id}", report.candidate_train))
    print("Verifica su dati mai visti:")
    print(_fmt_eval(f"campione {report.champion_id}", report.champion_holdout))
    print(_fmt_eval(f"candidato {report.candidate_id}", report.candidate_holdout))
    verdict = "PROMOSSO a nuovo campione" if report.promoted else "NON promosso"
    print(f"\nEsito: {verdict}: {report.reason}")
    if report.candidate_id != report.champion_id:
        base = params_to_dict(before)
        changes = {k: v for k, v in params_to_dict(report.candidate_params).items() if base.get(k) != v}
        print("Modifiche del candidato rispetto alla strategia di partenza:")
        for k, v in changes.items():
            print(f"  {k:22s} {base.get(k)} -> {v}")
    return 0


def cmd_strategies(args: argparse.Namespace, cfg: AppConfig) -> int:
    registry = StrategyRegistry(args.registry or cfg.evolution.registry_path)
    data = registry.data
    if not data["strategies"]:
        print("Registro vuoto: esegui prima `forex-trader evolve`.")
        return 0
    print(f"Campione attuale: {registry.champion_id}\n")
    for sid, s in data["strategies"].items():
        last = s["evaluations"][-1]
        mark = "*" if sid == registry.champion_id else " "
        ho = last.get("holdout", {})
        print(f"{mark} {sid}  origine {s['origin']:9s} genitore {s['parent'] or '-':10s} "
              f"holdout fitness {ho.get('fitness', 0):+.3f} trade {ho.get('trades', 0)} "
              f"rendimento {ho.get('return_pct', 0):+.2f}%")
        if s.get("rationale"):
            print(f"      ipotesi: {s['rationale']}")
    if data["promotions"]:
        print("\nPromozioni:")
        for pr in data["promotions"]:
            print(f"  {pr['at'][:16]}  {pr['from'] or '-'} -> {pr['to']}: {pr['reason']}")
    if args.show:
        params = registry.champion_params()
        print("\nParametri del campione:")
        desc = {g.name: g.description for g in GENOME}
        for k, v in params_to_dict(params).items():
            print(f"  {k:22s} {v!s:8s} {desc.get(k, '')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forex-trader", description="Trading Forex automatico con ragionamento sui trend")
    p.add_argument("--config", help="file TOML di configurazione")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--default-strategy", action="store_true",
                   help="ignora il campione evoluto e usa i parametri della configurazione")
    sub = p.add_subparsers(dest="command", required=True)

    reasoner_choices = ["rules", "claude", "hybrid"]

    b = sub.add_parser("backtest", help="simula la strategia su dati storici o sintetici")
    b.add_argument("--instrument", default="EUR_USD")
    b.add_argument("--csv", help="CSV H1 con time,open,high,low,close (senza: dati sintetici)")
    b.add_argument("--bars", type=int, default=4000)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--start-price", type=float, default=1.10)
    b.add_argument("--balance", type=float, default=10_000.0)
    b.add_argument("--spread", type=float, default=1.0, help="spread in pip")
    b.add_argument("--reasoner", choices=reasoner_choices, default="rules")
    b.add_argument("--max-llm-calls", type=int, default=30, help="tetto chiamate a Claude nel backtest")
    b.add_argument("--journal", help="salva il diario delle decisioni in questo file JSONL")

    a = sub.add_parser("analyze", help="analizza ora uno strumento e mostra il ragionamento")
    a.add_argument("--instrument", default="EUR_USD")
    a.add_argument("--reasoner", choices=reasoner_choices, default="hybrid")
    a.add_argument("--broker", choices=["mt5", "oanda"], help="broker da cui leggere i dati (default da config: mt5)")
    a.add_argument("--synthetic", action="store_true", help="usa dati sintetici invece del broker")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--live", action="store_true", help="consente un conto reale (richiede FOREX_ALLOW_LIVE=yes)")
    a.add_argument("--json", action="store_true")

    r = sub.add_parser("run", help="trading continuo su MetaTrader 5 o OANDA (conto demo di default)")
    r.add_argument("--broker", choices=["mt5", "oanda"], help="default da config: mt5")
    r.add_argument("--instruments", nargs="*")
    r.add_argument("--reasoner", choices=reasoner_choices)
    r.add_argument("--dry-run", action="store_true", help="ragiona e registra, senza inviare ordini")
    r.add_argument("--live", action="store_true", help="conto reale (richiede FOREX_ALLOW_LIVE=yes)")
    r.add_argument("--once", action="store_true", help="esegue un solo ciclo e termina")

    e = sub.add_parser("evolve", help="fa evolvere le strategie di ingresso e promuove la migliore")
    e.add_argument("--csv", action="append", metavar="STRUMENTO=FILE",
                   help="dati storici H1 (ripetibile), es. EUR_USD=dati/eurusd.csv")
    e.add_argument("--synthetic", action="store_true", help="usa dati sintetici (solo per prova)")
    e.add_argument("--bars", type=int, default=5000, help="barre sintetiche per strumento")
    e.add_argument("--broker", choices=["mt5", "oanda"], help="scarica lo storico dal broker (default)")
    e.add_argument("--live", action="store_true", help="consente un conto reale per leggere lo storico")
    e.add_argument("--instruments", nargs="*")
    e.add_argument("--generations", type=int)
    e.add_argument("--population", type=int)
    e.add_argument("--workers", type=int, help="processi paralleli")
    e.add_argument("--claude", type=int, metavar="N", help="chiedi a Claude N varianti ragionate (2 volte)")
    e.add_argument("--seed", type=int)
    e.add_argument("--registry", help="file del registro strategie")
    e.add_argument("--no-promote", action="store_true", help="valuta senza cambiare il campione")

    st = sub.add_parser("strategies", help="mostra campione, genealogia e promozioni")
    st.add_argument("--registry")
    st.add_argument("--show", action="store_true", help="mostra tutti i parametri del campione")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    if args.default_strategy:
        cfg.evolution.use_champion = False
    if args.command in ("backtest", "analyze", "run"):
        cfg.rules = resolve_rules(cfg)
    handlers = {"backtest": cmd_backtest, "analyze": cmd_analyze, "run": cmd_run,
                "evolve": cmd_evolve, "strategies": cmd_strategies}
    try:
        return handlers[args.command](args, cfg)
    except (MT5Error, OandaError) as exc:
        print(f"Errore broker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
