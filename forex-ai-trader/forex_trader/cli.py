"""Riga di comando: backtest, analisi singola e trading continuo."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

from .analysis import analyze
from .backtest import run_backtest
from .config import AppConfig, load_config
from .data import load_csv, synthetic_candles
from .engine import TradingEngine
from .fx import round_price
from .journal import Journal
from .llm_reasoner import ClaudeReasoner, HybridReasoner
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


def _live_broker(args: argparse.Namespace):
    from .brokers.oanda import OandaBroker

    return OandaBroker.from_env(live=args.live)


def cmd_analyze(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.synthetic:
        candles = synthetic_candles(bars=800, seed=args.seed)
        from .models import Account

        account = Account(10_000.0, 10_000.0, "USD")
    else:
        broker = _live_broker(args)
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
    broker = _live_broker(args)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forex-trader", description="Trading Forex automatico con ragionamento sui trend")
    p.add_argument("--config", help="file TOML di configurazione")
    p.add_argument("-v", "--verbose", action="store_true")
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
    a.add_argument("--synthetic", action="store_true", help="usa dati sintetici invece di OANDA")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--live", action="store_true", help="usa il conto reale OANDA (richiede FOREX_ALLOW_LIVE=yes)")
    a.add_argument("--json", action="store_true")

    r = sub.add_parser("run", help="trading continuo su OANDA (demo di default)")
    r.add_argument("--instruments", nargs="*")
    r.add_argument("--reasoner", choices=reasoner_choices)
    r.add_argument("--dry-run", action="store_true", help="ragiona e registra, senza inviare ordini")
    r.add_argument("--live", action="store_true", help="conto reale (richiede FOREX_ALLOW_LIVE=yes)")
    r.add_argument("--once", action="store_true", help="esegue un solo ciclo e termina")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    handlers = {"backtest": cmd_backtest, "analyze": cmd_analyze, "run": cmd_run}
    return handlers[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
