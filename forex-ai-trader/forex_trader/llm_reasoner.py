"""Ragionamento sui trend con Claude: l'LLM propone, il gestore del rischio dispone.

Claude riceve la fotografia quantitativa del mercato, il parere del modello a regole, le posizioni aperte e
gli esiti recenti, e restituisce una decisione strutturata con tesi, fattori, rischi e invalidazione.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

from .models import Decision, MarketSnapshot
from .reasoning import ReasoningContext
from .strategy import RuleBasedReasoner

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


def claude_credentials_available() -> bool:
    if any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    # Profilo salvato da `ant auth login`.
    return (Path.home() / ".config" / "anthropic").is_dir()

SYSTEM_PROMPT = """\
Sei un analista e trader Forex disciplinato. Il tuo compito è leggere i dati di mercato che ricevi, \
ragionare sul trend e sul contesto, e decidere se aprire una posizione (BUY/SELL) o restare fuori (HOLD).

Come ragionare:
- Parti dal timeframe superiore (H4) per la direzione, usa H1 per il tempismo di ingresso.
- Identifica il regime (trend, laterale, volatile, transizione) e scegli l'approccio coerente: \
pullback nel trend oppure ritorno alla media nel range. In regime volatile resta fuori.
- Considera livelli di supporto/resistenza, momentum (MACD, RSI), forza del trend (ADX, DI) ed estensione \
dalla EMA20 misurata in ATR.
- Il "parere quantitativo" è un secondo parere da un modello a regole: valutalo criticamente, non copiarlo.
- Tieni conto delle posizioni già aperte (evita esposizione duplicata o correlata) e degli esiti recenti: \
se gli ultimi trade sullo stesso setup hanno perso, chiediti cosa è cambiato.
- HOLD è la scelta corretta ogni volta che il vantaggio non è chiaro. Non esiste obbligo di operare.

Parametri di uscita:
- stop_loss_atr_multiple: distanza dello stop in multipli di ATR H1 (tipicamente 1.0-3.0), oltre il livello \
che invaliderebbe la tesi.
- take_profit_rr: rapporto rendimento/rischio del target (tipicamente 1.5-3.0), coerente con il livello \
successivo di resistenza/supporto.
- confidence: probabilità soggettiva (0-1) che la tesi si realizzi prima dello stop. Sii calibrato.

Il dimensionamento della posizione e i limiti di perdita sono applicati da un gestore del rischio separato \
che può rifiutare la tua proposta: concentrati sulla qualità dell'analisi. Scrivi in italiano, in modo conciso."""


class TradeDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    market_regime: str
    thesis: str
    supporting_factors: list[str]
    risks: list[str]
    invalidation: str
    stop_loss_atr_multiple: float
    take_profit_rr: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ClaudeSettings:
    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 16000
    timeout_s: float = 300.0


class ClaudeReasoner:
    """Chiama Claude per una decisione. In caso di errore o rifiuto ripiega sul ragionatore a regole."""

    def __init__(
        self,
        settings: ClaudeSettings | None = None,
        client: anthropic.Anthropic | None = None,
        fallback: RuleBasedReasoner | None = None,
    ) -> None:
        self.settings = settings or ClaudeSettings()
        self.client = client or anthropic.Anthropic(timeout=self.settings.timeout_s)
        self.fallback = fallback or RuleBasedReasoner()
        self.calls = 0

    def build_prompt(
        self, snapshot: MarketSnapshot, context: ReasoningContext, quant_view: Decision | None
    ) -> str:
        payload: dict[str, object] = {
            "mercato": snapshot.to_dict(),
            "conto": asdict(context.account),
            "posizioni_aperte": [
                {**asdict(p), "open_time": p.open_time.isoformat()} for p in context.open_positions
            ],
            "trade_recenti": [
                {"instrument": t.instrument, "lato": "long" if t.units > 0 else "short",
                 "pl": round(t.realized_pl, 2), "motivo_chiusura": t.reason,
                 "chiuso": t.close_time.isoformat()}
                for t in context.recent_trades
            ],
            "decisioni_recenti": context.recent_decisions,
        }
        if quant_view is not None:
            payload["parere_quantitativo"] = {
                "azione": quant_view.action,
                "punteggio": round(quant_view.score, 2),
                "tesi": quant_view.thesis,
                "pro": quant_view.supporting_factors,
                "contro": quant_view.risks,
            }
        return (
            f"Analizza {snapshot.instrument} e decidi. Dati (JSON):\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str)}"
        )

    def decide(
        self,
        snapshot: MarketSnapshot,
        context: ReasoningContext,
        quant_view: Decision | None = None,
    ) -> Decision:
        prompt = self.build_prompt(snapshot, context, quant_view)
        self.calls += 1
        try:
            response = self.client.beta.messages.parse(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.settings.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=TradeDecision,
            )
        except anthropic.APIConnectionError as exc:
            log.warning("Claude non raggiungibile (%s): uso il ragionatore a regole", exc)
            return self._fallback(snapshot, context, "connessione fallita")
        except anthropic.RateLimitError:
            log.warning("Rate limit Claude: uso il ragionatore a regole")
            return self._fallback(snapshot, context, "rate limit")
        except anthropic.APIStatusError as exc:
            log.error("Errore API Claude %s: %s", exc.status_code, exc.message)
            return self._fallback(snapshot, context, f"errore API {exc.status_code}")

        if response.stop_reason == "refusal":
            return self._fallback(snapshot, context, "richiesta rifiutata dal modello")
        parsed = response.parsed_output
        if parsed is None:
            return self._fallback(snapshot, context, f"risposta non valida ({response.stop_reason})")

        return Decision(
            action=parsed.action,
            confidence=_clamp(parsed.confidence, 0.0, 1.0),
            stop_atr_mult=_clamp(parsed.stop_loss_atr_multiple, 0.5, 5.0),
            take_profit_rr=_clamp(parsed.take_profit_rr, 0.5, 6.0),
            thesis=parsed.thesis,
            supporting_factors=parsed.supporting_factors,
            risks=parsed.risks,
            invalidation=parsed.invalidation,
            source=f"claude:{response.model}",
            score=quant_view.score if quant_view else 0.0,
            regime=parsed.market_regime,
        )

    def _fallback(self, snapshot: MarketSnapshot, context: ReasoningContext, why: str) -> Decision:
        d = self.fallback.decide(snapshot, context)
        return Decision(**{**asdict(d), "source": f"rules(fallback: {why})"})


HybridMode = Literal["consensus", "llm_final"]


class HybridReasoner:
    """Combina regole e Claude.

    - Le regole fanno da filtro: Claude viene consultato solo quando c'è un possibile setup
      (|punteggio| >= prefilter_score), così i costi restano contenuti.
    - consensus: si opera solo se Claude conferma la stessa direzione delle regole.
    - llm_final: decide Claude (le regole restano solo un parere e un filtro).
    """

    def __init__(
        self,
        llm: ClaudeReasoner,
        rules: RuleBasedReasoner | None = None,
        mode: HybridMode = "consensus",
        prefilter_score: float = 2.0,
        max_llm_calls: int | None = None,
    ) -> None:
        self.llm = llm
        self.rules = rules or RuleBasedReasoner()
        self.mode = mode
        self.prefilter_score = prefilter_score
        self.max_llm_calls = max_llm_calls

    def decide(self, snapshot: MarketSnapshot, context: ReasoningContext) -> Decision:
        quant = self.rules.decide(snapshot, context)
        budget_left = self.max_llm_calls is None or self.llm.calls < self.max_llm_calls
        if abs(quant.score) < self.prefilter_score or snapshot.regime == "volatile" or not budget_left:
            return quant

        llm = self.llm.decide(snapshot, context, quant_view=quant)
        if self.mode == "llm_final" or llm.action == "HOLD":
            return llm
        quant_dir = "BUY" if quant.score > 0 else "SELL"
        if llm.action != quant_dir:
            return Decision(
                **{**asdict(llm), "action": "HOLD",
                   "thesis": f"Disaccordo: Claude propone {llm.action}, le regole indicano {quant_dir}. "
                             f"Resto fuori. Tesi di Claude: {llm.thesis}",
                   "source": f"hybrid({llm.source})"}
            )
        return Decision(**{**asdict(llm), "source": f"hybrid({llm.source})"})
