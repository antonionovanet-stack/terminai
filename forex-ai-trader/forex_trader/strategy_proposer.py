"""Claude come ricercatore di strategie: legge i risultati dell'evoluzione e propone varianti ragionate.

Le proposte non hanno corsie preferenziali: vengono valutate con lo stesso backtest walk-forward e lo
stesso controllo su dati mai visti di qualsiasi altro candidato.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import anthropic
from pydantic import BaseModel, create_model

from .evolution import GENOME, Evaluation, params_from_dict, params_to_dict
from .llm_reasoner import ClaudeSettings
from .strategy import RuleParams

log = logging.getLogger(__name__)

ProposedParams = create_model(  # type: ignore[call-overload]
    "ProposedParams", **{g.name: (bool if g.is_bool else float, ...) for g in GENOME}
)


class Proposal(BaseModel):
    hypothesis: str
    params: ProposedParams  # type: ignore[valid-type]


class ProposalSet(BaseModel):
    diagnosis: str
    proposals: list[Proposal]


SYSTEM_PROMPT = """\
Sei un ricercatore quantitativo che migliora una strategia di ingresso sul Forex (timeframe H1 con \
direzione da H4). La strategia assegna un punteggio a ogni candela sommando pesi per condizioni di \
trend, pullback, momentum, RSI e, in mercato laterale, ritorno alla media; entra quando il punteggio \
supera min_score. Stop e target sono in multipli di ATR.

Ricevi: il significato e i limiti di ogni parametro, il campione attuale con i risultati per periodo, e \
i migliori candidati già testati. Il tuo compito:
1. Diagnostica perché il campione vince o perde nei diversi periodi (pochi trade? target troppo lontani? \
ingressi contro trend? un setup che perde sistematicamente?).
2. Proponi varianti diverse tra loro, ognuna guidata da un'ipotesi esplicita e verificabile. Preferisci \
modifiche mirate a pochi parametri rispetto a stravolgimenti casuali.

La metrica di valutazione è la robustezza statistica (t-statistic dei rendimenti per trade, media tra \
periodi meno la dispersione), penalizzata per drawdown oltre il 10% e per pochi trade: strategie con \
pochissimi trade non vengono premiate. Ogni proposta sarà verificata su dati che nessuno ha visto. \
Rispetta i limiti indicati. Scrivi in italiano, in modo conciso."""


def _genome_table() -> list[dict[str, object]]:
    return [
        {"nome": g.name, "min": g.low, "max": g.high, "tipo": "bool" if g.is_bool else "float",
         "significato": g.description}
        for g in GENOME
    ]


def _eval_view(ev: Evaluation) -> dict[str, object]:
    return {**ev.summary(), "periodi": [asdict(s) for s in ev.segments]}


def _diff(p: RuleParams, base: RuleParams) -> dict[str, object]:
    a, b = params_to_dict(p), params_to_dict(base)
    return {k: v for k, v in a.items() if v != b[k]}


class ClaudeStrategyProposer:
    def __init__(self, settings: ClaudeSettings | None = None, client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings or ClaudeSettings()
        self.client = client or anthropic.Anthropic(timeout=self.settings.timeout_s)

    def build_prompt(
        self, champion: RuleParams, champion_eval: Evaluation,
        leaderboard: list[tuple[RuleParams, Evaluation]], n: int,
    ) -> str:
        payload = {
            "parametri": _genome_table(),
            "campione": {"params": params_to_dict(champion), "risultati": _eval_view(champion_eval)},
            "migliori_testati": [
                {"modifiche_rispetto_al_campione": _diff(p, champion), "risultati": ev.summary()}
                for p, ev in leaderboard
            ],
        }
        return (f"Proponi {n} varianti della strategia. Dati (JSON):\n"
                f"{json.dumps(payload, ensure_ascii=False, default=str)}")

    def __call__(
        self, champion: RuleParams, champion_eval: Evaluation,
        leaderboard: list[tuple[RuleParams, Evaluation]], n: int,
    ) -> list[tuple[RuleParams, str]]:
        try:
            response = self.client.beta.messages.parse(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.settings.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self.build_prompt(champion, champion_eval, leaderboard, n)}],
                output_format=ProposalSet,
            )
        except anthropic.APIError as exc:
            log.warning("Claude non disponibile per le proposte: %s", exc)
            return []
        if response.stop_reason == "refusal" or response.parsed_output is None:
            log.warning("Nessuna proposta utilizzabile da Claude (%s)", response.stop_reason)
            return []
        result = response.parsed_output
        log.info("Diagnosi di Claude: %s", result.diagnosis)
        return [
            (params_from_dict(prop.params.model_dump()), prop.hypothesis)
            for prop in result.proposals[:n]
        ]
