"""Ragionatore deterministico: valuta il mercato con argomenti espliciti pro/contro e un punteggio.

Punteggio positivo = rialzista, negativo = ribassista. Si opera solo quando |punteggio| supera la soglia.
Pesi e soglie stanno tutti in RuleParams: sono il "DNA" che il modulo evolution.py fa evolvere.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Decision, MarketSnapshot
from .reasoning import ReasoningContext


@dataclass(frozen=True)
class RuleParams:
    min_score: float = 3.5
    trend_enabled: bool = True
    range_enabled: bool = True
    # Uscite
    trend_stop_atr: float = 1.5
    trend_rr: float = 2.0
    range_stop_atr: float = 1.2
    range_rr: float = 1.5
    # Ingresso in trend
    w_htf_trend: float = 1.5
    w_ltf_aligned: float = 1.5
    w_ltf_counter: float = 1.5
    pullback_atr: float = 0.6
    w_pullback: float = 1.0
    overextended_atr: float = 2.0
    w_overextended: float = 1.0
    w_momentum: float = 1.0
    adx_strong: float = 25.0
    w_adx: float = 0.5
    rsi_exhaustion: float = 72.0
    w_rsi_exhaustion: float = 1.0
    rsi_healthy_low: float = 45.0
    rsi_healthy_high: float = 65.0
    w_rsi_healthy: float = 0.5
    # Ingresso in range
    range_rsi_extreme: float = 32.0
    w_range_signal: float = 3.0
    w_range_against_htf: float = 1.5
    w_level: float = 0.5


class RuleBasedReasoner:
    def __init__(self, params: RuleParams | None = None) -> None:
        self.p = params or RuleParams()

    def decide(self, snapshot: MarketSnapshot, context: ReasoningContext | None = None) -> Decision:
        s, p = snapshot, self.p
        ltf, htf = s.ltf, s.htf
        pro: list[str] = []
        con: list[str] = []
        score = 0.0

        if s.regime == "volatile":
            return Decision(
                action="HOLD", confidence=0.0, stop_atr_mult=p.trend_stop_atr, take_profit_rr=p.trend_rr,
                thesis="Volatilità estrema (ATR oltre il 97° percentile): probabile notizia/shock, resto fuori.",
                risks=["Spread e slippage elevati", "Movimenti erratici che invalidano i segnali tecnici"],
                source="rules", score=0.0, regime=s.regime,
            )

        bias = {"up": 1, "down": -1, "flat": 0}[htf.trend]
        dist_ema20 = (ltf.close - ltf.ema20) / ltf.atr
        mode = "trend"

        if s.regime in ("trending", "transitional") and bias != 0 and p.trend_enabled:
            side = "rialzista" if bias > 0 else "ribassista"
            pro.append(f"Trend H4 {side} (prezzo/EMA20/EMA50 allineati, pendenza EMA50 {htf.ema50_slope_atr:+.2f} ATR)")
            score += p.w_htf_trend * bias

            if ltf.trend == htf.trend:
                score += p.w_ltf_aligned * bias
                pro.append("H1 allineato con H4 (sopra/sotto EMA20>EMA50>EMA200)")
            elif ltf.trend == "flat":
                con.append("H1 senza struttura di trend chiara")
            else:
                score -= p.w_ltf_counter * bias
                con.append("H1 in controtendenza rispetto a H4")

            if abs(dist_ema20) <= p.pullback_atr:
                score += p.w_pullback * bias
                pro.append(f"Prezzo in pullback sulla EMA20 H1 ({dist_ema20:+.2f} ATR): ingresso a buon prezzo")
            elif dist_ema20 * bias > p.overextended_atr:
                score -= p.w_overextended * bias
                con.append(f"Prezzo esteso dalla EMA20 ({dist_ema20:+.2f} ATR): rischio di inseguire il movimento")

            if ltf.macd_hist * bias > 0 and (ltf.macd_hist - ltf.macd_hist_prev) * bias > 0:
                score += p.w_momentum * bias
                pro.append("Momentum MACD a favore e in accelerazione")
            elif ltf.macd_hist * bias < 0:
                con.append("Momentum MACD contrario al trend")

            di_aligned = (ltf.plus_di > ltf.minus_di) if bias > 0 else (ltf.minus_di > ltf.plus_di)
            if ltf.adx >= p.adx_strong and di_aligned:
                score += p.w_adx * bias
                pro.append(f"ADX {ltf.adx:.0f} con DI direzionali coerenti: trend forte")

            rsi_extreme = ltf.rsi > p.rsi_exhaustion if bias > 0 else ltf.rsi < 100 - p.rsi_exhaustion
            healthy = (
                p.rsi_healthy_low <= ltf.rsi <= p.rsi_healthy_high if bias > 0
                else 100 - p.rsi_healthy_high <= ltf.rsi <= 100 - p.rsi_healthy_low
            )
            if rsi_extreme:
                score -= p.w_rsi_exhaustion * bias
                con.append(f"RSI {ltf.rsi:.0f} in eccesso: possibile esaurimento")
            elif healthy:
                score += p.w_rsi_healthy * bias
                pro.append(f"RSI {ltf.rsi:.0f} in zona sana per continuazione")

        elif s.regime == "ranging" and p.range_enabled:
            mode = "range"
            pro.append(f"Mercato laterale (ADX {ltf.adx:.0f}): strategia di ritorno alla media")
            if ltf.close <= ltf.bb_lower and ltf.rsi < p.range_rsi_extreme:
                score += p.w_range_signal
                pro.append(f"Prezzo sotto la banda di Bollinger inferiore con RSI {ltf.rsi:.0f}: ipervenduto")
            elif ltf.close >= ltf.bb_upper and ltf.rsi > 100 - p.range_rsi_extreme:
                score -= p.w_range_signal
                pro.append(f"Prezzo sopra la banda di Bollinger superiore con RSI {ltf.rsi:.0f}: ipercomprato")
            if score > 0 and bias < 0 or score < 0 and bias > 0:
                score -= p.w_range_against_htf if score > 0 else -p.w_range_against_htf
                con.append("Il ritorno alla media andrebbe contro il trend H4")
            if s.supports and score > 0 and abs(ltf.close - s.supports[0]) < ltf.atr:
                score += p.w_level
                pro.append(f"Vicino al supporto {s.supports[0]:.5f}")
            if s.resistances and score < 0 and abs(ltf.close - s.resistances[0]) < ltf.atr:
                score -= p.w_level
                pro.append(f"Vicino alla resistenza {s.resistances[0]:.5f}")
        else:
            con.append("Nessun setup attivo per questo regime: contesto ambiguo")

        magnitude = abs(score)
        stop = p.trend_stop_atr if mode == "trend" else p.range_stop_atr
        rr = p.trend_rr if mode == "trend" else p.range_rr

        if magnitude < p.min_score:
            return Decision(
                action="HOLD", confidence=round(min(1.0, magnitude / 6.0), 3), stop_atr_mult=stop,
                take_profit_rr=rr,
                thesis=f"Segnale insufficiente (punteggio {score:+.1f}, soglia {p.min_score:.1f}). Meglio attendere.",
                supporting_factors=pro, risks=con, source="rules", score=score, regime=s.regime,
            )

        # Confidenza relativa alla soglia: il filtro di rischio sulla confidenza minima resta coerente
        # anche quando l'evoluzione sposta la soglia.
        excess = (magnitude - p.min_score) / max(p.min_score, 1.0)
        confidence = round(min(1.0, 0.58 + 0.42 * excess), 3)
        action = "BUY" if score > 0 else "SELL"
        verb = "Acquisto" if action == "BUY" else "Vendita"
        if mode == "range":
            setup = "ritorno alla media in range"
        elif abs(dist_ema20) <= p.pullback_atr:
            setup = "pullback nel trend"
        else:
            setup = "continuazione del trend"
        return Decision(
            action=action,
            confidence=confidence,
            stop_atr_mult=stop,
            take_profit_rr=rr,
            thesis=f"{verb} {s.instrument}: {setup} (punteggio {score:+.1f}).",
            supporting_factors=pro,
            risks=con,
            invalidation=f"Chiusura oltre {stop:.2f} ATR contro la posizione (stop loss).",
            source="rules",
            score=score,
            regime=s.regime,
        )
