"""Ragionatore deterministico: valuta il mercato con argomenti espliciti pro/contro e un punteggio.

Punteggio positivo = rialzista, negativo = ribassista. Si opera solo quando |punteggio| supera la soglia.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Decision, MarketSnapshot
from .reasoning import ReasoningContext


@dataclass(frozen=True)
class RuleParams:
    min_score: float = 3.5
    trend_stop_atr: float = 1.5
    trend_rr: float = 2.0
    range_stop_atr: float = 1.2
    range_rr: float = 1.5
    pullback_atr: float = 0.6
    overextended_atr: float = 2.0


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

        if s.regime in ("trending", "transitional") and bias != 0:
            side = "rialzista" if bias > 0 else "ribassista"
            pro.append(f"Trend H4 {side} (prezzo/EMA20/EMA50 allineati, pendenza EMA50 {htf.ema50_slope_atr:+.2f} ATR)")
            score += 1.5 * bias

            if ltf.trend == htf.trend:
                score += 1.5 * bias
                pro.append("H1 allineato con H4 (sopra/sotto EMA20>EMA50>EMA200)")
            elif ltf.trend == "flat":
                con.append("H1 senza struttura di trend chiara")
            else:
                score -= 1.5 * bias
                con.append("H1 in controtendenza rispetto a H4")

            if abs(dist_ema20) <= p.pullback_atr:
                score += 1.0 * bias
                pro.append(f"Prezzo in pullback sulla EMA20 H1 ({dist_ema20:+.2f} ATR): ingresso a buon prezzo")
            elif dist_ema20 * bias > p.overextended_atr:
                score -= 1.0 * bias
                con.append(f"Prezzo esteso dalla EMA20 ({dist_ema20:+.2f} ATR): rischio di inseguire il movimento")

            if ltf.macd_hist * bias > 0 and (ltf.macd_hist - ltf.macd_hist_prev) * bias > 0:
                score += 1.0 * bias
                pro.append("Momentum MACD a favore e in accelerazione")
            elif ltf.macd_hist * bias < 0:
                con.append("Momentum MACD contrario al trend")

            di_aligned = (ltf.plus_di > ltf.minus_di) if bias > 0 else (ltf.minus_di > ltf.plus_di)
            if ltf.adx >= 25 and di_aligned:
                score += 0.5 * bias
                pro.append(f"ADX {ltf.adx:.0f} con DI direzionali coerenti: trend forte")

            rsi_extreme = ltf.rsi > 72 if bias > 0 else ltf.rsi < 28
            if rsi_extreme:
                score -= 1.0 * bias
                con.append(f"RSI {ltf.rsi:.0f} in eccesso: possibile esaurimento")
            elif (45 <= ltf.rsi <= 65) if bias > 0 else (35 <= ltf.rsi <= 55):
                score += 0.5 * bias
                pro.append(f"RSI {ltf.rsi:.0f} in zona sana per continuazione")

        elif s.regime == "ranging":
            mode = "range"
            pro.append(f"Mercato laterale (ADX {ltf.adx:.0f}): strategia di ritorno alla media")
            if ltf.close <= ltf.bb_lower and ltf.rsi < 32:
                score += 3.0
                pro.append(f"Prezzo sotto la banda di Bollinger inferiore con RSI {ltf.rsi:.0f}: ipervenduto")
            elif ltf.close >= ltf.bb_upper and ltf.rsi > 68:
                score -= 3.0
                pro.append(f"Prezzo sopra la banda di Bollinger superiore con RSI {ltf.rsi:.0f}: ipercomprato")
            if score > 0 and bias < 0 or score < 0 and bias > 0:
                score -= 1.5 if score > 0 else -1.5
                con.append("Il ritorno alla media andrebbe contro il trend H4")
            if s.supports and score > 0 and abs(ltf.close - s.supports[0]) < ltf.atr:
                score += 0.5
                pro.append(f"Vicino al supporto {s.supports[0]:.5f}")
            if s.resistances and score < 0 and abs(ltf.close - s.resistances[0]) < ltf.atr:
                score -= 0.5
                pro.append(f"Vicino alla resistenza {s.resistances[0]:.5f}")
        else:
            con.append("Nessun trend H4 definito e regime non laterale: contesto ambiguo")

        magnitude = abs(score)
        confidence = round(min(1.0, magnitude / 6.0), 3)
        stop = p.trend_stop_atr if mode == "trend" else p.range_stop_atr
        rr = p.trend_rr if mode == "trend" else p.range_rr

        if magnitude < p.min_score:
            return Decision(
                action="HOLD", confidence=confidence, stop_atr_mult=stop, take_profit_rr=rr,
                thesis=f"Segnale insufficiente (punteggio {score:+.1f}, soglia {p.min_score}). Meglio attendere.",
                supporting_factors=pro, risks=con, source="rules", score=score, regime=s.regime,
            )

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
            invalidation=f"Chiusura oltre {stop} ATR contro la posizione (stop loss).",
            source="rules",
            score=score,
            regime=s.regime,
        )
