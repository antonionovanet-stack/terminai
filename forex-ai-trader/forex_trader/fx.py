"""Convenzioni sugli strumenti Forex (formato OANDA: BASE_QUOTE, es. EUR_USD)."""

from __future__ import annotations


def split_instrument(instrument: str) -> tuple[str, str]:
    base, sep, quote = instrument.partition("_")
    if not sep or len(base) != 3 or len(quote) != 3:
        raise ValueError(f"Strumento non valido: {instrument!r} (atteso es. 'EUR_USD')")
    return base.upper(), quote.upper()


def pip_size(instrument: str) -> float:
    _, quote = split_instrument(instrument)
    return 0.01 if quote == "JPY" else 0.0001


def price_precision(instrument: str) -> int:
    _, quote = split_instrument(instrument)
    return 3 if quote == "JPY" else 5


def round_price(instrument: str, price: float) -> float:
    return round(price, price_precision(instrument))


def to_pips(instrument: str, price_distance: float) -> float:
    return price_distance / pip_size(instrument)
