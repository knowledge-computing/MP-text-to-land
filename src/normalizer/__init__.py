"""Local, deterministic PLSS numeral normalizer (thesis Plan B, replaces the GPT-4o step)."""

from .types import NER_FIELDS, SLOTS, TRS, Normalizer, Result, Span, spans_from_fields

__all__ = ["NER_FIELDS", "SLOTS", "TRS", "Normalizer", "Result", "Span", "spans_from_fields"]
