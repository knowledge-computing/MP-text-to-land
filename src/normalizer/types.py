"""
Shared types for PLSS (Township / Range / Section) numeral normalizers.

Every normalizer implements `Normalizer.normalize(spans, county, state) -> Result`, so the
evaluation harness can compare the rule-based normalizer, regex v1, the saved GPT-4o outputs and
later pyTRS / a local LLM / ByT5 through one interface.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

SLOTS = ("S", "T", "R")
# NER prediction columns in the try08 GeoJSONs -> slot label
NER_FIELDS = {"SECTION_p": "S", "TOWNSHIP_p": "T", "RANGE_p": "R"}


@dataclass(frozen=True)
class Span:
    """One NER span. start/end are character offsets into the source sentence when known."""
    label: str  # "S", "T" or "R"
    text: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class TRS:
    """A PLSS location. S is None for a township-only (6x6) reading; directions are "" when unknown."""
    T: int
    Tdir: str
    R: int
    Rdir: str
    S: Optional[int] = None

    def paper_triple(self):
        """The (sec, twp, rng, dir) tuple order used by TRIPLE_p and dev/recompute_table3.py."""
        return (self.S, self.T, self.R, self.Rdir)

    def township(self):
        return TRS(self.T, self.Tdir, self.R, self.Rdir, None)


@dataclass
class Result:
    """Ranked readings, best first; `reason` says why `ranked` is empty."""
    ranked: List[Tuple[TRS, float]] = field(default_factory=list)
    reason: Optional[str] = None
    info: dict = field(default_factory=dict)  # diagnostics for the error taxonomy (counts, all valid readings)

    @property
    def top(self) -> Optional[TRS]:
        return self.ranked[0][0] if self.ranked else None

    def top_k(self, k: Optional[int] = None) -> List[TRS]:
        return [t for t, _ in (self.ranked if k is None else self.ranked[:k])]


class Normalizer(Protocol):
    name: str

    def normalize(self, spans: Sequence[Span], county: Optional[str], state: Optional[str]) -> Result:
        ...


def spans_from_fields(props) -> List[Span]:
    """Build list-order spans (no offsets) from SECTION_p / TOWNSHIP_p / RANGE_p values (lists or strings)."""
    spans = []
    for col, label in NER_FIELDS.items():
        value = props.get(col)
        if hasattr(value, "tolist") and not isinstance(value, str):
            value = value.tolist()
        if isinstance(value, str):
            import ast
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = [value]
        if isinstance(value, (list, tuple)):
            spans.extend(Span(label, v) for v in value if isinstance(v, str) and v.strip())
    return spans
