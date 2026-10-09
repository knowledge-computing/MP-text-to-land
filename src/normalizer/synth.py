"""
Seeded synthetic Township / Range / Section phrases: the unit-test corpus for the normalizer.

Templates use generic legal-description phrasing (label words, number styles, directions, aliquot
prefixes, meridian suffixes). They are written by hand, not sampled from deed text. Each example
carries character-offset spans (as an NER model with offsets would give) and the gold TRS list.

Noise is applied after the base phrase is built, from separate per-example random streams, so the
clean and noisy corpora built with the same seed contain the same underlying phrases:
  * OCR noise (src/normalizer/ocr.py) per character, applied to every text segment;
  * span noise: trim or extend a span by a word, drop a span, add a spurious span from context.

    python -m src.normalizer.synth --toy --n 5               # print 5 toy examples
    python -m src.normalizer.synth --valid-json PATH --n 2000 --ocr-p 0.05 --out FILE.jsonl
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

from .numbers import int_to_words
from .ocr import ocr_noise
from .types import TRS, Span

LABEL_WORDS = {
    "S": ("Section", "Section", "Section", "Sec.", "Sec", "SECTION", "section", "Sect."),
    "T": ("Township", "Township", "Township", "Twp.", "Twp", "TOWNSHIP", "township", "T."),
    "R": ("Range", "Range", "Range", "Rge.", "Rng.", "RANGE", "range", "R."),
}
DIR_WORDS = {
    "N": ("North", "North", "N.", "N", "NORTH", "north"),
    "S": ("South", "South", "S.", "S", "SOUTH", "south"),
    "E": ("East", "East", "E.", "E", "EAST", "east"),
    "W": ("West", "West", "W.", "W", "WEST", "west"),
}
NUMBER_STYLES = (("digits", 0.4), ("words", 0.2), ("words_digits", 0.3), ("numbered", 0.1))
LAYOUTS = (("STR", 0.55), ("TRS", 0.08), ("TR", 0.1), ("multi_section", 0.1),
           ("multi_triple", 0.05), ("compact", 0.12))
JOINERS = (", ", ", ", ", ", " ", " in ", " of ", ",\n", "\n")
PREFIXES = (
    "", "", "",
    "All that part of the Northeast Quarter (NE 1/4) of the Southwest Quarter (SW 1/4) of ",
    "The South 40 acres of the West Half (W 1/2) of ",
    "Government Lot Three (3) of ",
    "That part of Lot 4, Block 2, lying in ",
)
SUFFIXES = (
    "", "", "",
    " of the {meridian} Principal Meridian",
    ", {side} of the {meridian} Principal Meridian",
    ", {county} County, {state}",
    ", according to the Government Survey thereof",
)
STATE_NAMES = {"MN": "Minnesota", "WI": "Wisconsin"}
MERIDIANS = {"MN": ("Fifth", "West"), "WI": ("Fourth", "East")}  # study-area meridian, range side
SPAN_OPS = ("trim_end", "extend_end", "extend_start", "drop")


@dataclass
class SynthExample:
    id: int
    text: str
    spans: List[Span]
    gold: List[TRS]
    county: str
    state: str
    layout: str
    styles: Dict[str, str] = field(default_factory=dict)
    noise: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id, "text": self.text, "county": self.county, "state": self.state,
            "layout": self.layout, "styles": self.styles, "noise": self.noise,
            "spans": [[s.label, s.text, s.start, s.end] for s in self.spans],
            "gold": [[g.T, g.Tdir, g.R, g.Rdir, g.S] for g in self.gold],
        }


def toy_valid() -> Dict[Tuple[str, str], List[TRS]]:
    """Fake county tables for tests and demos (made-up numbers, not real county extents). Like the real
    study counties, the MN county mixes two meridians' township numbering with shared range numbers
    (T27-28 and T114-115, both R22-23 W), and both counties have partial townships."""
    mn = [TRS(t, "N", r, "W", s) for t in (27, 28, 114, 115) for r in (22, 23) for s in range(1, 37)
          if not (t == 28 and r == 23 and s > 30)]
    wi = [TRS(t, "N", r, "E", s) for t in (7, 8) for r in (21, 22) for s in range(1, 37)
          if not (t == 8 and r == 22 and s <= 12)]
    return {("MN", "toyka"): mn, ("WI", "toywaukee"): wi}


def _pick(rng, weighted):
    names, weights = zip(*weighted)
    return rng.choices(names, weights)[0]


def _case(rng, words):
    return rng.choice((words, words.lower(), words.title(), words.upper(), words[:1].upper() + words[1:]))


def render_number(n, rng, style=None):
    style = style or _pick(rng, NUMBER_STYLES)
    words = _case(rng, int_to_words(n, joiner=rng.choice(("-", "-", " ", "")), hundred_and=rng.random() < 0.3))
    if style == "digits":
        return str(n), style
    if style == "words":
        return words, style
    if style == "words_digits":
        return f"{words}{rng.choice((' ', ' ', ''))}({n})", style
    return rng.choice((f"numbered {n}", f"Number {words} ({n})", f"No. {n}")), style


def render_slot(label, n, direction, rng, style=None):
    """One slot phrase, e.g. 'Township Thirty-one (31) North'. Direction is sometimes left out."""
    num, style = render_number(n, rng, style)
    text = f"{rng.choice(LABEL_WORDS[label])} {num}"
    if label in ("T", "R") and direction and rng.random() > 0.15:
        text += " " + rng.choice(DIR_WORDS[direction])
    return text, style


def render_compact(trs, rng):
    """Compact forms; returns segments [(text, label|None)]."""
    form = rng.choice(("STR_letters", "STR_dotted", "dashed"))
    if form == "STR_letters":
        return [(f"S{trs.S}", "S"), (" ", None), (f"T{trs.T}{trs.Tdir}", "T"), (" ", None), (f"R{trs.R}{trs.Rdir}", "R")]
    if form == "STR_dotted":
        return [(f"Sec. {trs.S}", "S"), (", ", None), (f"T. {trs.T} {trs.Tdir}.", "T"), (", ", None),
                (f"R. {trs.R} {trs.Rdir}.", "R")]
    return [(str(trs.S), "S"), ("-", None), (str(trs.T), "T"), ("-", None), (str(trs.R), "R")]


def _slots(trs, rng, with_section=True, order="STR"):
    segs, styles = [], {}
    labels = [lab for lab in order if with_section or lab != "S"]
    for i, lab in enumerate(labels):
        n = {"S": trs.S, "T": trs.T, "R": trs.R}[lab]
        d = {"S": "", "T": trs.Tdir, "R": trs.Rdir}[lab]
        text, styles[lab] = render_slot(lab, n, d, rng)
        if i:
            segs.append((rng.choice(JOINERS), None))
        segs.append((text, lab))
    return segs, styles


def _build(rng, county_key, table):
    """Base phrase: (segments, gold, layout, styles)."""
    state, county = county_key
    layout = _pick(rng, LAYOUTS)
    trs = rng.choice(table)
    styles = {}
    if layout in ("STR", "TRS"):
        segs, styles = _slots(trs, rng, order=layout)
        gold = [trs]
    elif layout == "TR":
        segs, styles = _slots(trs, rng, with_section=False)
        gold = [trs.township()]
    elif layout == "multi_section":
        others = [t for t in table if t.township() == trs.township() and t.S != trs.S]
        if not others:
            return _build(rng, county_key, [trs])  # falls back to another layout draw
        trs2 = rng.choice(others)
        rest, styles = _slots(trs, rng, with_section=False)
        num_style = _pick(rng, NUMBER_STYLES)
        n1, _ = render_number(trs.S, rng, num_style)
        n2, _ = render_number(trs2.S, rng, num_style)
        if rng.random() < 0.5:  # one span "Sections 5 and 6"
            segs = [(f"Sections {n1} and {n2}", "S")]
        else:
            segs = [(f"Section {n1}", "S"), (" and ", None), (f"Section {n2}", "S")]
        segs += [(rng.choice(JOINERS), None)] + rest
        styles["S"] = num_style
        gold = [trs, trs2]
    elif layout == "multi_triple":
        trs2 = rng.choice(table)
        segs1, styles = _slots(trs, rng)
        segs2, _ = _slots(trs2, rng)
        segs = segs1 + [(rng.choice(("; and ", ", and ", " and also ")), None)] + segs2
        gold = [trs, trs2]
    else:  # compact
        segs = render_compact(trs, rng)
        styles = {"S": "compact", "T": "compact", "R": "compact"}
        gold = [trs]
    prefix = rng.choice(PREFIXES)
    meridian, side = MERIDIANS.get(state, ("Fifth", "West"))
    suffix = rng.choice(SUFFIXES).format(county=county.title(), state=STATE_NAMES.get(state, state),
                                         meridian=meridian, side=side)
    segs = ([(prefix, None)] if prefix else []) + segs + ([(suffix, None)] if suffix else [])
    return segs, gold, layout, styles


def _assemble(segs):
    text, spans, pos = [], [], 0
    for seg, label in segs:
        if label:
            spans.append(Span(label, seg, pos, pos + len(seg)))
        text.append(seg)
        pos += len(seg)
    return "".join(text), spans


def _span_noise(text, spans, rng, p):
    """Perturb spans at word boundaries; returns (spans, ops)."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out, ops = [], []
    for s in spans:
        start, end = s.start, s.end
        if rng.random() < p:
            op = rng.choice(SPAN_OPS)
            inside = [w for w in words if w[0] >= start and w[1] <= end]
            after = [w for w in words if w[0] >= end]
            before = [w for w in words if w[1] <= start]
            if op == "drop":
                ops.append(f"{s.label}:drop")
                continue
            if op == "trim_end" and len(inside) >= 2:
                end = inside[-2][1]
            elif op == "extend_end" and after:
                end = after[min(len(after), rng.choice((1, 2))) - 1][1]
            elif op == "extend_start" and before:
                start = before[-1][0]
            else:
                op = "none"
            if op != "none":
                ops.append(f"{s.label}:{op}")
        out.append(Span(s.label, text[start:end], start, end))
    if rng.random() < p:  # spurious span from words outside every span
        covered = [(s.start, s.end) for s in out]
        free = [w for w in words if not any(a < w[1] and w[0] < b for a, b in covered)]
        if free:
            i = rng.randrange(len(free))
            j = min(len(free), i + rng.choice((1, 2, 3)))
            start, end = free[i][0], free[j - 1][1]
            if text[start:end].count("\n") == 0:
                label = rng.choice(("S", "T", "R"))
                out.append(Span(label, text[start:end], start, end))
                ops.append(f"{label}:spurious")
    out.sort(key=lambda s: s.start)
    return out, ops


def generate(valid: Mapping[Tuple[str, str], Sequence[TRS]], n: int, seed: int = 0,
             ocr_p: float = 0.0, span_p: float = 0.0) -> List[SynthExample]:
    """n examples. The base phrase of example i depends only on (seed, i); noise uses its own stream."""
    keys = sorted(k for k, v in valid.items() if v)
    if not keys:
        raise ValueError("valid table is empty")
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}:{i}")
        county_key = rng.choice(keys)
        segs, gold, layout, styles = _build(rng, county_key, list(valid[county_key]))
        noise = []
        if ocr_p > 0:
            nrng = random.Random(f"{seed}:{i}:ocr")
            noisy = [(ocr_noise(t, nrng, ocr_p), lab) for t, lab in segs]
            if noisy != segs:
                noise.append("ocr")
            segs = noisy
        text, spans = _assemble(segs)
        if span_p > 0:
            spans, ops = _span_noise(text, spans, random.Random(f"{seed}:{i}:span"), span_p)
            noise += ops
        out.append(SynthExample(i, text, spans, gold, county_key[1], county_key[0], layout, styles, noise))
    return out


def corpora(valid, n, seed=0, ocr_p=0.05, span_p=0.2):
    """The four standard test sets; all share the same base phrases."""
    return {
        "clean": generate(valid, n, seed),
        "ocr": generate(valid, n, seed, ocr_p=ocr_p),
        "span": generate(valid, n, seed, span_p=span_p),
        "ocr+span": generate(valid, n, seed, ocr_p=ocr_p, span_p=span_p),
    }


def load_valid_json(path):
    """Read the valid-TRS table written by dev/build_plss_table.py (public PLSS data only)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {(state, county): [TRS(t, td, r, rd, s) for t, td, r, rd, s in rows]
            for state, counties in data["valid"].items() for county, rows in counties.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic T/R/S phrases (synthetic only).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--toy", action="store_true", help="use the built-in fake county tables")
    src.add_argument("--valid-json", help="valid-TRS table from dev/build_plss_table.py")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ocr-p", type=float, default=0.0)
    ap.add_argument("--span-p", type=float, default=0.0)
    ap.add_argument("--out", help="write JSONL here instead of printing")
    args = ap.parse_args(argv)
    valid = toy_valid() if args.toy else load_valid_json(args.valid_json)
    examples = generate(valid, args.n, args.seed, args.ocr_p, args.span_p)
    lines = [json.dumps(e.to_dict(), ensure_ascii=False) for e in examples]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} examples", file=sys.stderr)
    else:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
