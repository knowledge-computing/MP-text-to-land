"""Top-1 plat matching.

A plat's identity is (county, n1(primary name)); several polygons can share one identity. Candidate pools
hold unique name keys (raw / n1 / n3, optionally including alias names) mapped to plat indices.

Selection for one deed (deterministic): for every predicted string (in NER order) and every candidate
with base score >= cutoff, final = base + k if the candidate's plat lies in a section of the deed's PLSS
readings (section="soft"); the best (final, base) wins, ties go to the earlier predicted string, then to the
candidate key that appears first in the pool. section="hard" keeps only in-section candidates when the deed
has readings and at least one candidate survives (else no filter).
"""
from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from .names import n1

SCORERS = {"ratio": fuzz.ratio, "WRatio": fuzz.WRatio, "token_set_ratio": fuzz.token_set_ratio}


@dataclass(frozen=True)
class MatchConfig:
    scorer: str = "ratio"
    norm: str = "n3"            # "raw" | "n1" | "n3"
    own_county: bool = True
    aliases: bool = False
    section: str = "none"       # "none" | "hard" | "soft"
    k: float = 0.0
    cutoff: float = 70.0

    def name(self):
        s = f"{self.scorer}|{self.norm}|{'county' if self.own_county else 'all'}|{'alias' if self.aliases else 'primary'}"
        return s + ("" if self.section == "none" else f"|{self.section}" + (f"{self.k:g}" if self.section == "soft" else ""))


@dataclass
class Match:
    ident: tuple
    plats: list          # all plat indices with this identity
    score: float
    final: float


class PlatIndex:
    def __init__(self, plats, norm3):
        """plats: list of dicts {county, name, aliases (list), geometry, sections (set of (S, T, R))}."""
        self.plats, self.norm3 = plats, norm3
        self.ident = [(p["county"], n1(p["name"])) for p in plats]
        self.by_ident = defaultdict(list)
        for i, idn in enumerate(self.ident):
            self.by_ident[idn].append(i)
        self._pools = {}

    def _form(self, s, norm):
        return s if norm == "raw" else n1(s) if norm == "n1" else self.norm3(s)

    def pool(self, county, cfg):
        key = (county if cfg.own_county else None, cfg.norm, cfg.aliases)
        if key not in self._pools:
            keys, members = [], defaultdict(list)
            for i, p in enumerate(self.plats):
                if cfg.own_county and p["county"] != county:
                    continue
                names = [p["name"]] + (list(p.get("aliases") or []) if cfg.aliases else [])
                for nm in names:
                    if not isinstance(nm, str) or len(nm.strip()) <= 1:
                        continue
                    k = self._form(nm, cfg.norm)
                    if k not in members:
                        keys.append(k)
                    if i not in members[k]:
                        members[k].append(i)
            self._pools[key] = (keys, dict(members))
        return self._pools[key]

    def in_section(self, plat_ids, triples):
        return bool(triples) and any(self.plats[i]["sections"] & triples for i in plat_ids)

    def top1(self, preds, county, cfg, triples=frozenset()):
        keys, members = self.pool(county, cfg)
        allowed = None
        if cfg.section == "hard" and triples:
            allowed = {j for j, k in enumerate(keys) if self.in_section(members[k], triples)} or None
        fn = SCORERS[cfg.scorer]
        best = None  # (final, base, -pred_order, -key_index, key)
        for order, p in enumerate(preds):
            q = self._form(p, cfg.norm)
            for k, sc, j in process.extract(q, keys, scorer=fn, score_cutoff=cfg.cutoff, limit=None, processor=None):
                if allowed is not None and j not in allowed:
                    continue
                bonus = cfg.k if cfg.section == "soft" and self.in_section(members[k], triples) else 0.0
                cand = (sc + bonus, sc, -order, -j, k)
                if best is None or cand[:4] > best[:4]:
                    best = cand
        if best is None:
            return None
        plat_ids = members[best[4]]
        idn = self.ident[min(plat_ids)]
        return Match(idn, self.by_ident[idn], best[1], best[0])

    def any_candidate(self, preds, county, cfg):
        """The paper's subdivision-level criterion: >= 1 plat name scores >= cutoff for any predicted string."""
        keys, _ = self.pool(county, cfg)
        fn = SCORERS[cfg.scorer]
        return any(process.extractOne(self._form(p, cfg.norm), keys, scorer=fn, score_cutoff=cfg.cutoff,
                                      processor=None) for p in preds)
