"""Per-county subdivision gazetteer.

An entry is one (county, normalized key); every source name whose key is identical collapses into it.
A surface is a string the matchers compare against: the entry key itself and every alias key.

  kind "name"  the key of a plat / addition name (plat layer primary or alias field, MnGeo plat_name,
               covenant add_mod). Gold names resolve through "name" surfaces only.
  kind "cov"   the key of a covenant's add_cov (as written in the deed) that was mapped to the entry.

Each surface keeps its contributors: "layer" for plat layers, else deed ids (county, doc_num) of the
covenant rows that produced it. Matching on a train deed can exclude surfaces whose only contributors are
that deed (leave-one-out), so the reranker does not learn from its own alias.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field

LAYER = ("layer", "")


@dataclass
class Entry:
    county: str
    key: str
    sources: set = field(default_factory=set)
    n_names: int = 0
    n_parcels: int = 0
    n_cov_rows: int = 0
    blocks: set = field(default_factory=set)
    lots: set = field(default_factory=set)
    pairs: set = field(default_factory=set)       # (block, lot)


@dataclass
class Surface:
    county: str
    text: str
    kind: str                                     # "name" | "cov"
    entries: set = field(default_factory=set)     # entry ids
    contributors: set = field(default_factory=set)
    sources: set = field(default_factory=set)


class Gazetteer:
    def __init__(self, normalizer):
        self.norm = normalizer
        self.entries, self._eid = [], {}
        self.surfaces, self._sid = [], {}

    # ---------------------------------------------------------------- build
    def entry(self, county, key):
        k = (county, key)
        if k not in self._eid:
            self._eid[k] = len(self.entries)
            self.entries.append(Entry(county, key))
        return self._eid[k]

    def surface(self, county, text, kind):
        k = (county, text, kind)
        if k not in self._sid:
            self._sid[k] = len(self.surfaces)
            self.surfaces.append(Surface(county, text, kind))
        return self.surfaces[self._sid[k]]

    def add_name(self, county, name, source, aliases=(), contributor=LAYER, n_parcels=0, blocks=(), lots=(),
                 pairs=()):
        key = self.norm(name)
        if not key:
            return None
        e = self.entry(county, key)
        ent = self.entries[e]
        ent.sources.add(source)
        ent.n_names += 1
        ent.n_parcels += n_parcels
        ent.blocks.update(b for b in blocks if b)
        ent.lots.update(x for x in lots if x)
        ent.pairs.update(p for p in pairs if p[0] and p[1])
        for text in [key] + [self.norm(a) for a in aliases]:
            if text:
                s = self.surface(county, text, "name")
                s.entries.add(e)
                s.contributors.add(contributor)
                s.sources.add(source)
        return e

    def add_cov_alias(self, county, add_cov, add_mod, source, contributor, block=None, lots=()):
        """A covenant row: add_mod becomes / joins an entry, add_cov becomes a "cov" surface of it."""
        e = self.add_name(county, add_mod, source, contributor=contributor,
                          blocks=[block], lots=lots, pairs=[(block, x) for x in lots])
        if e is None:
            return None
        self.entries[e].n_cov_rows += 1
        cov = self.norm(add_cov)
        if cov:
            s = self.surface(county, cov, "cov")
            s.entries.add(e)
            s.contributors.add(contributor)
            s.sources.add(source)
        return e

    # ---------------------------------------------------------------- lookup
    def county_surfaces(self, county):
        return [i for i, s in enumerate(self.surfaces) if s.county == county]

    def excluded(self, exclude):
        """Surface ids whose contributors are all in `exclude`."""
        if not hasattr(self, "_by_contrib"):
            self._by_contrib = defaultdict(set)
            for i, s in enumerate(self.surfaces):
                for c in s.contributors:
                    self._by_contrib[c].add(i)
        cand = set().union(*(self._by_contrib.get(c, set()) for c in exclude)) if exclude else set()
        return {i for i in cand if not self.usable(i, exclude)}

    def usable(self, sid, exclude=frozenset()):
        """False if every contributor of the surface is in `exclude` (leave-one-out)."""
        return not exclude or bool(self.surfaces[sid].contributors - set(exclude))

    def gold_entries(self, county, names, exclude=frozenset()):
        """Entries reachable from gold names through "name" surfaces of the county."""
        out = set()
        for n in names:
            sid = self._sid.get((county, self.norm(n), "name"))
            if sid is not None and self.usable(sid, exclude):
                out |= self.surfaces[sid].entries
        return out

    # ---------------------------------------------------------------- io
    def to_frames(self):
        import pandas as pd
        e = pd.DataFrame([{"entry_id": i, "county": x.county, "key": x.key, "sources": json.dumps(sorted(x.sources)),
                           "n_names": x.n_names, "n_parcels": x.n_parcels, "n_cov_rows": x.n_cov_rows,
                           "blocks": json.dumps(sorted(x.blocks)), "lots": json.dumps(sorted(x.lots)),
                           "pairs": json.dumps(sorted(x.pairs))} for i, x in enumerate(self.entries)])
        s = pd.DataFrame([{"surface_id": i, "county": x.county, "text": x.text, "kind": x.kind,
                           "entries": json.dumps(sorted(x.entries)), "sources": json.dumps(sorted(x.sources)),
                           "contributors": json.dumps(sorted(x.contributors))} for i, x in enumerate(self.surfaces)])
        return e, s

    @classmethod
    def from_frames(cls, e, s, normalizer):
        g = cls(normalizer)
        for r in e.to_dict("records"):
            ent = Entry(r["county"], r["key"], set(json.loads(r["sources"])), r["n_names"], r["n_parcels"],
                        r["n_cov_rows"], set(json.loads(r["blocks"])), set(json.loads(r["lots"])),
                        {tuple(p) for p in json.loads(r["pairs"])})
            g._eid[(ent.county, ent.key)] = len(g.entries)
            g.entries.append(ent)
        for r in s.to_dict("records"):
            sur = Surface(r["county"], r["text"], r["kind"], set(json.loads(r["entries"])),
                          {tuple(c) for c in json.loads(r["contributors"])}, set(json.loads(r["sources"])))
            g._sid[(sur.county, sur.text, sur.kind)] = len(g.surfaces)
            g.surfaces.append(sur)
        return g

    def summary(self):
        by = defaultdict(lambda: defaultdict(int))
        for x in self.entries:
            c = by[x.county]
            c["entries"] += 1
            for src in x.sources:
                c[f"entries_{src}"] += 1
            if {"partner", "mngeo"} <= x.sources:
                c["entries_partner_and_mngeo"] += 1
            if {"partner", "mclio"} <= x.sources:
                c["entries_partner_and_mclio"] += 1
            if x.pairs:
                c["entries_with_block_lot"] += 1
        for x in self.surfaces:
            by[x.county][f"surfaces_{x.kind}"] += 1
        return {k: dict(v) for k, v in by.items()}
