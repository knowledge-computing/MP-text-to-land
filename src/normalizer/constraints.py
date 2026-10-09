"""
PLSS constraints: which (T, R, S, Rdir) readings exist, per county, per state, or not checked at all.

The valid sets come from dev/build_plss_table.py (public PLSS layers + Census county boundaries):
exact (T, R, S) sets per county, never T/R min-max, because Dakota, Washington and Sherburne mix
townships numbered from two meridians (e.g. T26-28 and T112-115). Sections outside 1-36 are excluded
when the table is built. Township direction is N everywhere in the study area, so Tdir is not checked.

Levels (the no-county ablation uses "state" and "none"):
  county  the deed's collection county (cnty_name)
  state   the whole state layer (valid_state), or the union of the state's study counties if absent
  none    no PLSS check; only Section 1-36, Township 1-200, Range 1-80
"""

import json
from typing import Iterable, Optional

S_RANGE, T_RANGE, R_RANGE = (1, 36), (1, 200), (1, 80)


def in_range(T=None, R=None, S=None):
    ok = lambda v, lo_hi: v is None or lo_hi[0] <= v <= lo_hi[1]
    return ok(T, T_RANGE) and ok(R, R_RANGE) and ok(S, S_RANGE)


class Scope:
    """The set of readings that count as existing for one deed."""

    def __init__(self, rows: Iterable = (), name="", accept_all=False):
        self.name, self.accept_all = name, accept_all
        self.sections, self.townships, self.rdirs = set(), set(), set()
        for t, _td, r, rd, s in rows:
            self.sections.add((int(t), int(r), int(s), rd))
            self.townships.add((int(t), int(r), rd))
            if rd:
                self.rdirs.add(rd)

    def directions(self):
        return sorted(self.rdirs)

    def default_rdir(self):
        return next(iter(self.rdirs)) if len(self.rdirs) == 1 else ""

    def section_ok(self, T, R, S, Rdir=""):
        if not in_range(T, R, S):
            return False
        if self.accept_all:
            return True
        if Rdir:
            return (T, R, S, Rdir) in self.sections or (T, R, S, "") in self.sections
        return any((T, R, S, d) in self.sections for d in self.rdirs | {""})

    def township_ok(self, T, R, Rdir=""):
        if not in_range(T, R):
            return False
        if self.accept_all:
            return True
        if Rdir:
            return (T, R, Rdir) in self.townships or (T, R, "") in self.townships
        return any((T, R, d) in self.townships for d in self.rdirs | {""})


class Constraints:
    def __init__(self, valid, valid_state=None):
        """valid: {state: {county: [[T, Tdir, R, Rdir, S], ...]}}; valid_state: {state: [...]} (optional)."""
        self.valid = valid
        self.valid_state = valid_state or {}
        self._cache = {}

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["valid"], data.get("valid_state"))

    @classmethod
    def from_trs(cls, table):
        """{(state, county): [TRS, ...]} (e.g. synth.toy_valid()) -> Constraints."""
        valid = {}
        for (st, c), rows in table.items():
            valid.setdefault(st, {})[c] = [[t.T, t.Tdir, t.R, t.Rdir, t.S] for t in rows if t.S is not None]
        return cls(valid)

    def scope(self, county: Optional[str], state: Optional[str], level="county") -> Scope:
        key = (level, state, county)
        if key in self._cache:
            return self._cache[key]
        if level == "none":
            sc = Scope(name="none", accept_all=True)
        elif level == "county" and state in self.valid and county in self.valid[state]:
            sc = Scope(self.valid[state][county], name=f"county:{state}:{county}")
        elif state in self.valid_state:
            sc = Scope(self.valid_state[state], name=f"state:{state}")
        elif state in self.valid:
            sc = Scope([row for rows in self.valid[state].values() for row in rows], name=f"state_union:{state}")
        else:
            sc = Scope(name="none", accept_all=True)
        self._cache[key] = sc
        return sc
