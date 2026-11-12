"""Baseline matchers: deed query strings -> ranked gazetteer entries.

A deed is a dict {county, queries (raw strings), blocks (set), lots (set), exclude (set of contributor ids
to leave out; train only)}. Every matcher returns a ranked list of entry ids (best first).

  exact      entries with a surface equal to a normalized query
  token_set  rapidfuzz token_set_ratio, max over queries x surfaces (ties: ratio, then entry id)
  tfidf      cosine of char_wb 3-5-gram TF-IDF vectors, max over queries x surfaces
  rerank     TF-IDF top-k from the own county and top-k from all counties, re-scored by a logistic
             regression over features (string similarities, plat-number agreement, block/lot exists in
             the plat, county match; optional source/popularity priors)

(a)-(c) search the deed's own county only, as the 2025 matcher and the MP processor do.
"""
import math
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz, process

from .normalize import plat_numbers

TEXT_FEATURES = ["tfidf", "tfidf_gap", "tset", "ratio", "partial", "tsort", "wratio", "exact_name", "exact_cov"]
NUMBER_FEATURES = ["num_match", "num_conflict", "num_query_only", "num_entry_only"]
BLOCKLOT_FEATURES = ["bl_info", "block_in", "lot_in", "pair_in"]
COUNTY_FEATURES = ["county_match"]
PRIOR_FEATURES = ["log_parcels", "log_cov_rows", "src_partner", "src_mngeo", "src_public", "src_train"]
FEATURE_SETS = {
    "rerank": TEXT_FEATURES + NUMBER_FEATURES + BLOCKLOT_FEATURES + COUNTY_FEATURES,
    "rerank_noBL": TEXT_FEATURES + NUMBER_FEATURES + COUNTY_FEATURES,
    "rerank_prior": TEXT_FEATURES + NUMBER_FEATURES + BLOCKLOT_FEATURES + COUNTY_FEATURES + PRIOR_FEATURES,
}


class MatchIndex:
    def __init__(self, gaz, analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.gaz = gaz
        self.texts = [s.text for s in gaz.surfaces]
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram_range), min_df=min_df,
                                   sublinear_tf=sublinear_tf, lowercase=False)
        self.X = self.vec.fit_transform(self.texts)
        self.by_county = defaultdict(list)
        for i, s in enumerate(gaz.surfaces):
            self.by_county[s.county].append(i)
        self.by_county = {c: np.array(v) for c, v in self.by_county.items()}
        self.all_ids = np.arange(len(self.texts))
        self.entry_surfaces = defaultdict(list)
        for i, s in enumerate(gaz.surfaces):
            for e in s.entries:
                self.entry_surfaces[e].append(i)

    def queries(self, deed):
        qs = []
        for q in deed.get("queries") or []:
            k = self.gaz.norm(q)
            if k and k not in qs:
                qs.append(k)
        return qs

    def _pool(self, county, exclude):
        ids = self.by_county.get(county, np.array([], int)) if county is not None else self.all_ids
        if exclude:
            drop = self.gaz.excluded(exclude)
            if drop:
                ids = np.array([i for i in ids if i not in drop], int)
        return ids

    def _entry_max(self, ids, scores):
        out = {}
        for sid, sc in zip(ids, scores):
            for e in self.gaz.surfaces[sid].entries:
                if e not in out or sc > out[e]:
                    out[e] = sc
        return out

    # ---------------------------------------------------------------- (a) exact
    def exact(self, deed, k=5):
        qs, ids = self.queries(deed), self._pool(deed["county"], deed.get("exclude"))
        hits = defaultdict(lambda: [0, 0, 0])
        qset = set(qs)
        for sid in ids:
            s = self.gaz.surfaces[sid]
            if s.text in qset:
                for e in s.entries:
                    h = hits[e]
                    h[0] += 1
                    h[1] = max(h[1], s.kind == "name")
                    ent = self.gaz.entries[e]
                    h[2] = ent.n_parcels + ent.n_cov_rows
        return [e for e, _ in sorted(hits.items(), key=lambda kv: (-kv[1][0], -kv[1][1], -kv[1][2], kv[0]))][:k]

    # ---------------------------------------------------------------- (b) token_set
    def token_set(self, deed, k=5):
        qs, ids = self.queries(deed), self._pool(deed["county"], deed.get("exclude"))
        if not qs or not len(ids):
            return []
        texts = [self.texts[i] for i in ids]
        ts = process.cdist(qs, texts, scorer=fuzz.token_set_ratio).max(axis=0)
        rt = process.cdist(qs, texts, scorer=fuzz.ratio).max(axis=0)
        best = {}
        for sid, a, b in zip(ids, ts, rt):
            for e in self.gaz.surfaces[sid].entries:
                if e not in best or (a, b) > best[e]:
                    best[e] = (a, b)
        return [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))][:k]

    # ---------------------------------------------------------------- (c) tfidf
    def tfidf_scores(self, deed, county):
        qs, ids = self.queries(deed), self._pool(county, deed.get("exclude"))
        if not qs or not len(ids):
            return {}
        sims = (self.vec.transform(qs) @ self.X[ids].T).toarray().max(axis=0)
        return self._entry_max(ids, sims)

    def tfidf(self, deed, k=5):
        sc = self.tfidf_scores(deed, deed["county"])
        return [e for e, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))][:k]

    # ---------------------------------------------------------------- (d) rerank features
    def candidates(self, deed, k):
        own = self.tfidf_scores(deed, deed["county"])
        allc = self.tfidf_scores(deed, None)
        top = lambda d: [e for e, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))][:k]  # noqa: E731
        cands = list(dict.fromkeys(top(own) + top(allc)))
        return cands, {**allc, **own}

    def features(self, deed, cands, tf):
        qs, exclude = self.queries(deed), deed.get("exclude")
        best_tf = max(tf.values()) if tf else 0.0
        blocks, lots = set(deed.get("blocks") or ()), set(deed.get("lots") or ())
        rows = []
        for e in cands:
            ent = self.gaz.entries[e]
            sids = [s for s in self.entry_surfaces[e] if self.gaz.usable(s, exclude)]
            texts = [self.texts[s] for s in sids] or [ent.key]
            f = {"tfidf": tf.get(e, 0.0), "tfidf_gap": best_tf - tf.get(e, 0.0)}
            best_q, best_r = None, -1
            for name, fn in (("tset", fuzz.token_set_ratio), ("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio),
                             ("tsort", fuzz.token_sort_ratio), ("wratio", fuzz.WRatio)):
                m = process.cdist(qs, texts, scorer=fn) if qs else np.zeros((1, 1))
                f[name] = float(m.max()) / 100
                if name == "ratio" and qs:
                    qi, ti = np.unravel_index(int(m.argmax()), m.shape)
                    best_q, best_r = qs[qi], ti
            kinds = {(self.texts[s], self.gaz.surfaces[s].kind) for s in sids}
            f["exact_name"] = float(any((q, "name") in kinds for q in qs))
            f["exact_cov"] = float(any((q, "cov") in kinds for q in qs))
            qn = plat_numbers(best_q) if best_q else set()
            en = plat_numbers(texts[best_r]) if best_q else plat_numbers(ent.key)
            f["num_match"] = float(bool(qn & en))
            f["num_conflict"] = float(bool(qn) and bool(en) and not (qn & en))
            f["num_query_only"] = float(bool(qn) and not en)
            f["num_entry_only"] = float(bool(en) and not qn)
            info = bool(blocks or lots) and bool(ent.blocks or ent.lots)
            f["bl_info"] = float(info)
            f["block_in"] = float(info and bool(blocks & ent.blocks))
            f["lot_in"] = float(info and bool(lots & ent.lots))
            f["pair_in"] = float(info and any((b, x) in ent.pairs for b in blocks for x in lots))
            f["county_match"] = float(ent.county == deed["county"])
            f["log_parcels"] = math.log1p(ent.n_parcels)
            f["log_cov_rows"] = math.log1p(ent.n_cov_rows)
            for src in ("partner", "mngeo", "public", "train"):
                f[f"src_{src}"] = float(any(s.startswith(src) for s in ent.sources))
            rows.append(f)
        return rows


def collect(index, deeds, k):
    """Per deed: (candidate entry ids, tfidf scores, feature rows) for the rerankers."""
    out = []
    for d in deeds:
        cands, tf = index.candidates(d, k)
        out.append((cands, tf, index.features(d, cands, tf) if cands else []))
    return out


def fit_reranker(collected, gold, cols, C=1.0, class_weight="balanced", max_iter=2000, seed=0):
    """Logistic regression on candidate rows (label = candidate is a gold entry)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for (cands, _, rows), g in zip(collected, gold):
        X += [[r[c] for c in cols] for r in rows]
        y += [int(e in g) for e in cands]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                                                               random_state=seed))
    model.fit(np.array(X, float), np.array(y))
    return model


def rerank(model, cols, cands, tf, rows, k=5):
    if not cands:
        return []
    p = model.predict_proba(np.array([[r[c] for c in cols] for r in rows], float))[:, 1]
    order = sorted(range(len(cands)), key=lambda i: (-p[i], -tf.get(cands[i], 0.0), cands[i]))
    return [cands[i] for i in order][:k]


def coefficients(model, cols):
    return dict(zip(cols, [round(float(c), 4) for c in model[-1].coef_[0]]))
