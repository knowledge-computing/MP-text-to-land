"""
ONE-TIME test evaluation of the 12 extraction x matcher pipelines, exactly as pre-registered in
docs/preregistration_test.md (approved by Me 2026-11-11). Prints and writes aggregates only.

Order; it stops at the first failed check:
 1. refuse if results/extract_x_matcher/test_summary.json exists (run once);
 2. refuse unless section 11 of the pre-registration names the frozen code commit, HEAD contains it, the frozen
    files are unchanged since then (working tree included) and the pre-registration differs from its copy in that
    commit only in the section 11 hash line and the section 10 deviations log (the pre-registration is not in
    FROZEN: the hash is filled in by a second commit after the frozen one);
 3. refuse unless every extractor's test cache holds exactly the test deeds, its records carry the frozen tag /
    prompt / digest / weights, no record has thinking text, and a cache with error records had its single retry;
 4. reproduce try08 + matcher_v1.0 from the saved try08 test predictions with the frozen code and compare every
    count (per county: matcher, paper criterion, weak labels) with results/matcher/test_v1.json (section 4.2);
 5. evaluate the 12 pipelines with matcher_v1.0's metric code (dev/eval_matcher_val.evaluate; paper criterion per
    extractor). Per-deed centroid-in outcomes come from evaluate() on one deed at a time and must sum to the
    aggregate;
 6. exact McNemar + Holm: Family 1 (m = 5, each extractor vs try08, matcher_v1.0) and Family 2 (m = 6,
    matcher_v1.0 vs paper_b1 per extractor), two-sided, alpha 0.05 (section 6);
 7. write test_cells.csv, test_extraction.csv and test_comparisons.csv, then test_summary.json.

    mapprejudice/bin/python dev/subdiv_16_test_extract_x_matcher.py          # the single test run
    mapprejudice/bin/python dev/subdiv_16_test_extract_x_matcher.py --toy    # synthetic data only (temp folder)
"""
import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dev"))
import plat_layers as pl  # noqa: E402
import yaml  # noqa: E402
from eval_matcher_test import TEST_PRED, load_test_deeds, paper_original  # noqa: E402
from eval_matcher_val import evaluate, preds_of  # noqa: E402
from subdiv_14_extract_x_matcher import ExactIndex  # noqa: E402
from weak_labels import summarize, weak_labels  # noqa: E402

from src.matcher import PlatIndex, frozen  # noqa: E402
from src.subdiv_match import runguard  # noqa: E402
from src.subdiv_match.data import deed_key  # noqa: E402
from src.subdiv_match.stats import discordant, holm, mcnemar_exact  # noqa: E402

DESIGN = REPO_ROOT / "configs" / "extract_x_matcher.yaml"
PREREG = REPO_ROOT / "docs" / "preregistration_test.md"
REFERENCE = REPO_ROOT / "results" / "matcher" / "test_v1.json"
OUT = REPO_ROOT / "results" / "extract_x_matcher"
BASE = "try08_ner"
EXTRACTORS = ("try08_ner", "qwen2.5_7b_q4", "gemma4_e4b_q4", "gemma4_31b_q4", "qwen3.8_27b_q4", "gliner_large_v2.5")
MATCHERS = ("paper_b1", "matcher_v1.0")
ALPHA = 0.05
FROZEN = ["configs/extract_x_matcher.yaml", "configs/subdiv_match.yaml", "dev/private_paths.py", "dev/plat_layers.py",
          "dev/infer_try08.py", "dev/weak_labels.py", "dev/eval_matcher_val.py", "dev/eval_matcher_test.py",
          "dev/subdiv_09_llm_extract.py", "dev/subdiv_14_extract_x_matcher.py", "dev/subdiv_15_gliner_extract.py",
          "dev/subdiv_16_test_extract_x_matcher.py", "dev/msi/llm_eval.sbatch", "src/matcher", "src/subdiv_match"]
NO_TRS = frozenset()   # matcher_v1.0 has no section constraint


def pct(n, d):
    return round(100 * n / d, 2) if d else None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 2. frozen code
def frozen_commit(prereg_text):
    """The hash in the line '- Frozen code commit: <hash>' (None while it is still the placeholder)."""
    m = re.search(r"^- Frozen code commit: `?([0-9a-f]{7,40})`?\s*$", prereg_text, re.M)
    return m.group(1) if m else None


def prereg_core(prereg_text):
    """The pre-registration without the parts that may change after the frozen commit: the section 11 hash line and
    the body of the section 10 deviations log."""
    text = re.sub(r"^- Frozen code commit: .*$", "- Frozen code commit: _(hash)_", prereg_text, flags=re.M)
    return re.sub(r"(^## 10\. Deviations log\n).*?(?=^## 11\. )", r"\1", text, flags=re.M | re.S)


def check_frozen(prereg=PREREG, files=FROZEN):
    text = Path(prereg).read_text()
    commit = frozen_commit(text)
    if not commit:
        sys.exit("refused: docs/preregistration_test.md section 11 has no frozen code commit yet")
    git = lambda *a: subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True)  # noqa: E731
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        sys.exit(f"refused: HEAD does not contain the frozen commit {commit}")
    old = git("show", f"{commit}:{Path(prereg).resolve().relative_to(REPO_ROOT).as_posix()}")
    if old.returncode != 0:
        sys.exit("refused: the pre-registration is not in the frozen commit")
    if prereg_core(old.stdout) != prereg_core(text):
        sys.exit("refused: the pre-registration changed since the frozen commit outside the section 11 hash line "
                 "and the section 10 deviations log")
    if git("diff", "--quiet", commit, "--", *files).returncode != 0:
        sys.exit("refused: frozen files changed since the frozen commit (working tree included)")
    if git("status", "--porcelain", "--untracked-files=all", "--", *files).stdout.strip():
        sys.exit("refused: untracked or uncommitted files under the frozen paths")
    return commit, git("rev-parse", "HEAD").stdout.strip()


# ---------------------------------------------------------------- 3. caches
def expected_fields(ex):
    if ex["kind"] == "llm":
        return {"model": ex["ollama_tag"], "prompt_sha256": ex["prompt_sha256"],
                "ollama_manifest_digest": ex.get("ollama_digest"),
                "think": None if ex.get("think_field") == "omit" else False}
    return {"model": ex["model"], "revision": ex["revision"], "weights_sha256": ex["model_sha256"]}


def check_caches(extractors, keys, root):
    """-> (counts per extractor, problems). Counts only; names are never read out."""
    status, problems = {}, []
    for name, ex in extractors.items():
        if ex["kind"] == "ner":
            continue
        path = Path(root) / ex["test_cache"]
        st = runguard.cache_status(path, keys)
        marker = runguard.retry_marker(path)
        st["retry"] = json.loads(marker.read_text()) if marker.exists() else None
        status[name] = st
        if not runguard.is_complete(st):
            problems.append(f"{name}: cache incomplete (exists {st['exists']}, missing {st['missing']}, "
                            f"duplicates {st['duplicates']}, extra {st['extra']})")
            continue
        if st["errors"] and st["retry"] is None:
            problems.append(f"{name}: {st['errors']} error records and the single retry has not run")
        if st["with_thinking"]:
            problems.append(f"{name}: {st['with_thinking']} records with thinking text")
        if not (Path(root) / ex["test_run_log"]).exists():
            problems.append(f"{name}: run log {ex['test_run_log']} is missing (hardware and timing record)")
        recs = runguard.read_records(path)
        for field, want in expected_fields(ex).items():
            got = {r.get(field) for r in recs}
            if got != {want}:
                problems.append(f"{name}: field {field} has {len(got)} distinct value(s), not only the frozen one")
    return status, problems


def cached_names(path):
    """{deed key: (names, seconds)}; an error record (after the retry) counts as no extraction."""
    out = {}
    for r in runguard.read_records(path):
        out[runguard.key_of(r)] = ([] if "error" in r else list(r["names"]), r.get("seconds"))
    return out


# ---------------------------------------------------------------- 4. reproduction
def reproduce(deeds, labels, index, errors):
    """Same composition as dev/eval_matcher_test.run (the 2026-10-09 test run)."""
    return {"weak_labels": summarize(deeds, labels),
            "matcher": evaluate(deeds, labels, [NO_TRS] * len(deeds), index, frozen.CONFIG, errors),
            "paper_original": paper_original(deeds, labels, index, errors)}


def reproduction_diffs(got, ref):
    """Every count that differs (empty list = exact reproduction), and the number of values compared."""
    diffs, n = [], 0
    for part in ("matcher", "paper_original"):
        for g in sorted(set(got[part]) | set(ref[part])):
            a, b = got[part].get(g, {}).get("counts", {}), ref[part].get(g, {}).get("counts", {})
            for k in sorted(set(a) | set(b)):
                n += 1
                if a.get(k, 0) != b.get(k, 0):
                    diffs.append(f"{part}.{g}.{k}: {a.get(k, 0)} vs {b.get(k, 0)} in the reference")
    for g in sorted(set(got["weak_labels"]) | set(ref["weak_labels"])):
        a, b = got["weak_labels"].get(g, {}), ref["weak_labels"].get(g, {})
        for k in ("deeds", "status", "with_weak_label", "overlap_>=2_plats"):
            n += 1
            if a.get(k) != b.get(k):
                diffs.append(f"weak_labels.{g}.{k}: {a.get(k)} vs {b.get(k)} in the reference")
    return diffs, n


# ---------------------------------------------------------------- 5. pipelines
def with_names(deeds, names):
    """Deeds whose SUBDIVISION_p is replaced by an extractor's names (None = keep try08's own predictions)."""
    if names is None:
        return deeds
    return [{**d, "props": {**d["props"], "SUBDIVISION_p": list(names.get(deed_key(d["props"]), ([], None))[0])}}
            for d in deeds]


def outcome_vector(ds, labels, index, errors):
    """Per eligible deed: 1 if its centroid lies in the predicted plat (evaluate() on that deed alone), else 0."""
    out = []
    for i, lab in enumerate(labels):
        if lab["status"] == "eligible":
            r = evaluate([ds[i]], [lab], [NO_TRS], index, frozen.CONFIG, errors)
            out.append(int(r.get("ALL", {}).get("counts", {}).get("centroid_in", 0)))
    return out


def evaluate_pipelines(deeds, labels, indexes, preds_by_extractor, errors, per_deed_errors):
    cells, coverage, outcomes, paper = [], {}, {}, {}
    eligible = [lab["status"] == "eligible" for lab in labels]
    for name, names in preds_by_extractor.items():
        ds = with_names(deeds, names)
        has = [bool(preds_of(d["props"])) for d in ds]
        coverage[name] = {"deeds": len(ds), "with_extraction": sum(has), "coverage_pct": pct(sum(has), len(ds)),
                          "eligible": sum(eligible), "eligible_with_extraction": sum(h and e for h, e in zip(has, eligible)),
                          "coverage_eligible_pct": pct(sum(h and e for h, e in zip(has, eligible)), sum(eligible))}
        paper[name] = paper_original(ds, labels, indexes["matcher_v1.0"], errors)   # matcher-independent
        for m in MATCHERS:
            res = evaluate(ds, labels, [NO_TRS] * len(ds), indexes[m], frozen.CONFIG, errors)
            vec = outcome_vector(ds, labels, indexes[m], per_deed_errors)
            agg = res.get("ALL", {}).get("counts", {}).get("centroid_in", 0)
            if sum(vec) != agg:
                sys.exit(f"stop: per-deed outcomes ({sum(vec)}) differ from the aggregate ({agg}) for {name} + {m}")
            outcomes[(name, m)] = vec
            for county, r in sorted(res.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
                c, p = r["counts"], paper[name].get(county, {})
                cells.append({"extractor": name, "matcher": m, "county": county, "eligible_b": c.get("b_all", 0),
                              "with_extraction_a": c.get("a_with_pred", 0), "top1": c.get("top1", 0),
                              "centroid_in": c.get("centroid_in", 0),
                              "centroid_in_pct_b": r["centroid_in_pct_b"], "centroid_in_pct_a": r["centroid_in_pct_a"],
                              "plat_correct": c.get("plat_correct", 0),
                              "plat_acc_pct_b": r["plat_acc_b"], "plat_acc_pct_a": r["plat_acc_a"],
                              "contains_pct_b": r["contains_pct_b"], "intersects_pct_b": r["intersects_pct_b"],
                              "paper_criterion_pct_all": p.get("pct_all"),
                              "paper_criterion_hits": (p.get("counts") or {}).get("all_hit"),
                              "paper_criterion_n_all": (p.get("counts") or {}).get("all_with_pred")})
    return cells, coverage, outcomes, paper


# ---------------------------------------------------------------- 6. tests
def comparisons(outcomes, extractors=EXTRACTORS, base=BASE, alpha=ALPHA):
    families = {"family1_extractor": [((e, "matcher_v1.0"), (base, "matcher_v1.0")) for e in extractors if e != base],
                "family2_matcher": [((e, "matcher_v1.0"), (e, "paper_b1")) for e in extractors]}
    rows = []
    for fam, pairs in families.items():
        part = []
        for x, y in pairs:
            ox, oy = outcomes[x], outcomes[y]
            b, c = discordant(ox, oy)
            n = len(ox)
            part.append({"family": fam, "m": len(pairs), "x": " + ".join(x), "y": " + ".join(y), "n_deeds": n,
                         "centroid_in_x": sum(ox), "centroid_in_y": sum(oy), "pct_x": pct(sum(ox), n),
                         "pct_y": pct(sum(oy), n), "diff_points": round(100 * (sum(ox) - sum(oy)) / n, 2) if n else None,
                         "b_only_x": b, "c_only_y": c, "p": mcnemar_exact(b, c)})
        for r, adj in zip(part, holm([r["p"] for r in part])):
            r["p_holm"] = adj
            r["significant"] = adj < alpha
        rows += part
    return rows


# ---------------------------------------------------------------- 7. outputs
def run_log_info(ex, root):
    """Hardware and job counts from an extractor's test run log (one row per job)."""
    rows = list(csv.DictReader((Path(root) / ex["test_run_log"]).open()))
    hw = sorted({" / ".join(v for v in (r.get("gpus"), r.get("cpu"), r.get("device"), r.get("os")) if v) for r in rows})
    return {"test_jobs": sum(r.get("phase") == "pass" for r in rows), "test_retry_jobs": sum(r.get("phase") == "retry" for r in rows),
            "test_hardware": " | ".join(hw), "ollama_version": " | ".join(sorted({r["ollama_version"] for r in rows if r.get("ollama_version")})) or None}


def timing_row(name, ex, names, root):
    if ex["kind"] == "ner":
        run = json.loads((Path(root) / ex["test_timing"]).read_text())["run"]
        return {"sec_per_deed_mean": round(run["ner_seconds"] / max(1, run["counts"]["deeds"]), 3),
                "timing_source": f"{ex['test_timing']} (ner_seconds / deeds; model load excluded)"}
    s = [v[1] for v in names.values() if v[1] is not None]
    q = statistics.quantiles(s, n=10) if len(s) >= 10 else [None] * 9
    r3 = lambda v: round(v, 3) if v is not None else None  # noqa: E731
    return {"sec_per_deed_mean": r3(statistics.mean(s) if s else None), "sec_per_deed_median": r3(statistics.median(s) if s else None),
            "sec_per_deed_p90": r3(q[8]), "timing_source": f"{ex['test_cache']} (per-deed wall clock, {len(s)} deeds)",
            **run_log_info(ex, root)}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_all(deeds, plats, indexes, extractors, root, reference, out, meta):
    """Steps 3-7 (shared by the real run and --toy)."""
    t0, errors, per_deed_errors = time.time(), Counter(), Counter()
    keys = [deed_key(d["props"]) for d in deeds]
    if len(set(keys)) != len(keys):
        sys.exit(f"stop: {len(keys) - len(set(keys))} duplicate deed keys in the test predictions")
    status, problems = check_caches(extractors, set(keys), root)
    if problems:
        print("\n".join("  " + p for p in problems))
        sys.exit("refused: the test caches are not ready (see above); nothing was evaluated")
    labels = weak_labels(deeds, plats, errors)
    diffs, n_checked = reproduction_diffs(reproduce(deeds, labels, indexes["matcher_v1.0"], errors), reference)
    if diffs:
        print("\n".join("  " + d for d in diffs[:40]))
        sys.exit(f"stop: try08 + matcher_v1.0 does not reproduce the reference ({len(diffs)} of {n_checked} counts "
                 "differ); nothing else was evaluated (pre-registration section 4.2)")
    print(f"  reproduction of try08 + matcher_v1.0: {n_checked} counts identical", flush=True)
    names = {n: (None if ex["kind"] == "ner" else cached_names(Path(root) / ex["test_cache"])) for n, ex in extractors.items()}
    cells, coverage, outcomes, _ = evaluate_pipelines(deeds, labels, indexes, names, errors, per_deed_errors)
    comps = comparisons(outcomes, tuple(extractors))
    ext_rows = []
    for n, ex in extractors.items():
        st = status.get(n, {})
        retry = st.get("retry") or {}
        ext_rows.append({"extractor": n, "kind": ex["kind"], **coverage[n], "cache_errors_first_pass": retry.get("errors_before", st.get("errors")),
                         "cache_retried": retry.get("retried"), "cache_errors_final": st.get("errors"),
                         "with_thinking": st.get("with_thinking"), **timing_row(n, ex, names[n] or {}, root),
                         **{k: ex[k] for k in ("model_sha256", "prompt_sha256") if k in ex}})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "test_cells.csv", cells)
    write_csv(out / "test_extraction.csv", ext_rows)
    write_csv(out / "test_comparisons.csv", comps)
    summary = {**meta, "deeds": len(deeds), "eligible": sum(lab["status"] == "eligible" for lab in labels),
               "reproduction": {"reference": str(meta.get("reference_path")), "counts_compared": n_checked, "identical": True},
               "pipelines": [c for c in cells if c["county"] == "ALL"], "extraction": ext_rows, "comparisons": comps,
               "errors": dict(errors), "per_deed_errors": dict(per_deed_errors), "seconds": round(time.time() - t0, 1)}
    (out / "test_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    import pandas as pd
    pd.set_option("display.width", 250)
    cf = pd.DataFrame(cells)
    print(cf[cf.county == "ALL"][["extractor", "matcher", "eligible_b", "with_extraction_a", "centroid_in", "centroid_in_pct_b",
                                  "centroid_in_pct_a", "plat_acc_pct_b", "paper_criterion_pct_all"]].to_string(index=False))
    print(pd.DataFrame(ext_rows)[["extractor", "deeds", "with_extraction", "coverage_pct", "eligible_with_extraction",
                                  "cache_errors_first_pass", "cache_retried", "cache_errors_final", "sec_per_deed_mean"]].to_string(index=False))
    cp = pd.DataFrame(comps)
    cp["p"] = cp["p"].map(lambda v: f"{v:.3g}")
    cp["p_holm"] = cp["p_holm"].map(lambda v: f"{v:.3g}")
    print(cp[["family", "x", "y", "pct_x", "pct_y", "diff_points", "b_only_x", "c_only_y", "p", "p_holm", "significant"]].to_string(index=False))
    print("errors:", dict(errors), "| per-deed errors:", dict(per_deed_errors), "| seconds:", summary["seconds"])
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--toy", action="store_true", help="synthetic plats, deeds and caches only (reads no data)")
    args = ap.parse_args(argv)
    if args.toy:
        return toy()
    if (OUT / "test_summary.json").exists():
        sys.exit(f"refused: {(OUT / 'test_summary.json').relative_to(REPO_ROOT)} exists; the test evaluation runs once")
    commit, head = check_frozen()
    cfg = yaml.safe_load(DESIGN.read_text())
    extractors = {n: cfg["extractors"][n] for n in EXTRACTORS}
    plats, pairs = pl.load_plats()
    norm3 = frozen.frozen_normalizer(pairs)            # raises if the frozen name rules changed
    indexes = {"paper_b1": ExactIndex(plats, norm3), "matcher_v1.0": PlatIndex(plats, norm3)}
    deeds = load_test_deeds(TEST_PRED)
    meta = {"preregistration": str(PREREG.relative_to(REPO_ROOT)), "preregistration_sha256": sha256_file(PREREG),
            "design_sha256": sha256_file(DESIGN), "frozen_commit": commit, "head": head, "matcher": frozen.VERSION,
            "reference_path": str(REFERENCE.relative_to(REPO_ROOT)), "alpha": ALPHA}
    run_all(deeds, plats, indexes, extractors, REPO_ROOT, json.loads(REFERENCE.read_text()), OUT, meta)


# ---------------------------------------------------------------- --toy (synthetic only)
def toy():
    """End-to-end check of steps 3-7 on synthetic plats, deeds and caches, plus the refusal paths. Reads no data."""
    from shapely.geometry import MultiPolygon, box

    from src.matcher import NameNormalizer, learn_rules
    rng = random.Random(0)
    plat_boxes = {("alpha", "Maple Heights Addition"): (0, 0), ("alpha", "Maple Heights 2nd Addition"): (100, 0),
                  ("alpha", "Oak Park"): (0, 100), ("beta", "Pine Ridge"): (1000, 0), ("beta", "Cedar Glen"): (1100, 0)}
    plats = [{"county": c, "name": n, "geometry": box(x, y, x + 100, y + 100), "sections": set(), "aliases": []}
             for (c, n), (x, y) in plat_boxes.items()]
    norm = NameNormalizer(learn_rules([], 5))
    indexes = {"paper_b1": ExactIndex(plats, norm), "matcher_v1.0": PlatIndex(plats, norm)}
    variants = lambda n: [n, n.upper(), n.replace("Addition", "Addn"), n.replace("Heights", "Hts"), "Oak Park", ""]  # noqa: E731
    deeds = []
    for i in range(60):
        (county, name), (x, y) = rng.choice(list(plat_boxes.items()))
        px, py = x + rng.uniform(5, 85), y + rng.uniform(5, 85)
        geom = box(px, py, px + 5, py + 5)
        n_parts = 1
        if i == 59:
            geom, n_parts = MultiPolygon([geom, box(px + 7, py, px + 9, py + 2)]), 2
        pred = rng.choice(variants(name))
        deeds.append({"county": county, "geometry": geom, "n_parts": n_parts,
                      "props": {"cnty_name": county, "image_ids": f"['t{i}']", "SUBDIVISION_p": [pred] if pred else [],
                                "_true": name}})
    cfg = yaml.safe_load(DESIGN.read_text())
    extractors = {n: dict(cfg["extractors"][n]) for n in EXTRACTORS}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results" / "infer_try08").mkdir(parents=True)
        (root / extractors[BASE]["test_timing"]).write_text(json.dumps({"run": {"ner_seconds": 30.0, "counts": {"deeds": 60}}}))
        for k, (n, ex) in enumerate(extractors.items()):
            if ex["kind"] == "ner":
                continue
            path = root / ex["test_cache"]
            path.parent.mkdir(parents=True, exist_ok=True)
            fields = expected_fields(ex)
            with open(path, "w") as fh:
                for i, d in enumerate(deeds):
                    county, ids = deed_key(d["props"])
                    rec = {"county": county, "image_ids": sorted(ids), **fields, "seconds": round(rng.uniform(0.5, 4), 2)}
                    if k == 2 and i == 3:
                        rec.update(names=[], error="TimeoutError")                    # stays an error after the retry
                    else:
                        guess = rng.choice(variants(d["props"]["_true"]) if rng.random() < 0.6 + 0.05 * k else variants("Oak Park"))
                        rec.update(names=[guess] if guess else [], thinking_chars=0)
                    fh.write(json.dumps(rec) + "\n")
            if k == 2:
                runguard.retry_marker(path).write_text(json.dumps({"errors_before": 2, "retried": 2, "errors_after": 1}))
            log = root / ex["test_run_log"]
            log.parent.mkdir(parents=True, exist_ok=True)
            runguard.append_log(log, {"phase": "pass", "gpus": "TOY GPU" if "31b" in n or "27b" in n else "",
                                      "cpu": "toy cpu", "device": "mps" if ex["kind"] == "gliner" else "",
                                      "ollama_version": "0.40.3" if ex["kind"] == "llm" else ""})
        keys = {deed_key(d["props"]) for d in deeds}

        print("toy: refusal checks")
        g = extractors["gemma4_e4b_q4"]
        st, probs = check_caches(extractors, keys, root)
        print(f"  ready caches -> problems {len(probs)}")
        lines = (root / g["test_cache"]).read_text().splitlines(True)
        (root / g["test_cache"]).write_text("".join(lines[:-1]))
        print(f"  dropped one record -> {check_caches(extractors, keys, root)[1]}")
        bad = [json.dumps({**json.loads(lines[0]), "prompt_sha256": "0" * 64}) + "\n"] + lines[1:]
        (root / g["test_cache"]).write_text("".join(bad))
        print(f"  wrong prompt hash in one record -> {check_caches(extractors, keys, root)[1]}")
        (root / g["test_cache"]).write_text("".join(lines))
        gl = root / extractors["gliner_large_v2.5"]["test_cache"]
        gl_lines = gl.read_text().splitlines(True)
        gl.write_text("".join([json.dumps({**json.loads(gl_lines[0]), "names": [], "error": "RuntimeError"}) + "\n"] + gl_lines[1:]))
        print(f"  GLiNER error record, no retry marker -> {check_caches(extractors, keys, root)[1]}")
        gl.write_text("".join(gl_lines))
        log = root / extractors["qwen3.8_27b_q4"]["test_run_log"]
        log.rename(log.with_suffix(".bak"))
        print(f"  missing run log -> {check_caches(extractors, keys, root)[1]}")
        log.with_suffix(".bak").rename(log)
        print(f"  frozen-commit line: placeholder -> {frozen_commit('- Frozen code commit: _(hash)_')}, "
              f"filled -> {frozen_commit('- Frozen code commit: 1a2b3c4d')}")
        pre = "# P\n## 10. Deviations log\n\n(empty)\n\n## 11. Approval\n\n- Frozen code commit: _(hash)_\n"
        same = lambda t: prereg_core(t) == prereg_core(pre)  # noqa: E731
        print(f"  pre-registration vs frozen copy: hash filled -> same {same(pre.replace('_(hash)_', '1a2b3c4d'))}, "
              f"deviation logged -> same {same(pre.replace('(empty)', '- toy deviation'))}, "
              f"other text edited -> same {same(pre.replace('# P', '# Q'))}")

        labels = weak_labels(deeds, plats, Counter())
        reference = reproduce(deeds, labels, indexes["matcher_v1.0"], Counter())
        ref_bad = json.loads(json.dumps(reference))
        ref_bad["matcher"]["alpha"]["counts"]["centroid_in"] += 1
        print(f"  perturbed reference -> diffs {reproduction_diffs(reproduce(deeds, labels, indexes['matcher_v1.0'], Counter()), ref_bad)[0]}")
        try:
            run_all(deeds, plats, indexes, extractors, root, ref_bad, root / "out_bad", {"toy": True})
        except SystemExit as e:
            print(f"  run_all with the perturbed reference -> {e}; outputs written: {(root / 'out_bad').exists()}")

        print("toy: full run (synthetic)")
        summary = run_all(deeds, plats, indexes, extractors, root, reference, root / "out", {"toy": True})
        written = sorted(p.name for p in (root / "out").iterdir())
        fam = Counter(r["family"] for r in summary["comparisons"])
        print(f"toy: outputs {written} | comparisons per family {dict(fam)} | eligible {summary['eligible']} of {summary['deeds']}")


if __name__ == "__main__":
    main()
