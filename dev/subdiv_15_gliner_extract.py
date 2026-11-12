"""
Extractor for the extraction x matcher experiment (configs/extract_x_matcher.yaml): GLiNER zero-shot NER, run locally
(HF transformers weights from the local cache, HF_HUB_OFFLINE=1; deed text never leaves the machine). No training,
no tuning on val: labels, threshold and chunking are fixed in the config before any real run.

Input text per deed = the same keyword windows the LLM extractors read (dev/subdiv_09_llm_extract.windows, window /
max_chars from the config), cut into overlapping chunks on GLiNER's own word tokens (src/subdiv_match/span_extract)
so nothing is silently truncated; entities of any configured label are subdivision names. Results are cached per deed
in output/derived/subdiv_match/gliner_<split>__<slug>.jsonl (git-ignored; names are derived from deed text, never
printed) in the same record format as the LLM caches, so dev/subdiv_14_extract_x_matcher.py reads both. The run CSV
(results/extract_x_matcher/run_gliner_<split>__<slug>.csv) logs the model revision and weight sha256, the settings,
library versions, device and hardware. Prints counts and timing only.

    ~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_15_gliner_extract.py --toy          # synthetic text only
    ~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_15_gliner_extract.py --split val [--limit 20]
    ~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_15_gliner_extract.py --split test [--retry-errors]

TEST split (docs/preregistration_test.md, approved 2026-11-11): --limit is refused; an incomplete cache resumes but a
second full pass is refused; --retry-errors re-queries the error records once after the pass (then refused); the run
log gets one row per job (src/subdiv_match/runguard.py). The weights sha256 is checked at load, as for val.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / "MapPrejudice" / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"          # never contact the Hub during a run (model must be in the local cache)
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # not via src/__init__ (it needs dotenv)
sys.path.insert(0, str(REPO_ROOT / "dev"))
import yaml  # noqa: E402
from subdiv_09_llm_extract import windows  # noqa: E402
from subdiv_match import runguard  # noqa: E402
from subdiv_match.span_extract import merge_entities, token_chunks  # noqa: E402

EXTRACTOR = "gliner_large_v2.5"
TOY = [  # synthetic, written for this script (no deed text)
    "... conveyed unto the grantee: Lot Twelve (12), Block Four (4), Birchwood Terrace Third Addition, according to "
    "the plat on file, Toy County ...",
    "... the East 40 feet of Lot 3, Block 1, Lakeview Park, and Lot 2, Auditor's Subdivision No. 7, Sample County, "
    "State of Example ...",
    "... commencing at the northeast corner of Section 22, Township 115 North, Range 20 West; thence South 200 feet ...",
    "... Lot 9 Blk 6 REARRANGEMENT OF BLOCKS 5 AND 6, SUNNYSIDE HEIGHTS, in the Village of Toyburg ...",
    "... Lots 1 and 2, Block 3, Elm Grove Replat, subject to the restriction that said premises shall not be sold ...",
    # long input: the name sits past the first chunk, so chunking and merging are exercised
    ("WITNESSETH, that the party of the first part, in consideration of the sum of One Dollar, does grant, bargain, "
     "sell and convey unto the party of the second part, forever, all the tract of land lying in the County of "
     "Example and State of Toyland, ") * 12
    + "described as Lot 15, Block 2, Meadowbrook Gardens Second Addition, according to the recorded plat thereof. "
    + ("TO HAVE AND TO HOLD the same, together with all the hereditaments and appurtenances thereunto belonging. ") * 6,
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ex, device):
    """Load the pinned revision from the local cache; refuse if the weight file's sha256 differs from the config."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(ex["model"], revision=ex["revision"], local_files_only=True))
    got = sha256_file((snap / ex["weights_file"]).resolve())
    if got != ex["model_sha256"]:
        raise SystemExit(f"weights sha256 mismatch for {ex['weights_file']}: config {ex['model_sha256'][:12]}..., "
                         f"local {got[:12]}...")
    import torch
    from gliner import GLiNER
    torch.manual_seed(0)
    model = GLiNER.from_pretrained(str(snap), local_files_only=True)
    model.to(device)
    model.eval()
    return model, got


def extract(model, text, ex, counter):
    """-> names in order of appearance. Truncation warnings from GLiNER are counted (they must stay 0)."""
    chunks = token_chunks(text, ex["chunk_tokens"], ex["chunk_overlap"])
    if not chunks:
        return [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ents = model.inference([text[a:b] for a, b, _, _ in chunks], ex["labels"], flat_ner=ex["flat_ner"],
                               threshold=ex["threshold"], multi_label=False, batch_size=ex["batch_size"])
    counter["truncation_warnings"] += sum("truncated" in str(w.message) for w in caught)
    return merge_entities(text, chunks, ents), len(chunks)


def hardware(device):
    import torch
    info = {"python": platform.python_version(), "machine": platform.machine(), "os": platform.platform(),
            "device": device, "torch": torch.__version__}
    try:
        import gliner
        import transformers
        info.update(gliner=gliner.__version__, transformers=transformers.__version__)
        out = lambda *a: subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()  # noqa: E731
        if platform.system() == "Darwin":
            info["cpu"] = out("sysctl", "-n", "machdep.cpu.brand_string")
            info["memory_gb"] = round(int(out("sysctl", "-n", "hw.memsize")) / 2 ** 30, 1)
        else:
            info["cpu"] = platform.processor()
            info["gpus"] = out("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    except Exception as e:  # noqa: BLE001
        info["hardware_error"] = type(e).__name__
    return info


def pick_device(arg):
    import torch
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "subdiv_match.yaml"))
    ap.add_argument("--design", default=str(REPO_ROOT / "configs" / "extract_x_matcher.yaml"))
    ap.add_argument("--extractor", default=EXTRACTOR)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-errors", action="store_true", help="test only: the single retry of error records")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "extract_x_matcher"), help="run CSV folder")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--toy", action="store_true", help="run on the built-in synthetic texts only (reads no data)")
    args = ap.parse_args(argv)
    ex = yaml.safe_load(Path(args.design).read_text())["extractors"][args.extractor]
    if ex.get("kind") != "gliner":
        raise SystemExit(f"{args.extractor} is not a gliner extractor")
    test = args.split == "test"
    if args.retry_errors and not test:
        raise SystemExit("--retry-errors is for the test split only")
    if test and args.limit:
        raise SystemExit("refused: --limit is not allowed on test (each extractor runs over all test deeds once)")
    device = pick_device(args.device)
    t0 = time.time()
    model, weights_sha = load_model(ex, device)
    load_s = round(time.time() - t0, 1)
    stats = Counter()

    if args.toy:
        print(f"model {ex['model']}@{ex['revision'][:10]} weights sha256 ok | device {device} | load {load_s} s")
        secs = []
        for i, t in enumerate(TOY):
            outs = []
            for r in range(2):
                t1 = time.time()
                names, n_chunks = extract(model, windows(t, ex["window"], ex["max_chars"]), ex, stats)
                secs.append(time.time() - t1)
                outs.append(names)
                print(f"  toy{i} rep{r}: {secs[-1]:5.2f}s chunks={n_chunks} names={names}")
            stats["identical"] += outs[0] == outs[1]
        print(f"identical across 2 runs {stats['identical']}/{len(TOY)} | truncation warnings "
              f"{stats['truncation_warnings']} | s/call mean {sum(secs) / len(secs):.2f}, max {max(secs):.2f}")
        # chunking check: the long toy text without keyword windows, small chunks vs one chunk
        small, n_small = extract(model, TOY[-1], {**ex, "chunk_tokens": 64, "chunk_overlap": 16}, stats)
        whole, n_whole = extract(model, TOY[-1], {**ex, "chunk_tokens": 760, "chunk_overlap": 0}, stats)
        print(f"chunking check ({len(TOY[-1])} chars): {n_small} chunks -> {small} | {n_whole} chunk -> {whole} | "
              f"same {small == whole}")
        return

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from subdiv_match.config import load_config, ocr_root
    from subdiv_match.ocr import deed_texts
    cfg = load_config(args.config)
    P = cfg["paths"]
    texts = deed_texts(P["splits"][args.split], ocr_root(), stats)
    keys = sorted(texts, key=lambda k: (k[0], sorted(k[1])))
    random.Random(cfg["seed"]).shuffle(keys)
    if args.limit:
        keys = keys[:args.limit]
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", args.extractor)
    cache = P["derived"] / f"gliner_{args.split}__{slug}.jsonl"
    out = Path(args.out_dir) / f"run_gliner_{args.split}__{slug}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    settings = {k: ex[k] for k in ("model", "revision", "weights_file", "labels", "threshold", "flat_ner",
                                   "chunk_tokens", "chunk_overlap", "batch_size", "window", "max_chars")}
    base = {"split": args.split, "extractor": args.extractor, **settings, "labels": "|".join(ex["labels"]),
            "weights_sha256": weights_sha, **hardware(device), "model_load_seconds": load_s}

    def query(k):
        rec = {"county": k[0], "image_ids": sorted(k[1]), "model": ex["model"], "revision": ex["revision"],
               "weights_sha256": weights_sha, "device": device}
        t1 = time.time()
        try:
            names, n_chunks = extract(model, windows(texts[k], ex["window"], ex["max_chars"]), ex, stats)
            rec.update(names=names, chunks=n_chunks)
        except Exception as e:  # noqa: BLE001 - type only
            rec.update(names=[], error=type(e).__name__)
        rec["seconds"] = round(time.time() - t1, 3)
        return rec

    t_run = time.time()
    if args.retry_errors:
        counts = runguard.retry_errors(cache, keys, lambda r: query(runguard.key_of(r)))
        row = {**base, "phase": "retry", **counts, "truncation_warnings": stats["truncation_warnings"],
               "seconds": round(time.time() - t_run, 1)}
        runguard.append_log(out, row)
        print(row)
        return
    if test:
        runguard.guard_pass(cache, keys)
    done = set()
    if cache.exists():
        done = {runguard.key_of(r) for r in runguard.read_records(cache)}
    with cache.open("a") as fh:
        for i, k in enumerate(keys):
            if k in done:
                stats["cached"] += 1
                continue
            rec = query(k)
            if "error" in rec:
                stats[f"error:{rec['error']}"] += 1
            else:
                stats["ok"] += 1
                stats["with_name"] += bool(rec["names"])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(keys)} deeds, {round(time.time() - t_run)} s, {dict(stats)}", flush=True)
    row = {**base, "deeds_requested": len(keys), **stats, "seconds": round(time.time() - t_run, 1)}
    if test:
        st = runguard.cache_status(cache, keys)
        runguard.append_log(out, {"phase": "pass", **row, "cache_complete": runguard.is_complete(st),
                                  "cache_errors": st["errors"]})
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
