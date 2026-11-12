"""
Step 7c (baseline B3): a LOCAL LLM (Ollama on 127.0.0.1; no cloud API, deed text never leaves the machine)
reads each val deed's OCR text and extracts the subdivision / addition / plat name(s); dev/subdiv_10_eval.py then
matches them with the same rapidfuzz / TF-IDF matchers as the NER queries.

Input text: windows of +-W characters around legal-description keywords (addition, subdivision, plat, lot,
block, according to ...), merged and capped at --max-chars (the first --max-chars characters when no keyword
occurs). Prompt: instructions + three SYNTHETIC few-shot examples (no deed text in this file); JSON output
{"subdivisions": [...]}; temperature 0, fixed seed, thinking off (think = llm.think, sent explicitly; the
thinking text is never stored, only its length, which must be 0).

Results are cached per deed in output/derived/subdiv_match/llm_<split>.jsonl for the config model, and in
llm_<split>__<model slug>.jsonl for any other --model (git-ignored: extracted names are derived from deed text), so
runs can resume; --limit samples deeds with the config seed. Every record and the run CSV log the model, the Ollama
manifest digest, the Ollama version, the prompt hash and the host hardware (item 4 of the extraction x matcher
design, configs/extract_x_matcher.yaml). All models get the same prompt and parameters. Prints counts and timing
only.

TEST split (docs/preregistration_test.md, approved 2026-11-11): --split test refuses --limit, checks the model's GGUF
blob sha256 against configs/extract_x_matcher.yaml before reading any deed, resumes an incomplete cache but refuses a
second full pass (src/subdiv_match/runguard.py), and appends one row per job to the run log. After the pass,
--retry-errors re-queries the error records once (identical settings) and is then refused. --toy-think compares
think false with no think field on synthetic text only (decision 3; reads no data).

    ~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_09_llm_extract.py --split val [--model TAG] [--limit 150]
    ~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_09_llm_extract.py --toy-think [--model TAG]
    ~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_09_llm_extract.py --split test --model TAG [--retry-errors]
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # not via src/__init__ (it needs dotenv)
from subdiv_match import runguard  # noqa: E402
from subdiv_match.config import load_config, ocr_root  # noqa: E402
from subdiv_match.ocr import deed_texts  # noqa: E402



def _ollama_root():
    """Ollama server from OLLAMA_HOST (default 127.0.0.1:11434). Only a loopback host is accepted, so deed text can
    only go to a server on this machine (on MSI: the job's own server, on a per-job port)."""
    host = (os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434").removeprefix("http://")
    name, _, port = host.partition(":")
    if name not in ("127.0.0.1", "localhost") or not (port or "11434").isdigit():
        raise SystemExit("OLLAMA_HOST must be 127.0.0.1[:port] (local server only)")
    return f"http://127.0.0.1:{port or 11434}"


OLLAMA_ROOT = _ollama_root()
OLLAMA = OLLAMA_ROOT + "/api/chat"
KEYWORDS = re.compile(r"\b(?:addition|add\.|addn|subdivision|subd|sub\.|plat|replat|rearrangement|lots?|blocks?|"
                      r"according\s+to|outlots?|auditor)", re.I)
SYSTEM = (
    "You read the legal description in an OCR'd historical property deed and return the name of the recorded "
    "subdivision, addition or plat that the land lies in (for example 'Oakwood Second Addition', 'Auditor's "
    "Subdivision No. 9', 'Rearrangement of Block 3, Elm Grove'). Copy each name as written in the deed, without "
    "lot or block numbers, city, county or state. Return every distinct plat name if there are several. Return an "
    "empty list when the land is described only by metes and bounds or by section, township and range. "
    "Answer with JSON only: {\"subdivisions\": [\"...\"]}.")
FEW_SHOT = [  # synthetic examples written for this script
    ("... the following described real estate, to-wit: Lot Seven (7), Block Three (3), Maplewood Heights Second "
     "Addition to the City of Toyville, according to the recorded plat thereof, County of Example ...",
     {"subdivisions": ["Maplewood Heights Second Addition"]}),
    ("... all that part of the Southwest Quarter of Section 12, Township 30 North, Range 21 West, described as "
     "follows: beginning at the southwest corner thereof; thence North 89 degrees East 330 feet ...",
     {"subdivisions": []}),
    ("... Lots 4 and 5, Block 2, Rearrangement of Block 2, Toy Park, and Lot 1, Auditor's Subdivision No. 12, "
     "Example County ...",
     {"subdivisions": ["Rearrangement of Block 2, Toy Park", "Auditor's Subdivision No. 12"]}),
]


def windows(text, w, max_chars):
    t = re.sub(r"\s+", " ", text)
    spans = []
    for m in KEYWORDS.finditer(t):
        a, b = max(0, m.start() - w), min(len(t), m.end() + w)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    out = " ... ".join(t[a:b] for a, b in spans) if spans else t
    return out[:max_chars]


def prompt_sha256():
    return hashlib.sha256(json.dumps({"system": SYSTEM, "few_shot": FEW_SHOT, "keywords": KEYWORDS.pattern},
                                     sort_keys=True).encode()).hexdigest()


def _get(path, body=None, timeout=30):
    req = urllib.request.Request(OLLAMA_ROOT + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_info(model):
    """Model digest, Ollama version and host hardware (no deed data)."""
    info = {"model": model, "prompt_sha256": prompt_sha256(), "python": platform.python_version(),
            "machine": platform.machine(), "os": platform.platform()}
    try:
        info["ollama_version"] = _get("/api/version").get("version")
        tags = {m["name"]: m for m in _get("/api/tags").get("models", [])}
        info["ollama_manifest_digest"] = (tags.get(model) or {}).get("digest")
        show = _get("/api/show", {"model": model})
        d = show.get("details") or {}
        info.update(capabilities=",".join(show.get("capabilities") or []), quantization=d.get("quantization_level"),
                    weights_format=d.get("format"), runner=d.get("runner"), parameter_size=d.get("parameter_size"),
                    modelfile_params=" ".join((show.get("parameters") or "").split()))
    except Exception as e:  # noqa: BLE001
        info["ollama_info_error"] = type(e).__name__
    try:
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


def request_options(num_ctx, seed):
    """Generation options sent with every request (identical for all models)."""
    return {"temperature": 0, "seed": seed, "num_ctx": num_ctx}


TOY = [  # synthetic, written for this script (no deed text)
    "... conveyed unto the grantee: Lot Twelve (12), Block Four (4), Birchwood Terrace Third Addition, according to "
    "the plat on file, Toy County ...",
    "... the East 40 feet of Lot 3, Block 1, Lakeview Park, and Lot 2, Auditor's Subdivision No. 7, Sample County, "
    "State of Example ...",
    "... commencing at the northeast corner of Section 22, Township 115 North, Range 20 West; thence South 200 feet ...",
    "... Lot 9 Blk 6 REARRANGEMENT OF BLOCKS 5 AND 6, SUNNYSIDE HEIGHTS, in the Village of Toyburg ...",
    "... Lots 1 and 2, Block 3, Elm Grove Replat, subject to the restriction that said premises shall not be sold ...",
    "... all of Lot 14 in Block 22 of Rivermead Gardens 2nd Addn to the City of Example, according to the plat ...",
]
DESIGN = REPO_ROOT / "configs" / "extract_x_matcher.yaml"


def ask(model, text, num_ctx, seed, think=False, timeout=600):
    """think=None omits the think field from the request (decision 3 of the pre-registration)."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for x, y in FEW_SHOT:
        msgs += [{"role": "user", "content": x}, {"role": "assistant", "content": json.dumps(y)}]
    msgs.append({"role": "user", "content": text})
    body = {"model": model, "messages": msgs, "stream": False, "format": "json", "options": request_options(num_ctx, seed)}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.load(r)
    names = json.loads(js["message"]["content"]).get("subdivisions", [])
    return [n.strip() for n in names if isinstance(n, str) and n.strip()], js


def design_entry(model, design=DESIGN):
    """The extractor entry of configs/extract_x_matcher.yaml whose ollama_tag is `model` (None if absent)."""
    import yaml
    for name, ex in yaml.safe_load(Path(design).read_text())["extractors"].items():
        if ex.get("ollama_tag") == model:
            return name, ex
    return None, None


def think_setting(L, ex):
    """llm.think, unless the design entry says to omit the field (think_field: omit) -> None."""
    if ex and ex.get("think_field") == "omit":
        return None
    return bool(L.get("think", False))


def verify_weights(model, expected):
    """sha256 of the model's GGUF blob in the local Ollama store (OLLAMA_MODELS, default ~/.ollama/models) must equal
    `expected` (configs/extract_x_matcher.yaml). Reads no deed data."""
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    name, _, tag = model.partition(":")
    man = json.loads((root / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")).read_bytes())
    layers = [x for x in man["layers"] if x["mediaType"].endswith(".model")]
    if len(layers) != 1:
        raise SystemExit(f"refused: expected one model layer in the manifest, found {len(layers)}")
    h = hashlib.sha256()
    with open(root / "blobs" / layers[0]["digest"].replace(":", "-"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected or "sha256:" + got != layers[0]["digest"]:
        raise SystemExit(f"refused: model blob sha256 {got[:12]}... does not match the config ({str(expected)[:12]}...)")
    return got


def toy_think(model, L, seed):
    """Decision 3: each synthetic input is sent with think false and with no think field; prints counts and the
    synthetic outputs only."""
    stats = Counter()
    for i, t in enumerate(TOY):
        outs = {}
        for label, think in (("no_field", None), ("think_false", False)):
            try:
                names, js = ask(model, windows(t, L["window"], L["max_chars"]), L["num_ctx"], seed, think)
                outs[label] = (js["message"]["content"], names, len((js.get("message") or {}).get("thinking") or ""))
                stats[f"{label}_ok"] += 1
            except Exception as e:  # noqa: BLE001
                body = getattr(e, "read", lambda: b"")()[:200]
                outs[label] = None
                stats[f"{label}_error:{type(e).__name__}"] += 1
                print(f"  toy{i} {label}: {type(e).__name__} {getattr(e, 'code', '')} {body!r}")
        same = outs["no_field"] is not None and outs["think_false"] is not None and outs["no_field"][0] == outs["think_false"][0]
        stats["identical_content"] += same
        stats["thinking_chars>0"] += sum(1 for o in outs.values() if o and o[2])
        print(f"  toy{i}: identical {same} | no_field {outs['no_field'][1] if outs['no_field'] else None} | "
              f"think_false {outs['think_false'][1] if outs['think_false'] else None}")
    accepted = stats["think_false_ok"] == len(TOY)
    verdict = "think false" if accepted and stats["identical_content"] == len(TOY) else "no think field"
    print(f"model {model} | inputs {len(TOY)} | {dict(stats)} | decision 3 -> send {verdict}")
    return verdict


def query(model, text, L, seed, think, info):
    """One deed -> cache record (names stay in the record; nothing is printed)."""
    rec = {"model": model, "ollama_manifest_digest": info.get("ollama_manifest_digest"),
           "prompt_sha256": info["prompt_sha256"], "think": think}
    t1 = time.time()
    try:
        names, js = ask(model, windows(text, L["window"], L["max_chars"]), L["num_ctx"], seed, think)
        n_think = len((js.get("message") or {}).get("thinking") or "")   # length only; the text is dropped
        rec.update(names=names, prompt_tokens=js.get("prompt_eval_count"), out_tokens=js.get("eval_count"),
                   thinking_chars=n_think)
    except Exception as e:  # noqa: BLE001 - type only
        rec.update(names=[], error=type(e).__name__)
    rec["seconds"] = round(time.time() - t1, 2)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "subdiv_match.yaml"))
    ap.add_argument("--design", default=str(DESIGN))
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", help="Ollama tag (default: llm.model in the config)")
    ap.add_argument("--retry-errors", action="store_true", help="test only: the single retry of error records")
    ap.add_argument("--toy-think", action="store_true", help="synthetic think-field comparison (reads no data)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    L, P = cfg["llm"], cfg["paths"]
    model = args.model or L["model"]
    if "cloud" in model.lower():
        raise SystemExit("cloud model tags are not allowed (deed text must stay local)")
    ex_name, ex = design_entry(model, args.design)
    if args.toy_think:
        toy_think(model, L, cfg["seed"])
        return
    test = args.split == "test"
    if args.retry_errors and not test:
        raise SystemExit("--retry-errors is for the test split only")
    if test:
        if args.limit:
            raise SystemExit("refused: --limit is not allowed on test (each extractor runs over all test deeds once)")
        if ex is None or not ex.get("model_sha256"):
            raise SystemExit(f"refused: {model} has no frozen entry with model_sha256 in {Path(args.design).name}")
        verify_weights(model, ex["model_sha256"])
    think = think_setting(L, ex)
    info = {**run_info(model), "think": "omitted" if think is None else think,
            "options": json.dumps(request_options(L["num_ctx"], cfg["seed"]))}
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", model)
    cache = P["derived"] / (f"llm_{args.split}.jsonl" if model == L["model"] else f"llm_{args.split}__{slug}.jsonl")
    res = Path(P["results"]) / (f"step7_llm_{args.split}_run.csv" if model == L["model"]
                                else f"step7_llm_{args.split}_run__{slug}.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    texts = deed_texts(P["splits"][args.split], ocr_root())
    keys = sorted(texts, key=lambda k: (k[0], sorted(k[1])))
    random.Random(cfg["seed"]).shuffle(keys)   # fixed random order: any prefix (or an interrupted run) is a random sample
    if args.limit:
        keys = keys[:args.limit]
    t0 = time.time()

    if args.retry_errors:
        counts = runguard.retry_errors(cache, keys, lambda r: {
            "county": r["county"], "image_ids": r["image_ids"],
            **query(model, texts[runguard.key_of(r)], L, cfg["seed"], think, info)})
        row = {"split": args.split, "phase": "retry", "extractor": ex_name, **info, **counts,
               "seconds": round(time.time() - t0, 1)}
        runguard.append_log(res, row)
        print(row)
        return

    if test:
        runguard.guard_pass(cache, keys)
    done = set()
    if cache.exists():
        done = {runguard.key_of(r) for r in runguard.read_records(cache)}
    stats = Counter()
    with cache.open("a") as fh:
        for i, k in enumerate(keys):
            if k in done:
                stats["cached"] += 1
                continue
            rec = {"county": k[0], "image_ids": sorted(k[1]), **query(model, texts[k], L, cfg["seed"], think, info)}
            if "error" in rec:
                stats[f"error:{rec['error']}"] += 1
            else:
                stats["ok"] += 1
                stats["with_name"] += bool(rec["names"])
                stats["with_thinking"] += rec["thinking_chars"] > 0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(keys)} deeds, {round(time.time() - t0)} s, {dict(stats)}", flush=True)
    row = {"split": args.split, **info, "deeds_requested": len(keys), **stats,
           "seconds": round(time.time() - t0, 1)}
    if test:
        st = runguard.cache_status(cache, keys)
        runguard.append_log(res, {"phase": "pass", "extractor": ex_name, **row, "cache_complete": runguard.is_complete(st),
                                  "cache_errors": st["errors"]})
    else:
        with open(res, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
