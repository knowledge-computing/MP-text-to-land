#!/bin/bash
# LLM subdivision-name extraction on VAL with a local Ollama server inside the job (see dev/msi/MSI_RUNBOOK.md).
# One model per job; submit one job at a time, from the code folder:
#
#     source ~/.llm-eval.env                                            # exports SBATCH_ACCOUNT (private)
#     sbatch dev/msi/llm_eval.sbatch <ollama_tag> [limit]
#     e.g.  sbatch dev/msi/llm_eval.sbatch gemma4:31b-it-q4_K_M 20     # timing check, 20 val deeds
#     LLM_EVAL_DRY_RUN=1 sbatch dev/msi/llm_eval.sbatch <tag>           # setup checks only, reads no data
#     sbatch -p preempt-gpu dev/msi/llm_eval.sbatch <tag>              # other partition (command line wins)
#     sbatch --time=05:00:00 dev/msi/llm_eval.sbatch <tag> --split=test               # TEST pass (pre-registered)
#     sbatch --time=05:00:00 dev/msi/llm_eval.sbatch <tag> --split=test --retry-errors  # the single retry
#
# Options after the tag: a number (val deed limit; refused on test), --split=val|test (default val), --retry-errors
# (test only). Test rules (docs/preregistration_test.md): no limit, resume allowed, second full pass refused, one retry.
#
# Partition and GPU defaults are MSI's public settings (msigpu, one A100 40 GB). The account (group) is never in this
# file: sbatch reads it from SBATCH_ACCOUNT, exported by the private env file (LLM_EVAL_ENV, default ~/.llm-eval.env),
# which also sets OLLAMA_BIN, OLLAMA_MODELS and PYTHON for the job.
# The extraction prints counts and timing only; extracted names stay in the git-ignored cache under output/.
# Plain bash (no login shell, no modules): everything is called by full path, and the submit shell's conda / library
# settings are cleared below, because they break system tools on the compute node.
#SBATCH --job-name=llm-eval
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=llm-eval-%j.out
#SBATCH --error=llm-eval-%j.out

set -euo pipefail
umask 077                                   # everything this job writes is readable by the owner only

# Clean environment: sbatch copies the submit shell's variables into the job; conda's libraries and Python settings
# must not leak into system tools, Ollama or the venv Python.
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH PYTHONHOME
for v in $(compgen -e); do
  case "$v" in CONDA_*|_CE_*|MAMBA_*) unset "$v" ;; esac
done
CLEAN_PATH=""
IFS=: read -r -a _parts <<< "${PATH:-}"
for d in "${_parts[@]}"; do
  case "$d" in ""|*conda*|*mamba*) ;; *) CLEAN_PATH="${CLEAN_PATH:+$CLEAN_PATH:}$d" ;; esac
done
export PATH="${CLEAN_PATH:-/usr/bin:/bin}"
unset CLEAN_PATH _parts d v

TAG="${1:?usage: sbatch dev/msi/llm_eval.sbatch <ollama_tag> [limit] [--split=val|test] [--retry-errors]}"
shift
LIMIT="" SPLIT=val RETRY=""
for a in "$@"; do
  case "$a" in
    --split=val|--split=test) SPLIT="${a#--split=}" ;;
    --retry-errors) RETRY=--retry-errors ;;
    *) [[ "$a" =~ ^[0-9]+$ ]] || { echo "refused: unknown option $a"; exit 2; }; LIMIT="$a" ;;
  esac
done
case "$TAG" in *cloud*) echo "refused: cloud tag"; exit 2 ;; esac
case "$TAG" in *q4_K_M) ;; *) echo "refused: use a fully specified q4_K_M tag"; exit 2 ;; esac
if [ "$SPLIT" = test ] && [ -n "$LIMIT" ]; then echo "refused: no limit on test"; exit 2; fi
if [ -n "$RETRY" ] && [ "$SPLIT" != test ]; then echo "refused: --retry-errors is for --split=test only"; exit 2; fi
echo "split $SPLIT ${LIMIT:+limit $LIMIT }${RETRY}"

[ -n "${SBATCH_ACCOUNT:-}" ] || { echo "refused: source the env file (SBATCH_ACCOUNT) before sbatch"; exit 2; }
ENV_FILE="${LLM_EVAL_ENV:-$HOME/.llm-eval.env}"
[ -f "$ENV_FILE" ] || { echo "missing env file (set LLM_EVAL_ENV)"; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${OLLAMA_BIN:?set in the env file}" "${OLLAMA_MODELS:?set in the env file}" "${PYTHON:?set in the env file}"

cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the code folder}"
[ -f dev/subdiv_09_llm_extract.py ] || { echo "submit from the code folder"; exit 2; }

# GPU with >= 40 GB
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ "${GPU_MB:-0}" -ge 40000 ] || { echo "refused: GPU has ${GPU_MB:-0} MiB (< 40 GB)"; exit 3; }

# Ollama server for this job only: loopback, per-job port (the node may be shared), no cloud models
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
export OLLAMA_HOST="127.0.0.1:${PORT}" OLLAMA_NO_CLOUD=1 OLLAMA_MODELS
mkdir -p output/msi_logs output/derived/subdiv_match results/subdiv_match   # not copied to MSI; the extraction writes here
"$OLLAMA_BIN" serve > "output/msi_logs/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true; wait "$OLLAMA_PID" 2>/dev/null || true' EXIT

# Wait for the server (Python urllib, no curl): up to 120 s; stop early if the server process died
"$PYTHON" - "$OLLAMA_PID" <<'EOF'
import json, os, sys, time, urllib.request
pid, host, limit = int(sys.argv[1]), os.environ["OLLAMA_HOST"], 120
log = f"output/msi_logs/ollama-{os.environ.get('SLURM_JOB_ID', 'local')}.log"
t0 = time.time()
while True:
    try:
        version = json.load(urllib.request.urlopen(f"http://{host}/api/version", timeout=5)).get("version")
        break
    except Exception:  # noqa: BLE001 - server not up yet
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(f"ollama server exited before answering on {host}; see {log}")
    if time.time() - t0 > limit:
        sys.exit(f"timeout: ollama server did not answer on {host} within {limit} s; see {log}")
    time.sleep(2)
if not version:
    sys.exit(f"ollama server on {host} returned no version")
print(f"ollama {version} (server ready after {time.time() - t0:.0f} s)")
EOF

# The model must already be in OLLAMA_MODELS (pulled beforehand, see the runbook): local weights, Q4_K_M, sha256 ok
"$PYTHON" - "$TAG" <<'EOF'
import hashlib, json, os, sys, urllib.request
tag = sys.argv[1]
host = "http://" + os.environ["OLLAMA_HOST"]
req = urllib.request.Request(host + "/api/show", data=json.dumps({"model": tag}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    show = json.load(urllib.request.urlopen(req, timeout=60))
except Exception as e:
    sys.exit(f"model {tag} not available locally ({type(e).__name__}); pull it first")
if any("remote" in k for k in show):
    sys.exit("refused: remote model")
d = show.get("details") or {}
print("model", tag, "| quantization", d.get("quantization_level"), "| format", d.get("format"),
      "| params", d.get("parameter_size"), "| capabilities", ",".join(show.get("capabilities") or []))
if d.get("quantization_level") != "Q4_K_M":
    sys.exit("refused: not Q4_K_M")
name, _, t = tag.partition(":")
root = os.environ["OLLAMA_MODELS"]
man_path = os.path.join(root, "manifests", "registry.ollama.ai", "library", name, t)
raw = open(man_path, "rb").read()
man = json.loads(raw)
print("manifest sha256", hashlib.sha256(raw).hexdigest())
models = [layer for layer in man["layers"] if layer["mediaType"].endswith(".model")]
if len(models) != 1:
    sys.exit(f"refused: expected one GGUF model layer, found {len(models)}")
for layer in models:
    if layer["mediaType"].endswith(".model"):
        h = hashlib.sha256()
        with open(os.path.join(root, "blobs", layer["digest"].replace(":", "-")), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        ok = "sha256:" + h.hexdigest() == layer["digest"]
        print("model blob", layer["digest"], layer["size"], "verified" if ok else "MISMATCH")
        if not ok:
            sys.exit(1)
EOF

if [ "${LLM_EVAL_DRY_RUN:-0}" = 1 ]; then
  echo "dry run: setup checks passed; no data read"
  exit 0
fi

# Extraction: same frozen prompt and options as the local runs; think false comes from configs/subdiv_match.yaml.
# On test, subdiv_09 also checks the blob sha256 against configs/extract_x_matcher.yaml before reading any deed.
# The run CSV (results/subdiv_match/step7_llm_val_run__<slug>.csv) records ollama_version, the manifest digest and the GPU.
"$PYTHON" -u dev/subdiv_09_llm_extract.py --split "$SPLIT" --model "$TAG" ${LIMIT:+--limit "$LIMIT"} $RETRY
