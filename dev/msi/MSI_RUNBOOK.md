# MSI runbook: large-LLM extraction on val

For the large LLMs in `configs/extract_x_matcher.yaml` (`gemma4:31b-it-q4_K_M`, `qwen3.8:27b-q4_K_M`). The job runs
its own Ollama server on the compute node (loopback only, `OLLAMA_NO_CLOUD=1`), so deed text never leaves the node.
Val only; test is never run here.

Placeholders, filled in only on MSI or in your shell, never in a committed file:
`<MSI_USER>`, `<MSI_HOST>`, `<CODE_DIR>` (code folder), `<DATA_DIR>` (your copy of the private data),
`<OLLAMA_DIR>` (Ollama binary and models), `<VENV>`, `<GROUP>` (your MSI group, used as the Slurm account).
Use neutral folder names, for example `~/work/llm-eval`.

## 1. One-time setup on MSI

**Folders and permissions.** Only you should be able to read the code folder, the data and the outputs.

```bash
mkdir -p <CODE_DIR> <OLLAMA_DIR>/models
chmod 700 <CODE_DIR> <DATA_DIR>
chmod -R go-rwx <DATA_DIR>
ls -ld <CODE_DIR> <DATA_DIR>      # expect drwx------
getfacl <CODE_DIR> <DATA_DIR>     # expect no extra user or group entries
```

The job runs with `umask 077`, so everything it writes is owner-only.

**Ollama binary.** Use the same version as the local runs (0.40.3). On the login node:
1. Download the Linux amd64 archive for v0.40.3 from the Ollama GitHub releases page.
2. Check its sha256 against the list published with the release.
3. Unpack it into `<OLLAMA_DIR>`.

**Python.** The extraction needs PyYAML, pandas, numpy, pyogrio and geopandas. `pyogrio.read_dataframe()` refuses to
run without geopandas, even though the job reads no geometry. geopandas in turn loads shapely and pyproj, so pin those
too. All versions match the Mac:

```bash
module load python3            # module name: ask the lab; Python 3.11 matches the local runs
python3 -m venv <VENV>
<VENV>/bin/pip install PyYAML==6.0.3 pandas==3.0.6 numpy==2.4.6 pyogrio==0.13.0 \
  geopandas==1.2.0 shapely==2.2.0 pyproj==3.7.2
<VENV>/bin/python -c "import yaml, pandas, numpy, pyogrio, geopandas, shapely, pyproj; print('imports ok')"
```

These are all the job needs. Checked on 2026-11-08 by running the extraction on two synthetic deeds and listing every
third-party package it loaded. The other packages it loaded came in as dependencies of these, or are optional on the
Mac (pyarrow), so they aren't needed. The copied code also mentions spacy, rapidfuzz and scikit-learn, but only in
functions this job never calls, so don't install them.

**Private env file for the job.** Keep it outside the code folder.

```bash
cat > ~/.llm-eval.env <<'EOF'
export SBATCH_ACCOUNT=<GROUP>
OLLAMA_BIN=<OLLAMA_DIR>/bin/ollama
OLLAMA_MODELS=<OLLAMA_DIR>/models
PYTHON=<VENV>/bin/python
EOF
chmod 600 ~/.llm-eval.env
```

**Pull the two models.** This needs internet access, which compute nodes may not have. Use the login node, or an
interactive session if the lab prefers. The download is about 38 GB, so check your quota first (`quota -s`).

```bash
export OLLAMA_MODELS=<OLLAMA_DIR>/models OLLAMA_HOST=127.0.0.1:<FREE_PORT> OLLAMA_NO_CLOUD=1
<OLLAMA_DIR>/bin/ollama serve > /dev/null 2>&1 &
<OLLAMA_DIR>/bin/ollama pull gemma4:31b-it-q4_K_M
<OLLAMA_DIR>/bin/ollama pull qwen3.8:27b-q4_K_M
kill %1
```

## 2. Copy the code to MSI

Run this from the repo root on the laptop. It copies only the files the job needs. Never copy `.env`, `data/`,
`output/`, `results/` or `.git`.

```bash
rsync -av --relative --exclude '__pycache__' \
  configs/subdiv_match.yaml configs/extract_x_matcher.yaml dev/private_paths.py dev/infer_try08.py \
  dev/subdiv_09_llm_extract.py dev/msi/llm_eval.sbatch src/subdiv_match/ \
  <MSI_USER>@<MSI_HOST>:<CODE_DIR>/
```

`configs/extract_x_matcher.yaml` is needed since 2026-11-11: the extraction reads each model's frozen entry from it.

## 3. Private paths on MSI

**`<CODE_DIR>/.env`** (chmod 600). `dev/private_paths.py` reads it, the same way as on the laptop:

```
MP_DATA_ROOT=<DATA_DIR>
MP_RUN_DIR=<the 2025 run folder, absolute or relative to MP_DATA_ROOT>
MP_OCR_ROOT=<the OCR text folder, absolute or relative to MP_DATA_ROOT>
MP_RUN_FILES=<path to run_files.json, outside the code folder>
```

**`run_files.json`** (chmod 600, outside the code folder). The job only needs the val split:

```json
{"splits": {"val": "<val split file, relative to MP_RUN_DIR>"}}
```

Check the paths without printing any data:

```bash
cd <CODE_DIR>
<VENV>/bin/python -c "import os, sys; sys.path.insert(0, 'dev'); import private_paths as pp; os.fspath(pp.ocr_root()); assert os.path.isfile(pp.run_files()['splits']['val']), 'val split missing'; print('paths ok')"
```

## 4. Partition, GPU and account

Nothing needs editing in the job script.
- **Defaults** (public MSI settings): partition `msigpu`, `--gres=gpu:a100:1` (one A100, 40 GB). The job also
  refuses any GPU under 40 GB.
- **Another partition:** pass it on the command line, which overrides the script, e.g.
  `sbatch -p preempt-gpu dev/msi/llm_eval.sbatch <tag>`.
  - A preempted job can simply be resubmitted with the same command. The extraction writes each deed to the cache as
    it goes and skips cached deeds, so the run resumes where it stopped.
- **interactive-gpu:** override both the partition and the GPU type. This partition has only A40 and L40S GPUs (no
  A100), so the default `--gres=gpu:a100:1` would not match:

  ```bash
  sbatch -p interactive-gpu --gres=gpu:a40:1 dev/msi/llm_eval.sbatch <tag> 20
  ```

  Both GPUs have 48 GB, which passes the job's 40 GB check.
- **Memory:** if MSI rejects the job for memory, lower it on the command line, e.g.
  `sbatch --mem=40G dev/msi/llm_eval.sbatch <tag>`. The model runs on the GPU; host RAM mostly holds the Ollama
  server and the OCR text.
- **Account:** comes from `SBATCH_ACCOUNT` in `~/.llm-eval.env`. The job refuses to run if it wasn't set at submit
  time.

## 5. Submit one job at a time

Run each job from `<CODE_DIR>`, after `source ~/.llm-eval.env` in that shell (it sets the account). Wait until
`squeue -u $USER` is empty before submitting the next one.

**a. Setup check (reads no data).**

```bash
LLM_EVAL_DRY_RUN=1 sbatch dev/msi/llm_eval.sbatch gemma4:31b-it-q4_K_M
```

In `llm-eval-<jobid>.out`, check the GPU name and memory, the Ollama version, `quantization Q4_K_M`, the manifest
sha256, and `model blob sha256:… verified`. Send the two hashes to Claude so it can fill in the config entry.

**b. Timing check (20 val deeds).**

```bash
sbatch dev/msi/llm_eval.sbatch gemma4:31b-it-q4_K_M 20
```

Seconds per deed is `seconds` / `ok` in `results/subdiv_match/step7_llm_val_run__gemma4_31b_it_q4_K_M.csv`. The first
deed includes loading the model, so this overestimates. 491 deeds must fit in 2 h with room to spare.

**c. Full val run.**

```bash
sbatch dev/msi/llm_eval.sbatch gemma4:31b-it-q4_K_M
```

It resumes from the 20 cached deeds.

**d. Qwen.** Repeat a–c with `qwen3.8:27b-q4_K_M`. Its run CSV is `step7_llm_val_run__qwen3.8_27b_q4_K_M.csv`.

The `.out` log has counts and timing only. Extracted names stay in the git-ignored
`output/derived/subdiv_match/llm_val__<slug>.jsonl`, and the Ollama server log is in `output/msi_logs/`. Both are
owner-only.

## 6. Copy results back to the laptop (option A, decided by Me 2026-11-07)

Both MSI and the laptop already hold the raw deed data and are within the team, so the derived names files are copied
between them directly. Run these from the laptop, in the repo root.

**a. Aggregate run CSVs** (into the tracked results folder):

```bash
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/results/subdiv_match/step7_llm_val_run__gemma4_31b_it_q4_K_M.csv results/subdiv_match/
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/results/subdiv_match/step7_llm_val_run__qwen3.8_27b_q4_K_M.csv results/subdiv_match/
```

**b. Names files** (derived deed data). Copy them straight into the git-ignored folder, make them owner-only, and
never open them. Claude doesn't open them either.

```bash
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/output/derived/subdiv_match/llm_val__gemma4_31b_it_q4_K_M.jsonl output/derived/subdiv_match/
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/output/derived/subdiv_match/llm_val__qwen3.8_27b_q4_K_M.jsonl output/derived/subdiv_match/
chmod 600 output/derived/subdiv_match/llm_val__gemma4_31b_it_q4_K_M.jsonl output/derived/subdiv_match/llm_val__qwen3.8_27b_q4_K_M.jsonl
git check-ignore -q output/derived/subdiv_match/llm_val__gemma4_31b_it_q4_K_M.jsonl && echo ignored   # must print "ignored"
```

**c. Evaluation (Claude).** Claude runs only `mapprejudice/bin/python dev/subdiv_14_extract_x_matcher.py`, which
prints aggregates (centroid-in, plat accuracy, recall, seconds per deed) and writes
`results/extract_x_matcher/val_*.csv`. It reports `missing_in_cache` and `cached_errors` for each model; both should
be 0.

**d. Delete the names files on MSI** once the evaluation is done and the numbers are in EXPERIMENTS.md:

```bash
ssh <MSI_USER>@<MSI_HOST> 'cd <CODE_DIR> && rm output/derived/subdiv_match/llm_val__gemma4_31b_it_q4_K_M.jsonl output/derived/subdiv_match/llm_val__qwen3.8_27b_q4_K_M.jsonl && ls output/derived/subdiv_match/'
```

The laptop copies stay in `output/derived/subdiv_match/` (git-ignored, chmod 600) like the other extractors' caches.
They are never committed.

## 7. Test run (pre-registered; docs/preregistration_test.md, approved by Me 2026-11-11)

Only after the frozen code is committed and its hash is in section 11 of the pre-registration. Don't edit any frozen
file (the list is `FROZEN` in `dev/subdiv_16_test_extract_x_matcher.py`) until the test evaluation is done; the
evaluation refuses if one changed. The pre-registration itself may change only in the section 11 hash line and the
section 10 deviations log.

**a. Code.** On the laptop, `git status` must show no changes under the frozen paths. Then copy the code as in
section 2.

**b. Test data** (decision 4). Copy only the test split and the OCR pages its deeds use. [Mac]

```bash
python dev/msi/ocr_file_list.py --split test        # counts only; expect missing locally 0 and unmapped 0
```

Then copy the test split file and the OCR pages in `output/private/test_ocr_files.txt`, the same way as the val data,
into the same `<DATA_DIR>` folders. Add the test split to the MSI `run_files.json`:
`{"splits": {"val": "<val split file>", "test": "<test split file>"}}`. Then run the path check of section 3 and
`chmod -R go-rwx <DATA_DIR>`.

**c. Submit, one job at a time** (after `source ~/.llm-eval.env`, from `<CODE_DIR>`). Each test job asks for 5 hours:

```bash
sbatch --time=05:00:00 dev/msi/llm_eval.sbatch gemma4:31b-it-q4_K_M --split=test
```

- **Interrupted job:** resubmit the same command; it resumes from the cache.
- **Cache line error at the start:** if the resubmitted job stops with `cache line N is not valid JSON` (a kill during a
  write), don't fix it yourself. Tell Claude: removing that partial line is allowed but is logged as a deviation.
- **Once the pass is complete:** a further full pass is refused. The last `pass` row of
  `results/subdiv_match/step7_llm_test_run__gemma4_31b_it_q4_K_M.csv` shows `cache_complete True` and `cache_errors`.
  If `cache_errors` is above 0, run the single retry (decision 2) once:

  ```bash
  sbatch --time=05:00:00 dev/msi/llm_eval.sbatch gemma4:31b-it-q4_K_M --split=test --retry-errors
  ```

  If `cache_errors` is 0, no retry is needed.
- **Qwen:** repeat with `qwen3.8:27b-q4_K_M`. Its log is `step7_llm_test_run__qwen3.8_27b_q4_K_M.csv`.

On test, the job refuses a deed limit, and the extraction checks the model blob sha256 against
`configs/extract_x_matcher.yaml` before it reads any deed.

**d. Copy back** (option A, as in section 6). Copy the two test caches and, if they exist, their `.retry.json`
markers to `output/derived/subdiv_match/` (chmod 600, never opened). Copy the two test run logs to
`results/subdiv_match/`.

```bash
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/output/derived/subdiv_match/llm_test__gemma4_31b_it_q4_K_M.jsonl* output/derived/subdiv_match/
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/output/derived/subdiv_match/llm_test__qwen3.8_27b_q4_K_M.jsonl* output/derived/subdiv_match/
chmod 600 output/derived/subdiv_match/llm_test__*
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/results/subdiv_match/step7_llm_test_run__gemma4_31b_it_q4_K_M.csv results/subdiv_match/
scp <MSI_USER>@<MSI_HOST>:<CODE_DIR>/results/subdiv_match/step7_llm_test_run__qwen3.8_27b_q4_K_M.csv results/subdiv_match/
```

**e. Laptop extractors** [Mac], one at a time, never in parallel (section 4.1). Run each command to completion, then
its `--retry-errors` once if its log shows `cache_errors` above 0:

```bash
~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_09_llm_extract.py --split test                       # Qwen2.5-7B
~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_09_llm_extract.py --split test --model gemma4:e4b-it-q4_K_M
~/MapPrejudice/venvs/thesis-hf/bin/python dev/subdiv_15_gliner_extract.py --split test                    # GLiNER
```

**f. Evaluation** [Mac], once, after all five caches are complete:
`mapprejudice/bin/python dev/subdiv_16_test_extract_x_matcher.py`. It prints and writes aggregates only.

**g. Delete the MSI copies** after the evaluation: the test caches and markers, the test run logs, the test split
and the test OCR pages under `<DATA_DIR>`.

Cleanup note (2026-11-17 to 11-22): on 2026-11-17, right after the evaluation ran at 13:02, I deleted on MSI
`<DATA_DIR>` (`rm -rf`: the val and test splits, the OCR pages and `run_files.json`), the `*.jsonl` caches in
`<CODE_DIR>/output/derived/subdiv_match/` (`rm -f`; the folder was last changed at 13:30) and the Ollama models folder
`<OLLAMA_DIR>/models` (`rm -rf`). There were no retry markers. So at the 7g check (2026-11-21/22) the test caches, test
split and test OCR pages were already gone, and the laptop test caches are the only copies. They match their run logs
(2,436 deeds, 0 errors, 0 with thinking), have sha256 `6496ea3e…9d95` (Gemma 4 31B) and `92229e52…cf27` (Qwen3.8-27B),
and are in my Time Machine backup on an external drive (2026-11-22). The two MSI test run logs matched the laptop
copies by sha256 (committed in c5a60f9) and were then deleted, as were the two uploaded page lists and the empty data
folder. Only the two val run logs remain in `<CODE_DIR>/results/subdiv_match/`; no deed data or derived names are left
on MSI.
