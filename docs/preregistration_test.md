# Pre-registration: extraction method × matcher on the test split

**Status: APPROVED by Me on 2026-11-11 (version 2, plus the §4.1 laptop rule). Nothing in this document has been
run on test.** Before any test run, the code changes in §9 are made and tested on synthetic data, the frozen code
is committed, and its commit hash is filled in (§11).

Design and aggregate numbers only. This file holds no deed text, no extracted names and no private paths.

## 1. Question

How much does the extraction method (which reads the deed OCR text and returns subdivision name strings) and the
matcher (which maps those strings to a recorded plat) each contribute to locating a covenanted deed's plat?

Each pipeline is **one extraction method followed by one matcher**. There is no query fusion: a pipeline never
combines names from two extractors.

## 2. Test set

- **Split:** the frozen July 2025 6:2:2 test split, 2,436 unique deeds from five counties: Anoka, Dakota,
  Milwaukee, Olmsted and Sherburne. Washington has no test deeds.
- **Eligible deeds:** the deed's county has a partner plat layer, and the deed geometry is single-part. This is
  the same eligibility as `matcher_v1.0`'s test run: 2,435 of 2,436 deeds.

  | County | Eligible deeds |
  |---|---|
  | Anoka | 518 |
  | Dakota | 64 |
  | Milwaukee | 1,369 |
  | Olmsted | 347 |
  | Sherburne | 137 |
- **Deed key:** (county, set of image ids), as in all val runs. Every extractor is evaluated on the same 2,435
  eligible deeds.
- **How test has been used before** (stated so readers can judge how independent test is):
  1. The published paper evaluated its pipeline on this split.
  2. On 2026-10-08/09, before the test freeze, exploratory matcher work ran on test: an ablation, error
     diagnosis, a soft section bonus and a case-sensitivity check. It is labelled exploratory and did not choose
     any configuration.
  3. On 2026-10-09, try08 inference was re-run once on test to check that the saved predictions reproduce. All
     2,436 deeds were identical; no metrics were computed.
  4. On 2026-10-09, `matcher_v1.0` was evaluated once on test with try08 NER: 76.02% centroid-in (1,851 / 2,435),
     in `results/matcher/test_v1.json`.

  The LLM extractors, GLiNER and the `paper_b1` matcher have never been run on test.
- **Leakage notes:**
  - try08 was trained on the train split and selected on val, so its val numbers are optimistic. On test it saw
    no test deeds, but 10 test deeds share OCR pages with non-test deeds.
  - The LLMs and GLiNER are used zero-shot. They were not trained or tuned on any project deeds, and the prompt's
    few-shot examples are synthetic.

## 3. Pipelines (12)

| # | Extractor | Matcher | Test status |
|---|---|---|---|
| 1 | try08 NER | paper_b1 | to run (on the saved try08 test predictions; no new inference) |
| 2 | try08 NER | matcher_v1.0 | **done 2026-10-09 (76.02%); reused, not re-run** (see §4.2) |
| 3 | Qwen2.5-7B | paper_b1 | to run |
| 4 | Qwen2.5-7B | matcher_v1.0 | to run |
| 5 | Gemma 4 E4B | paper_b1 | to run |
| 6 | Gemma 4 E4B | matcher_v1.0 | to run |
| 7 | Gemma 4 31B | paper_b1 | to run |
| 8 | Gemma 4 31B | matcher_v1.0 | to run |
| 9 | Qwen3.8-27B | paper_b1 | to run |
| 10 | Qwen3.8-27B | matcher_v1.0 | to run |
| 11 | GLiNER large v2.5 | paper_b1 | to run |
| 12 | GLiNER large v2.5 | matcher_v1.0 | to run |

Each extractor runs **once** over the test deeds. Both matchers then read that one extraction, so each LLM and
GLiNER extraction serves two pipelines.

### 3.1 Matchers (frozen)

- **`paper_b1`:** a simple exact-match baseline. **It is not the paper's method**; the paper reported only the
  paper criterion (§5). It casefolds each predicted string, collapses whitespace and compares it with the partner
  plat primary names in the deed's own county. The first predicted string with a hit wins. Punctuation is kept
  and abbreviations are not expanded. Code: `ExactIndex` in `dev/subdiv_14_extract_x_matcher.py`.
- **`matcher_v1.0`:** frozen on 2026-10-09 (`src/matcher/frozen.py`, guarded by `test/test_matcher_frozen.py`).
  Settings:
  - rapidfuzz `ratio` scorer, n3 normalization (learned rules, minimum frequency 5), cutoff 70;
  - own county, primary names, fixed tie rule, no section constraint;
  - step-3 rules pinned by SHA-256: 41 learned pairs, 75 canonical tokens, 1 merge.

### 3.2 Extractors (frozen)

All four LLMs share these settings:
- **Prompt:** `dev/subdiv_09_llm_extract.py` (system text, 3 synthetic few-shot examples, keyword windows),
  prompt sha256 `0006b8791de5884df660b18e812f0025ecdc3b04e39d8cdd8633b90cf149469a`.
- **Input:** windows of ±250 characters around legal-description keywords, capped at 3,000 characters.
- **Request:** `format: "json"`, `think: false`, options `{temperature: 0, seed: 0, num_ctx: 4096}`. top_k and
  top_p are not sent. Qwen2.5-7B is the one exception for `think`; see §8, decision 3.
- **Server:** Ollama 0.40.3 on a loopback address only; deed text never leaves the machine.

| Extractor | Model / tag | Weights sha256 | Other pins | Hardware (as on val) |
|---|---|---|---|---|
| try08 NER | spaCy v3 + roberta-base encoder, transition NER head | (model already run; saved test predictions, reproduced 2026-10-09) | spaCy 3.7.5, spacy-transformers 1.3.9 | — (no new inference) |
| Qwen2.5-7B | `qwen2.5:7b-instruct-q4_K_M` | `2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730` | manifest `156bcbf7…b4c3` | Apple M1 Pro, 16 GB, Metal |
| Gemma 4 E4B | `gemma4:e4b-it-q4_K_M` | `370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c` | manifest `df5980f1…531b` | Apple M1 Pro, 16 GB, Metal |
| Gemma 4 31B | `gemma4:31b-it-q4_K_M` | `cd20f2b3f2e30239023189c04e65754b0cc7c38959665ac727461ee470df16cf` | manifest `859aadd7…98ca` | MSI, NVIDIA A40 46 GB |
| Qwen3.8-27B | `qwen3.8:27b-q4_K_M` | `f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d` | manifest `c3742ec5…07cc` | MSI, NVIDIA A40 46 GB |
| GLiNER large v2.5 | `gliner-community/gliner_large-v2.5@3d6d1760be1c591069f85f207fced9214df8b15f` | `b0a0f6f2109367a7157d7a0a00f8383d80ba232c1fd9c7fa88ee653011226c64` | label `[subdivision name]`, threshold 0.5, flat NER, 384-token chunks with 64 overlap, same keyword windows; torch 2.14.1, gliner 0.2.29, transformers 5.16.1 | Apple M1 Pro, MPS |

The full manifest hashes and other settings are in `configs/extract_x_matcher.yaml`, which is the authoritative
record. Every run checks the weights sha256 before it reads any deed. An LLM run refuses any tag that isn't
Q4_K_M or is a cloud model.

## 4. Execution protocol

### 4.1 Running each extractor once

- Each extractor runs over all 2,436 test deeds exactly once, on the hardware listed in §3.2.
  - Gemma 4 31B and Qwen3.8-27B run on MSI. Each test job asks for `--time=05:00:00`, and they run one at a time.
  - The three laptop extractors (Qwen2.5-7B, Gemma 4 E4B, GLiNER) also run one at a time, never in parallel, so
    their timing stays clean (Me, 2026-11-11).
  - If an A40 is not available on MSI, any GPU with at least 40 GB may be used. The GPU model is recorded.
  - GPU numerics can differ slightly between GPU models. That is accepted; the run is not repeated.
- **Interruptions:** an interrupted job is resubmitted with the same command and resumes from its cache. Deeds
  already in the cache are never re-queried. A partial last line in the cache (from a kill during a write) may be
  removed before resuming; this is logged as a deviation.
- **Failed deeds:** a deed whose extraction fails (for example a timeout or invalid JSON) is written to the cache
  as an error record.
  - After the full pass, and before any scoring, each deed with an error record is retried **once** with
    identical settings. Only those deeds are retried.
  - A successful retry replaces the error record.
  - A deed that fails again counts as "no extraction", which is a miss under every metric.
  - The run reports the errors from the first pass, the deeds retried, and the errors left after the retry.
- **No re-runs:** no extractor is re-run after any test metric has been seen, for any reason.
- **Per extractor, the run records:**
  - number of deeds;
  - errors;
  - deeds with thinking text (must be 0);
  - seconds per deed;
  - hardware;
  - Ollama version;
  - weights sha256 and prompt sha256.

### 4.2 try08 + matcher_v1.0 (reused; §8, decision 1)

This pipeline's test result is the 2026-10-09 run, and it is not run again. That run saved aggregates only. The
paired tests in Family 1 (§6) need its per-deed outcomes, so these are **recomputed** with the frozen code from
the same saved try08 test predictions. This is a deterministic reproduction, not a new run. Before it is used, the
reproduction must match `results/matcher/test_v1.json` exactly:
- 1,851 / 2,435 centroid-in;
- 2,162 deeds with a prediction;
- 2,142 deeds with a top-1;
- every per-county count.

If any number differs, the analysis stops and Me is told. Nothing is adjusted.

## 5. Metrics

All metrics are computed by `matcher_v1.0`'s pre-registered metric code (`dev/eval_matcher_val.evaluate`, applied
to test) for every pipeline.

- **Primary: centroid-in (b).** The share of all 2,435 eligible test deeds whose deed centroid lies inside the
  predicted plat. A deed with no extraction, no top-1 or a failed extraction counts as a miss.
- **Secondary** (descriptive only; no significance tests):
  - Centroid-in (a): the denominator is eligible deeds with at least one extracted string.
  - Plat accuracy vs the geometric weak label, (b) and (a). Weak labels come from volunteer polygons, so they are
    not independent ground truth.
  - Contains rate and intersects rate (b).
  - Extraction coverage: the share of deeds with at least one extracted string, over all 2,436 test deeds and
    over the 2,435 eligible ones.
  - Paper criterion: at least one predicted string reaches fuzz.ratio ≥ 70 with a plat name, using raw strings
    and candidates from all counties, with no geometry. It is computed per extractor (it doesn't depend on the
    matcher). The denominator is test deeds with a prediction. For try08 this reproduces the published 16.18%
    (350 / 2,162).
  - Per county: every metric above for each of the five counties. Dakota (64) and Sherburne (137) are small.
  - Seconds per deed and hardware. These are not comparable across hardware.

## 6. Statistical analysis

- **Per-deed outcome:** 1 if centroid-in, else 0 (the primary metric), over the 2,435 eligible deeds. Every
  comparison is paired on the same deeds.
- **Test:** exact two-sided McNemar.
  - Notation: b = deeds that only pipeline X gets right; c = deeds that only pipeline Y gets right; n = b + c.
  - p = min(1, 2 · Σ_{k=0}^{min(b,c)} C(n, k) · 0.5^n), and p = 1 when n = 0.
- **Holm correction within each family**, α = 0.05:
  - Sort the family's m p-values in ascending order: p(1) ≤ … ≤ p(m).
  - Adjusted p(i) = max over j ≤ i of min(1, (m − j + 1) · p(j)).
  - A comparison is significant when its adjusted p is below 0.05.
- **Family 1 (extractor), m = 5:** with `matcher_v1.0` fixed, each of Qwen2.5-7B, Gemma 4 E4B, Gemma 4 31B,
  Qwen3.8-27B and GLiNER vs try08 NER (pipelines 4, 6, 8, 10, 12 vs 2).
- **Family 2 (matcher), m = 6:** within each of the six extractors, `matcher_v1.0` vs `paper_b1` (pipelines 2 vs
  1, 4 vs 3, …, 12 vs 11).
- **Reported for every comparison:**
  - both centroid-in percentages and the difference in points;
  - b and c (the direction comes from these counts);
  - the raw p and the Holm-adjusted p.
- Secondary metrics and per-county numbers get no significance tests.
- No other comparisons are tested. Any further comparison is labelled exploratory.

**What val suggests** (exploratory, not hypotheses). These numbers are centroid-in (b) over 271 eligible val
deeds:

| Extractor | + paper_b1 | + matcher_v1.0 | Extraction coverage |
|---|---|---|---|
| try08 NER (optimistic on val) | 57.20 | 77.86 | 84.7% |
| Qwen2.5-7B | 51.29 | 74.91 | 91.4% |
| Gemma 4 E4B | 52.40 | 75.28 | 89.4% |
| Gemma 4 31B | 55.72 | 80.07 | 88.2% |
| Qwen3.8-27B | 55.35 | 78.23 | 87.4% |
| GLiNER large v2.5 | 48.71 | 67.90 | 83.9% |

On val, `matcher_v1.0` added 19.2 to 24.4 points for every extractor. The top three extractors were within 6
deeds of each other. **Both families are two-sided** (Me, 2026-11-11), even though val points to `matcher_v1.0`
in Family 2.

## 7. Reporting

- All 12 pipelines are reported with every metric in §5, whatever the outcome, including any that fail or that
  perform worse than expected.
- The results go in `results/extract_x_matcher/test_*.csv` and a summary JSON, which hold aggregates only.
- Names extracted from test deeds stay in git-ignored caches and are never printed.
- Any deviation from this document is logged in §10 with its date and reason.

## 8. Decisions (Me, 2026-11-11)

1. **Reusing try08 + matcher_v1.0:** its per-deed outcomes are recomputed from the saved try08 test predictions
   with the frozen code. The recomputation must reproduce `results/matcher/test_v1.json` exactly: 1,851 / 2,435
   centroid-in, 2,162 deeds with a prediction, 2,142 with a top-1, and every county. Otherwise the analysis
   stops (§4.2).
2. **Failed extractions:** one retry with identical settings, only for deeds with an error record, before any
   scoring. Deeds that still fail count as misses (§4.1).
3. **Qwen2.5-7B and `think`:** first, a toy test on synthetic text only. Each synthetic input is sent twice:
   once with `think: false` and once with no `think` field.
   - Use `think: false` on test only if Ollama accepts it for this model **and** every toy output is identical
     between the two requests.
   - Otherwise send no `think` field, as on val.
   - The toy-test result and the choice are recorded in the config before the test run.
   - **Result (2026-11-11):** Ollama accepted `think: false` for all 6 inputs, the message content was identical
     for 6 of 6, and there were 0 thinking characters. So Qwen2.5-7B sends `think: false` on test, like the other
     LLMs (`think_toy_test` in `configs/extract_x_matcher.yaml`).
4. **Test data on MSI:** the test split and the OCR texts of the 2,436 test deeds are copied to MSI for Gemma 4
   31B and Qwen3.8-27B. They are owner-only (chmod 700 folders), and their names caches come back to the laptop
   the same way as on val (option A). The MSI copies are deleted after the evaluation.
5. **Walltime:** `--time=05:00:00` for each large-model test job. At val speed (about 3.9 s per deed), 2,436
   deeds take about 2.6 h.

Both families of tests stay two-sided (§6).

## 9. Code changes (made 2026-11-11; tested on synthetic data only)

- **`dev/subdiv_09_llm_extract.py` and `dev/subdiv_15_gliner_extract.py`:**
  - Allow `--split test` only with a run-once guard. Resuming from the cache is allowed; starting a second full
    run is refused.
  - Add a one-time `--retry-errors` mode for decision 2. It re-queries only deeds with an error record, refuses
    to run twice, and logs the error counts before and after.
- **Qwen2.5-7B toy test** (decision 3): synthetic text only, with and without `think: false`. The comparison is
  printed as counts and identical/different only.
- **`dev/msi/llm_eval.sbatch`:** take the split as an argument (the default stays val). Large-model test jobs are
  submitted with `--time=05:00:00` on the command line, so the script's 2-hour default for val stays as it is.
- **`dev/msi/MSI_RUNBOOK.md`:** add the test steps: copy the test split and test OCR file list (decision 4),
  submit, copy the caches back, and delete the MSI copies.
- **A new test evaluation script** that:
  - refuses to run if its summary file exists;
  - refuses if any extractor cache is incomplete or still waiting for its retry;
  - reproduces and checks the 2026-10-09 try08 + `matcher_v1.0` numbers (§4.2);
  - evaluates all 12 pipelines;
  - computes the McNemar and Holm statistics in §6;
  - writes aggregates only.
- **Unit tests** for the McNemar and Holm code on toy numbers.
- The frozen code is committed before the test run, and its commit hash is recorded in §11.

Where they are:
- `src/subdiv_match/runguard.py` holds the run-once guard and the retry; `src/subdiv_match/stats.py` holds McNemar
  and Holm.
- The test modes are in `dev/subdiv_09_llm_extract.py` and `dev/subdiv_15_gliner_extract.py`. On test, the LLM run
  also checks the GGUF blob sha256 against the config before reading any deed.
- The split and retry options are in `dev/msi/llm_eval.sbatch`. `dev/msi/ocr_file_list.py` lists the OCR pages to
  copy, and `dev/msi/MSI_RUNBOOK.md` section 7 has the steps.
- The evaluation is `dev/subdiv_16_test_extract_x_matcher.py`. It refuses to run unless the frozen commit is in §11,
  the frozen files are unchanged since then, and this document differs from its copy in the frozen commit only in
  the §11 hash line and the §10 log. Its `--toy` mode runs on synthetic data.
- The unit tests are `StatsTest` and `RunGuardTest` in `test/test_subdiv_match.py`.

## 10. Deviations log

- 2026-11-17: no deviations. All five extractors ran once over the 2,436 test deeds with 0 error records, so no
  retry was needed; Gemma 4 31B and Qwen3.8-27B ran on MSI A40s with `--time=05:00:00`; the laptop extractors ran
  one at a time. try08 + `matcher_v1.0` reproduced `results/matcher/test_v1.json` exactly, and the evaluation ran once.
- 2026-11-22, MSI cleanup (§8 decision 4): on 2026-11-17, right after the evaluation (13:02), I deleted the MSI
  copies of the val and test splits and OCR pages, the two test caches and the Ollama models. The two MSI test run logs
  matched the committed laptop copies by sha256 and were deleted on 2026-11-22; nothing from test is left on MSI. The
  laptop test caches used by the evaluation are now the only copies (they match their run logs: 2,436 deeds, 0 errors;
  I backed them up). Not a deviation: decision 4 only requires deletion after the evaluation.

## 11. Approval

- Approved by Me: 2026-11-11
- Frozen code commit: b16a305566d88d97fffe1fc11fd7b1dc37ac911a

The frozen code and this document are committed together; a second commit then fills in only the hash above. After
that, only §10 may change.
