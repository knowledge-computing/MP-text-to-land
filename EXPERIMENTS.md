# Experiments log

Aggregate numbers only. No deed text or examples.
Primary metric: centroid-in over eligible deeds (val 271; test 2,435). Test run once, as pre-registered
(docs/preregistration_test.md), on 2026-11-17.

## Status
| Family | Method | Val done | Test done | Notes |
|---|---|---|---|---|
| Rules | regex-only extraction | [ ] | [ ] | lower bound, optional |
| Classic NER | try08 (main) | [Y] | [Y] | test reused (2026-10-09 run) |
| Transformer NER | roberta-base or ModernBERT | [ ] | [ ] | if time |
| Zero-shot NER | GLiNER zero-shot (gliner_large-v2.5) | [Y] | [Y] | must do |
| Open LLM (old gen) | Qwen2.5-7B q4 | [Y] | [Y] | laptop, Ollama |
| Open LLM (small) | Gemma 4 E4B | [Y] | [Y] | laptop |
| Open LLM (large) | Gemma 4 31B | [Y] | [Y] | MSI |
| Open LLM (new gen) | Qwen3.8-27B | [Y] | [Y] | MSI |
| Open LLM (optional) | gpt-oss-20b | [ ] | [ ] | MSI, if time; not in the pre-registration |

Matchers: paper_b1 / matcher_v1.0
All LLMs: same frozen prompt, temperature 0, thinking mode recorded.

## Decisions
- 2026-11-06: no query fusion; one extractor per pipeline, then one matcher.
- 2026-11-06: model list = Qwen2.5-7B, Gemma 4 E4B, Gemma 4 31B,
  Qwen3.8-27B (+ gpt-oss-20b optional), GLiNER zero-shot.
- 2026-11-11: test pre-registration approved (12 pipelines, McNemar + Holm); frozen code b16a305.

## Findings so far (val)
| Extractor | + paper_b1 | + matcher_v1.0 | Extraction recall | s/deed |
|---|---|---|---|---|
| try08 NER | 57.20 | 77.86 | 84.7% | 0.54 (CPU) |
| Qwen2.5-7B | 51.29 | 74.91 | 91.4% | 3.38 (Metal) |
| Gemma 4 E4B (q4_K_M) | 52.40 | 75.28 | 89.4% | 2.57 (Metal) |
| Gemma 4 31B (q4_K_M) | 55.72 | 80.07 | 88.2% | 3.91 (MSI A40; 2 jobs, 20 + 471 deeds) |
| Qwen3.8-27B (q4_K_M) | 55.35 | 78.23 | 87.4% | 3.84 (MSI A40; 2 jobs, 20 + 471 deeds) |
| GLiNER large v2.5 (zero-shot) | 48.71 | 67.90 | 83.9% | 0.89 (MPS) |

- Centroid-in over 271 eligible val deeds; matcher_v1.0 deeds: Gemma 4 31B 217, Qwen3.8-27B 212,
  try08 211, Gemma 4 E4B 204, Qwen2.5-7B 203, GLiNER 184.
- matcher_v1.0 adds 19.2–24.4 points over paper_b1 for all six extractors (GLiNER +19.2 to
  Gemma 4 31B +24.4). Relative to an exact-match baseline, the choice of matching method had a larger
  effect than the choice among the tested extractors (matcher_v1.0 range across extractors: 184–217 deeds,
  12.2 points).
- The top three (Gemma 4 31B, Qwen3.8-27B, try08) are within 6 deeds; no
  significance test on val.
- Large LLMs beat small ones (217/212 vs 204/203 deeds) through name quality,
  not recall: with a name, 85.8%/84.1% vs 79.7%/79.0%, while they find a name
  in about as many eligible deeds (253/252 vs 256/257).
- GLiNER finds names in the fewest deeds (232), but when it finds one,
  accuracy is close to the small LLMs (79.3%).
- try08's val number is optimistic (val was its dev set; 435/491 val deeds
  contain a training sentence). The LLMs and GLiNER were not trained on
  these deeds.
- Per county (matcher_v1.0, eligible deeds): Milwaukee (123) large LLMs 85.4%,
  try08 82.9%, small LLMs 78.9–80.5%; Anoka (85) LLMs 77.6–83.5%, try08 76.5%;
  Olmsted (54) try08 best at 74.1% (LLMs 59.3–72.2%). Dakota (8) and
  Sherburne (1) are too small to compare.
- Paper criterion (no geometry) ranges 9.6–16.6% and does not rank the
  pipelines like centroid-in (Qwen2.5-7B lowest at 9.6%).
- Speed is not comparable across hardware: try08 0.54 s (CPU), GLiNER
  0.89 s (MPS), small LLMs 2.6–3.4 s (Metal), large LLMs 3.8–3.9 s (A40).

## Test results (pre-registered, run once 2026-11-17)
`dev/subdiv_16_test_extract_x_matcher.py` → `results/extract_x_matcher/test_*.csv`, `test_summary.json`.
Centroid-in over all 2,435 eligible test deeds (deeds in brackets); no deviations from the pre-registration.

| Extractor | + paper_b1 | + matcher_v1.0 | Plat acc (v1.0) | Coverage | s/deed |
|---|---|---|---|---|---|
| try08 NER | 53.14 (1,294) | **76.02** (1,851) | 75.24 | 88.75% | 0.48 (CPU) |
| Qwen2.5-7B | 48.75 (1,187) | **75.56** (1,840) | 75.32 | 97.29% | 3.00 (Metal) |
| Gemma 4 E4B | 47.84 (1,165) | **74.58** (1,816) | 74.09 | 96.51% | 2.16 (Metal) |
| Gemma 4 31B | 54.13 (1,318) | **80.66** (1,964) | 80.41 | 96.18% | 3.83 (MSI A40) |
| Qwen3.8-27B | 53.22 (1,296) | **80.16** (1,952) | 79.92 | 95.89% | 3.83 (MSI A40) |
| GLiNER large v2.5 | 49.40 (1,203) | **70.14** (1,708) | 68.87 | 89.29% | 0.70 (MPS) |

- try08 + matcher_v1.0 reproduced `results/matcher/test_v1.json` exactly (114 counts), as required.
- Family 1 (each extractor + matcher_v1.0 vs try08 + matcher_v1.0; exact McNemar, Holm m = 5; b = only the
  extractor right, c = only try08 right):
  - Gemma 4 31B +4.64 points (b 197, c 84; p 1.26e-11, Holm 5.04e-11) and Qwen3.8-27B +4.15 (188 / 87; 1.07e-09,
    Holm 3.22e-09): significantly better than try08.
  - GLiNER −5.87 (149 / 292; 9.02e-12, Holm 4.51e-11): significantly worse.
  - Qwen2.5-7B −0.45 (176 / 187; p 0.60, Holm 0.60) and Gemma 4 E4B −1.44 (171 / 206; 0.0798, Holm 0.16): no
    significant difference.
- Family 2 (matcher_v1.0 vs paper_b1 within each extractor; Holm m = 6): matcher_v1.0 is better for all six,
  +20.7 to +26.9 points, with 507–657 deeds only it gets right vs 1–3 only paper_b1 gets right (largest Holm p 1.5e-148).
- Per county (matcher_v1.0, centroid-in): Milwaukee (1,369) large LLMs 89.0 / 88.8, try08 81.6; Anoka (518)
  Qwen3.8 90.0, Gemma E4B 88.4, Gemma 31B 88.0, try08 82.4, GLiNER 57.1; Olmsted (347) 56.5–63.4 for all;
  Dakota (64) 57.8–65.6; Sherburne (137) try08 44.5 vs 19.0–22.6 for the others.
- Paper criterion (raw strings, no geometry): try08 16.19% (the published 16.18%), the others 8.19–12.89%.
- Seconds per deed are not comparable across hardware; try08's test timing is from its 2026-10-09 inference run.
