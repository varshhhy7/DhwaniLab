# DhwaniLab

Benchmarking pretrained Indic ASR models on Telugu speech, under a single evaluation protocol.

![Language](https://img.shields.io/badge/python-3.11%2B-blue)
![Notebooks](https://img.shields.io/badge/jupyter-notebooks-orange)
![Dataset](https://img.shields.io/badge/dataset-IndicVoices%20Telugu-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Overview

DhwaniLab evaluates publicly released Telugu ASR models on the **same 3,295 utterances** from the IndicVoices validation split, scored through an **identical WER pipeline**, so that reported differences reflect the models rather than the harness.

Every model is run without an external language model, with per-utterance predictions written to CSV so that any metric — WER today, OIWER and CER next — can be recomputed from stored text without re-running inference.

## Results

**IndicVoices Telugu · validation split · 3,295 utterances · 37,350 reference words**

| Model | WER | OI-WER | Gain | S | D | I | Decoding |
|---|---|---|---|---|---|---|---|
| **IndicConformer** (`indic-conformer-600m-multilingual`) | **27.84%** | **26.08%** | −1.76 | 8,066 | 1,318 | 1,014 | greedy CTC |
| **IndicWav2Vec** (`te.pt`, Fairseq) | 54.56% | 51.64% | −2.93 | 15,955 | 2,915 | 1,508 | greedy CTC |
| **IndicWhisper** (`whisper-medium-te_alldata_multigpu`) | 56.74% | 53.41% | −3.35 | 14,410 | 4,702 | 2,079 | autoregressive |

IndicConformer leads by **26.7 percentage points** over the next best model. The ranking is
identical under both metrics — OI-WER reorders nothing. S/D/I are the WER-column counts.

Variants come from deterministic rules **plus** a locally-run `gemma3:4b`, filtered so a hallucinated
variant cannot forgive a real error — see [OI-WER methodology](#oi-wer-methodology).

All three models were scored against byte-identical references (verified: 3,295/3,295 matching) over
the same 37,350 reference words.

> **Denominator note.** OI-WER must strip the 62 `<unintelligible>` transcriber markers, which removes
> 61 tokens and takes the denominator from 37,350 to **37,289**. The gains above are therefore measured
> against a normalized-text WER on that same 37,289-word denominator (27.85% / 54.58% / 56.76%), not
> against the raw figures. Comparing across two denominators would misattribute part of the change to
> the metric.

### Reading the results

Two caveats matter for interpretation, and neither is visible in the WER column alone:

**Domain match.** IndicConformer is AI4Bharat's own model, trained on IndicVoices. The IndicVoices validation split is therefore in-domain for it, while IndicWhisper was fine-tuned on a different Telugu mixture and faces a domain shift. The defensible claim is *"IndicConformer substantially outperforms IndicWhisper on IndicVoices Telugu validation"* — not that it is the better Telugu ASR model in general. Establishing that would require a corpus neutral to both.

**Decoding asymmetry.** IndicConformer ran greedy CTC with no language model. IndicWhisper used autoregressive generation, whose decoder carries an implicit internal LM. The comparison is between the models as shipped and as typically used, not between their acoustic encoders in isolation.

**Error profiles differ in kind, not only in degree.** IndicWhisper's 4,627 deletions and 2,004 insertions are the familiar Whisper failure mode — dropped content and hallucinated repetition on short utterances — rather than uniform mis-recognition. IndicConformer produced 2 empty predictions across the split.

## Repository structure

```
DhwaniLab/
├── notebooks/
│   ├── indicvoices_baseline.ipynb      IndicWhisper baseline
│   ├── indicconformer_baseline.ipynb   IndicConformer baseline
│   ├── indicwav2vec_baseline.ipynb     IndicWav2Vec baseline (isolated Fairseq env)
│   └── model_comparison.ipynb          Integrity checks + unified WER recomputation
├── scripts/
│   ├── render_flow_diagram.py          Pipeline diagram as SVG/PNG (correct text)
│   ├── generate_flow_image.py          Same diagram via an image model (text garbles)
│   └── project_flow_prompt.txt         Whiteboard-style prompt for image models
├── oiwer/
│   ├── oiwer.py                        Variant-aware alignment engine
│   ├── rules.py                        Telugu ITN, merge/split, normalization
│   ├── variant_filter.py               Guardrail for LLM-proposed variants
│   ├── gen_variants_ollama.py          Variant generation via local Ollama
│   ├── score_llm.py                    Rules vs LLM vs combined scoring
│   ├── crosscheck_official.py          Our engine vs AI4Bharat's scorer
│   ├── run_oiwer.py                    Rule-only ablation runner
│   ├── test_oiwer.py                   Engine unit tests (9)
│   ├── test_variant_filter.py          Filter unit tests (11)
│   └── vendor/                         AI4Bharat oiwer_core.py + Telugu prompt (MIT)
├── results/
│   ├── indicwhisper/  indicconformer/  indicwav2vec/
│   │   └── indicvoices_telugu_valid.csv
│   └── oiwer/
│       ├── variants_gemma3_4b.jsonl    LLM-proposed variants, one line per utterance
│       ├── oiwer_llm_full.csv/.json    Rules vs LLM vs combined, all three models
│       └── engine_crosscheck.csv       Agreement with AI4Bharat's scorer
├── reports/
│   ├── gen_report.py                   Builds the PDF from the results files
│   └── DhwaniLab_Progress_Report.pdf
└── README.md
```

### Result files

`indicvoices_telugu_valid.csv` holds one row per utterance — 3,295 rows per model:

| Column | Meaning |
|---|---|
| `index` | position in the streamed validation split, 0–3294 |
| `speaker_id` | speaker identifier from IndicVoices |
| `duration` | utterance length in seconds |
| `reference` | ground-truth transcript, whitespace-normalized |
| `prediction` | model output, whitespace-normalized |
| `wer` | per-utterance WER |
| `substitutions`, `deletions`, `insertions` | per-utterance error counts |

Because predictions are stored as text, any additional metric — OIWER, CER, per-speaker breakdowns — can be computed from these files without re-running inference on a GPU.

## Evaluation protocol

Identical across every model:

| Component | Choice |
|---|---|
| Dataset | `ai4bharat/IndicVoices`, config `telugu`, split `valid`, streamed |
| Reference field | `normalized` |
| Text normalization | whitespace collapse only — no case folding, no punctuation stripping |
| Metric | corpus WER via `jiwer.process_words`, aggregated over the full split |
| Aggregation | corpus-level (total errors ÷ total reference words), never a mean of per-utterance WERs |
| Language model | none |
| Audio | mono, resampled to 16 kHz |

Corpus-level aggregation is deliberate: averaging per-utterance WER lets short utterances dominate and is not comparable to published figures.

Each run writes `index, speaker_id, duration, reference, prediction, wer, substitutions, deletions, insertions` per utterance, checkpointing to CSV every 25 samples. Runs resume by index, so an interrupted evaluation continues rather than restarting.

## Models

**IndicWhisper** — `whisper-medium-te_alldata_multigpu`, distributed as a standard HuggingFace checkpoint via AI4Bharat's object store. Loads with `WhisperProcessor` and `WhisperForConditionalGeneration`; decoded with `language="te", task="transcribe"`.

**IndicConformer** — `ai4bharat/indic-conformer-600m-multilingual`, a **gated** HuggingFace repository requiring access approval. Loaded with `trust_remote_code=True`; the remote code resolves to an ONNX Runtime model (`model_onnx.IndicASRModel`), so weights live outside the PyTorch module graph — `model.parameters()` is legitimately empty and `.to(device)` is a no-op. Called as `model(waveform, "te", "ctc")`.

**IndicWav2Vec** — AI4Bharat never published Telugu weights in HuggingFace format
(`ai4bharat/indicwav2vec_v1_telugu` is an empty repository), so the only Telugu artifact is a Fairseq
`.pt` checkpoint (3.53 GB) requiring Python 3.10 and torch 1.13. That stack cannot coexist with the
runtime the other two models use, so it runs in an **isolated uv virtual environment** driven by a
subprocess decoder, with audio pre-dumped to disk so the old environment never touches `datasets`.
Decoded with greedy CTC (Viterbi), no language model.

Note that AI4Bharat's official IndicWav2Vec pipeline uses **KenLM + lexicon decoding**. We
deliberately skipped that so all three models are compared without an external LM; our 54.56% is
therefore an LM-free figure and is not directly comparable to published numbers that use LM decoding.

This packaging difference generalizes: models released as artifacts for a stable framework are cheap
to benchmark, while models welded to their training framework drag that framework's entire dependency
era into the evaluation runtime.

## OI-WER methodology

OI-WER accepts a **set of valid spellings** at each reference position instead of a single string, so
a prediction matching any accepted variant counts as a hit rather than a substitution. The engine
(`oiwer/oiwer.py`) aligns a segment lattice against the prediction by dynamic programming; a variant
may span multiple words, so one reference word can match two predicted words and vice versa. The
denominator stays the original reference word count, keeping WER and OI-WER directly comparable.

Variants come from two independent sources, scored separately so each contribution is visible.

### Source A — deterministic rules

| Rule | Effect |
|---|---|
| Inverse text normalization | Telugu number words ↔ digits (`ఎనిమిది` ↔ `8`), incl. compounds to lakh/crore |
| Compound merge / split | `మూడువేల` ↔ `మూడు వేల` — pure string concatenation, no lexicon needed |
| Annotation tags | `<unintelligible>` (62 markers) removed from references |
| Surface normalization | Unicode NFC, punctuation, replacement chars, Latin case folding |

### Source B — a local LLM, behind a filter

The paper generates variants with Gemini-2.5-Pro and has native speakers review them. With no LLM API
available, `gemma3:4b` — the largest Gemma that fits in 5 GB of VRAM — was run locally over all 3,295
references via Ollama, prompted with **AI4Bharat's own expert-verified Telugu guideline examples**
(vendored from their repository into `oiwer/vendor/prompt_telugu.py`).

A 4B model is not a substitute for Gemini plus human review, and it behaved accordingly: alongside
genuine spelling variants it produced translations, synonyms and invented words. This matters more
than it looks, because **a bad variant can only ever make a score better** — alignment takes the
minimum over variants, so an unfiltered hallucination silently forgives a real error. Every proposal
therefore passes three gates (`oiwer/variant_filter.py`) before it can touch a score:

| Gate | Rule | Why |
|---|---|---|
| **Anchoring** | The group is dropped entirely unless it contains the reference word copied exactly | Kills groups where the model rewrote or invented the original; keeps the reference word count fixed so the denominator cannot drift |
| **Consonant skeleton** | Strip Telugu vowel signs, virama and anusvara from both; skeletons must agree within 1 edit | Spelling variation preserves the skeleton; a different word does not |
| **Edit ratio** | Character edit distance ≤ 34% of the reference word's length, or identical once spaces are removed | Admits compound splits and merges; rejects distant rewrites |

Accepted: `రీచ్ → రిచ్` (same skeleton, one edit), `తెలంగాణలో → తెలంగాణ లో` (compound split).
Rejected: `కావడానికి → చేయడానికి` (different word), `ప్లేసెస్ → ప్రదేశాలు` (translation), `నేను → నా`.

| LLM proposal stage | Count |
|---|---|
| Word groups proposed by `gemma3:4b` | 36,613 |
| Groups anchored to a reference span | 32,782 |
| Alternative spellings offered | 12,247 |
| Alternatives that passed the filter | 6,765 |
| Reference spans actually augmented | 6,658 |

The filter rejected **45%** of what the model proposed.

### Rules vs LLM

`python oiwer/score_llm.py`

| Configuration | IndicConformer | IndicWav2Vec | IndicWhisper |
|---|---|---|---|
| WER, raw text (37,350 words) | 27.839% | 54.560% | 56.736% |
| WER, normalized (37,289 words) | 27.845% | 54.576% | 56.762% |
| OI-WER, rules only | 26.925% | 52.485% | 54.338% |
| OI-WER, LLM variants only | 26.834% | 53.498% | 55.606% |
| OI-WER, rules + LLM | **26.083%** | **51.643%** | **53.407%** |

Substitutions removed relative to the normalized-text baseline:

| Variant source | IndicConformer | IndicWav2Vec | IndicWhisper |
|---|---|---|---|
| Rules only | 184 | 486 | 618 |
| LLM only | 306 | 336 | 388 |
| Rules + LLM | 462 | 770 | 961 |

The two sources are **nearly additive** — they correct different things. Rules handle number
formatting and word-boundary placement; the LLM reaches matra and vowel-length differences and
loanword spellings, which no string rule can produce without a lexicon. ITN removed 124 substitutions
from IndicWhisper and **zero** from IndicConformer, which never emits digits: the digit-formatting
penalty was entirely one-sided.

### Validation against AI4Bharat's own scorer

AI4Bharat released their reference implementation after this work began. It is vendored at
`oiwer/vendor/oiwer_core.py` (MIT, upstream commit recorded) and both engines were run over identical
inputs across the full split — `python oiwer/crosscheck_official.py`:

| Configuration | Model | Our engine | AI4Bharat scorer | Δ |
|---|---|---|---|---|
| no variants | IndicConformer | 27.845% | 27.845% | **0.000** |
| no variants | IndicWav2Vec | 54.576% | 54.576% | **0.000** |
| no variants | IndicWhisper | 56.762% | 56.762% | **0.000** |
| rules + LLM | IndicConformer | 26.083% | 26.898% | 0.815 |
| rules + LLM | IndicWav2Vec | 51.643% | 53.584% | 1.942 |
| rules + LLM | IndicWhisper | 53.407% | 55.392% | 1.984 |

**With variants switched off the two engines agree exactly on all 3,295 utterances** — the strongest
available evidence that our alignment and counting are correct.

With variants enabled our figures are lower, and the reason is structural rather than a disagreement
about the data: `oiwer_core.py` greedily commits to a single multi-word variant *before* running its
dynamic program, scoring candidates by how many hypothesis words they contain. Ours searches all
variants inside the program, so it is optimal by construction. The effect is not marginal — under
their scorer the rule variants deliver **zero** improvement for IndicConformer and IndicWav2Vec
(27.845% and 54.576%, identical to the no-variant baseline), because the greedy step never selects
the digit or merged form.

### Honest limits

- **This is still an approximation of the published metric.** The paper pairs Gemini-2.5-Pro with
  review by 61 native speakers across seven variation categories. This pairs hand-written rules with
  an unreviewed 4B local model behind a deliberately strict filter — which rejects genuine variants
  along with bad ones. Read these figures as a **lower bound** on the true OI-WER gain.
- **No native speaker has reviewed anything yet** — neither the Telugu number lexicon nor the 6,765
  accepted LLM variants. This is the cheapest available improvement in confidence; the accepted
  variants are stored and can be reviewed as a flat list.
- **The LLM variants are unvalidated against a gold set.** We can show what the filter rejects, but
  without AI4Bharat's own annotations we cannot measure what fraction of true variants it misses.

## Reproducing

**Prerequisites**

- A GPU runtime (Colab or equivalent) and a HuggingFace account
- Accepted terms for `ai4bharat/IndicVoices`
- Approved access to `ai4bharat/indic-conformer-600m-multilingual`

**Order**

1. `notebooks/indicvoices_baseline.ipynb` — IndicWhisper
2. `notebooks/indicconformer_baseline.ipynb` — IndicConformer
3. `notebooks/indicwav2vec_baseline.ipynb` — IndicWav2Vec (isolated Fairseq environment)
4. `notebooks/model_comparison.ipynb` — verifies all three CSVs and recomputes WER (CPU only)
5. `python oiwer/run_oiwer.py` — rule-only OI-WER ablation (CPU only, runs in seconds)
6. `python oiwer/test_oiwer.py` — 9 engine tests, incl. exact reproduction of the published WER figures
7. `python oiwer/test_variant_filter.py` — 11 filter tests, incl. the specific gemma hallucinations
8. `python oiwer/gen_variants_ollama.py` — LLM variants (see below; ~2 h, resumable, CPU/GPU local)
9. `python oiwer/score_llm.py` — rules vs LLM vs combined, all three models (CPU, ~1 min)
10. `python oiwer/crosscheck_official.py` — agreement with AI4Bharat's scorer (CPU, needs `numpy` + `indic-nlp-library`)

Steps 8–10 are optional: `results/oiwer/variants_gemma3_4b.jsonl` is committed, so step 9 runs
without regenerating anything.

**LLM variant generation.** Requires [Ollama](https://ollama.com) with `ollama pull gemma3:4b`. The
generator is resumable — it keys on utterance index and skips what is already in the output file, so
an interrupted run continues rather than restarting. Throughput depends on `OLLAMA_NUM_PARALLEL`; the
default of 1 serialises requests regardless of `--workers`. On an RTX 3050 6 GB (33/35 layers
offloaded, 4 parallel slots) the full split took roughly two hours at 0.5 utterances/second.

Variants depend only on the **reference**, never on a model's predictions, so they are generated once
and reused for all three models — and remain valid for any model evaluated later.

`model_comparison.ipynb` scores over the **intersection** of indices present across all three CSVs, so it produces an honest comparison even while one run is still in progress.

**Environment note.** IndicConformer's inference path needs only `transformers`, `datasets`, `onnxruntime`, `huggingface_hub` and `jiwer`. Installing `nemo_toolkit[asr]` is unnecessary and actively harmful in Colab: it pins `protobuf==3.20.3`, which breaks `transformers.modeling_utils`, downgrades `transformers`, and cascades into `huggingface_hub`/`datasets` import failures.

## Roadmap

- [x] IndicWhisper baseline on IndicVoices Telugu
- [x] IndicConformer baseline on IndicVoices Telugu
- [x] Unified comparison with reference-integrity verification
- [x] OI-WER, rule-based approximation following [arXiv:2603.00941](https://arxiv.org/abs/2603.00941)
- [x] OI-WER with LLM-generated variants (`gemma3:4b`, filtered) over the full split
- [x] Validation against AI4Bharat's released `oiwer_core.py` — exact agreement without variants
- [ ] Native-speaker review of the accepted variants and the Telugu number lexicon
- [ ] Full OI-WER with a frontier LLM + human review (all 7 categories)
- [ ] Error analysis: WER vs duration, per-speaker breakdown, substitution pairs, code-mixing and numerals
- [ ] CER alongside WER
- [ ] Paired significance testing over utterances
- [x] IndicWav2Vec baseline (isolated Fairseq environment)
- [ ] IndicConformer RNNT decoding as a third configuration
- [ ] IndicWav2Vec with KenLM + lexicon decoding, matching AI4Bharat's official pipeline
- [x] Commit per-utterance result CSVs for full reproducibility

## References

- [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) — AI4Bharat
- [IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) — AI4Bharat
- [IndicWhisper / Vistaar](https://github.com/AI4Bharat/vistaar) — AI4Bharat
- [jiwer](https://github.com/jitsi/jiwer) — WER computation
- [Towards Orthographically-Informed Evaluation of Speech Recognition Systems for Indian Languages](https://arxiv.org/abs/2603.00941) — defines OIWER
- [OIWER benchmarking framework](https://github.com/AI4Bharat/OIWER-Orthographically-Informed-Benchmarking-for-ASR) — AI4Bharat reference implementation

## Acknowledgements

All models and datasets are the work of [AI4Bharat](https://ai4bharat.iitm.ac.in/). This repository contains evaluation code only.
