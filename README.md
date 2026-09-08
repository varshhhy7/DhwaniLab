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

| Model | WER | OI-WER | S | D | I | Decoding |
|---|---|---|---|---|---|---|
| **IndicConformer** (`indic-conformer-600m-multilingual`) | **27.84%** | **26.92%** | 8,072 | 1,315 | 1,011 | greedy CTC |
| **IndicWav2Vec** (`te.pt`, Fairseq) | 54.56% | 52.48% | 15,915 | 2,935 | 1,528 | greedy CTC |
| **IndicWhisper** (`whisper-medium-te_alldata_multigpu`) | 56.74% | 54.34% | 14,560 | 4,627 | 2,004 | autoregressive |

IndicConformer leads by **26.7 percentage points** over the next best model. The ranking is
identical under both metrics. OI-WER here is a rule-based approximation &mdash; see
[OI-WER methodology](#oi-wer-methodology).

All three models were scored against byte-identical references (verified: 3,295/3,295 matching) over the same 37,350 reference words.

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
│   ├── model_comparison.ipynb          Integrity checks + unified WER recomputation
│   └── oiwer_evaluation.ipynb          Optional LLM-based variant generation
├── oiwer/
│   ├── oiwer.py                        Variant-aware alignment engine
│   ├── rules.py                        Telugu ITN, merge/split, normalization
│   ├── run_oiwer.py                    Ablation runner
│   └── test_oiwer.py                   Unit tests
├── results/
│   ├── indicwhisper/  indicconformer/  indicwav2vec/
│   │   └── indicvoices_telugu_valid.csv
│   └── oiwer/
│       └── oiwer_comparison.csv
├── reports/
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

Variants are generated by **deterministic rules**, not an LLM:

| Rule | Effect |
|---|---|
| Inverse text normalization | Telugu number words ↔ digits (`ఎనిమిది` ↔ `8`), incl. compounds to lakh/crore |
| Compound merge / split | `మూడువేల` ↔ `మూడు వేల` — pure string concatenation, no lexicon needed |
| Annotation tags | `<unintelligible>` (60 utterances) removed from references |
| Surface normalization | Unicode NFC, punctuation, replacement chars, Latin case folding |

Per-rule ablation (`python oiwer/run_oiwer.py`):

| Configuration | IndicConformer | IndicWav2Vec | IndicWhisper |
|---|---|---|---|
| WER (baseline) | 27.839% | 54.560% | 56.736% |
| + strip `<unintelligible>` | 27.845% | 54.576% | 56.765% |
| + surface normalization | 27.845% | 54.576% | 56.762% |
| + ITN numbers | 27.839% | 54.386% | 56.236% |
| + merge / split = **OI-WER** | **26.925%** | **52.485%** | **54.338%** |

The gain comes almost entirely from substitutions (232 / 522 / 661 removed), the expected signature
since accepting a variant can only convert a wrong word into a correct one. ITN removed 124
substitutions from IndicWhisper and **zero** from IndicConformer, which never emits digits — the
digit-formatting penalty was entirely one-sided.

**This is an approximation, not a reproduction of the published metric.** It covers 2 of the paper's
7 variation categories; the remaining five (matra/diacritic, loanword spellings, phonetic, ligature,
sandhi) need a lexicon or an LLM. It is closer to the paper's WER-SN baseline than to full OI-WER,
which is why our 0.9–2.4 point gains are below the paper's 6.3-point average. A local `gemma3:4b`
was evaluated for variant generation and rejected: roughly half its proposed Telugu variants were
hallucinated or omitted.

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
5. `python oiwer/run_oiwer.py` — OI-WER with per-rule ablation (CPU only, runs in seconds)
6. `python oiwer/test_oiwer.py` — 9 tests, incl. exact reproduction of the published WER figures

`model_comparison.ipynb` scores over the **intersection** of indices present in both CSVs, so it produces an honest comparison even while one run is still in progress.

**Environment note.** IndicConformer's inference path needs only `transformers`, `datasets`, `onnxruntime`, `huggingface_hub` and `jiwer`. Installing `nemo_toolkit[asr]` is unnecessary and actively harmful in Colab: it pins `protobuf==3.20.3`, which breaks `transformers.modeling_utils`, downgrades `transformers`, and cascades into `huggingface_hub`/`datasets` import failures.

## Roadmap

- [x] IndicWhisper baseline on IndicVoices Telugu
- [x] IndicConformer baseline on IndicVoices Telugu
- [x] Unified comparison with reference-integrity verification
- [x] OI-WER, rule-based approximation following [arXiv:2603.00941](https://arxiv.org/abs/2603.00941)
- [ ] Full OI-WER with LLM-generated + human-reviewed variants (all 7 categories)
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
