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

| Model | Corpus WER | Substitutions | Deletions | Insertions | Decoding |
|---|---|---|---|---|---|
| **IndicConformer** (`indic-conformer-600m-multilingual`) | **27.84%** | 8,072 | 1,315 | 1,011 | greedy CTC |
| **IndicWhisper** (`whisper-medium-te_alldata_multigpu`) | 56.74% | 14,560 | 4,627 | 2,004 | autoregressive |

IndicConformer leads by **28.9 percentage points**.

Both models were scored against byte-identical references (verified: 3,295/3,295 matching) over the same 37,350 reference words.

### Reading the results

Two caveats matter for interpretation, and neither is visible in the WER column alone:

**Domain match.** IndicConformer is AI4Bharat's own model, trained on IndicVoices. The IndicVoices validation split is therefore in-domain for it, while IndicWhisper was fine-tuned on a different Telugu mixture and faces a domain shift. The defensible claim is *"IndicConformer substantially outperforms IndicWhisper on IndicVoices Telugu validation"* — not that it is the better Telugu ASR model in general. Establishing that would require a corpus neutral to both.

**Decoding asymmetry.** IndicConformer ran greedy CTC with no language model. IndicWhisper used autoregressive generation, whose decoder carries an implicit internal LM. The comparison is between the models as shipped and as typically used, not between their acoustic encoders in isolation.

**Error profiles differ in kind, not only in degree.** IndicWhisper's 4,627 deletions and 2,004 insertions are the familiar Whisper failure mode — dropped content and hallucinated repetition on short utterances — rather than uniform mis-recognition. IndicConformer produced 2 empty predictions across the split.

## Repository structure

```
DhwaniLab/
├── notebooks/
│   ├── indicvoices_baseline.ipynb      IndicWhisper baseline, end to end
│   ├── indicconformer_baseline.ipynb   IndicConformer baseline, end to end
│   └── model_comparison.ipynb          Integrity checks + unified WER recomputation
├── results/
│   ├── indicwhisper/
│   │   └── indicvoices_telugu_valid.csv
│   └── indicconformer/
│       ├── indicvoices_telugu_valid.csv
│       └── indicconformer_summary.csv
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

### Excluded: IndicWav2Vec

IndicWav2Vec was evaluated and dropped. AI4Bharat never published Telugu weights in HuggingFace format — `ai4bharat/indicwav2vec_v1_telugu` is an empty repository — leaving a Fairseq `.pt` checkpoint as the only Telugu artifact. That checkpoint requires Python 3.10, torch 1.13, omegaconf 2.0.6 and hydra 1.0.7, which cannot coexist with the modern runtime the rest of the pipeline uses. Running it would require an isolated virtual environment, a subprocess decoder, a pre-dumped audio corpus and hand-written CTC decoding — substantial scaffolding that would not change the finding.

This is a packaging constraint, not a modeling one, and it generalizes: models released as artifacts for a stable framework are cheap to benchmark, while models welded to their training framework drag that framework's entire dependency era into the evaluation runtime.

## Reproducing

**Prerequisites**

- A GPU runtime (Colab or equivalent) and a HuggingFace account
- Accepted terms for `ai4bharat/IndicVoices`
- Approved access to `ai4bharat/indic-conformer-600m-multilingual`

**Order**

1. `notebooks/indicvoices_baseline.ipynb` — IndicWhisper, writes `results/indicwhisper/indicvoices_telugu_valid.csv`
2. `notebooks/indicconformer_baseline.ipynb` — IndicConformer, writes `results/indicconformer/indicvoices_telugu_valid.csv`
3. `notebooks/model_comparison.ipynb` — verifies both CSVs and recomputes WER side by side (CPU is sufficient)

`model_comparison.ipynb` scores over the **intersection** of indices present in both CSVs, so it produces an honest comparison even while one run is still in progress.

**Environment note.** IndicConformer's inference path needs only `transformers`, `datasets`, `onnxruntime`, `huggingface_hub` and `jiwer`. Installing `nemo_toolkit[asr]` is unnecessary and actively harmful in Colab: it pins `protobuf==3.20.3`, which breaks `transformers.modeling_utils`, downgrades `transformers`, and cascades into `huggingface_hub`/`datasets` import failures.

## Roadmap

- [x] IndicWhisper baseline on IndicVoices Telugu
- [x] IndicConformer baseline on IndicVoices Telugu
- [x] Unified comparison with reference-integrity verification
- [ ] OIWER (Orthographically-Informed WER), following [arXiv:2603.00941](https://arxiv.org/abs/2603.00941)
- [ ] Error analysis: WER vs duration, per-speaker breakdown, substitution pairs, code-mixing and numerals
- [ ] CER alongside WER
- [ ] Paired significance testing over utterances
- [ ] IndicConformer RNNT decoding as a third configuration
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
