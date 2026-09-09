import argparse
import csv
import functools
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import oiwer_core
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

from oiwer import align_segments, Counts
from rules import build_spans, canonical, strip_tags
from score_llm import load_predictions, load_variants, MODELS, VARIANTS_PATH
from variant_filter import anchor_slots, merge_into_spans


@functools.lru_cache(maxsize=None)
def _normalizer(lang_code):
    return IndicNormalizerFactory().get_normalizer(lang_code)


def _fast_normalize(sentence, lang_code):
    import string
    sentence = sentence.translate(
        str.maketrans("", "", string.punctuation + "।۔'-")
    )
    return _normalizer(lang_code).normalize(sentence)


oiwer_core.normalize_sentence = _fast_normalize


def variant_lists(reference, llm_slots, use_rules, use_llm):
    spans, tokens = build_spans(reference, enable_itn=use_rules)

    if use_llm and llm_slots:
        accepted, _ = anchor_slots(tokens, llm_slots, canonical)
        if accepted:
            spans, _ = merge_into_spans(spans, accepted)

    return [[" ".join(v) for v in variants] for _, _, variants in spans], tokens


def run(rows, variants, use_rules, use_llm, limit=None):
    ours = Counts()
    theirs = Counts()

    for n, row in enumerate(rows):
        if limit and n >= limit:
            break

        reference = str(row["reference"])
        prediction = str(row["prediction"])
        slots = variants.get(int(row["index"])) if use_llm else None

        lists, tokens = variant_lists(reference, slots, use_rules, use_llm)
        pred_tokens = [t for t in canonical(strip_tags(prediction)).split() if t]

        segments = [[v.split() for v in group] for group in lists]
        _, subs, dels, ins, hits = align_segments(
            segments, pred_tokens, allow_merge_split=use_rules
        )
        ours = ours + Counts(hits, subs, dels, ins, len(tokens))

        if lists:
            _, _, _, _, errors, _ = oiwer_core.oiwer(
                " ".join(pred_tokens), lists, "telugu"
            )
            insertions, deletions, substitutions = errors
        else:
            insertions = len(pred_tokens)
            deletions = substitutions = 0
        theirs = theirs + Counts(
            0, substitutions, deletions, insertions, len(tokens)
        )

    return ours, theirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=str(VARIANTS_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "results" / "oiwer" / "engine_crosscheck.csv"))
    args = parser.parse_args()

    variants = load_variants(args.variants)
    available = [
        m for m in MODELS
        if (ROOT / "results" / m / "indicvoices_telugu_valid.csv").exists()
    ]

    cases = [
        ("no variants", False, False),
        ("rules only", True, False),
        ("rules + LLM", True, True),
    ]

    print("engine cross-check against AI4Bharat oiwer_core.py")
    print("variant records:", len(variants), " utterance limit:", args.limit or "all")
    print()
    print("{:14} {:16} {:>10} {:>10} {:>8}".format(
        "configuration", "model", "ours %", "official %", "delta"))
    print("-" * 64)

    written = []
    for label, use_rules, use_llm in cases:
        for model in available:
            rows = load_predictions(model)
            ours, theirs = run(rows, variants, use_rules, use_llm, args.limit)
            print("{:14} {:16} {:10.3f} {:10.3f} {:8.3f}".format(
                label, model, ours.wer * 100, theirs.wer * 100,
                (theirs.wer - ours.wer) * 100))
            written.append([
                model, label, round(ours.wer * 100, 3), round(theirs.wer * 100, 3),
                ours.substitutions, theirs.substitutions,
                ours.deletions, theirs.deletions,
                ours.insertions, theirs.insertions,
                ours.reference_words,
            ])
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "configuration", "ours_percent", "official_percent",
            "ours_substitutions", "official_substitutions",
            "ours_deletions", "official_deletions",
            "ours_insertions", "official_insertions", "reference_words",
        ])
        writer.writerows(written)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
