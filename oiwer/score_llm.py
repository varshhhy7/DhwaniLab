import argparse
import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oiwer import align_segments, Counts
from rules import build_spans, canonical, strip_tags
from variant_filter import anchor_slots, merge_into_spans

MODELS = ["indicconformer", "indicwav2vec", "indicwhisper"]
VARIANTS_PATH = ROOT / "results" / "oiwer" / "variants_gemma3_4b.jsonl"


def load_predictions(model):
    path = ROOT / "results" / model / "indicvoices_telugu_valid.csv"
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_variants(path):
    table = {}
    if not Path(path).exists():
        return table
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record.get("slots"):
                table[int(record["index"])] = record["slots"]
    return table


def build_case(reference, llm_slots, use_rules, use_llm, stats):
    spans, tokens = build_spans(reference, enable_itn=use_rules)

    if use_llm and llm_slots:
        accepted, slot_stats = anchor_slots(tokens, llm_slots, canonical)
        for key, value in slot_stats.items():
            stats[key] = stats.get(key, 0) + value
        if accepted:
            spans, merged = merge_into_spans(spans, accepted)
            stats["spans_augmented"] = stats.get("spans_augmented", 0) + merged

    return [variants for _, _, variants in spans], tokens


def score(rows, variants, use_rules, use_llm, raw=False, subset=None):
    total = Counts()
    stats = {}

    for row in rows:
        if subset is not None and int(row["index"]) not in subset:
            continue
        reference = str(row["reference"])
        prediction = str(row["prediction"])

        if raw:
            ref_tokens = reference.split()
            pred_tokens = prediction.split()
            segments = [[[t]] for t in ref_tokens]
        else:
            slots = variants.get(int(row["index"])) if use_llm else None
            segments, ref_tokens = build_case(
                reference, slots, use_rules, use_llm, stats
            )
            pred_tokens = [t for t in canonical(strip_tags(prediction)).split() if t]

        _, subs, dels, ins, hits = align_segments(
            segments, pred_tokens, allow_merge_split=use_rules
        )
        total = total + Counts(hits, subs, dels, ins, len(ref_tokens))

    return total, stats


CONFIGS = [
    ("WER (published protocol)", dict(raw=True, use_rules=False, use_llm=False)),
    ("WER (normalized text)", dict(raw=False, use_rules=False, use_llm=False)),
    ("OIWER rules only", dict(raw=False, use_rules=True, use_llm=False)),
    ("OIWER LLM only", dict(raw=False, use_rules=False, use_llm=True)),
    ("OIWER rules + LLM", dict(raw=False, use_rules=True, use_llm=True)),
]

BASELINE = CONFIGS[1][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=str(VARIANTS_PATH))
    parser.add_argument("--out", default=str(ROOT / "results" / "oiwer" / "oiwer_llm_comparison.csv"))
    parser.add_argument("--subset", action="store_true",
                        help="score only the utterances that have LLM variants")
    args = parser.parse_args()

    variants = load_variants(args.variants)
    available = [
        m for m in MODELS
        if (ROOT / "results" / m / "indicvoices_telugu_valid.csv").exists()
    ]

    subset = set(variants) if args.subset else None

    print("variant records loaded:", len(variants))
    print("scoring scope:", "LLM-covered subset" if subset else "full corpus")
    print()
    print("{:28} {:16} {:>9} {:>7} {:>7} {:>7} {:>9}".format(
        "configuration", "model", "WER %", "S", "D", "I", "refwords"))
    print("-" * 90)

    table = {}
    stats_by_config = {}

    for label, options in CONFIGS:
        for model in available:
            rows = load_predictions(model)
            total, stats = score(rows, variants, subset=subset, **options)
            table[(label, model)] = total
            if stats:
                stats_by_config[(label, model)] = stats
            print("{:28} {:16} {:9.3f} {:7} {:7} {:7} {:9}".format(
                label, model, total.wer * 100, total.substitutions,
                total.deletions, total.insertions, total.reference_words))
        print()

    print("=" * 90)
    print("deltas measured against '{}' (same denominator)".format(BASELINE))
    print()
    for model in available:
        b = table[(BASELINE, model)]
        print("{:16} baseline {:6.2f}%".format(model, b.wer * 100))
        for label, _ in CONFIGS[2:]:
            t = table[(label, model)]
            print("    {:22} {:6.2f}%   -{:.2f} pts   {} fewer substitutions".format(
                label, t.wer * 100, (b.wer - t.wer) * 100,
                b.substitutions - t.substitutions))

    print()
    for key, stats in sorted(stats_by_config.items()):
        if key[0].endswith("LLM") and key[1] == available[0]:
            print("LLM slot statistics ({}):".format(key[0]))
            for name in ["slots", "anchored", "variants_offered", "variants_kept", "spans_augmented"]:
                print("  {:18} {}".format(name, stats.get(name, 0)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "configuration", "wer_percent", "substitutions",
            "deletions", "insertions", "reference_words",
        ])
        for model in available:
            for label, _ in CONFIGS:
                t = table[(label, model)]
                writer.writerow([
                    model, label, round(t.wer * 100, 3), t.substitutions,
                    t.deletions, t.insertions, t.reference_words,
                ])
    summary = {
        "variant_records": len(variants),
        "scope": "subset" if subset else "full",
        "utterances": len(subset) if subset else len(load_predictions(available[0])),
        "configurations": {
            label: {
                model: {
                    "wer_percent": round(table[(label, model)].wer * 100, 3),
                    "substitutions": table[(label, model)].substitutions,
                    "deletions": table[(label, model)].deletions,
                    "insertions": table[(label, model)].insertions,
                    "reference_words": table[(label, model)].reference_words,
                }
                for model in available
            }
            for label, _ in CONFIGS
        },
        "llm_slot_statistics": stats_by_config.get(
            (CONFIGS[-1][0], available[0]), {}
        ),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print()
    print("wrote", out_path)
    print("wrote", out_path.with_suffix(".json"))


if __name__ == "__main__":
    main()
