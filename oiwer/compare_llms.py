import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from score_llm import MODELS, load_predictions, load_variants, score

RESULTS = ROOT / "results" / "oiwer"

DEFAULT_SOURCES = [
    ("gemma3:4b", RESULTS / "variants_gemma3_4b.jsonl"),
    ("gemma3n:e2b", RESULTS / "variants_gemma3n_e2b.jsonl"),
]


def coverage(path):
    records = 0
    with_slots = 0
    if not Path(path).exists():
        return 0, 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                if json.loads(line).get("slots"):
                    with_slots += 1
            except Exception:
                continue
    return records, with_slots


def main():
    parser = argparse.ArgumentParser(
        description="Compare OI-WER across different variant-generating models."
    )
    parser.add_argument("--source", action="append", default=None,
                        metavar="LABEL=PATH",
                        help="repeatable; defaults to the two gemma runs")
    parser.add_argument("--out", default=str(RESULTS / "llm_model_comparison.csv"))
    parser.add_argument("--matched", action="store_true",
                        help="score only utterances covered by every source, so "
                             "generators with different coverage stay comparable")
    args = parser.parse_args()

    if args.source:
        sources = []
        for item in args.source:
            label, _, path = item.partition("=")
            sources.append((label, Path(path)))
    else:
        sources = [(label, path) for label, path in DEFAULT_SOURCES if path.exists()]

    if not sources:
        sys.exit("no variant files found")

    available = [
        m for m in MODELS
        if (ROOT / "results" / m / "indicvoices_telugu_valid.csv").exists()
    ]

    rows = {model: load_predictions(model) for model in available}
    cache = {label: load_variants(path) for label, path in sources}

    subset = None
    if args.matched:
        covered = [set(cache[label]) for label, _ in sources]
        subset = set.intersection(*covered) if covered else set()

    baseline = {}
    for model in available:
        total, _ = score(rows[model], {}, use_rules=False, use_llm=False,
                         subset=subset)
        baseline[model] = total

    rules = {}
    for model in available:
        total, _ = score(rows[model], {}, use_rules=True, use_llm=False,
                         subset=subset)
        rules[model] = total

    print("variant-generator comparison")
    print()
    for label, path in sources:
        records, with_slots = coverage(path)
        print("  {:14} {:,} records, {:,} with variants  ({})".format(
            label, records, with_slots, path.name))
    if subset is not None:
        print()
        print("  matched scope: {:,} utterances covered by every source, "
              "{:,} reference words".format(
                  len(subset), baseline[available[0]].reference_words))
    print()

    header = "{:16} {:>10} {:>10}".format("model", "WER", "rules")
    for label, _ in sources:
        header += " {:>13} {:>13}".format(label[:12] + " LLM", label[:12] + " both")
    print(header)
    print("-" * len(header))

    table = []
    for model in available:
        line = "{:16} {:10.3f} {:10.3f}".format(
            model, baseline[model].wer * 100, rules[model].wer * 100)
        record = {
            "model": model,
            "wer_percent": round(baseline[model].wer * 100, 3),
            "rules_percent": round(rules[model].wer * 100, 3),
        }
        for label, _ in sources:
            variants = cache[label]
            llm_only, stats = score(rows[model], variants, use_rules=False,
                                    use_llm=True, subset=subset)
            both, _ = score(rows[model], variants, use_rules=True,
                            use_llm=True, subset=subset)
            line += " {:13.3f} {:13.3f}".format(
                llm_only.wer * 100, both.wer * 100)
            key = label.replace(":", "_")
            record[key + "_llm_only_percent"] = round(llm_only.wer * 100, 3)
            record[key + "_both_percent"] = round(both.wer * 100, 3)
            record[key + "_variants_offered"] = stats.get("variants_offered", 0)
            record[key + "_variants_kept"] = stats.get("variants_kept", 0)
        print(line)
        table.append(record)

    print()
    for label, _ in sources:
        _, stats = score(rows[available[0]], cache[label], use_rules=True,
                         use_llm=True, subset=subset)
        offered = stats.get("variants_offered", 0)
        kept = stats.get("variants_kept", 0)
        rejected = round(100.0 * (offered - kept) / offered) if offered else 0
        print("{:14} groups {:,}  anchored {:,}  offered {:,}  kept {:,}  "
              "rejected {}%".format(
                  label, stats.get("slots", 0), stats.get("anchored", 0),
                  offered, kept, rejected))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for record in table for k in record},
                    key=lambda k: (k != "model", k))
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
