import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oiwer import align_segments, Counts
from rules import canonical, strip_tags, build_segments

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = [m for m in ["indicconformer", "indicwav2vec", "indicwhisper"]
          if (RESULTS / m / "indicvoices_telugu_valid.csv").exists()]


def load(model):
    path = RESULTS / model / "indicvoices_telugu_valid.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(rows, drop_tags, normalize, enable_itn, merge_split=False):
    total = Counts()
    changed = []

    for row in rows:
        reference = str(row["reference"])
        prediction = str(row["prediction"])

        if drop_tags:
            reference = strip_tags(reference)

        if normalize:
            ref_tokens = [t for t in canonical(reference).split() if t]
            pred_tokens = [t for t in canonical(prediction).split() if t]
        else:
            ref_tokens = reference.split()
            pred_tokens = prediction.split()

        if enable_itn:
            segments, ref_tokens = build_segments(reference, enable_itn=True)
        else:
            segments = [[[t]] for t in ref_tokens]

        _, subs, dels, ins, hits = align_segments(
            segments, pred_tokens, allow_merge_split=merge_split
        )

        total = total + Counts(hits, subs, dels, ins, len(ref_tokens))

    return total


CONFIGS = [
    ("WER (baseline)", False, False, False, False),
    ("+ strip <unintelligible>", True, False, False, False),
    ("+ normalize (punct/NFC)", True, True, False, False),
    ("+ ITN numbers", True, True, True, False),
    ("+ merge/split = OIWER", True, True, True, True),
]


def main():
    print(f"{'configuration':30} {'model':16} {'WER %':>9} {'S':>7} {'D':>7} {'I':>7} {'refwords':>9}")
    print("-" * 92)

    table = {}

    for label, tags, norm, itn, ms in CONFIGS:
        for model in MODELS:
            rows = load(model)
            total = score(rows, tags, norm, itn, ms)
            table[(label, model)] = total
            print(f"{label:30} {model:16} {total.wer*100:9.3f} "
                  f"{total.substitutions:7} {total.deletions:7} {total.insertions:7} "
                  f"{total.reference_words:9}")
        print()

    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)

    base = CONFIGS[0][0]
    final = CONFIGS[-1][0]

    for model in MODELS:
        b = table[(base, model)]
        f = table[(final, model)]
        print(f"{model:16} WER {b.wer*100:6.2f}%  ->  OIWER {f.wer*100:6.2f}%   "
              f"(-{(b.wer-f.wer)*100:.2f} pts, {b.substitutions-f.substitutions} fewer substitutions)")

    print()
    ranked_wer = sorted(MODELS, key=lambda m: table[(base, m)].wer)
    ranked_oi = sorted(MODELS, key=lambda m: table[(final, m)].wer)
    spread_wer = (table[(base, ranked_wer[-1])].wer - table[(base, ranked_wer[0])].wer) * 100
    spread_oi = (table[(final, ranked_oi[-1])].wer - table[(final, ranked_oi[0])].wer) * 100
    print(f"ranking under WER  : {' < '.join(ranked_wer)}")
    print(f"ranking under OIWER: {' < '.join(ranked_oi)}")
    print(f"best-worst spread, WER  : {spread_wer:.2f} points")
    print(f"best-worst spread, OIWER: {spread_oi:.2f} points")


if __name__ == "__main__":
    main()
