import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from oiwer import score_utterance, score_corpus, normalize_segments


def test_exact_match():
    c = score_utterance("hello world", "hello world")
    assert (c.hits, c.substitutions, c.deletions, c.insertions) == (2, 0, 0, 0)
    assert c.wer == 0.0


def test_substitution():
    c = score_utterance("hello world", "hello there")
    assert (c.hits, c.substitutions, c.deletions, c.insertions) == (1, 1, 0, 0)


def test_deletion():
    c = score_utterance("hello big world", "hello world")
    assert (c.hits, c.substitutions, c.deletions, c.insertions) == (2, 0, 1, 0)


def test_insertion():
    c = score_utterance("hello world", "hello big world")
    assert (c.hits, c.substitutions, c.deletions, c.insertions) == (2, 0, 0, 1)


def test_variant_accepted():
    segments = normalize_segments([["can"], ["i"], ["find"], ["passbook", "pass book"]])
    c = score_utterance("can i find passbook", "can i find pass book", segments)
    assert c.substitutions == 0
    assert c.insertions == 0
    assert c.wer == 0.0


def test_variant_multiword_to_single():
    segments = normalize_segments([["EPFO", "E P F O"]])
    c = score_utterance("EPFO", "E P F O", segments)
    assert c.errors == 0


def test_numeral_variant():
    segments = normalize_segments([["56849", "five six eight four nine"]])
    c = score_utterance("56849", "five six eight four nine", segments)
    assert c.errors == 0


def test_variant_does_not_forgive_real_error():
    segments = normalize_segments([["passbook", "pass book"]])
    c = score_utterance("passbook", "notebook", segments)
    assert c.substitutions == 1


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_reproduces_published_wer():
    root = Path(__file__).resolve().parent.parent / "results"

    expected = {
        "indicconformer": 27.839357,
        "indicwhisper": 56.736278,
    }

    for model, target in expected.items():
        path = root / model / "indicvoices_telugu_valid.csv"
        if not path.exists():
            print(f"SKIP {model}: {path} not found")
            continue

        rows = load_csv(path)
        total, _ = score_corpus(rows)
        got = total.wer * 100

        print(
            f"{model:16} rows={len(rows)}  "
            f"WER={got:.6f}%  expected={target:.6f}%  "
            f"S={total.substitutions} D={total.deletions} I={total.insertions} "
            f"refwords={total.reference_words}"
        )

        assert abs(got - target) < 0.001, f"{model}: {got} != {target}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0

    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")

    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
