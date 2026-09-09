import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rules import canonical
from variant_filter import admissible, anchor_slots, merge_into_spans, skeleton


def test_skeleton_drops_matras():
    assert skeleton("రీచ్") == skeleton("రిచ్")


def test_accepts_matra_variant():
    assert admissible("రీచ్", "రిచ్")


def test_accepts_compound_split():
    assert admissible(
        "తెలంగాణలో",
        "తెలంగాణ లో",
    )


def test_rejects_synonym():
    assert not admissible(
        "కావడానికి",
        "చేయడానికి",
    )


def test_rejects_translation():
    assert not admissible(
        "ప్లేసెస్",
        "ప్రదేశాలు",
    )


def test_rejects_different_word():
    assert not admissible("నేను", "నా")


def test_slot_without_anchor_is_dropped():
    ref = ["ఇండియా"]
    slots = [["ఇండీయా"]]
    accepted, stats = anchor_slots(ref, slots, canonical)
    assert accepted == {}
    assert stats["anchored"] == 0


def test_extra_hallucinated_slot_is_dropped():
    ref = ["ట్రాఫిక్", "లేకుండా"]
    slots = [
        ["ట్రాఫిక్"],
        ["లేకుండా"],
        ["ఇలానా"],
    ]
    accepted, stats = anchor_slots(ref, slots, canonical)
    assert accepted == {}
    assert stats["slots"] == 3
    assert stats["anchored"] == 2


def test_anchored_variant_is_attached():
    ref = ["హలో"]
    slots = [["హలో", "హెలో"]]
    accepted, stats = anchor_slots(ref, slots, canonical)
    assert accepted == {(0, 1): {("హెలో",)}}
    assert stats["variants_kept"] == 1


def test_merge_into_spans_keeps_reference_length():
    ref = ["a", "b", "c"]
    spans = [(0, 1, [["a"]]), (1, 2, [["b"]]), (2, 3, [["c"]])]
    accepted = {(0, 2): {("ab",)}}
    merged, count = merge_into_spans(spans, accepted)
    assert count == 1
    assert merged[0] == (0, 2, [["a", "b"], ["ab"]])
    assert sum(e - s for s, e, _ in merged) == len(ref)


def test_itn_span_not_clobbered():
    spans = [(0, 2, [["రెండు", "వేల"], ["2000"]])]
    accepted = {(0, 1): {("x",)}}
    merged, count = merge_into_spans(spans, accepted)
    assert count == 0
    assert merged == spans


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0

    for t in tests:
        try:
            t()
            print("PASS  " + t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  " + t.__name__ + ": " + str(e))

    print()
    print("{}/{} passed".format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
