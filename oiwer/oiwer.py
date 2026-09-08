import unicodedata
from dataclasses import dataclass


def normalize_text(text):
    return unicodedata.normalize("NFC", str(text).strip())


def tokenize(text):
    return normalize_text(text).split()


@dataclass
class Counts:
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    @property
    def errors(self):
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self):
        if self.reference_words == 0:
            return 0.0
        return self.errors / self.reference_words

    def __add__(self, other):
        return Counts(
            self.hits + other.hits,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_words + other.reference_words,
        )


def _local_align(variant, window, allow_merge_split=False):
    m = len(variant)
    n = len(window)

    if (
        allow_merge_split
        and m
        and n
        and (m > 1 or n > 1)
        and "".join(variant) == "".join(window)
    ):
        return (0, 0, 0, 0, m)

    table = [[None] * (n + 1) for _ in range(m + 1)]
    table[0][0] = (0, 0, 0, 0, 0)

    for j in range(1, n + 1):
        c, s, d, i, h = table[0][j - 1]
        table[0][j] = (c + 1, s, d, i + 1, h)

    for k in range(1, m + 1):
        c, s, d, i, h = table[k - 1][0]
        table[k][0] = (c + 1, s, d + 1, i, h)

        for j in range(1, n + 1):
            if variant[k - 1] == window[j - 1]:
                c, s, d, i, h = table[k - 1][j - 1]
                best = (c, s, d, i, h + 1)
            else:
                c, s, d, i, h = table[k - 1][j - 1]
                best = (c + 1, s + 1, d, i, h)

            c, s, d, i, h = table[k - 1][j]
            cand = (c + 1, s, d + 1, i, h)
            if cand[0] < best[0]:
                best = cand

            c, s, d, i, h = table[k][j - 1]
            cand = (c + 1, s, d, i + 1, h)
            if cand[0] < best[0]:
                best = cand

            table[k][j] = best

    return table[m][n]


def _better(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return b if b[0] < a[0] else a


def align_segments(segments, pred_tokens, window_slack=2, allow_merge_split=False):
    n_pred = len(pred_tokens)
    n_seg = len(segments)

    best = [[None] * (n_pred + 1) for _ in range(n_seg + 1)]
    best[0][0] = (0, 0, 0, 0, 0)

    for j in range(1, n_pred + 1):
        c, s, d, i, h = best[0][j - 1]
        best[0][j] = (c + 1, s, d, i + 1, h)

    for k in range(1, n_seg + 1):
        variants = segments[k - 1] or [[]]

        for j in range(0, n_pred + 1):
            current = None

            if j > 0 and best[k][j - 1] is not None:
                c, s, d, i, h = best[k][j - 1]
                current = _better(current, (c + 1, s, d, i + 1, h))

            for variant in variants:
                m = len(variant)
                max_window = m + window_slack
                lowest = max(0, j - max_window)

                for i_start in range(lowest, j + 1):
                    base = best[k - 1][i_start]
                    if base is None:
                        continue

                    lc, ls, ld, li, lh = _local_align(
                        variant, pred_tokens[i_start:j], allow_merge_split
                    )

                    candidate = (
                        base[0] + lc,
                        base[1] + ls,
                        base[2] + ld,
                        base[3] + li,
                        base[4] + lh,
                    )
                    current = _better(current, candidate)

            best[k][j] = current

    return best[n_seg][n_pred]


def score_utterance(reference, prediction, segments=None, allow_merge_split=False):
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if segments is None:
        segments = [[[token]] for token in ref_tokens]

    _, subs, dels, ins, hits = align_segments(
        segments, pred_tokens, allow_merge_split=allow_merge_split
    )

    return Counts(
        hits=hits,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_words=len(ref_tokens),
    )


def score_corpus(rows):
    total = Counts()
    per_utterance = []

    for row in rows:
        counts = score_utterance(
            row["reference"],
            row["prediction"],
            row.get("segments"),
        )
        per_utterance.append(counts)
        total = total + counts

    return total, per_utterance


def normalize_segments(raw_segments):
    segments = []

    for slot in raw_segments:
        variants = []
        seen = set()

        for variant in slot:
            tokens = tuple(tokenize(variant))
            if tokens and tokens not in seen:
                seen.add(tokens)
                variants.append(list(tokens))

        if variants:
            segments.append(variants)

    return segments
