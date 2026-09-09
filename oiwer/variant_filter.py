import unicodedata

MARK_CATEGORIES = ("Mn", "Mc")


def skeleton(word):
    return "".join(
        c for c in unicodedata.normalize("NFC", word)
        if unicodedata.category(c) not in MARK_CATEGORIES
    )


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current

    return previous[-1]


def admissible(anchor, variant, skeleton_slack=1, ratio=0.34, floor=2):
    a = anchor.replace(" ", "")
    b = variant.replace(" ", "")

    if not b:
        return False
    if a == b:
        return True
    if len(b) > 2 * len(a) + 2:
        return False
    if edit_distance(skeleton(a), skeleton(b)) > skeleton_slack:
        return False
    if edit_distance(a, b) > max(floor, int(ratio * len(a))):
        return False

    return True


def anchor_slots(ref_tokens, llm_slots, canonical, lookahead=3, max_variants=8):
    accepted = {}
    stats = {"slots": 0, "anchored": 0, "variants_offered": 0, "variants_kept": 0}
    cursor = 0

    for slot in llm_slots:
        stats["slots"] += 1
        members = []
        for raw in slot:
            member = " ".join(t for t in canonical(raw).split() if t)
            if member and member not in members:
                members.append(member)
        if not members:
            continue

        best = None
        limit = min(cursor + lookahead + 1, len(ref_tokens))
        for start in range(cursor, limit):
            for member in members:
                tokens = member.split()
                end = start + len(tokens)
                if end <= len(ref_tokens) and ref_tokens[start:end] == tokens:
                    if best is None or start < best[0] or (start == best[0] and end > best[1]):
                        best = (start, end, member)
        if best is None:
            continue

        stats["anchored"] += 1
        start, end, anchor = best
        anchor_join = "".join(ref_tokens[start:end])

        keep = []
        for member in members:
            if member == anchor:
                continue
            stats["variants_offered"] += 1
            if admissible(anchor_join, member):
                keep.append(tuple(member.split()))

        if keep:
            bucket = accepted.setdefault((start, end), set())
            for variant in keep[:max_variants]:
                bucket.add(variant)
            stats["variants_kept"] += len(keep[:max_variants])

        cursor = end

    return accepted, stats


def merge_into_spans(spans, accepted):
    by_range = {(s, e): i for i, (s, e, _) in enumerate(spans)}
    starts = {s: i for i, (s, e, _) in enumerate(spans)}
    merged = 0
    drop = set()
    additions = {}

    for (start, end), extras in sorted(accepted.items()):
        if (start, end) in by_range:
            additions.setdefault(by_range[(start, end)], set()).update(extras)
            merged += 1
            continue

        if start not in starts:
            continue

        index = starts[start]
        covered = []
        cursor = start
        while cursor < end and index < len(spans):
            s, e, variants = spans[index]
            if s != cursor or len(variants) != 1 or len(variants[0]) != 1:
                covered = []
                break
            covered.append(index)
            cursor = e
            index += 1

        if not covered or cursor != end:
            continue

        keep = covered[0]
        tokens = []
        for i in covered:
            tokens.extend(spans[i][2][0])
        spans[keep] = (start, end, [tokens])
        for i in covered[1:]:
            drop.add(i)
        additions.setdefault(keep, set()).update(extras)
        merged += 1

    for index, extras in additions.items():
        s, e, variants = spans[index]
        seen = {tuple(v) for v in variants}
        for extra in sorted(extras):
            if extra not in seen:
                seen.add(extra)
                variants = variants + [list(extra)]
        spans[index] = (s, e, variants)

    return [span for i, span in enumerate(spans) if i not in drop], merged
