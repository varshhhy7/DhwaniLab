import re
import unicodedata

UNITS = {
    "సున్న": 0, "సున్నా": 0,
    "ఒకటి": 1, "ఒక": 1, "ఒక్క": 1,
    "రెండు": 2, "రెండ": 2,
    "మూడు": 3,
    "నాలుగు": 4, "నాల్గు": 4,
    "అయిదు": 5, "ఐదు": 5,
    "ఆరు": 6,
    "ఏడు": 7,
    "ఎనిమిది": 8,
    "తొమ్మిది": 9,
    "పది": 10,
    "పదకొండు": 11,
    "పన్నెండు": 12,
    "పదమూడు": 13, "పదమూడ": 13,
    "పద్నాలుగు": 14, "పధ్నాలుగు": 14,
    "పదిహేను": 15,
    "పదహారు": 16,
    "పదిహేడు": 17,
    "పద్దెనిమిది": 18,
    "పంతొమ్మిది": 19, "పందొమ్మిది": 19,
    "ఇరవై": 20,
    "ముప్పై": 30, "ముప్ఫై": 30,
    "నలభై": 40, "నలబై": 40,
    "యాభై": 50, "ఏభై": 50,
    "అరవై": 60,
    "డెబ్బై": 70, "డెభై": 70, "డెబ్భై": 70,
    "ఎనభై": 80, "ఎనబై": 80,
    "తొంభై": 90, "తొంబై": 90,
}

LOAN_UNITS = {
    "జీరో": 0, "వన్": 1, "టూ": 2, "త్రీ": 3, "ఫోర్": 4, "ఫైవ్": 5,
    "సిక్స్": 6, "సెవెన్": 7, "ఎయిట్": 8, "నైన్": 9, "టెన్": 10,
    "ఎలెవెన్": 11, "ట్వెల్వ్": 12,
    "ట్వంటీ": 20, "థర్టీ": 30, "ఫోర్టీ": 40, "ఫిఫ్టీ": 50,
    "సిక్స్టీ": 60, "సెవెంటీ": 70, "ఎయిటీ": 80, "నైంటీ": 90,
}

MULTIPLIERS = {
    "వంద": 100, "వందల": 100, "వందలు": 100, "నూరు": 100, "హండ్రెడ్": 100,
    "వెయ్యి": 1000, "వేయి": 1000, "వేల": 1000, "వేలు": 1000, "థౌజండ్": 1000,
    "లక్ష": 100000, "లక్షల": 100000, "లక్షలు": 100000,
    "కోటి": 10000000, "కోట్ల": 10000000, "కోట్లు": 10000000,
}

ALL_UNITS = {**UNITS, **LOAN_UNITS}

UNINTELLIGIBLE = re.compile(r"<[^>]*>")
REPLACEMENT_CHAR = "�"


def strip_tags(text):
    return UNINTELLIGIBLE.sub(" ", str(text))


def canonical(token, strip_punctuation=True, fold_case=True):
    token = unicodedata.normalize("NFC", str(token))
    token = token.replace(REPLACEMENT_CHAR, "")

    if strip_punctuation:
        token = "".join(
            c for c in token
            if not unicodedata.category(c).startswith(("P", "S"))
        )

    if fold_case:
        token = token.casefold()

    return token


def is_number_token(token):
    return token in ALL_UNITS or token in MULTIPLIERS


def span_value(tokens):
    result = 0
    current = 0
    seen = False

    for token in tokens:
        if token in ALL_UNITS:
            current += ALL_UNITS[token]
            seen = True
        elif token in MULTIPLIERS:
            multiplier = MULTIPLIERS[token]
            seen = True
            if multiplier >= 1000:
                result += (current if current else 1) * multiplier
                current = 0
            else:
                current = (current if current else 1) * multiplier
        else:
            return None

    if not seen:
        return None

    return result + current


def number_spans(tokens, max_span=8):
    spans = []
    i = 0
    n = len(tokens)

    while i < n:
        if not is_number_token(tokens[i]):
            i += 1
            continue

        end = i
        while end + 1 < n and is_number_token(tokens[end + 1]) and (end + 1 - i) < max_span:
            end += 1

        spans.append((i, end + 1))
        i = end + 1

    return spans


def build_spans(reference, enable_itn=True):
    tokens = [t for t in canonical(strip_tags(reference)).split() if t]

    if not tokens:
        return [], tokens

    if not enable_itn:
        return [(i, i + 1, [[t]]) for i, t in enumerate(tokens)], tokens

    number = number_spans(tokens)
    if not number:
        return [(i, i + 1, [[t]]) for i, t in enumerate(tokens)], tokens

    rebuilt = []
    cursor = 0

    for start, end in number:
        while cursor < start:
            rebuilt.append((cursor, cursor + 1, [[tokens[cursor]]]))
            cursor += 1

        span_tokens = tokens[start:end]
        value = span_value(span_tokens)

        variants = [list(span_tokens)]

        if value is not None:
            digits = str(value)
            if digits not in span_tokens:
                variants.append([digits])

            if end - start > 1:
                joined = "".join(str(ALL_UNITS[t]) for t in span_tokens if t in ALL_UNITS)
                if joined and joined != digits and len(joined) == end - start:
                    variants.append([joined])

        rebuilt.append((start, end, variants))
        cursor = end

    while cursor < len(tokens):
        rebuilt.append((cursor, cursor + 1, [[tokens[cursor]]]))
        cursor += 1

    return rebuilt, tokens


def build_segments(reference, enable_itn=True):
    spans, tokens = build_spans(reference, enable_itn=enable_itn)
    return [variants for _, _, variants in spans], tokens


def prepare(reference, prediction, enable_itn=True, drop_tags=True):
    if drop_tags:
        reference = strip_tags(reference)

    segments, ref_tokens = build_segments(reference, enable_itn=enable_itn)

    pred_tokens = [t for t in canonical(prediction).split() if t]

    return segments, ref_tokens, pred_tokens
