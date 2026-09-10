import argparse
import csv
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "vendor"))

import prompt_telugu as P

ENDPOINT = "http://127.0.0.1:11434/api/generate"
NEWLINE = chr(10)

COMPACT_TASK = """You are a Telugu linguist. For the Telugu sentence below, list the
alternative WRITTEN SPELLINGS of each word.

Only spelling variation is allowed. Never replace a word with a synonym, a
translation, or a different word. The meaning and the pronunciation must stay
the same. Allowed changes are: phonetic spelling differences, matra and
diacritic differences, ligature differences, alternative spellings of loan
words, and splitting or merging a compound word.

Rules:
1. Output one list per word, in the same order as the sentence.
2. The first element of every list must be the original word, copied exactly.
3. If a word has no alternative spelling, output a list with just that word.
4. Output only a JSON list of lists. No explanation.

Reference examples of the allowed variation types:"""


def build_prompt(sentence, language="Telugu", style="compact"):
    if style == "official":
        parts = [
            P.TASK_OVERVIEW_PROMPT.format(language=language),
            P.TASK_INSTRUCTION_PROMPT.format(language=language),
            P.GUIDELINES_TELUGU,
            "Sentence: " + sentence + NEWLINE + "Output:",
        ]
    else:
        parts = [
            COMPACT_TASK,
            P.GUIDELINES_TELUGU,
            "Sentence: " + sentence + NEWLINE + "Output:",
        ]
    return (NEWLINE + NEWLINE).join(parts)


def call_ollama(sentence, model, timeout, num_ctx, num_predict, style="compact"):
    body = json.dumps(
        {
            "model": model,
            "prompt": build_prompt(sentence, style=style),
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload.get("response", "")


def _coerce(obj, depth=0):
    if depth > 4:
        return []
    if isinstance(obj, dict):
        for value in obj.values():
            slots = _coerce(value, depth + 1)
            if slots:
                return slots
        return []
    if not isinstance(obj, list):
        return []
    slots = []
    for item in obj:
        if isinstance(item, str):
            slots.append([item])
        elif isinstance(item, list) and all(isinstance(x, str) for x in item):
            slots.append([x for x in item])
        else:
            return []
    return slots


ARTIFACTS = {
    ord("▁"): " ",
    ord(" "): " ",
    ord("​"): None,
    ord("﻿"): None,
}


def parse_variants(text):
    text = (text or "").translate(ARTIFACTS).strip()
    if not text:
        return []
    try:
        return _coerce(json.loads(text))
    except Exception:
        pass
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            return _coerce(json.loads(match.group(0)))
        except Exception:
            return []
    return []


def load_references(csv_path, limit=None):
    rows = []
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["index"]), row["reference"]))
            if limit and len(rows) >= limit:
                break
    return rows


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except Exception:
                continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--out", default=str(ROOT / "results" / "oiwer" / "variants_gemma3_4b.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--style", default="compact", choices=["compact", "official"])
    args = parser.parse_args()

    source = ROOT / "results" / "indicconformer" / "indicvoices_telugu_valid.csv"
    references = load_references(source, args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    pending = [(i, ref) for i, ref in references if i not in done]

    sys.stderr.write(
        "total={} done={} pending={} model={} workers={}{}".format(
            len(references), len(done), len(pending), args.model, args.workers, NEWLINE
        )
    )
    if not pending:
        return

    lock = threading.Lock()
    handle = open(out_path, "a", encoding="utf-8")
    started = time.time()
    counters = {"ok": 0, "empty": 0, "error": 0}

    def work(item):
        index, reference = item
        record = {"index": index, "reference": reference, "slots": [], "status": "ok"}
        try:
            raw = call_ollama(
                reference, args.model, args.timeout, args.num_ctx, args.num_predict, args.style
            )
            slots = parse_variants(raw)
            record["slots"] = slots
            if not slots:
                record["status"] = "empty"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)[:200]
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in pending]
        for n, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            counters[record["status"]] = counters.get(record["status"], 0) + 1
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + NEWLINE)
                if n % 25 == 0:
                    handle.flush()
                    elapsed = time.time() - started
                    rate = n / elapsed
                    remaining = (len(pending) - n) / rate if rate else 0
                    sys.stderr.write(
                        "{}/{} ok={} empty={} error={} {:.2f} utt/s eta {:.1f} min{}".format(
                            n,
                            len(pending),
                            counters.get("ok", 0),
                            counters.get("empty", 0),
                            counters.get("error", 0),
                            rate,
                            remaining / 60.0,
                            NEWLINE,
                        )
                    )
                    sys.stderr.flush()

    handle.close()
    sys.stderr.write("finished {}{}".format(counters, NEWLINE))


if __name__ == "__main__":
    main()
