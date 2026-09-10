import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "oiwer" / "oiwer_llm_full.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W, H = 1520, 1210

NAVY = "#1a2b4c"
SLATE = "#3d4b63"
GREY = "#7c8798"
LINE = "#9aa4b2"
INK = "#10233f"
PAPER = "#ffffff"
BLUE_FILL = "#eaf0f9"
BLUE_EDGE = "#2f5c9e"
GREEN_FILL = "#e8f4ec"
GREEN_EDGE = "#1c7c3f"
AMBER_FILL = "#fdf1e3"
AMBER_EDGE = "#b7791f"
RED_FILL = "#fdecec"
RED_EDGE = "#b03030"
GREY_FILL = "#f2f4f7"

FONT = "Segoe UI, Inter, Helvetica Neue, Arial, sans-serif"
MONO = "Consolas, Menlo, monospace"

MODELS = ["indicconformer", "indicwav2vec", "indicwhisper"]

out = []


def esc(text_value):
    return (str(text_value).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, content, size=12, fill=INK, weight="normal", anchor="middle",
         family=FONT):
    out.append(
        '<text x="{:.1f}" y="{:.1f}" font-family="{}" font-size="{}" '
        'font-weight="{}" fill="{}" text-anchor="{}">{}</text>'
        .format(x, y, family, size, weight, fill, anchor, esc(content))
    )


def box(x, y, w, h, title, lines=(), fill=GREY_FILL, edge=LINE, title_size=15,
        line_size=11.5, line_fill=SLATE, mono_lines=False, accent=None):
    out.append(
        '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" rx="8" '
        'fill="{}" stroke="{}" stroke-width="1.6"/>'
        .format(x, y, w, h, fill, edge)
    )
    cx = x + w / 2.0
    if not lines:
        text(cx, y + h / 2.0 + 5, title, title_size, NAVY, "600")
        return

    block = 22 + 17 * len(lines)
    top = y + (h - block) / 2.0 + 18
    text(cx, top, title, title_size, NAVY, "600")
    ty = top + 24
    for line in lines:
        text(cx, ty, line, line_size, line_fill,
             family=MONO if mono_lines else FONT)
        ty += 17
    if accent:
        text(cx, ty + 4, accent, 11.5, AMBER_EDGE, "600")


def cylinder(x, y, w, h, title, subtitle):
    ry = 13.0
    out.append(
        '<path d="M{x:.1f},{top:.1f} a{rx:.1f},{ry:.1f} 0 0 1 {w:.1f},0 '
        'v{body:.1f} a{rx:.1f},{ry:.1f} 0 0 1 -{w:.1f},0 z" '
        'fill="{fill}" stroke="{edge}" stroke-width="1.6"/>'
        .format(x=x, top=y + ry, rx=w / 2.0, ry=ry, w=w, body=h - 2 * ry,
                fill=BLUE_FILL, edge=BLUE_EDGE)
    )
    out.append(
        '<path d="M{x:.1f},{top:.1f} a{rx:.1f},{ry:.1f} 0 0 0 {w:.1f},0" '
        'fill="none" stroke="{edge}" stroke-width="1.6"/>'
        .format(x=x, top=y + ry, rx=w / 2.0, ry=ry, w=w, edge=BLUE_EDGE)
    )
    cx = x + w / 2.0
    text(cx, y + h / 2.0 + 2, title, 15, NAVY, "600")
    text(cx, y + h / 2.0 + 21, subtitle, 11.5, SLATE)


def line(d, color=LINE, width=1.6):
    out.append('<path d="{}" fill="none" stroke="{}" stroke-width="{}"/>'
               .format(d, color, width))


def head(x, y, direction="down", color=LINE, size=7.0):
    span = size * 0.66
    if direction == "down":
        pts = [(x - span, y - size), (x + span, y - size), (x, y)]
    elif direction == "left":
        pts = [(x + size, y - span), (x + size, y + span), (x, y)]
    else:
        pts = [(x - size, y - span), (x - size, y + span), (x, y)]
    out.append('<polygon points="{}" fill="{}"/>'.format(
        " ".join("{:.1f},{:.1f}".format(px, py) for px, py in pts), color))


def arrow_down(x, y1, y2, color=LINE):
    line("M{},{} V{}".format(x, y1, y2), color)
    head(x, y2, "down", color)


def load_numbers():
    if not SUMMARY.exists():
        return None
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cfg = data["configurations"]
    slots = data.get("llm_slot_statistics", {})
    offered = slots.get("variants_offered", 0)
    kept = slots.get("variants_kept", 0)
    return {
        "wer": [cfg["WER (normalized text)"][m]["wer_percent"] for m in MODELS],
        "oiwer": [cfg["OIWER rules + LLM"][m]["wer_percent"] for m in MODELS],
        "rules": [cfg["OIWER rules only"][m]["wer_percent"] for m in MODELS],
        "llm": [cfg["OIWER LLM only"][m]["wer_percent"] for m in MODELS],
        "words": cfg["WER (normalized text)"][MODELS[0]]["reference_words"],
        "generator": data.get("variant_source", "local LLM").replace("_", ":"),
        "rejected": round(100.0 * (offered - kept) / offered) if offered else 0,
        "offered": offered,
        "kept": kept,
    }


def fmt(values):
    return "   /   ".join("{:.1f}".format(v) for v in values)


def build(n):
    out.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'viewBox="0 0 {w} {h}">'.format(w=W, h=H)
    )
    out.append('<rect width="{}" height="{}" fill="{}"/>'.format(W, H, PAPER))

    text(60, 52, "DhwaniLab — Telugu ASR Benchmarking", 25, NAVY, "700", "start")
    text(60, 78, "Phase 1 evaluation pipeline: three pretrained models, scored "
                 "with WER and Orthographically-Informed WER",
         13, SLATE, "normal", "start")
    line("M60,92 H{}".format(W - 60))

    cylinder(590, 118, 340, 76, "IndicVoices Telugu · valid split",
             "3,295 utterances · {:,} reference words".format(n["words"]))

    line("M760,194 V214")
    line("M340,214 H1180")
    for cx in (340, 760, 1180):
        arrow_down(cx, 214, 244)

    box(160, 244, 360, 92, "IndicWhisper",
        ["whisper-medium-te · HuggingFace", "autoregressive decoding"],
        BLUE_FILL, BLUE_EDGE)
    box(580, 244, 360, 92, "IndicConformer",
        ["indic-conformer-600m · ONNX", "greedy CTC"],
        BLUE_FILL, BLUE_EDGE)
    box(1000, 244, 360, 92, "IndicWav2Vec",
        ["te.pt · Fairseq, isolated venv", "greedy CTC, no language model"],
        BLUE_FILL, BLUE_EDGE)

    for cx in (340, 760, 1180):
        line("M{},336 V358".format(cx))
    line("M340,358 H1180")
    arrow_down(760, 358, 386)

    box(310, 386, 900, 70, "results/<model>/indicvoices_telugu_valid.csv",
        ["index · speaker_id · duration · reference · prediction · S/D/I"],
        GREY_FILL, LINE, title_size=14, mono_lines=True)

    text(985, 478, "the reference column only — never the predictions", 11, GREY)
    line("M760,456 V486")
    line("M380,486 H1140")
    arrow_down(380, 486, 516)
    arrow_down(1140, 486, 516)

    box(120, 516, 520, 132, "Deterministic rules · oiwer/rules.py",
        ["Telugu number words  →  digits  (ITN)",
         "compound merge / split by string equality",
         "NFC, punctuation, <unintelligible> removal"],
        GREEN_FILL, GREEN_EDGE)

    box(880, 516, 520, 76, n["generator"] + " · local, via Ollama",
        ["prompted with AI4Bharat's expert-verified Telugu guidelines"],
        AMBER_FILL, AMBER_EDGE)

    arrow_down(1140, 592, 616)

    box(880, 616, 520, 148, "oiwer/variant_filter.py",
        ["1.  group must contain the original word",
         "2.  consonant skeleton within 1 edit",
         "3.  short words (≤ 4 chars): skeleton must match exactly",
         "4.  edit distance ≤ 34%, or equal without spaces"],
        AMBER_FILL, AMBER_EDGE,
        accent="{:,} offered  →  {:,} kept  ·  {}% rejected".format(
            n["offered"], n["kept"], n["rejected"]))

    line("M380,648 V800 H752")
    line("M1140,764 V800 H768")
    arrow_down(760, 800, 830)

    box(460, 830, 600, 84, "Variant lattice",
        ["each reference word carries a set of accepted spellings"],
        GREY_FILL, LINE)

    arrow_down(760, 914, 940)

    box(460, 940, 600, 84, "Dynamic-programming alignment",
        ["oiwer/oiwer.py · fewest edits over the lattice"],
        GREY_FILL, NAVY, mono_lines=True)

    box(1130, 940, 330, 84, "Cross-check",
        ["vs AI4Bharat oiwer_core.py", "exact agreement, no variants"],
        BLUE_FILL, BLUE_EDGE, title_size=14, line_size=10.5)
    line("M1130,982 H1074", BLUE_EDGE)
    head(1068, 982, "left", BLUE_EDGE)

    line("M760,1024 V1048")
    line("M615,1048 H915")
    arrow_down(615, 1048, 1074)
    arrow_down(915, 1048, 1074)

    box(470, 1074, 290, 88, "WER",
        [fmt(n["wer"]), "conformer / wav2vec / whisper"],
        RED_FILL, RED_EDGE, line_size=13, line_fill=INK)
    box(770, 1074, 290, 88, "OI-WER",
        [fmt(n["oiwer"]), "conformer / wav2vec / whisper"],
        GREEN_FILL, GREEN_EDGE, line_size=13, line_fill=INK)

    text(60, 1190,
         "Variants depend on the reference alone, so they are generated once and "
         "reused for all three models.     Rules only: {}.     LLM only: {}."
         .format(fmt(n["rules"]), fmt(n["llm"])),
         11.5, GREY, anchor="start")

    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "reports" / "project_flow.svg"))
    parser.add_argument("--png", action="store_true",
                        help="also write a PNG (needs svglib + rlPyCairo)")
    args = parser.parse_args()

    numbers = load_numbers()
    if numbers is None:
        sys.exit("missing {} — run oiwer/score_llm.py first".format(SUMMARY))

    svg = build(numbers)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print("wrote", out_path)

    if args.png:
        try:
            from reportlab.graphics import renderPM
            from svglib.svglib import svg2rlg
        except ImportError:
            print("skipped PNG: pip install svglib rlPyCairo")
            return
        png_path = out_path.with_suffix(".png")
        drawing = svg2rlg(str(out_path))
        renderPM.drawToFile(drawing, str(png_path), fmt="PNG", dpi=144)
        print("wrote", png_path)


if __name__ == "__main__":
    main()
