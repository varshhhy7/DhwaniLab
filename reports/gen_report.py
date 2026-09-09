# -*- coding: utf-8 -*-
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "oiwer"
OUT = str(ROOT / "reports" / "DhwaniLab_Progress_Report.pdf")

TELUGU_FONT = "Helvetica"
_FONT_PATH = "C:\\Windows\\Fonts\\Nirmala.ttc"

for _idx in (0, 1, 2):
    try:
        pdfmetrics.registerFont(TTFont("Nirmala", _FONT_PATH, subfontIndex=_idx))
        TELUGU_FONT = "Nirmala"
        break
    except Exception:
        continue

print("telugu font:", TELUGU_FONT)

MODELS = ["indicconformer", "indicwav2vec", "indicwhisper"]
LABELS = {
    "indicconformer": "IndicConformer",
    "indicwav2vec": "IndicWav2Vec",
    "indicwhisper": "IndicWhisper",
}

summary = json.loads((RESULTS / "oiwer_llm_full.json").read_text(encoding="utf-8"))
CFG = summary["configurations"]
SLOTS = summary["llm_slot_statistics"]

RAW = "WER (published protocol)"
NORM = "WER (normalized text)"
RULES = "OIWER rules only"
LLM = "OIWER LLM only"
BOTH = "OIWER rules + LLM"


def val(config, model, field="wer_percent"):
    return CFG[config][model][field]


def pct(config, model):
    return "{:.2f}%".format(val(config, model))


def delta(config, model):
    return "{:+.2f}".format(val(config, model) - val(NORM, model))


crosscheck = []
_cross_path = RESULTS / "engine_crosscheck.csv"
if _cross_path.exists():
    with open(_cross_path, encoding="utf-8") as handle:
        crosscheck = list(csv.DictReader(handle))


def cross(config, model, field):
    for row in crosscheck:
        if row["configuration"] == config and row["model"] == model:
            return row[field]
    return "-"


navy = colors.HexColor("#1a2b4c")
slate = colors.HexColor("#3d4b63")
light = colors.HexColor("#f2f4f7")
midgray = colors.HexColor("#c9cfd8")
green = colors.HexColor("#1c7c3f")
codebg = colors.HexColor("#eef1f5")

styles = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold",
                            fontSize=17, leading=20, textColor=navy, alignment=TA_LEFT, spaceAfter=2),
    "sub": ParagraphStyle("s", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=13, textColor=slate),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, leading=14, textColor=navy, spaceBefore=11, spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=styles["Heading3"], fontName="Helvetica-Bold",
                         fontSize=9.8, leading=12, textColor=slate, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("b", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=9, leading=12.5, textColor=colors.HexColor("#1a1a1a"), spaceAfter=4),
    "bullet": ParagraphStyle("bu", parent=styles["Normal"], fontName="Helvetica",
                             fontSize=9, leading=12.3, leftIndent=11, spaceAfter=3),
    "code": ParagraphStyle("c", parent=styles["Normal"], fontName="Courier",
                           fontSize=8.4, leading=11.5, textColor=colors.HexColor("#10233f"),
                           backColor=codebg, borderPadding=5, spaceBefore=3, spaceAfter=5),
    "meta": ParagraphStyle("m", parent=styles["Normal"], fontName="Helvetica",
                           fontSize=8.5, leading=11, textColor=slate, alignment=TA_CENTER),
    "cap": ParagraphStyle("cap", parent=styles["Normal"], fontName="Helvetica-Oblique",
                          fontSize=8, leading=10.5, textColor=slate, spaceAfter=6),
    "te": ParagraphStyle("te", parent=styles["Normal"], fontName=TELUGU_FONT,
                         fontSize=10, leading=17, textColor=colors.HexColor("#10233f"),
                         backColor=codebg, borderPadding=6, spaceBefore=3, spaceAfter=6),
}

story = []
TE = lambda s: "<font face='{}' size='10'>{}</font>".format(TELUGU_FONT, s)
P = lambda t, s="body": story.append(Paragraph(t, S[s]))
B = lambda t: story.append(Paragraph("&bull;&nbsp;&nbsp;" + t, S["bullet"]))
SP = lambda h=4: story.append(Spacer(1, h))


def table(data, widths, align_center_from=1, highlight_rows=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
        ("GRID", (0, 0), (-1, -1), 0.4, midgray),
        ("ALIGN", (align_center_from, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in (highlight_rows or []):
        style += [("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"),
                  ("TEXTCOLOR", (1, r), (1, r), green)]
    t.setStyle(TableStyle(style))
    story.append(t)


P("DhwaniLab &mdash; Telugu ASR Benchmarking", "title")
P("Phase 1 report &middot; three pretrained AI4Bharat models on IndicVoices Telugu &middot; WER and OI-WER", "sub")
SP(5)
story.append(HRFlowable(width="100%", thickness=1.1, color=navy, spaceAfter=8))

P("1. Objective", "h2")
P("Benchmark publicly released Telugu ASR models under a single identical evaluation protocol, "
  "and evaluate them with both standard Word Error Rate (WER) and Orthographically-Informed WER "
  "(OI-WER), so that reported differences reflect the models rather than the measurement. "
  "This is Phase 1; Phase 2 will build a new Telugu dataset and fine-tune these models on it.")

P("2. Experimental setup", "h2")
table(
    [["Component", "Choice"],
     ["Dataset", "ai4bharat/IndicVoices, config telugu, split valid (streamed)"],
     ["Size", "3,295 utterances / 37,350 reference words"],
     ["Reference field", "normalized"],
     ["Audio", "mono, resampled to 16 kHz"],
     ["Language model", "none, for any of the three models"],
     ["Output", "per-utterance CSV: index, speaker_id, duration, reference, prediction, S/D/I"]],
    [32 * mm, 134 * mm], align_center_from=0)
P("The validation split is the only evaluation split IndicVoices publishes; there is no separate "
  "test split (train = 269,300 utterances, valid = 3,295).", "cap")

P("Models evaluated", "h3")
B("<b>IndicWhisper</b> (<font face='Courier' size='8'>whisper-medium-te_alldata_multigpu</font>) &mdash; "
  "standard HuggingFace checkpoint, decoded autoregressively with language=te, task=transcribe.")
B("<b>IndicConformer</b> (<font face='Courier' size='8'>ai4bharat/indic-conformer-600m-multilingual</font>) &mdash; "
  "gated repository, loaded with trust_remote_code; resolves to an ONNX Runtime model, decoded with greedy CTC.")
B("<b>IndicWav2Vec</b> (<font face='Courier' size='8'>te.pt</font>, 3.53 GB Fairseq checkpoint) &mdash; no HuggingFace "
  "Telugu weights exist, so it runs in an isolated Python 3.10 / torch 1.13 virtual environment via a subprocess "
  "decoder, with audio pre-dumped to disk. Greedy CTC (Viterbi), no language model.")

P("3. How WER is computed", "h2")

P("3.1 Per utterance", "h3")
P("An <i>utterance</i> is one audio clip with one reference transcript &mdash; the unit the dataset ships and the "
  "unit each model transcribes. Each utterance is scored independently. Both the reference and the prediction "
  "are whitespace-normalized and split into word tokens. The two token sequences are then aligned using "
  "<b>minimum edit distance</b> (word-level Levenshtein alignment), which finds the alignment requiring the "
  "fewest edits. Each aligned position is classified as:")
B("<b>Hit</b> &mdash; reference word and predicted word are identical.")
B("<b>Substitution (S)</b> &mdash; a reference word was replaced by a different predicted word.")
B("<b>Deletion (D)</b> &mdash; a reference word has no counterpart in the prediction (word missed).")
B("<b>Insertion (I)</b> &mdash; a predicted word has no counterpart in the reference (word invented).")

P("3.2 Corpus aggregation", "h3")
P("Per-utterance counts are summed across all 3,295 utterances <i>before</i> dividing. The reported figure is "
  "therefore a corpus-level rate:")
P("WER = ( &Sigma;S + &Sigma;D + &Sigma;I ) / &Sigma;N<br/>"
  "where N = number of reference words in an utterance", "code")
P("<b>This is deliberately not the mean of per-utterance WERs.</b> Averaging per-utterance rates gives a "
  "one-word utterance the same weight as a forty-word utterance, which inflates the influence of short "
  "utterances and produces numbers that are not comparable with published results. Per-utterance WER is "
  "still stored in the CSVs, but it is used only for error analysis, never for the headline figure.")

P("3.3 Ensuring the three models are comparable", "h3")
B("All three models were run over the same streamed split, so utterance <i>index</i> aligns across the result files.")
B("References were compared position by position across the runs: <b>3,295 of 3,295 matched exactly.</b> "
  "Had they differed, the WERs would not have been measuring the same task.")
B("All three are scored against the identical denominator of 37,350 reference words.")
B("A separate notebook recomputes the WERs from the raw stored predictions; it reproduced the original "
  "IndicWhisper figure exactly, confirming the harness itself introduces no difference.")

P("3.4 One denominator subtlety", "h3")
P("The references contain 62 <font face='Courier' size='8'>&lt;unintelligible&gt;</font> transcriber markers. "
  "Removing them &mdash; which OI-WER must do, since no model can produce them &mdash; drops 61 tokens, taking the "
  "denominator from 37,350 to <b>37,289</b>. Every OI-WER figure in this report is therefore compared against a "
  "<b>normalized-text WER computed on that same 37,289-word denominator</b>, not against the raw 37,350-word "
  "figure. Comparing across two different denominators would misattribute part of the change to the metric.")

P("4. How OI-WER is computed", "h2")
P("OI-WER addresses a known problem in Indian-language ASR evaluation: many predictions marked wrong are "
  "in fact legitimate alternative spellings. The published method (arXiv:2603.00941) replaces each single "
  "reference word with a <b>set of acceptable variants</b>; a prediction matching any variant is counted as "
  "a hit rather than a substitution.")

P("4.1 Algorithm", "h3")
P("The reference is represented as an ordered sequence of <i>segments</i>, each carrying one or more accepted "
  "token sequences. Alignment then runs as a dynamic program over (segment index, prediction position), "
  "where each segment may be matched by any of its variants &mdash; and a variant may span multiple words, so "
  "one reference word can legitimately match two predicted words and vice versa. The error counting and "
  "corpus aggregation are otherwise identical to Section 3, and <b>the denominator remains the reference "
  "word count</b>, so WER and OI-WER stay directly comparable.")

P("4.2 Where the variants come from", "h3")
P("Two independent sources are used, and the report separates their contributions rather than reporting only "
  "the combined figure.")

P("Source A &mdash; deterministic rules", "h3")
B("<b>Inverse Text Normalization (ITN)</b> &mdash; spans of Telugu number words are parsed to their numeric "
  "value and the digit form is added as an accepted variant, so " + TE("ఎనిమిది") + " and "
  "<font face='Courier' size='8'>8</font> match. The parser handles units, teens, tens, and the multipliers "
  "hundred / thousand / lakh / crore, including compound values such as 399,999.")
B("<b>Compound merging and splitting</b> &mdash; a reference token matches a group of predicted tokens (or the "
  "reverse) when their concatenations are identical. This is pure string equality, requiring no lexicon, and "
  "handles Telugu agglutination.")
B("<b>Annotation tag removal and surface normalization</b> &mdash; &lt;unintelligible&gt; markers are removed; "
  "Unicode NFC, punctuation and symbol stripping, and case folding for embedded Latin text are applied.")

P("Source B &mdash; a local LLM (gemma3:4b via Ollama)", "h3")
P("The paper generates variants with Gemini-2.5-Pro and then has native speakers review them. No LLM API was "
  "available for this project, so the largest Gemma that fits in 5 GB of VRAM was run locally over all 3,295 "
  "references, prompted with AI4Bharat's own expert-verified Telugu guideline examples, taken from their "
  "published repository.")
P("A 4B model is not a substitute for Gemini plus human review, and it behaved accordingly: alongside genuine "
  "spelling variants it produced translations, synonyms, and invented words. This matters more than it might "
  "appear, because <b>a bad variant can only ever make a score look better</b> &mdash; the alignment takes the "
  "minimum over variants, so an unfiltered hallucination silently forgives a real error. Every LLM-proposed "
  "variant is therefore passed through a filter before it can affect any score:")
B("<b>Anchoring.</b> A proposed group is discarded entirely unless it contains the reference word itself, "
  "copied exactly. This drops groups where the model rewrote or invented the original, and it keeps the "
  "reference word count fixed, so the denominator cannot drift.")
B("<b>Consonant skeleton.</b> Telugu vowel signs, virama and anusvara are stripped from both strings; the "
  "remaining consonant skeletons must agree within one edit. Spelling variation preserves the skeleton; "
  "substituting a different word does not.")
B("<b>Edit ratio.</b> Character edit distance must be within 34% of the reference word's length, or the two "
  "must be identical once spaces are removed (which admits compound splits and merges).")
P("Worked examples, all taken from actual gemma3:4b output on this dataset: "
  + TE("రీచ్") + " &rarr; " + TE("రిచ్") + " is accepted (same skeleton, one edit); "
  + TE("తెలంగాణలో") + " &rarr; " + TE("తెలంగాణ లో") + " is accepted (compound split); "
  + TE("కావడానికి") + " &rarr; " + TE("చేయడానికి") + " is rejected (different word); "
  + TE("ప్లేసెస్") + " &rarr; " + TE("ప్రదేశాలు") + " is rejected (translation); "
  + TE("నేను") + " &rarr; " + TE("నా") + " is rejected (different word).", "body")

if SLOTS:
    table(
        [["LLM proposal stage", "Count"],
         ["Word groups proposed by gemma3:4b", "{:,}".format(SLOTS.get("slots", 0))],
         ["Groups anchored to a reference span", "{:,}".format(SLOTS.get("anchored", 0))],
         ["Alternative spellings offered", "{:,}".format(SLOTS.get("variants_offered", 0))],
         ["Alternatives that passed the filter", "{:,}".format(SLOTS.get("variants_kept", 0))],
         ["Reference spans actually augmented", "{:,}".format(SLOTS.get("spans_augmented", 0))]],
        [70 * mm, 30 * mm], align_center_from=1)
    _offered = SLOTS.get("variants_offered", 0)
    _kept = SLOTS.get("variants_kept", 0)
    if _offered:
        P("The filter rejected {:.0f}% of what the model proposed.".format(
            100.0 * (_offered - _kept) / _offered), "cap")

P("5. Results", "h2")

P("5.1 Headline", "h3")
table(
    [["Model", "WER", "OI-WER", "Gain", "Decoding"],
     ["IndicConformer", pct(NORM, "indicconformer"), pct(BOTH, "indicconformer"),
      delta(BOTH, "indicconformer"), "greedy CTC"],
     ["IndicWav2Vec", pct(NORM, "indicwav2vec"), pct(BOTH, "indicwav2vec"),
      delta(BOTH, "indicwav2vec"), "greedy CTC"],
     ["IndicWhisper", pct(NORM, "indicwhisper"), pct(BOTH, "indicwhisper"),
      delta(BOTH, "indicwhisper"), "autoregressive"]],
    [38 * mm, 26 * mm, 26 * mm, 22 * mm, 34 * mm], highlight_rows=[1])
P("WER here is the normalized-text figure on the 37,289-word denominator (Section 3.4). Ranking is identical "
  "under both metrics: IndicConformer &lt; IndicWav2Vec &lt; IndicWhisper &mdash; OI-WER reorders nothing.", "cap")

P("5.2 Rules against LLM", "h3")
P("Because the two variant sources are independent, each was scored alone and then together.")
table(
    [["Configuration"] + [LABELS[m] for m in MODELS],
     ["WER, raw text (37,350 words)"] + [pct(RAW, m) for m in MODELS],
     ["WER, normalized (37,289 words)"] + [pct(NORM, m) for m in MODELS],
     ["OI-WER, rules only"] + [pct(RULES, m) for m in MODELS],
     ["OI-WER, LLM variants only"] + [pct(LLM, m) for m in MODELS],
     ["OI-WER, rules + LLM"] + [pct(BOTH, m) for m in MODELS]],
    [52 * mm, 32 * mm, 32 * mm, 32 * mm], highlight_rows=[5])

P("Substitutions removed relative to the normalized-text baseline:", "body")
table(
    [["Variant source"] + [LABELS[m] for m in MODELS],
     ["Rules only"] + ["{:,}".format(val(NORM, m, "substitutions") - val(RULES, m, "substitutions"))
                       for m in MODELS],
     ["LLM only"] + ["{:,}".format(val(NORM, m, "substitutions") - val(LLM, m, "substitutions"))
                     for m in MODELS],
     ["Rules + LLM"] + ["{:,}".format(val(NORM, m, "substitutions") - val(BOTH, m, "substitutions"))
                        for m in MODELS]],
    [52 * mm, 32 * mm, 32 * mm, 32 * mm])
P("The improvement is almost entirely in substitutions, which is the expected signature: accepting a variant "
  "can only convert a wrong word into a correct one.", "cap")

P("5.3 What the variants corrected", "h3")
B("<b>The digit penalty was entirely one-sided.</b> ITN removed 124 substitutions from IndicWhisper and zero "
  "from IndicConformer, because IndicConformer never emits digits while IndicWhisper does so in 129 utterances. "
  "Cases such as reference &lsquo;eight&rsquo; against prediction &lsquo;8&rsquo; were perfect recognitions previously scored as "
  "complete errors.")
B("<b>Merge/split was the larger rule effect</b>, as expected for an agglutinative language: a single Telugu "
  "word written by the model with an internal space was previously counted as two separate errors.")
B("<b>The LLM contributed a different class of correction</b> &mdash; matra and vowel-length differences and "
  "loanword spellings, which no string rule can produce without a lexicon. This is the category the rules "
  "cannot reach, which is why the two sources are largely additive.")

P("Representative corrections &mdash; reference against prediction, identical words differing only in spacing, "
  "number format, or vowel length:", "body")
P("హైదరాబాద్కి &nbsp;/&nbsp; "
  "హైదరాబాద్ కి"
  " &nbsp;&nbsp;&middot;&nbsp;&nbsp; "
  "మూడువేల &nbsp;/&nbsp; మూడు వేల"
  " &nbsp;&nbsp;&middot;&nbsp;&nbsp; "
  "ఎనిమిది &nbsp;/&nbsp; 8"
  " &nbsp;&nbsp;&middot;&nbsp;&nbsp; "
  "రీచ్ &nbsp;/&nbsp; రిచ్", "te")

P("6. Verification of the scoring code", "h2")
P("The OI-WER engine was written from scratch, so it was validated before being trusted.")

P("6.1 Against the authors' own implementation", "h3")
P("AI4Bharat have since released their reference scorer. It is vendored into this repository "
  "(<font face='Courier' size='8'>oiwer/vendor/oiwer_core.py</font>, MIT) and both engines were run over the "
  "identical inputs.")
if crosscheck:
    table(
        [["Configuration", "Model", "Our engine", "AI4Bharat scorer"]]
        + [["no variants", LABELS[m], cross("no variants", m, "ours_percent") + "%",
            cross("no variants", m, "official_percent") + "%"] for m in MODELS]
        + [["rules + LLM", LABELS[m], cross("rules + LLM", m, "ours_percent") + "%",
            cross("rules + LLM", m, "official_percent") + "%"] for m in MODELS],
        [34 * mm, 34 * mm, 30 * mm, 34 * mm])
    P("With variants switched off the two engines agree to the digit on all three models &mdash; the strongest "
      "available evidence that our alignment and counting are correct. With variants enabled our figures are "
      "slightly lower, because the AI4Bharat scorer greedily commits to one multi-word variant <i>before</i> "
      "running its dynamic program, while ours searches all variants inside the program and is therefore "
      "optimal by construction.", "cap")

P("6.2 Other checks", "h3")
B("With trivial single-variant segments the engine reduces to standard WER and reproduces both published "
  "figures to six decimal places (27.839357% and 56.736278%), matching the independent jiwer computation.")
B("Nine unit tests cover exact match, substitution, deletion, insertion, multi-word variants, numeral "
  "variants, and a negative case confirming a genuine error is still counted as an error.")
B("Eleven further unit tests pin down the LLM variant filter, including the specific hallucinations observed "
  "in gemma3:4b output &mdash; a translation, a synonym, a different word, an invented extra group &mdash; each "
  "asserted to be rejected, and genuine matra and compound-split variants asserted to be accepted.")
B("The invariant OI-WER &le; WER was verified to hold for every configuration.")

P("7. Interpreting the model gap", "h2")
B("<b>Domain match.</b> IndicConformer is AI4Bharat's own model, trained on IndicVoices, so this validation "
  "split is in-domain for it; IndicWhisper was fine-tuned on a different Telugu mixture and faces a domain "
  "shift. The supportable claim is that IndicConformer is stronger <i>on this benchmark</i>, not that it is a "
  "better Telugu ASR model in general.")
B("<b>Decoding asymmetry.</b> IndicConformer and IndicWav2Vec used greedy CTC with no language model; "
  "IndicWhisper generates autoregressively with an implicit internal language model.")
B("<b>Different failure modes.</b> IndicWhisper's much higher deletion and insertion counts indicate dropped "
  "content and hallucinated repetition on short utterances, rather than uniform mis-recognition.")
B("<b>OI-WER helps the weaker models most.</b> The gain grows with the error rate, which is what the metric is "
  "for: a model that is merely spelling things differently is penalised more heavily by plain WER the more "
  "output it produces.")

P("8. Limitations", "h2")
B("<b>This is still an approximation of OI-WER, not a reproduction of it.</b> The published method pairs "
  "Gemini-2.5-Pro with review by 61 native speakers across seven variation categories. Ours pairs hand-written "
  "rules with an unreviewed 4B local model behind a conservative filter. The filter is deliberately strict, so "
  "it will reject genuine variants as well as bad ones; our figures should be read as a <b>lower bound</b> on "
  "the true OI-WER gain.")
B("<b>No native-speaker review has taken place.</b> Neither the Telugu number lexicon nor the accepted LLM "
  "variants have been checked by a Telugu speaker. This is the single most valuable next step, and it is "
  "cheap: the accepted variants are stored and can be reviewed as a flat list.")
B("<b>The LLM variants are unvalidated against a gold set.</b> We can show what the filter rejects, but "
  "without AI4Bharat's own annotations we cannot measure what fraction of true variants it misses.")
B("<b>No human-perceived WER reference.</b> The paper validates OI-WER against post-edited human transcripts; "
  "we have no such reference, so we cannot measure how much closer our metric sits to human judgement.")

P("9. Note on IndicWav2Vec decoding", "h2")
P("AI4Bharat never published Telugu IndicWav2Vec weights in HuggingFace format &mdash; the hosted repository is "
  "empty &mdash; so the only Telugu artifact is a Fairseq checkpoint tied to Python 3.10 and torch 1.13. It was "
  "evaluated by running that stack in an isolated virtual environment, driven as a subprocess, with audio "
  "pre-dumped to disk so the old environment never touches the modern data libraries. The run completed all "
  "3,295 utterances at roughly 8 utterances per second.")
P("One caveat matters for comparison with published figures: AI4Bharat's official IndicWav2Vec pipeline decodes "
  "with KenLM and a lexicon. That was deliberately skipped here so that all three models are compared without an "
  "external language model. Our figure is therefore LM-free and should not be read against published "
  "numbers obtained with LM decoding.")
P("A decoding detail worth recording: in Fairseq's CTC convention the blank symbol is the dictionary's "
  "bos index, not pad. Filtering the wrong index leaves every blank frame in the output, which renders as "
  "literal markup and produces a near-100% error rate that looks like model failure rather than a decoding bug.")

P("10. Next steps", "h2")
B("Native-speaker review of the accepted LLM variants and the Telugu number lexicon &mdash; the cheapest "
  "available improvement in confidence.")
B("Confirm whether AI4Bharat's orthographically-informed IndicVoices benchmark (Telugu: ~2.6K utterances, "
  "92.7K variations, native-speaker reviewed) can be obtained directly &mdash; this would replace our "
  "approximation with the reference annotations.")
B("Error analysis from the stored predictions: WER against utterance duration, per-speaker breakdown, "
  "substitution patterns, and CER alongside WER. No GPU inference required.")
B("Optionally re-run IndicWav2Vec with KenLM + lexicon decoding to match AI4Bharat's official pipeline.")
B("Phase 2: Demucs source separation, VAD segmentation, YouTube subtitles, MFA alignment, a 10-hour Telugu "
  "dataset, and fine-tuning all three models &mdash; re-evaluating with the same WER and OI-WER harness so the "
  "improvement from fine-tuning is measurable per model.")

SP(8)
story.append(HRFlowable(width="100%", thickness=0.6, color=midgray, spaceAfter=4))
P("Repository: github.com/varshhhy7/DhwaniLab &nbsp;&middot;&nbsp; Prepared by Mani Varshith &nbsp;&middot;&nbsp; DhwaniLab", "meta")

doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=15 * mm, bottomMargin=13 * mm,
                        leftMargin=17 * mm, rightMargin=17 * mm,
                        title="DhwaniLab Phase 1 Report", author="Mani Varshith")
doc.build(story)
print("written:", OUT)
