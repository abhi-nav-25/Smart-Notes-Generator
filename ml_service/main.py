import io
import json
import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI(title="Smart Notes Generator - ML Service")

GEMINI_MODEL = "gemini-3.6-flash"


class NotesRequest(BaseModel):
    text: str


class QuizRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Text extraction / cleanup helpers (generic - not tied to any document type)
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    normalized = text.replace('\u00a0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'(?<!\n)(?=\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{2,100})', '\n', normalized)

    cleaned_lines = []
    for raw_line in normalized.split('\n'):
        spaced_columns = re.split(r'\s{2,}', raw_line.strip())
        if '\t' in raw_line:
            line = re.sub(r'\t+', ': ', raw_line).strip()
        elif len(spaced_columns) == 2:
            line = ': '.join(spaced_columns)
        else:
            line = re.sub(r'\s+', ' ', raw_line).strip()
        if not line or re.fullmatch(r'function\s+description:?', line, re.IGNORECASE):
            continue

        line = re.sub(r'\s*:\s*', ': ', line)
        cleaned_lines.append(line)

    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'(?im)^function\s+description\s*:\s*', '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def prepare_source_text(text: str, max_chars: int = 200000) -> str:
    """
    Removes unnecessary repetition and caps the source size before it is
    sent to Gemini. max_chars is a generous safety cap, not a working
    limit - Gemini's context window comfortably handles typical document
    sizes in a single call, so no chunking is needed.
    """
    text = clean_extracted_text(text)

    lines = []
    seen = set()

    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()

        if not line:
            continue

        key = line.lower()

        if key not in seen:
            lines.append(line)
            seen.add(key)

    cleaned = '\n'.join(lines)

    # Strip a document title when it's glued onto the first numbered topic
    # or the first question-style sentence (generic across document types).
    cleaned = re.sub(
        r"^(?:Introduction to|Introduction:|Title:).*?(?=\s+\d+\.\s+|\s+What is\b|\s+What are\b)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    # Remove numbering from topic lines
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", "", cleaned)

    # Strip generic worksheet/exam-style instruction lines ("Understand X.",
    # "Study for the quiz on Y.") that sometimes precede the real content.
    cleaned = re.sub(
        r'^(?:(?:Understand|Learn|Read|Use this text)[^.?!]*[.?!]\s*)+',
        '',
        cleaned,
        flags=re.IGNORECASE
    ).strip()

    instruction_patterns = [
        r"^understand\b",
        r"^learn\b",
        r"^read\b",
        r"^use this text\b",
        r"^prepare for\b",
        r"^study for\b",
    ]

    filtered_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in instruction_patterns):
            continue
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines)
    return cleaned[:max_chars]


# ---------------------------------------------------------------------------
# Gemini client / call helper (shared by summary, notes, and quiz)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_gemini_client():
    env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend", ".env")
    )
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API key is not configured.",
        )

    return genai.Client(api_key=api_key)


def _call_gemini(
    prompt: str,
    *,
    response_mime_type: str = "text/plain",
    temperature: float = 0.3,
    max_output_tokens: int = 2000,
    retries: int = 3,
    retry_delay_seconds: int = 5,
) -> str:
    client = _get_gemini_client()
    response = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type=response_mime_type,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            break
        except Exception as exc:
            error_text = str(exc)

            if "503" not in error_text and "UNAVAILABLE" not in error_text:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini generation failed: {error_text}",
                ) from exc

            print(f"Gemini temporarily unavailable. Retry {attempt + 1}/{retries}")
            if attempt < retries - 1:
                time.sleep(retry_delay_seconds)
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Gemini is temporarily unavailable. Please try again shortly.",
                ) from exc

    if response is None or not getattr(response, "text", None):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned an empty response.",
        )

    return response.text


# ---------------------------------------------------------------------------
# Generic text-quality helpers (apply to output from any source document)
# ---------------------------------------------------------------------------

def _clean_summary_output(summary: str) -> str:
    sentences = [
        part.strip()
        for part in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', summary))
        if part.strip()
    ]

    unique_sentences = []
    seen = set()

    for sentence in sentences:
        sentence = re.sub(r'^(?:summary|overview)\s*:\s*', '', sentence, flags=re.IGNORECASE).strip()
        key = sentence.lower()
        if sentence and key not in seen:
            unique_sentences.append(sentence)
            seen.add(key)

    return ' '.join(unique_sentences)


def _remove_instruction_sentences(text: str) -> str:
    instruction_patterns = [
        r"^understand\b",
        r"^learn\b",
        r"^read\b",
        r"^use this text\b",
        r"^prepare for\b",
        r"^study for\b",
        r"^review\b",
    ]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(re.match(pattern, sentence, flags=re.IGNORECASE) for pattern in instruction_patterns):
            continue
        filtered.append(sentence)

    return " ".join(filtered)


def _remove_similar_sentences(text: str) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    selected = []
    selected_words = []

    for sentence in sentences:
        words = set(re.findall(r"[a-z]{4,}", sentence.lower()))
        is_similar = False

        for previous_words in selected_words:
            if not words:
                continue
            overlap = len(words & previous_words) / len(words)
            if overlap >= 0.65:
                is_similar = True
                break

        if not is_similar:
            selected.append(sentence)
            selected_words.append(words)

    return " ".join(selected)


def _remove_repeated_phrases(text: str) -> str:
    return re.sub(r"\b(\w+(?:\s+\w+){1,4})\s+\1\b", r"\1", text, flags=re.IGNORECASE)


def _build_extractive_summary(text: str) -> str:
    """Pure sentence-extraction fallback, used only if Gemini is unavailable."""
    content_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^\d+(?:\.\d+)*[.)]?\s+[^.!?]+$', stripped):
            continue
        if re.match(r'^(understand|learn|read|use this text|prepare for|study for)\b', stripped, flags=re.IGNORECASE):
            continue
        content_lines.append(stripped)

    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', ' '.join(content_lines)) if s.strip()
    ]

    selected = []
    seen = set()

    for sentence in sentences:
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) == 10:
            break

    return ' '.join(selected)


def _summary_needs_fallback(summary: str, source: str) -> bool:
    if not summary:
        return True

    summary_words = summary.split()
    if len(summary_words) < 40:
        return True
    if len(summary_words) > 300:
        return True

    summary_sentences = [s for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
    if len(summary_sentences) < 2:
        return True

    source_words = set(re.findall(r"[A-Za-z]{5,}", source.lower()))
    summary_words_set = set(re.findall(r"[A-Za-z]{5,}", summary.lower()))
    overlap = len(source_words & summary_words_set)

    # Reject summaries that contain too little source-specific content.
    return overlap < 8


def _summary_is_instruction_heavy(summary: str) -> bool:
    instruction_patterns = [
        r"^understand\b",
        r"^learn\b",
        r"^read\b",
        r"^use this text\b",
        r"^prepare for\b",
        r"^study for\b",
    ]

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]
    if not sentences:
        return True

    instruction_count = sum(
        any(re.match(pattern, sentence, flags=re.IGNORECASE) for pattern in instruction_patterns)
        for sentence in sentences
    )

    return instruction_count >= max(2, len(sentences) // 2)


def _select_note_sentences(sentences: list, start: int, end: int) -> list:
    return [f"- {s}" for s in sentences[start:end] if len(s.split()) >= 5]


def _build_markdown_notes(text: str) -> str:
    """Pure sentence-extraction fallback, used only if Gemini is unavailable."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    overview = " ".join(sentences[:3])
    overview = re.sub(
        r"^(?:Introduction to|Introduction:|Title:)\s+.*?(?=\s+1\.\s+|\s+What is\b|\s+What are\b)",
        "",
        overview,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    key_concepts = _select_note_sentences(sentences, 3, 8)
    components = _select_note_sentences(sentences, 8, 12)
    details = _select_note_sentences(sentences, 12, 16)
    applications = _select_note_sentences(sentences, 16, 20)

    revision = []
    for sentence in sentences[-5:]:
        if len(sentence.split()) < 5:
            continue
        already_used = any(
            sentence.lower().strip() in item.lower().strip()
            or item.lower().strip() in sentence.lower().strip()
            for item in sentences[:20]
        )
        if not already_used:
            revision.append(f"- {sentence}")

    return f"""# Study Notes

## Overview

{overview or "Overview not available."}

## Key Concepts

{chr(10).join(key_concepts) or "- Not covered in the source document."}

## Main Components or Classification

{chr(10).join(components) or "- Not covered in the source document."}

## Important Details

{chr(10).join(details) or "- Not covered in the source document."}

## Examples or Applications

{chr(10).join(applications) or "- Not covered in the source document."}

## Quick Revision

{chr(10).join(revision) or "- Not covered in the source document."}
"""


def notes_are_valid(notes: str) -> bool:
    required_sections = ["Overview", "Key Concepts", "Quick Revision"]
    return (
        isinstance(notes, str)
        and len(notes.split()) >= 40
        and all(section.lower() in notes.lower() for section in required_sections)
    )


def _notes_need_fallback(notes: str, source: str) -> bool:
    if not notes or len(notes.split()) < 40:
        return True

    source_terms = set(re.findall(r"[A-Za-z]{5,}", source.lower()))
    note_terms = set(re.findall(r"[A-Za-z]{5,}", notes.lower()))
    overlap = len(source_terms & note_terms)

    return overlap < 12


# ---------------------------------------------------------------------------
# Markdown cleanup (generic - applies to notes generated from ANY document)
# ---------------------------------------------------------------------------

def _unescape_markdown(text: str) -> str:
    """
    Some LLMs occasionally escape Markdown punctuation even where it isn't
    needed (e.g. "\\##", "\\*"), which then shows up to the reader as a
    literal backslash instead of proper formatting. Strip the escaping
    wherever it occurs, regardless of what document it came from.
    """
    text = re.sub(r'\\+(#{1,6})', r'\1', text)
    text = re.sub(r'\\([*_`~>+.!-])', r'\1', text)
    return text


def _ensure_headings_on_own_line(text: str) -> str:
    """
    If a heading marker ends up mid-line (e.g. exposed after unescaping),
    push it onto its own line so it renders as an actual heading.
    """
    return re.sub(r'(?<!\n)(?<!\A)(#{1,6}\s+\S)', r'\n\n\1', text)


# A small, topic-agnostic list of linking/definition verbs used to spot a
# heading-like phrase that has been glued directly onto the sentence that
# follows it (e.g. "Network Topologies A network topology is ..."). None of
# these verbs are tied to any particular subject matter.
_HEADING_VERB_RE = (
    r'(?:is|are|was|were|refers to|means|describes|consists of|includes|'
    r'provides|defines|represents|involves|enables?|allows?|continues?\s+to|'
    r'helps?|contains?|comprises?)'
)

# Matches: an optional bullet marker, then a short run (1-6 words) of
# Title-Case words sitting at the very start of a line, immediately
# followed by what looks like the start of a new sentence (a capitalized
# word, then a short run of lowercase words, then a linking/definition
# verb). This is the generic signature of a heading that got merged into
# the paragraph or bullet that should follow it.
_GLUED_HEADING_RE = re.compile(
    r'^(?P<indent>[ \t]*)(?P<bullet>[-*]\s+)?'
    r'(?P<heading>(?:[A-Z][\w/&-]*\s+){0,5}[A-Z][\w/&-]*)\s+'
    r'(?P<sentence>[A-Z][\w/&-]*(?:\s+[a-z][\w/&-]*){0,8}?\s+' + _HEADING_VERB_RE + r'\b.*)$',
    re.MULTILINE,
)


def _repair_glued_headings(text: str) -> str:
    """
    Repairs headings that were merged into the following text or into a
    bullet point (Requirement A). Purely structural/pattern-based - it
    never references specific heading names or topics, so it behaves the
    same way regardless of what the document is about. Known limitation:
    a sentence that happens to open with a multi-word proper-noun subject
    can occasionally be mis-split; this only matters at the very start of
    a line, which keeps the false-positive surface small.
    """
    def _replace(match):
        heading = match.group('heading').strip()
        sentence = match.group('sentence').strip()
        if len(heading.split()) > 6 or len(heading) > 60 or len(heading) < 2:
            return match.group(0)
        return f"\n\n### {heading}\n\n{sentence}"

    return _GLUED_HEADING_RE.sub(_replace, text)


def _convert_tables_to_bullets(text: str) -> str:
    """
    Converts any pipe-delimited Markdown table in the text into a plain
    bulleted list, using the required format:
    "- **<first column>:** <remaining columns>." The first row of the
    block is treated as a header and skipped; the first column of every
    data row becomes the concept, and the remaining columns (joined) become
    its explanation. Tables must never be reproduced as Markdown tables.
    This only touches blocks that already look tabular (multiple
    '|'-delimited rows), so it never invents a table - or bullets - from
    ordinary prose, and it never fabricates a value for a missing cell.
    """
    lines = text.split('\n')
    output = []
    i = 0
    n = len(lines)

    def _cells(row):
        return [c.strip() for c in row.strip().strip('|').split('|')]

    separator_re = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$')

    while i < n:
        line = lines[i]
        if line.count('|') >= 1:
            block = [line]
            j = i + 1
            while j < n and lines[j].count('|') >= 1:
                block.append(lines[j])
                j += 1

            # Drop an optional Markdown separator row ("---|---").
            data_rows = [row for row in block if not separator_re.match(row)]

            bullet_lines = []
            if len(data_rows) >= 2:
                for row in data_rows[1:]:
                    cells = _cells(row)
                    if not cells or not cells[0]:
                        continue
                    concept = cells[0]
                    explanation_cells = [c for c in cells[1:] if c]
                    if not explanation_cells:
                        continue
                    explanation = ' '.join(explanation_cells).strip()
                    bullet_lines.append(f"- **{concept}:** {explanation}")

            if bullet_lines:
                output.extend(bullet_lines)
                i = j
                continue

            # Not a clear enough table structure - leave the lines as they
            # were rather than guessing at a reconstruction.
            output.extend(block)
            i = j
            continue

        output.append(line)
        i += 1

    return '\n'.join(output)


def _bullets_to_pseudo_sentences(text: str) -> str:
    """
    Turns "- **Concept:** Explanation" bullet lines (as produced by
    _convert_tables_to_bullets) into plain "Concept: Explanation."
    sentences, so the local sentence-extraction fallback below can still
    surface information that originally came from a source table.
    """
    def _replace(match):
        concept = match.group(1).strip()
        explanation = match.group(2).strip()
        if not explanation.endswith(('.', '!', '?')):
            explanation += '.'
        return f"{concept}: {explanation}"

    return re.sub(r'^-\s+\*\*(.+?):\*\*\s*(.+)$', _replace, text, flags=re.MULTILINE)


def _prepare_source_for_fallback(text: str) -> str:
    """
    Pre-processes source text for the local (non-Gemini) fallback
    generators so that table content is represented as plain sentence-like
    lines instead of being silently dropped by the extractive approach.
    """
    return _bullets_to_pseudo_sentences(_convert_tables_to_bullets(text))


# Generic list-context words that indicate a run of single digits is
# actually separate enumerated items (e.g. "steps 1 2 3") rather than a
# number that got split apart by OCR/PDF extraction. None of these are
# specific to any one document's subject matter.
_DIGIT_LIST_CONTEXT_RE = (
    r'(?:figure|figures|table|tables|step|steps|chapter|chapters|section|'
    r'sections|question|questions|option|options|type|types|item|items|'
    r'number|numbers|point|points|rule|rules|level|levels|phase|phases|'
    r'stage|stages)\s*:?\s*$'
)

_SPLIT_DIGITS_RE = re.compile(r'(?<![\d.])\d(?:\s+\d){2,}(?![\d.])')


def _repair_split_digit_numbers(text: str) -> str:
    """
    Detects a run of 3+ standalone single digits separated by whitespace
    (e.g. "1 9 6") and joins them into one number ("196"). This pattern
    is an unambiguous signature of OCR/PDF extraction corruption -
    natural language essentially never contains three or more consecutive
    standalone single-digit numbers. Runs preceded by a generic listy word
    ("step 1 2 3") are left untouched since those really are separate
    items, not a corrupted number (Requirement F).
    """
    def _replace(match):
        preceding = text[max(0, match.start() - 20):match.start()].lower()
        if re.search(_DIGIT_LIST_CONTEXT_RE, preceding):
            return match.group(0)
        return re.sub(r'\s+', '', match.group(0))

    return _SPLIT_DIGITS_RE.sub(_replace, text)


def _remove_duplicate_note_lines(notes: str) -> str:
    lines = notes.splitlines()
    result = []
    seen = set()

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip()).lower()
        if not normalized:
            result.append(line)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)

    return "\n".join(result)


def _remove_duplicate_sections(notes: str) -> str:
    sections = re.split(r"(?=^##\s+)", notes, flags=re.MULTILINE)

    result = []
    seen_content = set()

    for section in sections:
        normalized = re.sub(r"\s+", " ", section.strip()).lower()
        if not normalized:
            continue

        content = re.sub(r"^##\s+[^\n]+", "", normalized, count=1).strip()

        if content and content in seen_content:
            continue
        if content:
            seen_content.add(content)

        result.append(section.strip())

    return "\n\n".join(result)


def _normalize_markdown_spacing(notes: str) -> str:
    notes = notes.replace("\r\n", "\n").replace("\r", "\n")
    notes = "\n".join(line.rstrip() for line in notes.splitlines())
    notes = re.sub(r"\n{3,}", "\n\n", notes)
    notes = re.sub(r"(?m)^\s*(#{1,6}\s+[^\n]+)\s*$", r"\n\1\n", notes)
    notes = re.sub(r"\n{3,}", "\n\n", notes)
    return notes.strip()


def _remove_document_title(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    title_pattern = re.compile(r"^(?:Introduction to|Introduction:|Title:)\s+.+$", flags=re.IGNORECASE)
    return "\n".join(line for line in lines if not title_pattern.match(line.strip())).strip()


def clean_notes_output(notes: str) -> str:
    """
    Cleans common formatting problems in generated notes. Every rule here
    is generic (title-glue, escaped markdown, glued headings, table-to-
    bullet conversion, split-digit corruption, duplicate lines/sections,
    stray code fences) and applies regardless of what the source document
    is about.
    """
    if not isinstance(notes, str) or not notes.strip():
        return "# Study Notes\n\nNo notes could be generated."

    notes = notes.strip()

    notes = re.sub(r"(?m)^\s*\d+\.\s+", "", notes)

    notes = re.sub(
        r"^(?:Introduction to|Introduction:|Title:).*?(?=\s+\d+\.\s+|\s+What is\b|\s+What are\b)",
        "",
        notes,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    notes = _unescape_markdown(notes)
    notes = _ensure_headings_on_own_line(notes)

    # Repair headings that were merged into surrounding text or bullets,
    # then convert any table the model returned into readable bullets.
    notes = _repair_glued_headings(notes)
    notes = _convert_tables_to_bullets(notes)
    notes = _repair_split_digit_numbers(notes)

    notes = re.sub(r"(?m)^(#{1,6}\s+.+)\n+\1\s*$", r"\1", notes, flags=re.IGNORECASE)
    notes = _remove_document_title(notes)

    if not notes:
        return "# Study Notes\n\nNo notes could be generated."

    if notes.startswith("```"):
        lines = notes.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        notes = "\n".join(lines).strip()

    unwanted_starts = [
        "Here are the notes:",
        "Here are the notes",
        "These are the notes:",
        "These are the notes",
    ]
    for phrase in unwanted_starts:
        if notes.lower().startswith(phrase.lower()):
            notes = notes[len(phrase):].strip()

    if not notes.startswith("#"):
        lines = notes.splitlines()
        first_line = lines[0].strip() if lines else "Study Notes"
        remaining_lines = "\n".join(lines[1:])
        notes = f"# {first_line}\n\n{remaining_lines}".strip()

    notes = _remove_duplicate_note_lines(notes)
    notes = _remove_duplicate_sections(notes)
    notes = _normalize_markdown_spacing(notes)

    # Second pass in case normalization exposed more escaped markdown.
    notes = _unescape_markdown(notes)
    notes = re.sub(r"\n{3,}", "\n\n", notes)

    return notes.strip()


# ---------------------------------------------------------------------------
# Source-grounding validation (generic - checks notes against the source
# document itself, never against topic-specific rules or facts)
# ---------------------------------------------------------------------------

def _line_is_grounded(line: str, source_words: set, source_acronyms: set) -> bool:
    """
    A line counts as grounded if it shares enough substantive vocabulary
    with the source document, and if it does not introduce an acronym (a
    named algorithm, protocol, etc.) that never appears in the source at
    all - a generic signal of injected outside knowledge. Top-level "##"
    section headings and the standard "not covered" placeholder are always
    kept untouched; only actual content lines and content sub-headings
    ("###" and deeper) are checked.
    """
    stripped = line.strip()
    if not stripped:
        return True
    if re.match(r'^##(?!#)\s+', stripped):
        return True
    if stripped.lstrip('-* ').strip().lower() == 'not covered in the source document.':
        return True

    heading_match = re.match(r'^#{3,6}\s+(.*)$', stripped)
    check_text = heading_match.group(1) if heading_match else stripped

    # Acronym check: a capitalized short token (e.g. a named algorithm or
    # protocol abbreviation) that never appears in the source at all is a
    # strong, topic-agnostic sign of injected external knowledge.
    line_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', check_text))
    if line_acronyms and not line_acronyms.issubset(source_acronyms):
        return False

    content_words = set(re.findall(r'[a-z]{4,}', check_text.lower()))
    if not content_words:
        return True

    overlap = len(content_words & source_words)
    overlap_ratio = overlap / len(content_words)

    # Short lines need a higher overlap ratio, since a single unrelated
    # word can dominate a short line's word count.
    threshold = 0.35 if len(content_words) > 6 else 0.5
    return overlap_ratio >= threshold


def _strip_ungrounded_lines(notes: str, source: str) -> str:
    """
    Validation step (Requirement C): removes individual lines whose
    vocabulary doesn't meaningfully overlap with the source document, or
    that introduce an acronym absent from the source entirely. This is how
    invented/hallucinated content - including unsupported named concepts -
    gets filtered out without discarding the entire notes document.
    """
    source_words = set(re.findall(r'[a-z]{4,}', source.lower()))
    source_acronyms = set(re.findall(r'\b[A-Z]{2,6}\b', source))
    if not source_words:
        return notes

    kept_lines = [
        line for line in notes.splitlines()
        if _line_is_grounded(line, source_words, source_acronyms)
    ]
    return '\n'.join(kept_lines)


def _fill_empty_sections(notes: str) -> str:
    """
    After grounding validation removes unsupported lines, a section can
    be left with nothing but its heading. Make that explicit instead of
    leaving a blank section under the heading.
    """
    sections = re.split(r'(?=^##\s+)', notes, flags=re.MULTILINE)
    rebuilt = []

    for section in sections:
        stripped_section = section.strip()
        if not stripped_section:
            continue
        if stripped_section.startswith('##'):
            lines = stripped_section.splitlines()
            heading = lines[0]
            body = '\n'.join(lines[1:]).strip()
            if not body:
                body = 'Not covered in the source document.'
            rebuilt.append(f"{heading}\n\n{body}")
        else:
            rebuilt.append(stripped_section)

    return '\n\n'.join(rebuilt)


def _shared_long_ngram_exists(text_a: str, text_b: str, max_window: int = 15, min_window: int = 8) -> bool:
    """
    Checks whether text_a contains a long word-sequence that also appears
    verbatim in text_b. Used to detect near-verbatim copying regardless of
    topic - it's purely a structural (shared n-gram length) check.
    """
    words_a = re.findall(r'\w+', text_a.lower())
    if len(words_a) < min_window:
        return False

    joined_b = ' '.join(re.findall(r'\w+', text_b.lower()))

    window_ceiling = min(max_window, len(words_a))
    for window in range(window_ceiling, min_window - 1, -1):
        for start in range(0, len(words_a) - window + 1):
            phrase = ' '.join(words_a[start:start + window])
            if phrase in joined_b:
                return True
    return False


def _overview_is_copied(overview_text: str, source: str) -> bool:
    """
    Flags an Overview that is essentially lifted from the source rather
    than summarized (Requirement E).
    """
    return _shared_long_ngram_exists(overview_text, source, max_window=15, min_window=8)


def _summary_is_near_verbatim_copy(summary: str, source: str) -> bool:
    """
    Flags a summary that is essentially copied from the source rather than
    paraphrased. Uses a longer minimum shared run than the Overview check,
    since some technical-term overlap in a summary is expected and fine.
    """
    return _shared_long_ngram_exists(summary, source, max_window=20, min_window=12)


def _condensed_overview_fallback(source: str) -> str:
    """Local, no-API-call fallback for an Overview that copied the source."""
    extractive = _build_extractive_summary(_prepare_source_for_fallback(source))
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', extractive) if s.strip()]
    return ' '.join(sentences[:2]) or "Overview not available."


def _fix_overview_section(notes: str, source: str) -> str:
    """
    Checks the Overview section specifically for verbatim copying and
    replaces it with a short, locally-generated condensed version if so
    (Requirement E). Leaves the rest of the notes untouched.
    """
    match = re.search(
        r'(##\s+Overview\s*\n+)(.*?)(?=\n##\s+|\Z)', notes, flags=re.DOTALL | re.IGNORECASE
    )
    if not match:
        return notes

    heading, body = match.group(1), match.group(2).strip()
    if body and _overview_is_copied(body, source):
        body = _condensed_overview_fallback(source)

    return notes[:match.start()] + heading + body + '\n\n' + notes[match.end():]


def _ensure_quick_revision_populated(notes: str) -> str:
    """
    If Quick Revision ended up empty (e.g. every candidate point was
    removed during grounding validation), populate it with a handful of
    short points pulled from whichever other sections actually have
    supported content, instead of leaving it blank (Requirement: Quick
    Revision must not be empty when supported content exists elsewhere).
    Nothing new is introduced - points are copied from already-validated
    sections.
    """
    match = re.search(
        r'(##\s+Quick Revision\s*\n+)(.*?)(?=\n##\s+|\Z)', notes, flags=re.DOTALL | re.IGNORECASE
    )
    if not match:
        return notes

    heading, body = match.group(1), match.group(2).strip()
    if body and body.lower() != 'not covered in the source document.':
        return notes

    donor_sections = [
        'Key Concepts',
        'Main Components or Classification',
        'Important Details',
        'Examples or Applications',
    ]
    revision_points = []
    for section_name in donor_sections:
        section_match = re.search(
            rf'##\s+{re.escape(section_name)}\s*\n+(.*?)(?=\n##\s+|\Z)',
            notes,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not section_match:
            continue
        section_body = section_match.group(1).strip()
        bullet_lines = [
            l.strip() for l in section_body.splitlines()
            if l.strip().startswith('-') and l.strip().lower() != '- not covered in the source document.'
        ]
        if bullet_lines:
            revision_points.append(bullet_lines[0])
        if len(revision_points) >= 5:
            break

    new_body = '\n'.join(revision_points) if revision_points else 'Not covered in the source document.'
    return notes[:match.start()] + heading + new_body + '\n\n' + notes[match.end():]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUMMARY_PROMPT_TEMPLATE = """
Summarize the source document below, clearly and accurately, for a student
studying this material.

Rules:
- Write flowing prose (2 to 4 short paragraphs). Do not use a list or any
  Markdown formatting.
- Paraphrase the source in your own words. Do not copy sentences verbatim
  from the source document.
- Use only information explicitly present in the source document below.
  Do not add external knowledge, examples, or facts that are not stated
  in the source.
- Do not add a conclusion unless the source document itself states one.
- Do not repeat words, phrases, or sentences.
- Preserve important technical terms from the document.
- Combine similar points instead of repeating them.
- Focus only on the main ideas and important details.
- Do not start with a phrase like "This document is about" - begin
  directly with the content.
- Return only the summary text, nothing else.

SOURCE DOCUMENT:
{source_text}
"""

NOTES_PROMPT_TEMPLATE = """
You are an expert academic note-maker. Convert the source document below
into clear, exam-friendly study notes.

Use exactly this Markdown structure, adapting the CONTENT (not the
headings) to whatever the document is actually about:

# Study Notes

## Overview
A concise, paraphrased explanation of the main topic, based only on the
source below.

## Key Concepts
- Definitions, principles, and terminology explicitly present in the
  source.

## Main Components or Classification
- Types, categories, stages, or components explicitly described in the
  source. Do not force unrelated content into this section.

## Important Details
- Other source-supported facts, rules, or explanations that do not fit
  naturally into the sections above.

## Examples or Applications
- Examples or applications, only if the source actually contains them.

## Quick Revision
- 3 to 5 short points that restate information already covered above.
- Do not introduce anything new in this section.

TABLE HANDLING:
- If the source contains a table, read every row and column carefully.
- Do NOT reproduce the table as a Markdown table.
- Convert each table row into a bullet point using this exact format:
  - **<first column>:** <remaining columns, combined into one sentence>.
- Treat the first column as the concept and the remaining columns as its
  explanation.
- Preserve every readable row. Do not merge unrelated rows and do not
  invent missing values.
- Place these bullets under whichever section best matches their content
  (usually Key Concepts, Main Components or Classification, or Important
  Details).

CONTENT RULES:
- Use only information explicitly present in the source document below,
  including information from any tables. Do not add external knowledge,
  examples, explanations, algorithms, definitions, applications, named
  concepts, or conclusions that are not stated in the source.
- Place each fact under the single most relevant section based on its
  meaning. Do not assign content to a section merely because of where it
  appeared in the source (for example, do not just take "the first few
  sentences" for one section and "the next few" for another).
- Do not repeat the same fact in more than one section, except that Quick
  Revision may restate points already made elsewhere.
- If a section has no relevant content in the source, write exactly
  "Not covered in the source document." under that heading.

FORMATTING RULES:
- Every heading must be on its own line. Never merge a heading into the
  sentence or bullet that follows it (for example, never write
  "Network Topologies A network topology is..." on one line - the heading
  and the sentence must be separated).
- Never split a multi-digit number across separate characters or words
  (write "196", never "1 9 6").
- Do not escape Markdown characters with a backslash - write "##" and
  "**", never "\\##" or "\\**".
- Do not wrap the output in code fences.
- Do not include an introductory phrase such as "Here are the notes".
- Return only the Markdown notes, nothing else.

SOURCE DOCUMENT:
{source_text}
"""

QUIZ_PROMPT_TEMPLATE = """
You are an expert university-level assessment designer.

Read the source document and create exactly 5 high-quality multiple-choice
questions.

The questions must test understanding, not simple word matching.

Include one question from each of these types:
- definition/application
- concept identification
- comparison
- scenario-based question
- factual understanding

Create four plausible, closely related options. Make the incorrect options
realistic near-misses based on the source material. Do not use random,
absurd, unrelated, or obviously false distractors. Exactly one option must
be fully correct. Verify the answer and all distractors against the source
before returning the result.

Return ONLY valid JSON in this exact format:
{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option 1",
                "Option 2",
                "Option 3",
                "Option 4"
            ],
            "answer": 0,
            "explanation": "Why this answer is correct"
        }}
    ]
}}

Rules:
- Exactly 5 questions.
- Exactly 4 options per question.
- Options must be distinct.
- All options must belong to the same category and be based on the document.
- Only one option can be correct.
- The answer must be an integer from 0 to 3.
- Every question must be based on the source document.
- Do not invent facts that are not present in the source document.
- Avoid generic questions.
- Do not copy complete sentences directly from the document.
- Do not repeat the document title, headings, or raw extracted text as options.
- Do not use placeholder options.
- Generate meaningful, close, plausible distractors.
- Do not repeat questions.
- Explanations must be concise and document-based.
- Return raw JSON only.

SOURCE DOCUMENT:
{source_text}
"""


# ---------------------------------------------------------------------------
# Quiz JSON parsing / validation
# ---------------------------------------------------------------------------

def _parse_quiz_json(raw_quiz: str):
    candidate = raw_quiz.strip()
    candidate = re.sub(r'^```(?:json)?\s*', '', candidate)
    candidate = re.sub(r'\s*```$', '', candidate).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', candidate, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    questions = payload.get('questions') if isinstance(payload, dict) else None
    if not isinstance(questions, list) or len(questions) != 5:
        return None

    validated = []
    seen_questions = set()
    for item in questions:
        if not isinstance(item, dict):
            return None
        options = item.get('options')
        answer = item.get('answer', item.get('correctAnswer'))
        question = item.get('question', '').strip() if isinstance(item.get('question'), str) else ''
        question_key = re.sub(r'\s+', ' ', question).lower()
        if (
            not question
            or len(question) > 240
            or not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len({option.strip().lower() for option in options}) != 4
            or isinstance(answer, bool)
            or not isinstance(answer, int)
            or answer < 0
            or answer > 3
            or not isinstance(item.get('explanation'), str)
            or not item['explanation'].strip()
            or question_key in seen_questions
        ):
            return None
        seen_questions.add(question_key)
        validated.append({
            'question': question,
            'options': [option.strip() for option in options],
            'answer': answer,
            'explanation': item['explanation'].strip(),
        })
    return {'questions': validated}


def _quiz_payload_is_grounded(payload: dict, source_text: str) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get('questions'), list):
        return False

    source_words = set(re.findall(r"[a-z]{5,}", source_text.lower()))
    for item in payload['questions']:
        if not isinstance(item, dict):
            return False

        question = item.get('question')
        explanation = item.get('explanation')
        options = item.get('options')

        if not isinstance(question, str) or not isinstance(explanation, str) or not isinstance(options, list):
            return False

        fields = [question, *options, explanation]
        if any(len(field) > 240 for field in fields if isinstance(field, str)):
            return False

        # Reject generic questions and source headings used as answers.
        generic_patterns = [
            r"which statement is best supported",
            r"which statement accurately reflects",
            r"what is discussed in the document",
            r"according to the document",
            r"what is the main topic"
        ]
        question_lower = question.lower()
        if any(re.search(pattern, question_lower) for pattern in generic_patterns):
            return False

        normalized_source = re.sub(r"\s+", " ", source_text.lower()).strip()
        normalized_options = [re.sub(r"\s+", " ", option.lower()).strip() for option in options]
        if any(option in normalized_source and len(option.split()) >= 4 for option in normalized_options):
            return False

        question_words = set(re.findall(r"[a-z]{5,}", question.lower()))
        explanation_words = set(re.findall(r"[a-z]{5,}", explanation.lower()))
        option_words = set(re.findall(r"[a-z]{5,}", " ".join(options).lower()))

        if not question_words or not explanation_words or not option_words:
            return False

        question_overlap = len(question_words & source_words)
        explanation_overlap = len(explanation_words & source_words)
        option_overlap = len(option_words & source_words)

        # Require meaningful grounding across the question, options, and explanation.
        if question_overlap < 2 or explanation_overlap < 3 or option_overlap < 3:
            return False

    return True


def _generate_quiz(text: str):
    prompt = QUIZ_PROMPT_TEMPLATE.format(source_text=text)
    raw_quiz = _call_gemini(
        prompt,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=4000,
    )

    parsed = _parse_quiz_json(raw_quiz)
    if parsed is None:
        print("Gemini quiz generation failed: invalid JSON shape")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned an invalid quiz.",
        )

    if not _quiz_payload_is_grounded(parsed, text):
        print("Gemini quiz generation failed: not grounded in source document")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned a quiz that wasn't well grounded in the source document.",
        )

    print("Gemini quiz generated successfully")
    return parsed


# ---------------------------------------------------------------------------
# Summary / notes generation (Gemini primary, extractive fallback)
# ---------------------------------------------------------------------------

def _postprocess_generated_summary(raw_summary: str) -> str:
    summary = _clean_summary_output(raw_summary)
    summary = _remove_instruction_sentences(summary)
    summary = re.sub(r"\s+", " ", summary).strip()
    summary = _remove_similar_sentences(summary)
    summary = _remove_repeated_phrases(summary)
    return summary


def _generate_summary_from_text(cleaned_text: str) -> str:
    summary = ""
    try:
        raw_summary = _call_gemini(
            SUMMARY_PROMPT_TEMPLATE.format(source_text=cleaned_text),
            temperature=0.3,
            max_output_tokens=700,
        )
        summary = _postprocess_generated_summary(raw_summary)
    except HTTPException as exc:
        print(f"Gemini summary generation failed, using fallback: {exc.detail}")

    if (
        _summary_needs_fallback(summary, cleaned_text)
        or _summary_is_instruction_heavy(summary)
        or _summary_is_near_verbatim_copy(summary, cleaned_text)
    ):
        fallback_source = _prepare_source_for_fallback(cleaned_text)
        summary = _build_extractive_summary(fallback_source)
        summary = _remove_instruction_sentences(summary)
        summary = _remove_similar_sentences(summary)
        summary = _remove_repeated_phrases(summary)

    return re.sub(r"\s+", " ", summary).strip()


def _generate_notes_from_text(cleaned_text: str) -> str:
    notes = ""
    try:
        raw_notes = _call_gemini(
            NOTES_PROMPT_TEMPLATE.format(source_text=cleaned_text),
            temperature=0.3,
            max_output_tokens=1800,
        )
        notes = clean_notes_output(raw_notes)

        # --- Source-grounding validation pipeline (Requirement C) ---------
        # These steps check the model's output against the source document
        # itself - never against topic-specific facts or rules - and strip
        # or repair anything that isn't actually supported by the source.
        notes = _strip_ungrounded_lines(notes, cleaned_text)
        notes = _fill_empty_sections(notes)
        notes = _fix_overview_section(notes, cleaned_text)
        notes = _ensure_quick_revision_populated(notes)
        notes = _normalize_markdown_spacing(notes)
    except HTTPException as exc:
        print(f"Gemini notes generation failed, using fallback: {exc.detail}")
        notes = ""

    if not notes_are_valid(notes) or _notes_need_fallback(notes, cleaned_text):
        fallback_source = _prepare_source_for_fallback(cleaned_text)
        notes = clean_notes_output(_build_markdown_notes(fallback_source))

    return notes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ML service is running"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    is_pdf_mime = file.content_type == "application/pdf"
    is_pdf_ext = file.filename and file.filename.lower().endswith(".pdf")

    if not (is_pdf_mime or is_pdf_ext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a PDF."
        )

    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)

        extracted_pages = []
        for page in reader.pages:
            text = page.extract_text()
            extracted_pages.append(text if text else "")

        full_text = clean_extracted_text("\n".join(extracted_pages))
        page_count = len(reader.pages)

        return {"text": full_text, "pages": page_count}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted PDF file."
        )


@app.post("/generate-summary")
def generate_summary(request: NotesRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required to generate a summary."
        )

    cleaned_text = prepare_source_text(text)
    summary = _generate_summary_from_text(cleaned_text)
    return {"summary": summary}


@app.post("/generate-notes")
def generate_notes(request: NotesRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required to generate notes."
        )

    cleaned_text = prepare_source_text(text)
    notes = _generate_notes_from_text(cleaned_text)
    return {"notes": notes}


@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required to generate quiz questions."
        )

    return _generate_quiz(prepare_source_text(text))