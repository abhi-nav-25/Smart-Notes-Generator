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
    is generic (title-glue, escaped markdown, duplicate lines/sections,
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
# Prompts
# ---------------------------------------------------------------------------

SUMMARY_PROMPT_TEMPLATE = """
Summarize the source document below, clearly and accurately, for a student
studying this material.

Rules:
- Write flowing prose (2 to 4 short paragraphs). Do not use a list or any
  Markdown formatting.
- Do not repeat words, phrases, or sentences.
- Do not invent information that is not in the source document.
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
A short explanation of the main topic.

## Key Concepts
- The important concepts, terms, and definitions found in the document.

## Main Components or Classification
- Types, categories, steps, stages, or components described in the
  document, if any are present.

## Important Details
- Important facts, rules, formulas, or explanations from the document.

## Examples or Applications
- Examples or applications, only if the document actually contains them.

## Quick Revision
- The most important points already covered above, restated briefly.
- Do not introduce anything new in this section.

Rules:
- Use only information from the source document below. Do not invent facts.
- If a section has no relevant content in the source, write exactly
  "Not covered in the source document." under that heading.
- Avoid repeating the same point across multiple sections.
- Do not include an introductory phrase such as "Here are the notes".
- Do not wrap the output in code fences.
- Do not escape Markdown characters with a backslash - write "##", never
  "\\##".
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

    if _summary_needs_fallback(summary, cleaned_text) or _summary_is_instruction_heavy(summary):
        summary = _build_extractive_summary(cleaned_text)
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
    except HTTPException as exc:
        print(f"Gemini notes generation failed, using fallback: {exc.detail}")

    if not notes_are_valid(notes) or _notes_need_fallback(notes, cleaned_text):
        notes = clean_notes_output(_build_markdown_notes(cleaned_text))

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