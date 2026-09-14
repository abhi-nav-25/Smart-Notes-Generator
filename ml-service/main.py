import io
import json
import re
from functools import lru_cache

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

app = FastAPI(title="Smart Notes Generator - ML Service")


class NotesRequest(BaseModel):
    text: str


class QuizRequest(BaseModel):
    text: str


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

def prepare_source_text(text: str, max_chars: int = 12000) -> str:
    """
    Removes unnecessary repetition and limits the source size
    before sending it to the language model.
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
    return cleaned[:max_chars]

def _generate_model_text(
    text: str,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int
):
    tokenizer, model = get_summarizer()

    full_prompt = f"""
{prompt}

SOURCE DOCUMENT:
{text}
"""

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

def _clean_summary_output(summary: str) -> str:
    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', summary)) if part.strip()]
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        sentence = re.sub(r'^(?:summary|overview)\s*:\s*', '', sentence, flags=re.IGNORECASE).strip()
        key = sentence.lower()
        if sentence and key not in seen:
            unique_sentences.append(sentence)
            seen.add(key)
    return ' '.join(unique_sentences)


def _build_extractive_summary(text: str) -> str:
    content_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r'^\d+(?:\.\d+)*[.)]?\s+[^.!?]+$', stripped):
            continue
        content_lines.append(stripped)
    sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', ' '.join(content_lines)) if sentence.strip()]
    selected = []
    seen = set()
    for sentence in sentences:
        if re.fullmatch(r'\d+(?:\.\d+)*[.)]?\s+[^.!?]+', sentence):
            continue
        key = sentence.lower()
        if key not in seen:
            selected.append(sentence)
            seen.add(key)
        if len(selected) == 4:
            break
    return ' '.join(selected)


def _summary_needs_fallback(summary: str, source: str) -> bool:
    source_sentences = [sentence for sentence in re.split(r'(?<=[.!?])\s+', source) if sentence.strip()]
    if not summary or len(summary.split()) < 15 or len(source_sentences) < 2:
        return True
    first_sentence_words = {
        word.lower() for word in re.findall(r'[A-Za-z]{5,}', source_sentences[0])
    }
    summary_words = {word.lower() for word in re.findall(r'[A-Za-z]{5,}', summary)}
    return len(first_sentence_words & summary_words) < 2

def _build_markdown_notes(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    overview = " ".join(sentences[:3])

    key_concepts = []
    for sentence in sentences[3:10]:
        sentence = sentence.strip()
        if len(sentence.split()) >= 5:
            key_concepts.append(f"- {sentence}")

    quick_revision = []
    for sentence in sentences[-4:]:
        sentence = sentence.strip()
        if len(sentence.split()) >= 5:
            quick_revision.append(f"- {sentence}")

    return f"""# Study Notes

## Overview
{overview}

## Key Concepts
{chr(10).join(key_concepts)}

## Quick Revision
{chr(10).join(quick_revision)}
"""

def notes_are_valid(notes: str) -> bool:
    required_sections = [
        "Overview",
        "Key Concepts",
        "Quick Revision"
    ]

    return (
        isinstance(notes, str)
        and len(notes.split()) >= 40
        and all(
            section.lower() in notes.lower()
            for section in required_sections
        )
    )


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
        answer = item.get('answer')
        question = item.get('question', '').strip() if isinstance(item.get('question'), str) else ''
        question_key = re.sub(r'\s+', ' ', question).lower()
        if (
            not question
            or len(question) > 240
            or not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len({option.strip().lower() for option in options}) != 4  # NEW: options must be distinct
            or isinstance(answer, bool)                                   # NEW: bool is an int subclass, exclude it
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

@lru_cache(maxsize=1)
def get_summarizer():
    model_name = "google/flan-t5-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to load summarization model '{model_name}': {error}",
        ) from error

def _generate_quiz(text: str):
    print("Using Gemini quiz generation")

    prompt = f"""
        You are an expert university-level assessment designer.

        Read the source document and create exactly 5 high-quality multiple-choice questions.

        The questions must test understanding, not simple word matching.

        Include a balanced combination of:
        - definition
        - conceptual understanding
        - function or purpose
        - classification or comparison
        - application or scenario

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
        - Only one option can be correct.
        - The answer must be an integer from 0 to 3.
        - Every question must be based on the source document.
        - Avoid generic questions such as "Which statement is supported by the document?".
        - Do not copy complete sentences directly from the document.
        - Do not use placeholder options.
        - Generate meaningful distractors related to the topic.
        - Do not repeat questions.
        - Explanations must be concise and document-based.
        - Return raw JSON only.
        
        SOURCE DOCUMENT:
        {text}
    """

    env_path = os.path.join(os.path.dirname(__file__), '..', 'backend', '.env')
    load_dotenv(env_path)
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        print("Gemini quiz generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API key is not configured.",
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
                max_output_tokens=1800,
            ),
        )

        if not response or not getattr(response, 'text', None):
            print("Gemini quiz generation failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini quiz generation failed.",
            )

        parsed = _parse_quiz_json(response.text)
        if parsed is None:
            print("Gemini quiz generation failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an invalid quiz.",
            )

        source_words = set(re.findall(r"[a-z]{5,}", text.lower()))
        grounded = True
        for item in parsed["questions"]:
            normalized_source = re.sub(r'\s+', ' ', text.lower()).strip()
            fields = [item['question'], *item['options'], item['explanation']]
            if any(len(field) > 240 or field.lower().strip() in normalized_source for field in fields):
                grounded = False
                break

            question_words = set(re.findall(r"[a-z]{5,}", item["question"].lower()))
            explanation_words = set(re.findall(r"[a-z]{5,}", item["explanation"].lower()))
            question_overlap = len(question_words & source_words)
            explanation_overlap = len(explanation_words & source_words)

            if question_overlap < 2 or explanation_overlap < 3:
                grounded = False
                break

        if not grounded:
            print("Gemini quiz generation failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an invalid quiz.",
            )

        print("Gemini quiz generated successfully")
        return parsed

    except HTTPException:
        raise
    except Exception as exc:
        print("Gemini quiz generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini quiz generation failed.",
        ) from exc

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

        return {
            "text": full_text,
            "pages": page_count
        }
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
    summary_prompt = """
    You are an expert academic summarizer.

    Create a high-quality academic summary of the source document.

    Your summary must:
    1. Explain the central topic clearly.
    2. Include the most important definitions and concepts.
    3. Explain major classifications, processes, or components when present.
    4. Include important relationships and examples from the document.
    5. End with the overall significance or conclusion.
    6. Use your own words.
    7. Avoid copying sentences from the source.
    8. Do not invent information.
    9. Do not include headings, bullet points, numbering, or markdown.
    10. Write one well-connected paragraph of approximately 120 to 180 words.

    Return only the summary.
    """

    cleaned_text = prepare_source_text(text)
    summary = _clean_summary_output(_generate_model_text(
        cleaned_text,
        summary_prompt,
        max_new_tokens=220,
        min_new_tokens=100,
    ))
    if not summary or len(summary.split()) < 60:
        summary = _build_extractive_summary(cleaned_text)
    return {"summary": summary}


@app.post("/generate-notes")
def generate_notes(request: NotesRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required to generate notes."
        )
    notes_prompt = """
    You are an expert academic note-making assistant.

    Convert the source document into well-organized, exam-oriented study notes.

    Use exactly this structure:

    ## Overview
    Write a short explanation of the topic.

    ## Key Concepts
    - Explain the important concepts in your own words.
    - Include definitions where relevant.

    ## Main Components or Classification
    - Explain the major types, components, stages, or categories if present.

    ## Important Details
- Include important properties, functions, advantages, limitations, or relationships.

    ## Examples or Applications
    - Include examples only when supported by the document.

    ## Quick Revision
    - Give 4 to 6 concise revision points.

    Rules:
    - Use Markdown headings and bullet points.
    - Do not copy the source document line by line.
    - Do not repeat the same idea.
    - Do not create empty sections.
    - Do not invent facts.
    - Keep the notes concise but academically meaningful.
    - Preserve technical terminology from the source.
    """
    cleaned_text = prepare_source_text(text)
    notes = _generate_model_text(
        cleaned_text,
        notes_prompt,
        max_new_tokens=500,
        min_new_tokens=180,
    )
    if (
        not isinstance(notes, str)
        or "Write a short summary" in notes
        or "Rules:" in notes
        or len(notes.split()) < 80
    ):
        notes = _build_markdown_notes(cleaned_text)
    if not notes_are_valid(notes):
        notes = _build_markdown_notes(cleaned_text)

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

