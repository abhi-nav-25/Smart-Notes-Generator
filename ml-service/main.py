import io
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pypdf import PdfReader

app = FastAPI(title="Smart Notes Generator - ML Service")


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

        full_text = "\n".join(extracted_pages)
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

