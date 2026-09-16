from reportlab.pdfgen import canvas
from fastapi.testclient import TestClient
from main import app
import os

# 1. Generate a large PDF
pdf_path = "large_test.pdf"
c = canvas.Canvas(pdf_path)
for page in range(5):
    text_y = 800
    for line in range(40): # 40 lines per page
        c.drawString(50, text_y, f"This is line {line} on page {page}. " + "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data. "*2)
        text_y -= 20
    c.showPage()
c.save()

print(f"Generated {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")

# 2. Test the extraction endpoint
client = TestClient(app)
with open(pdf_path, "rb") as f:
    response = client.post("/extract-text", files={"file": ("large_test.pdf", f, "application/pdf")})

assert response.status_code == 200, f"Extraction failed: {response.text}"
data = response.json()
text = data["text"]
word_count = len(text.split())
print(f"Extracted {data['pages']} pages with {word_count} words.")

# 3. Test the summary chunking endpoint
# This should print "Processing PDF chunks" multiple times
print("\n--- Testing Summary Generation (will trigger chunking) ---")
summary_res = client.post("/generate-summary", json={"text": text})
if summary_res.status_code == 200:
    print(f"Success! Summary generated length: {len(summary_res.json()['summary'])} chars.")
else:
    print(f"Summary generation failed: {summary_res.text}")

# 4. Test the notes chunking endpoint
print("\n--- Testing Notes Generation (will trigger chunking) ---")
notes_res = client.post("/generate-notes", json={"text": text})
if notes_res.status_code == 200:
    print(f"Success! Notes generated length: {len(notes_res.json()['notes'])} chars.")
else:
    print(f"Notes generation failed: {notes_res.text}")
