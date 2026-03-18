from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.model import predict_ensemble, analyze_sentences
from app.file_handler import extract_text

# Create FastAPI app
app = FastAPI(title="AI Text Detector")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ── Request model ─────────────────────────────────────
class TextRequest(BaseModel):
    text: str

# ── Routes ────────────────────────────────────────────
@app.get("/")
def home():
    with open("app/static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health():
    return {"status": "healthy"}

# ── Predict from raw text ─────────────────────────────
@app.post("/predict-text")
def predict_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="Text too short — need at least 50 characters")

    result = predict_ensemble(request.text)
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "tfidf_score": result["tfidf_score"],
        "roberta_score": result["roberta_score"],
        "combined_score": result["combined_score"],
        "text_length": len(request.text)
    }

# ── Predict from uploaded file ────────────────────────
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    # Check file type
    allowed = [".pdf", ".docx", ".txt"]
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PDF, DOCX or TXT"
        )

    # Check file size (max 5MB)
    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 5MB")

    # Extract text
    try:
        text = extract_text(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check extracted text
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Not enough text found in file")

    # Run prediction
    result = predict_ensemble(text)
    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "tfidf_score": result["tfidf_score"],
        "roberta_score": result["roberta_score"],
        "combined_score": result["combined_score"],
        "text_length": len(text),
        "text_preview": text[:200] + "..."
    }
# ── Sentence level analysis ───────────────────────────
@app.post("/analyze-sentences")
def analyze_text_sentences(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="Text too short")

    # Overall prediction
    overall = predict_ensemble(request.text)

    # Sentence level
    sentences = analyze_sentences(request.text)

    return {
        "overall": overall,
        "sentences": sentences,
        "total_sentences": len(sentences),
        "ai_sentences": sum(1 for s in sentences if s["is_ai"]),
        "human_sentences": sum(1 for s in sentences if not s["is_ai"])
    }