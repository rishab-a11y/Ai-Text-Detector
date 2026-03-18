import torch
import pickle
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# ── STEP 1: Setup ─────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── STEP 2: Load TF-IDF model ─────────────────────────
print("Loading TF-IDF model...")
with open("model.pkl", "rb") as f:
    tfidf_model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
print("✅ TF-IDF model loaded!")

# ── STEP 3: Load RoBERTa model ────────────────────────
print("Loading RoBERTa model...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2
)
roberta_model.load_state_dict(
    torch.load("roberta_model.pt", map_location=device, weights_only=True)
)
roberta_model = roberta_model.to(device)
roberta_model.eval()
print("✅ RoBERTa model loaded!")

# ── STEP 4: Prediction functions ──────────────────────
def predict_tfidf(text):
    text_tfidf = vectorizer.transform([text])
    prob = tfidf_model.predict_proba(text_tfidf)[0]
    return prob[1]

def predict_roberta(text):
    encoding = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = roberta_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()

def predict_ensemble(text):
    tfidf_prob = predict_tfidf(text)
    roberta_prob = predict_roberta(text)
    combined_prob = (tfidf_prob * 0.4) + (roberta_prob * 0.6)

    prediction = "AI" if combined_prob > 0.5 else "Human"
    confidence = combined_prob * 100 if combined_prob > 0.5 else (1 - combined_prob) * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "tfidf_score": round(tfidf_prob * 100, 2),
        "roberta_score": round(roberta_prob * 100, 2),
        "combined_score": round(combined_prob * 100, 2)
    }
def analyze_sentences(text):
    """Analyze each sentence individually"""
    import re

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    results = []
    for sentence in sentences:
        try:
            tfidf_prob = predict_tfidf(sentence)
            roberta_prob = predict_roberta(sentence)
            combined = (tfidf_prob * 0.4) + (roberta_prob * 0.6)
            results.append({
              "sentence": sentence,
             "ai_probability": round(float(combined) * 100, 1),
             "is_ai": bool(combined > 0.5)
            })
        except:
            results.append({
                "sentence": sentence,
                "ai_probability": 50.0,
                "is_ai": False
            })

    return results