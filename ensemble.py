import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, classification_report

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
    """Get AI probability from TF-IDF model"""
    text_tfidf = vectorizer.transform([text])
    prob = tfidf_model.predict_proba(text_tfidf)[0]
    return prob[1]  # probability of AI

def predict_roberta(text):
    """Get AI probability from RoBERTa model"""
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
        return probs[0][1].item()  # probability of AI

def predict_ensemble(text, tfidf_weight=0.4, roberta_weight=0.6):
    """Combine both models"""
    tfidf_prob = predict_tfidf(text)
    roberta_prob = predict_roberta(text)

    # Weighted combination
    combined_prob = (tfidf_prob * tfidf_weight) + (roberta_prob * roberta_weight)

    prediction = 1 if combined_prob > 0.5 else 0
    label = "AI" if prediction == 1 else "Human"

    return {
        "prediction": label,
        "confidence": combined_prob * 100 if prediction == 1 else (1 - combined_prob) * 100,
        "tfidf_score": tfidf_prob * 100,
        "roberta_score": roberta_prob * 100,
        "combined_score": combined_prob * 100
    }

# ── STEP 5: Test on examples ──────────────────────────
print("\n" + "=" * 60)
print("ENSEMBLE MODEL TEST")
print("=" * 60)

test_examples = [
    ("When I was young, my grandmother used to tell me stories about her childhood. She grew up in a small village where everyone knew each other. Life was simple but hard.", "Human"),
    ("In today's rapidly evolving technological landscape, artificial intelligence has emerged as a transformative force reshaping industries.", "AI"),
    ("The results of the experiment were surprising. We expected the temperature to rise gradually but it spiked suddenly.", "Human"),
    ("I honestly didn't know what to write for this assignment. My teacher said write about summer but like what do I even say lol.", "Human"),
    ("Furthermore, it is worth noting that sustainable practices require a multifaceted approach encompassing innovation.", "AI"),
    ("bro i was so tired today i forgot to eat lunch and my code wasnt working for 2 hours i wanted to give up", "Human"),
]

correct = 0
for text, true_label in test_examples:
    result = predict_ensemble(text)
    status = "✅" if result['prediction'] == true_label else "❌"
    if result['prediction'] == true_label:
        correct += 1

    print(f"\n{status} True: {true_label} | Predicted: {result['prediction']} ({result['confidence']:.1f}%)")
    print(f"   TF-IDF:   {result['tfidf_score']:.1f}% AI")
    print(f"   RoBERTa:  {result['roberta_score']:.1f}% AI")
    print(f"   Combined: {result['combined_score']:.1f}% AI")
    print(f"   Text: {text[:70]}...")

print("\n" + "=" * 60)
print(f"ENSEMBLE SCORE: {correct}/{len(test_examples)} correct")
print("=" * 60)

# ── STEP 6: Evaluate on full test set ─────────────────
print("\nEvaluating on full test set (500 samples)...")
print("This will take 5-8 minutes...")

test_df = pd.read_csv("test_clean.csv").sample(500, random_state=42)
all_preds = []
all_labels = []

for idx, row in test_df.iterrows():
    result = predict_ensemble(row['text'])
    pred = 1 if result['prediction'] == "AI" else 0
    all_preds.append(pred)
    all_labels.append(row['label'])

    if len(all_preds) % 100 == 0:
        print(f"  Processed {len(all_preds)}/500...")

print("\n" + "=" * 60)
print("FULL TEST SET RESULTS")
print("=" * 60)
print(classification_report(
    all_labels, all_preds,
    target_names=['Human', 'AI']
))

# Save ensemble functions for later use in web app
print("✅ Ensemble model ready for web app!")