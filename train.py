import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── STEP 1: Load cleaned data ────────────────────────
print("Loading cleaned data...")
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")

X_train = train_df['text']
y_train = train_df['label']
X_test = test_df['text']
y_test = test_df['label']

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── STEP 2: Convert text to numbers (TF-IDF) ─────────
print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer(
    max_features=10000,   # use top 10,000 words
    ngram_range=(1, 2),   # use single words AND pairs of words
    stop_words='english'  # ignore common words like "the", "is"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# ── STEP 3: Train the model ───────────────────────────
print("\nTraining model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # ← treats both classes equally
    C=0.5                     # ← makes model less overconfident
)
model.fit(X_train_tfidf, y_train)
print("Model trained!")

# ── STEP 4: Evaluate the model ────────────────────────
print("\nEvaluating model...")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Human', 'AI']))

# ── STEP 5: Save the model ────────────────────────────
print("\nSaving model...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved as model.pkl and vectorizer.pkl")