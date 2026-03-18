import pickle
import pandas as pd

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the test set (model has NEVER seen this data)
test_df = pd.read_csv("test_clean.csv")

# Pick 5 real human and 5 real AI examples from test set
human_samples = test_df[test_df['label'] == 0].head(5)
ai_samples = test_df[test_df['label'] == 1].head(5)
samples = pd.concat([human_samples, ai_samples]).reset_index(drop=True)

print("=" * 60)
print("REAL DATA TEST REPORT")
print("=" * 60)

correct = 0

for i, row in samples.iterrows():
    text = row['text']
    true_label = "Human" if row['label'] == 0 else "AI"

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0]

    predicted_label = "AI" if prediction == 1 else "Human"
    conf_score = confidence[1] if prediction == 1 else confidence[0]
    is_correct = predicted_label == true_label

    if is_correct:
        correct += 1
        status = "✅ CORRECT"
    else:
        status = "❌ WRONG"

    print(f"\nExample {i+1}:")
    print(f"  Text preview: {text[:100]}...")
    print(f"  True Label:   {true_label}")
    print(f"  Predicted:    {predicted_label} ({conf_score*100:.1f}% confidence)")
    print(f"  Result:       {status}")

print("\n" + "=" * 60)
print(f"FINAL SCORE: {correct}/10 correct")
print(f"Accuracy: {correct/10*100:.1f}%")
print("=" * 60)