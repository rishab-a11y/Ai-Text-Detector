import pickle
import pandas as pd

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ── Test Examples ─────────────────────────────────────
test_examples = [
    {
        "text": "When I was young, my grandmother used to tell me stories about her childhood. She grew up in a small village where everyone knew each other. Life was simple but hard. They didn't have electricity or running water, but she always said those were the happiest days of her life.",
        "true_label": "Human",
        "style": "Personal story"
    },
    {
        "text": "In today's rapidly evolving technological landscape, artificial intelligence has emerged as a transformative force that is reshaping industries and redefining human potential across multiple domains of human endeavor.",
        "true_label": "AI",
        "style": "Formal AI essay"
    },
    {
        "text": "The results of the experiment were surprising. We expected the temperature to rise gradually, but instead it spiked suddenly at the 10 minute mark. This suggests that the reaction is more complex than we initially thought.",
        "true_label": "Human",
        "style": "Scientific writing"
    },
    {
        "text": "Climate change is one of the most pressing issues of our time. It is essential that we take immediate action to reduce carbon emissions and transition to renewable energy sources in order to preserve our planet for future generations.",
        "true_label": "AI",
        "style": "AI climate essay"
    },
    {
        "text": "I honestly didn't know what to write for this assignment. My teacher said to write about my summer but like what do I even say. I went to my cousins house and we played video games the whole time. It was fun I guess.",
        "true_label": "Human",
        "style": "Student writing"
    },
    {
        "text": "Furthermore, it is worth noting that the implementation of sustainable practices in urban environments requires a multifaceted approach that encompasses both technological innovation and behavioral change among citizens.",
        "true_label": "AI",
        "style": "Formal AI paragraph"
    },
]

# ── Run Predictions ───────────────────────────────────
print("=" * 60)
print("MODEL TEST REPORT")
print("=" * 60)

correct = 0
results = []

for i, example in enumerate(test_examples):
    text_tfidf = vectorizer.transform([example['text']])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0]

    predicted_label = "AI" if prediction == 1 else "Human"
    conf_score = confidence[1] if prediction == 1 else confidence[0]
    is_correct = predicted_label == example['true_label']

    if is_correct:
        correct += 1
        status = "✅ CORRECT"
    else:
        status = "❌ WRONG"

    print(f"\nExample {i+1}: {example['style']}")
    print(f"  True Label:  {example['true_label']}")
    print(f"  Predicted:   {predicted_label} ({conf_score*100:.1f}% confidence)")
    print(f"  Result:      {status}")

    results.append({
        "style": example['style'],
        "true_label": example['true_label'],
        "predicted": predicted_label,
        "confidence": f"{conf_score*100:.1f}%",
        "correct": is_correct
    })

# ── Summary ───────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FINAL SCORE: {correct}/{len(test_examples)} correct")
print(f"Accuracy on these examples: {correct/len(test_examples)*100:.1f}%")
print("=" * 60)

# Save report
results_df = pd.DataFrame(results)
results_df.to_csv("test_report.csv", index=False)
print("\n✅ Report saved as test_report.csv")