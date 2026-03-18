import pickle

# Load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded! Let's test it.\n")

# Try your own text!
while True:
    print("Enter any text (or type 'quit' to exit):")
    user_text = input("> ")

    if user_text.lower() == 'quit':
        break

    # Convert text to numbers and predict
    text_tfidf = vectorizer.transform([user_text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf)[0]

    if prediction == 1:
        print(f"🤖 AI WRITTEN   (confidence: {confidence[1]*100:.1f}%)\n")
    else:
        print(f"👤 HUMAN WRITTEN (confidence: {confidence[0]*100:.1f}%)\n")