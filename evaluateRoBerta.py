import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ── STEP 1: Setup ────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── STEP 2: Load tokenizer & model ───────────────────
print("Loading RoBERTa model...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)
model.load_state_dict(torch.load("roberta_model.pt", map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("✅ Model loaded!")

# ── STEP 3: Dataset class ─────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ── STEP 4: Load test data ────────────────────────────
print("\nLoading test data...")
test_df = pd.read_csv("test_clean.csv").sample(500, random_state=42)
test_dataset = TextDataset(test_df['text'], test_df['label'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ── STEP 5: Get predictions ───────────────────────────
print("Running predictions...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ── STEP 6: Full report ───────────────────────────────
print("\n" + "=" * 55)
print("ROBERTA EVALUATION REPORT")
print("=" * 55)
print(classification_report(
    all_labels, all_preds,
    target_names=['Human', 'AI']
))

# ── STEP 7: Confusion matrix ──────────────────────────
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(f"                 Predicted Human  Predicted AI")
print(f"Actual Human          {cm[0][0]}              {cm[0][1]}")
print(f"Actual AI             {cm[1][0]}              {cm[1][1]}")

# ── STEP 8: Test on specific examples ─────────────────
print("\n" + "=" * 55)
print("BIAS TEST — Same examples as before")
print("=" * 55)

test_examples = [
    ("When I was young, my grandmother used to tell me stories about her childhood. She grew up in a small village where everyone knew each other. Life was simple but hard.", "Human"),
    ("In today's rapidly evolving technological landscape, artificial intelligence has emerged as a transformative force reshaping industries.", "AI"),
    ("The results of the experiment were surprising. We expected the temperature to rise gradually, but instead it spiked suddenly.", "Human"),
    ("I honestly didn't know what to write for this assignment. My teacher said to write about my summer but like what do I even say.", "Human"),
    ("Furthermore, it is worth noting that the implementation of sustainable practices requires a multifaceted approach.", "AI"),
]

for text, true_label in test_examples:
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    predicted = "AI" if pred == 1 else "Human"
    status = "✅" if predicted == true_label else "❌"

    print(f"\n{status} True: {true_label} | Predicted: {predicted} ({confidence*100:.1f}%)")
    print(f"   Text: {text[:80]}...")