import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time

# Define TextDataset class here too
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

# ── STEP 1: Setup ────────────────────────────────────
print("Setting up...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# ---- Loading Datasets
print("Loading datasets...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_df = pd.read_csv("train_clean.csv").sample(3000, random_state=42)
test_df = pd.read_csv("test_clean.csv").sample(500, random_state=42)

print("Tokenizing... (2-3 mins)")
train_dataset = TextDataset(train_df['text'], train_df['label'], tokenizer)
test_dataset = TextDataset(test_df['text'], test_df['label'], tokenizer)
print("✅ Datasets ready!")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ── STEP 2: Load RoBERTa model ───────────────────────
print("\nLoading RoBERTa model... (may take a minute)")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2        # 2 classes: Human or AI
)
model = model.to(device)
print("✅ Model loaded!")

# ── STEP 3: Setup optimizer ──────────────────────────
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,
    num_training_steps=total_steps
)

# ── STEP 4: Training function ────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Progress update every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f} "
                  f"| Acc: {correct/total*100:.1f}%")

    return total_loss / len(loader), correct / total

# ── STEP 5: Evaluation function ──────────────────────
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ── STEP 6: Train for 3 epochs ───────────────────────
print("\n🚀 Starting training...")
print("=" * 50)

best_accuracy = 0

for epoch in range(3):
    print(f"\nEpoch {epoch+1}/3")
    start_time = time.time()

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    val_acc = evaluate(model, test_loader, device)

    elapsed = time.time() - start_time
    print(f"\n✅ Epoch {epoch+1} Complete!")
    print(f"   Train Loss:     {train_loss:.4f}")
    print(f"   Train Accuracy: {train_acc*100:.2f}%")
    print(f"   Val Accuracy:   {val_acc*100:.2f}%")
    print(f"   Time taken:     {elapsed/60:.1f} mins")

    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), "roberta_model.pt")
        print(f"   💾 New best model saved! ({val_acc*100:.2f}%)")

print("\n" + "=" * 50)
print(f"🎉 Training Complete!")
print(f"Best Validation Accuracy: {best_accuracy*100:.2f}%")
print("Model saved as roberta_model.pt")