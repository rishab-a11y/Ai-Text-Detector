import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pickle

# ── STEP 1: Load cleaned data ────────────────────────
print("Loading cleaned data...")
train_df = pd.read_csv("train_clean.csv")
test_df = pd.read_csv("test_clean.csv")

# Use a smaller sample for faster training on CPU
# (full dataset would take too long without GPU)
train_df = train_df.sample(3000, random_state=42)
test_df = test_df.sample(500, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")

# ── STEP 2: Load tokenizer ───────────────────────────
print("\nLoading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# ── STEP 3: Create Dataset class ─────────────────────
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

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',    # pad short texts
            truncation=True,         # cut long texts
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ── STEP 4: Create datasets ──────────────────────────
print("Tokenizing training data... (may take 2-3 mins)")
train_dataset = TextDataset(
    train_df['text'],
    train_df['label'],
    tokenizer
)

print("Tokenizing test data...")
test_dataset = TextDataset(
    test_df['text'],
    test_df['label'],
    tokenizer
)

# ── STEP 5: Create dataloaders ───────────────────────
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print(f"\n✅ Data prepared successfully!")
print(f"Training batches: {len(train_loader)}")
print(f"Testing batches:  {len(test_loader)}")

# Save datasets for use in training
torch.save(train_dataset, "train_dataset.pt")
torch.save(test_dataset, "test_dataset.pt")
print("✅ Datasets saved as train_dataset.pt and test_dataset.pt")

# ── STEP 6: Inspect one batch ────────────────────────
print("\n--- Inspecting one sample ---")
sample = train_dataset[0]
print(f"Input IDs shape:      {sample['input_ids'].shape}")
print(f"Attention mask shape: {sample['attention_mask'].shape}")
print(f"Label:                {sample['label']}")