import pandas as pd
import re
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv("train_v2_drcat_02.csv")

# Keep only columns we need
df = df[['text', 'label']]

# ── STEP 1: Clean the text ──────────────────────────
def clean_text(text):
    text = str(text)                            # make sure it's a string
    text = text.replace('\n', ' ')              # remove newlines
    text = re.sub(r'\s+', ' ', text)            # remove extra spaces
    text = text.strip()                         # remove leading/trailing spaces
    return text

print("Cleaning text...")
df['text'] = df['text'].apply(clean_text)

# ── STEP 2: Remove very short texts ─────────────────
df = df[df['text'].apply(len) > 100]
print(f"Rows after removing short texts: {len(df)}")

# ── STEP 3: Balance the dataset ─────────────────────
human_df = df[df['label'] == 0]
ai_df = df[df['label'] == 1]

# Take equal samples from both
min_count = min(len(human_df), len(ai_df))
human_df = human_df.sample(min_count, random_state=42)
ai_df = ai_df.sample(min_count, random_state=42)

balanced_df = pd.concat([human_df, ai_df]).reset_index(drop=True)
print(f"Balanced dataset size: {len(balanced_df)}")
print(f"Label distribution:\n{balanced_df['label'].value_counts()}")

# ── STEP 4: Split into Train & Test ─────────────────
train_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,        # 80% train, 20% test
    random_state=42,
    stratify=balanced_df['label']
)

print(f"\nTraining set size: {len(train_df)}")
print(f"Testing set size:  {len(test_df)}")

# ── STEP 5: Save the cleaned data ───────────────────
train_df.to_csv("train_clean.csv", index=False)
test_df.to_csv("test_clean.csv", index=False)

print("\n✅ Done! Saved train_clean.csv and test_clean.csv")