import pandas as pd

df = pd.read_csv("train_v2_drcat_02.csv")

# Show a human written example
print("=== HUMAN WRITTEN (label=0) ===")
human_example = df[df['label'] == 0].iloc[0]['text']
print(human_example[:500])

print("\n=== AI WRITTEN (label=1) ===")
ai_example = df[df['label'] == 1].iloc[0]['text']
print(ai_example[:500])

# Basic statistics
print("\n=== TEXT LENGTH STATS ===")
df['text_length'] = df['text'].apply(len)
print(df.groupby('label')['text_length'].mean())