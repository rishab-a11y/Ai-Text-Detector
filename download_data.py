import pandas as pd

print("Loading dataset...")

df = pd.read_csv("train_v2_drcat_02.csv")

print("Dataset loaded successfully!")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows:")
print(df.head(3))

print("\nLabel distribution:")
print(df['label'].value_counts())