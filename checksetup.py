import torch
from transformers import RobertaTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nLoading RoBERTa tokenizer... (first time downloads ~500MB)")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print("✅ RoBERTa tokenizer loaded successfully!")

# Test it
sample = "This is a test sentence to check if everything works."
tokens = tokenizer(sample, return_tensors="pt")
print(f"\nSample tokenized successfully!")
print(f"Token IDs shape: {tokens['input_ids'].shape}")