from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Prepare a test input sentence (e.g., "Hello, world!")
input_text = "Hello, world!"

# Tokenize the input text and convert it to input IDs
inputs = tokenizer(input_text, return_tensors="pt")  # Return tensors in PyTorch format

# Forward pass through the model
with torch.no_grad():  # Disable gradient calculation since we are only doing inference
    outputs = model(**inputs)

# Output model's hidden states (for the last layer)
print(outputs.last_hidden_state)
