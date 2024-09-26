import warnings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Suppress FutureWarnings and other warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Function to compute sentence embeddings by pooling token embeddings (CLS token)
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Pooling strategy: Use the hidden state of the [CLS] token as the sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
    return cls_embedding

# Example subject and abstract
subject = "Artificial Intelligence in Healthcare"
abstract = """
Artificial intelligence (AI) is transforming healthcare with its ability to analyze complex medical data and assist in diagnosis. 
AI models, especially in medical imaging, have shown promise in detecting diseases like cancer and predicting patient outcomes.
"""

# Get embeddings
subject_embedding = get_sentence_embedding(subject)
abstract_embedding = get_sentence_embedding(abstract)

# 2. **Measure Semantic Similarity Using Cosine Similarity**

# Compute cosine similarity between subject and abstract embeddings
similarity = F.cosine_similarity(subject_embedding, abstract_embedding)
print(f"Cosine Similarity: {similarity.item():.4f}")
