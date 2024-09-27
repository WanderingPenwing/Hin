from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel

# Load the SciBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

print("* Tokenizer")

model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

print("* Scibert model")

# Define a KeyBERT model using SciBERT embeddings
kw_model = KeyBERT(model=model)

print("* Keybert model")
# Define the subject from which to extract keywords
subject = "tig welding of inconel 625 and influences on micro structures"

# Extract keywords from the subject
keywords = kw_model.extract_keywords(subject, keyphrase_ngram_range=(1, 2), stop_words='english', use_maxsum=True)

# Print extracted keywords
for keyword, score in keywords:
    print(f"Keyword: {keyword}, Score: {score:.4f}")
