from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from itertools import combinations

# Load the SciBERT model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#print("* Tokenizer")

#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
#print("* Scibert model")

# Define a KeyBERT model using SciBERT embeddings
#kw_model = KeyBERT(model=model)
kw_model = KeyBERT()

print("* Keybert model")
# Define the subject from which to extract keywords
subject = "Thermo-Mechanical Impact of temperature oscillations on bonding and metallization for SiC MOSFETs soldered on ceramic substrate"

# Extract keywords from the subject
keywords = kw_model.extract_keywords(subject, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.7)

# Print extracted keywords
for keyword, score in keywords:
    print(f"Keyword: {keyword}, Score: {score:.4f}")

print("-"*40)
sorted_keywords = sorted(keywords, key=lambda x: -x[1])
text_keywords = [x[0] for x in sorted_keywords]

queries = []

for r in range(1, len(text_keywords) + 1):  # r is the length of combinations
    comb = combinations(text_keywords, r)
    queries.extend(comb)

#print([" OR ".join(query) for query in queries])

text_queries = [" OR ".join(query) for query in queries]

text_queries.append(subject)

print(text_queries)
