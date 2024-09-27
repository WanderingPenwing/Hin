import warnings
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
import torch
import torch.nn.functional as F
import requests
import progressbar
from itertools import combinations
from datetime import datetime

#subject = input("Enter subject : ")
subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"

current_time = datetime.now().strftime("%m-%d_%H-%M")

file_path = f"logs/run_{current_time}.log"

content = f"# Hin run, {current_time}\n\nSubject : {subject}\n\n"

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]

# Suppress FutureWarnings and other warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("\n### Fetching Data ###\n")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

print("* Got tokenizer")

model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

print("* Got model")

kw_model = KeyBERT()

print("* Got Keybert")

# Function to compute sentence embeddings by pooling token embeddings (CLS token)
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Pooling strategy: Use the hidden state of the [CLS] token as the sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
    return cls_embedding

# Function to compute cosine similarity
def compute_similarity(embedding1, embedding2):
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

print("\n### Getting Keywords ###\n")

keywords = kw_model.extract_keywords(subject, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True, diversity=0.7)

print("* keywords extracted")
sorted_keywords = sorted(keywords, key=lambda x: -x[1])
text_keywords = [x[0] for x in sorted_keywords]

content += f"## Keywords\n\n{text_keywords}\n\n"

queries = []

for r in range(1, len(text_keywords) + 1):
    comb = combinations(text_keywords, r)
    queries.extend(comb)

final_queries = [subject] + ["\"" + "\" OR \"".join(query) + "\"" for query in queries]

#final_queries.ins(subject)

print("* query generated")

print("\n### Fetching Web data ###\n")

# Define the SearxNG instance URL and search query
searxng_url = "https://search.penwing.org/search"

results = []

web_bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
        maxval=len(final_queries)).start()

progress = 0

content += f"## Queries\n\n"

for query in final_queries :
    params = {
        "q": query,  # Your search query
        "format": "json",         # Requesting JSON format
        "categories": "science",  # You can specify categories (optional)
    }

    response = requests.get(searxng_url, params=params)

    if response.status_code == 200:
        data = response.json()
    
        # List to store results with similarity scores
        scored_results = []
    
        results.extend(data.get("results", []))

        content += f"{query};\n"

        if query == subject:
            test_content = ""
            for result in data.get("results", []):
                test_content+= result['title'] + "\n---\n"
            with open("test.log", 'w') as file:
                file.write(test_content)
    else:
        print(f"Error: {response.status_code}")

    progress += 1
    web_bar.update(progress)
    
print("\n\n### Starting result processing (",len(results),") ###\n")

subject_embedding = get_sentence_embedding(subject)

print("* Tokenized subject\n")

scored_results_urls = []
scored_results = []

bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
    maxval=len(results)).start()

progress = 0

found_original = False

# Process each result
for result in results :
    progress += 1
    bar.update(progress)
        
    title = result['title']
    url = result['url']
    snippet = result['content']

    if title == subject :
        found_original = True

    if url in scored_results_urls :
        continue
        
    scored_results_urls.append(url)
    
    # Get embedding for the snippet (abstract)
    #result_embedding = get_sentence_embedding(snippet)
    result_embedding = get_sentence_embedding(title)
    
    # Compute similarity between subject and snippet
    similarity = compute_similarity(subject_embedding, result_embedding)
    
    # Store the result with its similarity score
    scored_results.append({
        'title': title,
        'url': url,
        'snippet': snippet,
        'similarity': similarity
    })

if found_original :
    print("\n* Found Original Article")
    

# Sort the results by similarity (highest first)
top_results = sorted(scored_results, key=lambda x: x['similarity'], reverse=True)

print("\n\n### Done ###\n")

# Print the top 10 results
for idx, result in enumerate(top_results[:10], 1):
    print(f"Rank {idx} ({result['similarity']:.4f}):")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}")
    print("-" * 40)

# Define the file path with the current time in the filename


content += "\n## Results\n\n"

for result in top_results :
    content += f"Title: {result['title']}\nURL: {result['url']}\n\n"

# Create and save the file
with open(file_path, 'w') as file:
    file.write(content)
