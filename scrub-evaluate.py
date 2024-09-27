import warnings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import requests
import progressbar


# Me
#subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"
#query = "composite viscoelastic damping"

# Anne
#subject = "State of the art on the identification of wood structure natural frequencies. Influence of the mechanical properties and interest in sensitivity analysis as prospects for reverse identification method of wood elastic properties."
#query = "wood frequency analysis mechanical properties"

# Axel
#subject = "Characterization of SiC MOSFET using double pulse test method."
#query = "SiC MOSFET double pulse test"

# Paul
#subject = "Thermo-Mechanical Impact of temperature oscillations on bonding and metallization for SiC MOSFETs soldered on ceramic substrate"
#query = "thermo mechanical model discrete bonding SiC MOSFET"

# Jam
subject = "tig welding of inconel 625 and influences on micro structures"
query = "tig welding inconel 625"

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

# Define the SearxNG instance URL and search query
searxng_url = "https://search.penwing.org/search"  # Replace with your instance URL
params = {
    "q": query,  # Your search query
    "format": "json",         # Requesting JSON format
    "categories": "science",  # You can specify categories (optional)
}

# Send the request to SearxNG API
response = requests.get(searxng_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    print("* Got search results")
    # Parse the JSON response
    data = response.json()
    
    subject_embedding = get_sentence_embedding(subject)

    print("* Tokenized subject")

    print("\n### Starting result processing ###\n")
    # List to store results with similarity scores
    scored_results = []

    results = data.get("results", [])
    progress = 0
     
    bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
        maxval=len(results)).start()
    
    # Process each result
    for result in results :
        title = result['title']
        url = result['url']
        snippet = result['content']
        
        # Get embedding for the snippet (abstract)
        snippet_embedding = get_sentence_embedding(snippet)
        
        # Compute similarity between subject and snippet
        similarity = compute_similarity(subject_embedding, snippet_embedding)
        
        # Store the result with its similarity score
        scored_results.append({
            'title': title,
            'url': url,
            'snippet': snippet,
            'similarity': similarity
        })

        progress += 1
        bar.update(progress)
    
    # Sort the results by similarity (highest first)
    top_results = sorted(scored_results, key=lambda x: x['similarity'], reverse=True)[:10]

    print("\n### Done ###\n")
    # Print the top 10 results
    for idx, result in enumerate(top_results, 1):
        print(f"Rank {idx} ({result['similarity']:.4f}):")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet']}")
        print("-" * 40)
else:
    print(f"Error: {response.status_code}")
