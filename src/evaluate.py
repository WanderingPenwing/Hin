from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import progressbar

print("\n### Fetching SciBert ###\n")

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


def sort_results(subject, results):

    print("\n### Starting result processing (",len(results),") ###\n")
    
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
        
    # Sort the results by similarity (highest first)
    sorted_results = sorted(scored_results, key=lambda x: x['similarity'], reverse=True)

    log = "## Results\n\n"
    
    for result in sorted_results :
        log += f"Title: {result['title']}\nURL: {result['url']}\n\n"

    return sorted_results, log
