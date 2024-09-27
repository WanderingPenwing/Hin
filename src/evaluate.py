from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import progressbar
import math

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


def score_results(subject, results, weights):
    
    subject_embedding = get_sentence_embedding(subject)
    print("* Tokenized subject\n")
    
    scored_results_urls = []
    scored_results = []

    print("* Started scoring results\n")
    
    bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
        maxval=len(results)).start()
    
    progress = 0
    title_score_bounds = [1, 0]
    snippet_score_bounds = [1, 0]
    
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
        title_embedding = get_sentence_embedding(title)
        snippet_embedding = get_sentence_embedding(snippet)
        
        # Compute similarity between subject and result
        title_score = compute_similarity(subject_embedding, title_embedding)
        snippet_score = compute_similarity(subject_embedding, snippet_embedding)

        if title_score < title_score_bounds[0] :
            title_score_bounds[0] = title_score
        if title_score > title_score_bounds[1] :
            title_score_bounds[1] = title_score
        if snippet_score < snippet_score_bounds[0] :
            snippet_score_bounds[0] = snippet_score
        if snippet_score > snippet_score_bounds[1] :
            snippet_score_bounds[1] = snippet_score
        
        # Store the result with its similarity score
        scored_results.append({
            'title': title,
            'url': url,
            'snippet': snippet,
            'title-score': title_score,
            'snippet-score': snippet_score
        })

    log = f"Score bounds : T{title_score_bounds} # S{snippet_score_bounds}\n\n"
    print("\n\n* Scored results\n")
    
    normalized_results = []
    for result in scored_results:
        title_score = (result['title-score'] - title_score_bounds[0]) / (title_score_bounds[1] - title_score_bounds[0])
        snippet_score = (result['snippet-score'] - snippet_score_bounds[0]) / (snippet_score_bounds[1] - snippet_score_bounds[0])
        
        score = math.pow(math.pow(title_score, weights[0]) * math.pow(snippet_score, weights[1]), 1 / (weights[0] + weights[1]))
        
        normalized_results.append({
            'title': result['title'],
            'url': result['url'],
            'snippet': result['snippet'],
            'score': score,
        })

    log += f"Weights : {weights};\n\n"
    
    return normalized_results, log


def sort_results(subject, results, weights):

    print("\n### Starting result processing (",len(results),") ###\n")

    log = "\n---\n\n## Scoring\n\n"
    
    scored_results, score_log = score_results(subject, results, weights)

    log += score_log

    # Sort the results by similarity (highest first)
    sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

    return sorted_results, log
