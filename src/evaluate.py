from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import progressbar
import math
import warnings

# Suppress FutureWarnings and other warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CLS_POOLING = 1
MEAN_POOLING = 2
MAX_POOLING = 3

print("\n### Fetching SciBert ###\n")

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

print("* Got tokenizer")

model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

print("* Got model")

def get_subject_output(subject):
    subject_inputs = tokenizer(subject, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        subject_outputs = model(**subject_inputs)

    return subject_outputs

# Function to compute the embedding with a selected pooling method
def compute_similarity(subject_outputs, compare_text, pooling_method):
    # Tokenize the input texts
    compare_inputs = tokenizer(compare_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Compute embeddings for both the subject and the comparison text
    with torch.no_grad():
        compare_outputs = model(**compare_inputs)

    # Pooling strategies
    def cls_pooling(output):
        return output.last_hidden_state[:, 0, :]  # CLS token is at index 0

    def mean_pooling(output):
        return output.last_hidden_state.mean(dim=1)  # Mean of all token embeddings

    def max_pooling(output):
        return output.last_hidden_state.max(dim=1).values  # Max of all token embeddings

    # Choose pooling strategy based on the input integer
    if pooling_method == CLS_POOLING:
        subject_embedding = cls_pooling(subject_outputs)
        compare_embedding = cls_pooling(compare_outputs)
    elif pooling_method == MEAN_POOLING:
        subject_embedding = mean_pooling(subject_outputs)
        compare_embedding = mean_pooling(compare_outputs)
    elif pooling_method == MAX_POOLING:
        subject_embedding = max_pooling(subject_outputs)
        compare_embedding = max_pooling(compare_outputs)
    else:
        raise ValueError("Pooling method must be 1 (CLS), 2 (Mean), or 3 (Max).")

    return F.cosine_similarity(subject_embedding, compare_embedding).item()


def score_results(subject, results, weights, pooling):
    
    subject_model_output = get_subject_output(subject)
    print("* Tokenized subject\n")
    
    scored_results_titles = []
    scored_results = []

    print("* Started scoring results\n")
    
    bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
        maxval=len(results)).start()
    
    progress = 0
    
    title_score_bounds = [1, 0]
    snippet_score_bounds = [1, 0]

    title_pooling = pooling[0]
    snippet_pooling = pooling[1]

    
    log = f"Weights : {weights};\n\nPooling : {pooling}\n\n"
        
    # Process each result
    for result in results :
        progress += 1
        bar.update(progress)

        if not ("content" in result) :
            continue
            
        title = result['title']
        url = result['url']
        snippet = result['content']
    
        if title in scored_results_titles :
            continue
            
        scored_results_titles.append(title)
        
        # Compute similarity between subject and result

        title_score, snippet_score = 1, 1
        
        if weights[0] != 0 :
            title_score = compute_similarity(subject_model_output, title, title_pooling)
        if weights[1] != 0 :
            snippet_score = compute_similarity(subject_model_output, snippet, snippet_pooling)

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

    log += f"Score bounds : T{title_score_bounds} # S{snippet_score_bounds}\n\n"
    print("\n\n* Scored results\n")
    
    normalized_results = []
    for result in scored_results:
        title_score, snippet_score = 1, 1

        if weights[0] != 0 :
            title_score = (result['title-score'] - title_score_bounds[0]) / (title_score_bounds[1] - title_score_bounds[0])
        if weights[1] != 0 :
            snippet_score = (result['snippet-score'] - snippet_score_bounds[0]) / (snippet_score_bounds[1] - snippet_score_bounds[0])
        
        score = math.pow(math.pow(title_score, weights[0]) * math.pow(snippet_score, weights[1]), 1 / (weights[0] + weights[1]))
        
        normalized_results.append({
            'title': result['title'],
            'url': result['url'],
            'snippet': result['snippet'],
            'score': score,
        })
    
    return normalized_results, log


def sort_results(subject, results, weights, pooling):

    print("\n### Starting result processing (",len(results),") ###\n")

    log = "\n---\n\n## Scoring\n\n"
    
    scored_results, score_log = score_results(subject, results, weights, pooling)

    log += score_log

    # Sort the results by similarity (highest first)
    sorted_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

    return sorted_results, log
