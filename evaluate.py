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
subject = "Experiments, numerical models and optimization of carbon-epoxy plates damped by a frequency-dependent interleaved viscoelastic layer"
abstract = """
The research work presented in this paper aims to optimise the dynamic response of a carbon-epoxy plate by including into the laminate one frequency-dependent interleaved viscoelastic layer. To keep an acceptable bending stiffness, some holes are created in the viscoelastic layer, thus facilitating the resin through layer pene- tration during the co-curing manufacturing process. Plates including (or not) one perforated (or non-perforated) viscoelastic layer are manufactured and investigated experimentally and numerically. First, static and dynamic tests are performed on sandwich coupons to characterise the stiffness and damping properties of the plates in a given frequency range. Resulting mechanical properties are then used to set-up a finite element model and simulate the plate dynamic response. In parallel, fre- quency response measurements are carried out on the manufactured plates, then successfully confronted to the numerical results. Finally, a design of experiments is built based on a limited number on numerical simulations to find the configuration of bridges that maximises the damping while keeping a stiffness higher than half the stiffness of the equivalent undamped plate."""

# Get embeddings
subject_embedding = get_sentence_embedding(subject)
abstract_embedding = get_sentence_embedding(abstract)

# 2. **Measure Semantic Similarity Using Cosine Similarity**

# Compute cosine similarity between subject and abstract embeddings
similarity = F.cosine_similarity(subject_embedding, abstract_embedding)
print(f"Cosine Similarity: {similarity.item():.4f}")
