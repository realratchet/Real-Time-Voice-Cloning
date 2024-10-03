import re
import torch
import numpy as np
from pathlib import Path
from functools import lru_cache

# Function to compute cosine similarity for a new embedding against multiple embeddings
def compute_cosine_similarities(new_embedding, precomputed_embeddings):
    # Compute cosine similarities
    return torch.nn.functional.cosine_similarity(new_embedding, precomputed_embeddings)

# Function to compute Euclidean distances for a new embedding against multiple embeddings
def compute_euclidean_distances(new_embedding, precomputed_embeddings):
    # Compute Euclidean distances
    return torch.norm(precomputed_embeddings - new_embedding, dim=1)

# Find the closest embedding based on a metric
def _find_closest_embedding(new_embedding, precomputed_embeddings, metric=None):
    metric = metric or 'cosine'

    if metric == 'cosine':
        similarities = compute_cosine_similarities(new_embedding, precomputed_embeddings)
        closest_index = torch.argmax(similarities).item()  # Find index of the highest similarity
    elif metric == 'euclidean':
        distances = compute_euclidean_distances(new_embedding, precomputed_embeddings)
        closest_index = torch.argmin(distances).item()  # Find index of the lowest distance
    else:
        raise NotImplementedError(f"Invalid metric: '{metric}'")

    return closest_index

@lru_cache(1)
def load_embeddings(path, device):
    path = Path(path)
    # Find all .npy files matching the pattern "embed-*.npy"
    embeds = list(path.glob("embed-*.npy"))
    
    # Check if any embeddings were found
    if not embeds:
        raise ValueError(f"No embedding files found in the directory '{path}'.")

    embeds.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))

    # Load the embeddings from the .npy files and stack them into a single array
    all_embeddings = []
    for embed_file in embeds:
        # Load each embedding
        embedding = np.load(embed_file)
        # Ensure the embedding is of the correct shape (256,)
        if embedding.shape != (256,):
            raise ValueError(f"Embedding in file '{embed_file}' does not have the shape (256,).")
        all_embeddings.append(embedding)

    # Convert the list of embeddings to a NumPy array
    stacked = np.stack(all_embeddings)  # Shape will be [n, 256] where n is the number of embeddings

    return torch.tensor(stacked, dtype=torch.float32, device=device)

@lru_cache(1)
def get_device():
    # Check if CUDA is available and set the device accordingly
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_closest_embedding(embedding, path_embeds, metric = None, device = None):
    device = get_device()
    precomputed_embeds = load_embeddings(path_embeds, device)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=device)

    closest_idx = _find_closest_embedding(embedding_tensor, precomputed_embeds, metric)
    closest = precomputed_embeds[closest_idx].cpu().numpy()

    return closest

def merge_embeds(v1, v2, weight):
    # Ensure that the weight is between 0 and 1
    if not (0 <= weight <= 1):
        raise ValueError("Weight must be between 0 and 1.")
    
    # Calculate the weighted average
    return (1 - weight) * v1 + weight * v2

if __name__ == "__main__":
    dir_embeds = Path("/media/ratchet/hdd/Dataset/liepa2-corpus/cml_compliant_liepa2_result/embeds")

    new_embed = np.load(dir_embeds / "embed-1.npy")

    closest = find_closest_embedding(new_embed, dir_embeds)
    
    dist = np.linalg.norm(new_embed - closest)

    assert dist < 1e-10, "wrong"