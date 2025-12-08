import numpy as np
from numpy.linalg import norm
from typing import List

def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalizes an embedding to have a unit length.
    Args:
        embedding (list[float]): The embedding to be normalized.
    Returns:
        list[float]: The normalized embedding.
    """

    embedding_values_np = np.array(embedding)
    normalized_embedding = embedding_values_np / norm(embedding_values_np)

    print(f"Normed embedding length: {len(normalized_embedding)}")
    print(f"Norm of normed embedding: {np.linalg.norm(normalized_embedding):.6f}") # Should be very close to 1
    return normalized_embedding.tolist()