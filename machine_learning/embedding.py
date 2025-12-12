import numpy as np
from numpy.linalg import norm
from typing import List
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from machine_learning.embedding import normalize_embedding
from tqdm.auto import tqdm


# Load environment variables from the project root .env file, if present.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-embedding-001")
GEMINI_EMBEDDING_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:batchEmbedContents" # embedContent
) 
DIMENSIONALITY = 768  # GOOGLE recommends using 768, 1536, or 3072 output dimensions.


def create_embeddings(array_of_texts: list[str]) -> list[list[float]]:
    """
    Creates an embedding for a text description.
    Args:
        text_description (str): The text description to be embedded.
    Returns:
        list[float]: The embedding.
    """
    embeddings = []
    batch_size = 25  # at most 25 requests can be in one batch

    # Loop over the texts in chunks of batch_size
    for i in tqdm(range(0, len(array_of_texts), batch_size)):
        batch = array_of_texts[i : i + batch_size]
        response = requests.post(
            GEMINI_EMBEDDING_URL,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={
                "requests": [
                    {
                        "model": GEMINI_MODEL,
                        "task_type": "CLASSIFICATION",
                        "content": {"parts": [{"text": item}]},
                        "output_dimensionality": DIMENSIONALITY,
                    }
                    for item in batch
                ]
            },
        )

        embedding_values = response.json()["embeddings"]
        normalized_embeddings = [
            normalize_embedding(row["values"]) for row in embedding_values
        ]

        # Convert numpy types to native Python types for Pydantic validation
        embeddings+=normalized_embeddings # concatenate the lists
    return embeddings

        
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
    return [float(x) for x in normalized_embedding.tolist()]