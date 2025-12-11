import logging
from schemas.embedding_schema import EmbeddingResponseSchema, EmbeddingListSchema, ErrorSchema
from config import embedding_tag
from config import app
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from machine_learning.embedding import normalize_embedding
from tqdm.auto import tqdm
tqdm.pandas()

# Load environment variables from the project root .env file, if present.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'models/gemini-embedding-001')
GEMINI_EMBEDDING_URL = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:embedContent"
DIMENSIONALITY = 768 # GOOGLE recommends using 768, 1536, or 3072 output dimensions.

logger = logging.getLogger(__name__)


@app.post('/embedding', tags=[embedding_tag],
          responses={"200": EmbeddingResponseSchema, "409": ErrorSchema, "400": ErrorSchema})
def embedding(body: EmbeddingListSchema):
    """
    Creates an embedding for a transaction description, using the Google Gemini API.

    REF: https://ai.google.dev/gemini-api/docs/embeddings#python
    """
    try:
        embeddings = []
        batch_size = 25  # at most 100 requests can be in one batch
        array_of_texts = [item.description for item in body.descriptions]

        # Loop over the texts in chunks of batch_size
        for i in tqdm(range(0, len(array_of_texts), batch_size)):
            batch = array_of_texts[i : i + batch_size]
            response = requests.post(
            GEMINI_EMBEDDING_URL,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={
                "model": GEMINI_MODEL,
                "task_type": "CLASSIFICATION",
                "content": {"parts": [{"text": item.description} for item in body.descriptions]},
                "output_dimensionality": DIMENSIONALITY,
            },
            )
            embedding_values = response.json()['embedding']['values']
            normalized_embeddings = [normalize_embedding(row) for row in embedding_values]

            # Convert numpy types to native Python types for Pydantic validation
            embeddings.extend([float(x) for x in normalized_embeddings])

        # for item in body.descriptions:
        #     response = requests.post(
        #         GEMINI_EMBEDDING_URL,
        #         headers={"x-goog-api-key": GEMINI_API_KEY},
        #         json={
        #             "model": GEMINI_MODEL,
        #             "task_type": "CLASSIFICATION",
        #             "content": {"parts": [{"text": item.description}]},
        #             "output_dimensionality": DIMENSIONALITY,
        #         },
        #     )
        #     logger.debug(f"Embedding created for transaction: '{item.description}'")
        #     embedding_values = response.json()['embedding']['values']
        #     normalized_embedding = normalize_embedding(embedding_values)
        #     # Convert numpy types to native Python types for Pydantic validation
        #     embeddings.append([float(x) for x in normalized_embedding])

        return {"embeddings": embeddings}, 200

    except Exception as e:
        error_msg = f"Error creating embedding for transaction '{body}': {e}"   
        logger.warning(error_msg)
        return {"message": error_msg}, 400