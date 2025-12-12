import logging
from schemas.embedding_schema import (
    EmbeddingResponseSchema,
    EmbeddingListSchema,
    ErrorSchema,
)
from config import embedding_tag
from config import app
from machine_learning.embedding import create_embeddings


logger = logging.getLogger(__name__)

@app.post(
    "/embedding",
    tags=[embedding_tag],
    responses={"200": EmbeddingResponseSchema, "409": ErrorSchema, "400": ErrorSchema},
)
def embedding(body: EmbeddingListSchema):
    """
    Creates an embedding for a transaction description, using the Google Gemini API.

    REF: https://ai.google.dev/gemini-api/docs/embeddings#python
    """
    try:
        text_descriptions = [item.description for item in body.descriptions]
        embeddings = create_embeddings(text_descriptions)

        return {"embeddings": embeddings}, 200

    except Exception as e:
        error_msg = f"Error creating embedding for transaction '{body}': {e}"
        logger.warning(error_msg)
        return {"message": error_msg}, 400
