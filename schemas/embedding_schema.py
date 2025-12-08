from pydantic import BaseModel
from typing import List


class EmbeddingSchema(BaseModel):
    """ Schema for creating an embedding for a transaction description
    """
    description:str

class EmbeddingListSchema(BaseModel):
    """ Schema for creating embeddings for a list of transaction descriptions
    """
    descriptions:List[EmbeddingSchema]

class EmbeddingResponseSchema(BaseModel):
    """ Schema for the response of creating embeddings for a list of transaction descriptions
    """
    embeddings:List[List[float]]

class ErrorSchema(BaseModel):
    """ Schema for the error response
    """
    error:str
    status_code:int