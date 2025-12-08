from flask_openapi3 import OpenAPI, Info, Tag
from flask_cors import CORS

info = Info(title="  API", version="0.0.1", description="API for creating embeddings for transaction descriptions")
app = OpenAPI(__name__, info=info)
CORS(app)

# definindo tags
home_tag = Tag(name="Docs", 
                description="Go to API doc")

embedding_tag = Tag(name="Embedding",
                    description="Creates an embedding for a transaction description, using the Google Gemini API")