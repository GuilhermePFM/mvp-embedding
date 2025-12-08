import pytest
from itertools import count

from machine_learning.embedding import normalize_embedding
import apis.embedding as embedding_api
from app import app


class DummyResponse:
    """Simple stub for the Gemini API response."""

    def __init__(self, values):
        self.values = values

    def json(self):
        return {"embeddings": [{"values": self.values}]}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_embedding_api_returns_normalized_vectors(client, monkeypatch):
    raw_embeddings = [
        [3.0, 4.0],
        [1.0, 2.0, 2.0],
    ]
    call_index = count()

    def fake_post(url, headers=None, json=None):
        idx = next(call_index)
        return DummyResponse(raw_embeddings[idx])

    monkeypatch.setattr(embedding_api.requests, "post", fake_post)

    payload = {
        "descriptions": [
            {"description": "Compra de alimentos"},
            {"description": "Pagamento de conta de luz"},
        ]
    }

    expected = [[float(x) for x in normalize_embedding(values)] for values in raw_embeddings]

    response = client.post("/embedding", json=payload, content_type="application/json")

    assert response.status_code == 200
    assert response.get_json() == {"embeddings": expected}


def test_embedding_api_handles_external_failure(client, monkeypatch):
    def fake_post(*args, **kwargs):
        raise RuntimeError("Upstream Gemini failure")

    monkeypatch.setattr(embedding_api.requests, "post", fake_post)

    payload = {"descriptions": [{"description": "Compra de alimentos"}]}

    response = client.post("/embedding", json=payload, content_type="application/json")

    assert response.status_code == 400
    body = response.get_json()
    assert "message" in body
    assert "Upstream Gemini failure" in body["message"]

def test_integration(client):
    """Test the integration of the embedding API with the Gemini API"""

    payload = {"descriptions": [{"description": "Compra de alimentos"}]}

    response = client.post("/embedding", json=payload, content_type="application/json")

    assert response.status_code == 200
    body = response.get_json()

    assert "embeddings" in body
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == 768
    assert all(isinstance(x, float) for x in body["embeddings"][0])
    assert all(0 <= x <= 1 for x in body["embeddings"][0])