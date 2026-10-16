import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the context manager runs the startup (lifespan) hook
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_classify_endpoint(client):
    response = client.post("/v1/classify", json={"subject": "Error", "body": "Compressor stops with alarm E104"})
    assert response.status_code == 200
    assert response.json()["category"] == "technical"
    assert "x-request-id" in response.headers


def test_chat_endpoint(client):
    response = client.post("/v1/chat", json={"question": "What does alarm E212 mean?", "user_groups": ["service"]})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "answered" and body["citations"]


def test_request_validation(client):
    assert client.post("/v1/classify", json={"body": ""}).status_code == 422
