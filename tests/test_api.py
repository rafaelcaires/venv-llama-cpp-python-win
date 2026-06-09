from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.app import app, get_llm


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.chat.return_value = "Mocked response"
    m.generate.return_value = "Mocked completion"
    m.stream_chat.return_value = iter([
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    return m


@pytest.fixture
def client(mock_llm):
    app.dependency_overrides[get_llm] = lambda: mock_llm
    with patch("src.app.download_model"), patch("src.app.LLMInference", return_value=mock_llm):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


# ---------- /health ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------- /v1/models ----------

def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert "id" in data["data"][0]


# ---------- /v1/chat/completions ----------

def test_chat_completions(client, mock_llm):
    r = client.post("/v1/chat/completions", json={
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Mocked response"


def test_chat_completions_respects_params(client, mock_llm):
    client.post("/v1/chat/completions", json={
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "temperature": 0.2,
    })
    mock_llm.chat.assert_called_once()
    _, kwargs = mock_llm.chat.call_args
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.2


def test_chat_completions_streaming(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        lines = [line for line in r.iter_lines() if line]
    assert any("Hello" in line for line in lines)
    assert any("[DONE]" in line for line in lines)


# ---------- /v1/completions ----------

def test_completions(client, mock_llm):
    r = client.post("/v1/completions", json={
        "model": "qwen2.5-7b-instruct",
        "prompt": "Once upon a time",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "Mocked completion"


# ---------- auth ----------

def test_api_key_accepted(client):
    with patch("src.app.config.API_KEY", "secret"):
        r = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_api_key_rejected(client):
    with patch("src.app.config.API_KEY", "secret"):
        r = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
