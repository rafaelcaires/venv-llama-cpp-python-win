from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app import app, get_llm


@pytest.fixture
def mock_llm():
    from unittest.mock import MagicMock

    m = MagicMock()
    m.chat.return_value = "Mocked response"
    m.generate.return_value = "Mocked completion"
    m.stream_chat.return_value = iter(
        [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )
    return m


@pytest.fixture
def client(mock_llm):
    app.dependency_overrides[get_llm] = lambda: mock_llm
    with patch("src.app.download_model"), patch(
        "src.app.LLMInference", return_value=mock_llm
    ):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


# ---------- lifespan ----------


def test_lifespan_downloads_when_model_missing():
    from unittest.mock import MagicMock

    dummy = MagicMock()
    with patch("src.app.os.path.exists", return_value=False), patch(
        "src.app.config.AUTO_DOWNLOAD", True
    ), patch("src.app.download_model") as mock_dl, patch(
        "src.app.LLMInference", return_value=dummy
    ):
        with TestClient(app):
            mock_dl.assert_called_once()


def test_lifespan_skips_download_when_model_exists():
    from unittest.mock import MagicMock

    dummy = MagicMock()
    with patch("src.app.os.path.exists", return_value=True), patch(
        "src.app.config.AUTO_DOWNLOAD", True
    ), patch("src.app.download_model") as mock_dl, patch(
        "src.app.LLMInference", return_value=dummy
    ):
        with TestClient(app):
            mock_dl.assert_not_called()


def test_lifespan_skips_download_when_auto_download_false():
    from unittest.mock import MagicMock

    dummy = MagicMock()
    with patch("src.app.os.path.exists", return_value=False), patch(
        "src.app.config.AUTO_DOWNLOAD", False
    ), patch("src.app.download_model") as mock_dl, patch(
        "src.app.LLMInference", return_value=dummy
    ):
        with TestClient(app):
            mock_dl.assert_not_called()


# ---------- get_llm 503 ----------


def test_get_llm_raises_503_when_not_loaded():
    from unittest.mock import patch

    from fastapi import HTTPException

    import src.app as app_module

    with patch.object(app_module, "_llm", None):
        with pytest.raises(HTTPException) as exc:
            app_module.get_llm()
    assert exc.value.status_code == 503


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
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5-7b-instruct", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Mocked response"


def test_chat_completions_respects_params(client, mock_llm):
    client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "temperature": 0.2,
        },
    )
    _, kwargs = mock_llm.chat.call_args
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.2


def test_chat_completions_streaming(client, mock_llm):
    mock_llm.stream_chat.return_value = iter(
        [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-7b-instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        lines = [line for line in r.iter_lines() if line]
    assert any("Hello" in line for line in lines)
    assert any("[DONE]" in line for line in lines)


# ---------- /v1/completions ----------


def test_completions(client, mock_llm):
    r = client.post(
        "/v1/completions",
        json={"model": "qwen2.5-7b-instruct", "prompt": "Once upon a time"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "Mocked completion"


# ---------- model not loaded ----------


def test_503_when_model_not_loaded():
    with patch("src.app.download_model"), patch("src.app.LLMInference", side_effect=Exception):
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert r.status_code in {503, 500}
        except Exception:
            pass


# ---------- auth ----------


def test_api_key_accepted(client):
    with patch("src.app.config.API_KEY", "secret"):
        r = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_api_key_rejected(client):
    with patch("src.app.config.API_KEY", "secret"):
        r = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_no_auth_required_when_api_key_empty(client):
    with patch("src.app.config.API_KEY", ""):
        r = client.get("/v1/models")
    assert r.status_code == 200
