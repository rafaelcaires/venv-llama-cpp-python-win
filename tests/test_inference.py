from unittest.mock import patch

import pytest

from src.inference import LLMInference


@pytest.fixture
def mock_llama():
    with patch("src.inference.Llama") as MockLlama:
        yield MockLlama


def test_generate_returns_text(mock_llama):
    mock_llama.return_value.return_value = {"choices": [{"text": "Paris"}]}
    llm = LLMInference(model_path="fake.gguf")
    assert llm.generate("Capital of France?") == "Paris"


def test_generate_passes_params(mock_llama):
    mock_llama.return_value.return_value = {"choices": [{"text": "ok"}]}
    llm = LLMInference(model_path="fake.gguf")
    llm.generate("test", max_tokens=128, temperature=0.5)
    mock_llama.return_value.assert_called_once_with(
        "test", max_tokens=128, temperature=0.5, echo=False
    )


def test_chat_returns_message(mock_llama):
    mock_llama.return_value.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    llm = LLMInference(model_path="fake.gguf")
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Hello!"


def test_chat_passes_messages(mock_llama):
    mock_llama.return_value.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    llm = LLMInference(model_path="fake.gguf")
    messages = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hello"}]
    llm.chat(messages, max_tokens=256, temperature=0.3)
    mock_llama.return_value.create_chat_completion.assert_called_once_with(
        messages=messages, max_tokens=256, temperature=0.3
    )


def test_stream_chat_returns_generator(mock_llama):
    chunks = [
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    mock_llama.return_value.create_chat_completion.return_value = iter(chunks)
    llm = LLMInference(model_path="fake.gguf")
    result = list(llm.stream_chat([{"role": "user", "content": "Hey"}]))
    assert result == chunks


def test_llama_initialized_with_correct_params(mock_llama):
    LLMInference(model_path="fake.gguf", n_ctx=2048, n_threads=4, n_gpu_layers=0, n_batch=256)
    mock_llama.assert_called_once_with(
        model_path="fake.gguf",
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        n_batch=256,
        verbose=False,
    )


def test_generate_resets_on_runtime_error(mock_llama):
    mock_llama.return_value.return_value = None
    mock_llama.return_value.side_effect = RuntimeError("llama_decode returned -3")
    llm = LLMInference(model_path="fake.gguf")
    with pytest.raises(RuntimeError):
        llm.generate("prompt")
    mock_llama.return_value.reset.assert_called_once()


def test_chat_resets_on_runtime_error(mock_llama):
    err = RuntimeError("llama_decode returned -3")
    mock_llama.return_value.create_chat_completion.side_effect = err
    llm = LLMInference(model_path="fake.gguf")
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "hi"}])
    mock_llama.return_value.reset.assert_called_once()


def test_stream_chat_resets_on_runtime_error(mock_llama):
    def _failing_iter(*_args: object, **_kwargs: object):
        raise RuntimeError("llama_decode returned -3")
        yield  # make it a generator

    mock_llama.return_value.create_chat_completion.side_effect = _failing_iter
    llm = LLMInference(model_path="fake.gguf")
    with pytest.raises(RuntimeError):
        list(llm.stream_chat([{"role": "user", "content": "hi"}]))
    mock_llama.return_value.reset.assert_called_once()
