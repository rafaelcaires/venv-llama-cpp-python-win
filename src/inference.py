from __future__ import annotations

from collections.abc import Generator
from typing import Any

from llama_cpp import Llama  # type: ignore[import-not-found]

from src import config


class LLMInference:
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        n_ctx: int = config.N_CTX,
        n_threads: int = config.N_THREADS,
        n_gpu_layers: int = config.N_GPU_LAYERS,
        n_batch: int = config.N_BATCH,
    ) -> None:
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        try:
            result = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
            return result["choices"][0]["text"]  # type: ignore[index]
        except RuntimeError:
            self.llm.reset()
            raise

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        try:
            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result["choices"][0]["message"]["content"]  # type: ignore[index]
        except RuntimeError:
            self.llm.reset()
            raise

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> Generator[dict[str, Any], None, None]:
        try:
            yield from self.llm.create_chat_completion(  # type: ignore[misc]
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        except RuntimeError:
            self.llm.reset()
            raise
