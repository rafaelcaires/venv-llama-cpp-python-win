from __future__ import annotations

from typing import List, Dict
from llama_cpp import Llama
from src import config


class LLMInference:
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        n_ctx: int = config.N_CTX,
        n_threads: int = config.N_THREADS,
        n_gpu_layers: int = config.N_GPU_LAYERS,
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        """Raw text completion."""
        result = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
        return result["choices"][0]["text"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        """Chat completion using the OpenAI-compatible messages format."""
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"]
