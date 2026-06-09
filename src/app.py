from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Generator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src import config
from src.downloader import download_model
from src.inference import LLMInference
from src.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ModelInfo,
    ModelList,
)

_llm: LLMInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    if config.AUTO_DOWNLOAD and not os.path.exists(config.MODEL_PATH):
        download_model()
    _llm = LLMInference()
    yield
    _llm = None


app = FastAPI(
    title="llama-cpp-python OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- dependencies ----------

def get_llm() -> LLMInference:
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _llm


def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    if not config.API_KEY:
        return
    if authorization != f"Bearer {config.API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------- routes ----------

@app.get("/health")
def health():
    return {"status": "ok", "model": config.MODEL_ID, "loaded": _llm is not None}


@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
def list_models():
    return ModelList(data=[ModelInfo(id=config.MODEL_ID, created=int(time.time()))])


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(verify_api_key)],
)
def chat_completions(
    req: ChatCompletionRequest,
    llm: LLMInference = Depends(get_llm),
):
    messages = [m.model_dump() for m in req.messages]
    max_tokens = req.max_tokens or config.MAX_TOKENS
    temperature = req.temperature if req.temperature is not None else config.TEMPERATURE

    if req.stream:
        return StreamingResponse(
            _stream_chat(llm, messages, max_tokens, temperature, req.model),
            media_type="text/event-stream",
        )

    text = llm.chat(messages, max_tokens=max_tokens, temperature=temperature)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop")],
    )


@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    dependencies=[Depends(verify_api_key)],
)
def completions(
    req: CompletionRequest,
    llm: LLMInference = Depends(get_llm),
):
    max_tokens = req.max_tokens or config.MAX_TOKENS
    temperature = req.temperature if req.temperature is not None else config.TEMPERATURE
    text = llm.generate(req.prompt, max_tokens=max_tokens, temperature=temperature)
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[CompletionChoice(text=text, index=0, finish_reason="stop")],
    )


# ---------- streaming helper ----------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_chat(
    llm: LLMInference,
    messages: list,
    max_tokens: int,
    temperature: float,
    model: str,
) -> Generator[str, None, None]:
    call_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield _sse({
        "id": call_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })

    for chunk in llm.stream_chat(messages, max_tokens=max_tokens, temperature=temperature):
        delta = chunk["choices"][0]["delta"]
        finish = chunk["choices"][0]["finish_reason"]
        if "content" in delta:
            yield _sse({
                "id": call_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": delta["content"]}, "finish_reason": finish}],
            })

    yield _sse({
        "id": call_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"
