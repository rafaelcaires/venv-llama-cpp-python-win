from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

# ---------- requests ----------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-7b-instruct"
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = "qwen2.5-7b-instruct"
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False


# ---------- responses ----------

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage = Usage()


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str]


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage = Usage()


# ---------- models endpoint ----------

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
