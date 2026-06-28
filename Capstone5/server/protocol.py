"""
OpenAI-Compatible API Protocol

We implement a subset of the OpenAI Completions API so existing clients
(LangChain, openai-python library, curl scripts) work without modification.

Reference: https://platform.openai.com/docs/api-reference/completions
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model:              str
    prompt:             Union[str, List[str]]
    max_tokens:         int  = 16
    temperature:        float = 1.0
    top_p:              float = 1.0
    top_k:              int   = -1
    n:                  int   = 1
    stop:               Optional[Union[str, List[str]]] = None
    stream:             bool  = False
    repetition_penalty: float = 1.0
    seed:               Optional[int] = None
    user:               Optional[str] = None


class CompletionChoice(BaseModel):
    text:          str
    index:         int
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs:      None = None


class UsageInfo(BaseModel):
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int


class CompletionResponse(BaseModel):
    id:      str  = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object:  str  = "text_completion"
    created: int  = Field(default_factory=lambda: int(time.time()))
    model:   str
    choices: List[CompletionChoice]
    usage:   UsageInfo


class CompletionStreamDelta(BaseModel):
    text:         str
    index:        int
    finish_reason: Optional[str] = None


class CompletionStreamChunk(BaseModel):
    id:      str
    object:  str = "text_completion"
    created: int
    model:   str
    choices: List[CompletionStreamDelta]


# Chat completions (subset)

class ChatMessage(BaseModel):
    role:    Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model:       str
    messages:    List[ChatMessage]
    max_tokens:  int   = 256
    temperature: float = 1.0
    top_p:       float = 1.0
    stream:      bool  = False


class ChatChoice(BaseModel):
    index:         int
    message:       ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id:      str  = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object:  str  = "chat.completion"
    created: int  = Field(default_factory=lambda: int(time.time()))
    model:   str
    choices: List[ChatChoice]
    usage:   UsageInfo


class ModelCard(BaseModel):
    id:         str
    object:     str = "model"
    created:    int = Field(default_factory=lambda: int(time.time()))
    owned_by:   str = "mini-vllm"


class ModelList(BaseModel):
    object: str = "list"
    data:   List[ModelCard]


class ErrorResponse(BaseModel):
    object:  str = "error"
    message: str
    type:    str
    code:    Optional[int] = None
