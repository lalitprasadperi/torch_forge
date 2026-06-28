"""
Mini vLLM API Server — OpenAI-Compatible REST Endpoint

Runs a FastAPI server that exposes:
  POST /v1/completions      — text completion (with streaming support)
  POST /v1/chat/completions — chat completion (messages → text)
  GET  /v1/models           — list available models
  GET  /health              — health check

The LLMEngine runs in a background asyncio thread (run_in_executor) so
the HTTP server stays responsive while the GPU is working.

USAGE:
  python server/api_server.py --model nano --port 8000

  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "mini-gpt", "prompt": "Hello world", "max_tokens": 50}'
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, "/home/jmd/Desktop/TrainHard/LearnTorch/Capstone5")

from engine.llm_engine import LLMEngine
from model.config import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig
from server.protocol import (
    ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChunk,
    CompletionStreamDelta, ErrorResponse, ModelCard, ModelList, UsageInfo,
)


# ── Minimal tokenizer for demo (byte-level) ────────────────────────────────────

class ByteTokenizer:
    """
    Trivial byte-level tokenizer for demo purposes.
    vocab_size = 256, each byte is one token.
    Replace with tiktoken or HuggingFace tokenizer for real use.
    """
    vocab_size = 256

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list) -> str:
        try:
            return bytes(token_ids).decode("utf-8", errors="replace")
        except Exception:
            return ""


# ── Global engine instance ─────────────────────────────────────────────────────

_engine: Optional[LLMEngine] = None
_model_name: str = "mini-gpt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _model_name
    args = _parse_args()
    _model_name = args.model

    print(f"[Server] Loading model: {args.model}")
    tokenizer = ByteTokenizer()

    if args.model == "nano":
        model_config = ModelConfig.nano()
        model_config.vocab_size = tokenizer.vocab_size
    elif args.model == "gpt2-small":
        model_config = ModelConfig.gpt2_small()
    else:
        model_config = ModelConfig.nano()
        model_config.vocab_size = tokenizer.vocab_size

    cache_config     = CacheConfig(
        block_size      = args.block_size,
        num_gpu_blocks  = args.num_gpu_blocks,
        num_cpu_blocks  = args.num_cpu_blocks,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs           = args.max_num_seqs,
        max_num_batched_tokens = args.max_num_batched_tokens,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _engine = LLMEngine.from_config(model_config, cache_config, scheduler_config,
                                     tokenizer, device=device)
    print(f"[Server] Ready on port {args.port}")
    yield
    print("[Server] Shutting down")


app = FastAPI(title="Mini vLLM", version="0.1.0", lifespan=lifespan)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "stats": _engine.stats if _engine else {}}


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelCard(id=_model_name)])


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")

    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    sp = SamplingParams(
        temperature        = request.temperature,
        top_k              = request.top_k,
        top_p              = request.top_p,
        max_tokens         = request.max_tokens,
        repetition_penalty = request.repetition_penalty,
        seed               = request.seed,
    )

    if request.stream:
        return StreamingResponse(
            _stream_completion(prompt, sp, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming: run to completion
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, _engine.generate, prompt, sp)

    completion_tokens = len(output.outputs[0].token_ids)
    prompt_tokens     = len(_engine.tokenizer.encode(prompt))

    return CompletionResponse(
        model   = request.model,
        choices = [CompletionChoice(
            text         = output.outputs[0].text,
            index        = 0,
            finish_reason = output.outputs[0].finish_reason,
        )],
        usage = UsageInfo(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Convert chat messages to a prompt string and call completions."""
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")

    # Naive prompt construction
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"

    sp = SamplingParams(
        temperature = request.temperature,
        top_p       = request.top_p,
        max_tokens  = request.max_tokens,
    )

    loop   = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, _engine.generate, prompt, sp)
    text   = output.outputs[0].text.strip()

    completion_tokens = len(output.outputs[0].token_ids)
    prompt_tokens     = len(_engine.tokenizer.encode(prompt))

    return ChatCompletionResponse(
        model   = request.model,
        choices = [ChatChoice(
            index        = 0,
            message      = ChatMessage(role="assistant", content=text),
            finish_reason = output.outputs[0].finish_reason,
        )],
        usage = UsageInfo(
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
        ),
    )


# ── Streaming helper ───────────────────────────────────────────────────────────

async def _stream_completion(
    prompt: str,
    sp:     SamplingParams,
    model:  str,
) -> AsyncGenerator[str, None]:
    """Yield SSE chunks as the engine generates tokens."""
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created    = int(time.time())
    loop       = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        for token_text in _engine.stream(prompt, sp):
            loop.call_soon_threadsafe(queue.put_nowait, token_text)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(None, _run)

    while True:
        token_text = await queue.get()
        if token_text is None:
            break
        chunk = CompletionStreamChunk(
            id      = request_id,
            created = created,
            model   = model,
            choices = [CompletionStreamDelta(text=token_text, index=0)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


# ── CLI ────────────────────────────────────────────────────────────────────────

_args = None

def _parse_args():
    global _args
    if _args is not None:
        return _args
    parser = argparse.ArgumentParser(description="Mini vLLM Server")
    parser.add_argument("--model",               default="nano")
    parser.add_argument("--port",         type=int, default=8000)
    parser.add_argument("--host",                 default="0.0.0.0")
    parser.add_argument("--block-size",   type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=512)
    parser.add_argument("--num-cpu-blocks", type=int, default=128)
    parser.add_argument("--max-num-seqs",  type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run("server.api_server:app", host=args.host, port=args.port, reload=False)
