import time
import uuid
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from aetheris.api.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatCompletionChoice, ChatMessage, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    CompletionRequest, CompletionResponse, CompletionChoice,
    ModelList, ModelCard
)
from aetheris.inference import InferenceEngine

app = FastAPI(title="Aetheris API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: InferenceEngine = None

def get_engine():
    global engine
    if engine is None:
        # Defaults, ideally loaded from config/env
        engine = InferenceEngine()
    return engine

@app.on_event("startup")
async def startup_event():
    get_engine()

@app.get("/")
async def root():
    return {"status": "running", "message": "Aetheris API is active. Use /v1/chat/completions for inference."}

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id="aetheris-hybrid-mamba-moe")])

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    engine = get_engine()
    
    # Simple prompt construction from messages
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if request.stream:
        async def event_generator():
            yield json.dumps(ChatCompletionChunk(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None
                )]
            ).model_dump())

            # Offload synchronous generation to a thread to avoid blocking the event loop
            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            import threading
            stop_event = threading.Event()

            def producer():
                try:
                    # Run the synchronous generator
                    for token in engine.generate(
                        prompt=prompt,
                        max_new_tokens=request.max_tokens or 100,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        repetition_penalty=1.0 + request.frequency_penalty,
                        stream=True
                    ):
                        if stop_event.is_set():
                            break
                        # Schedule the put() coroutine on the main loop
                        asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                except Exception as e:
                    print(f"Generation error: {e}")
                finally:
                    # Signal done
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    
                    yield json.dumps(ChatCompletionChunk(
                        id=request_id,
                        created=created_time,
                        model=request.model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=token),
                            finish_reason=None
                        )]
                    ).model_dump())
                
                yield json.dumps(ChatCompletionChunk(
                    id=request_id,
                    created=created_time,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop"
                    )]
                ).model_dump())
                
                yield "[DONE]"
            finally:
                stop_event.set()

        return EventSourceResponse(event_generator())

    else:
        generated_text = engine.generate_full(
            prompt=prompt,
            max_new_tokens=request.max_tokens or 100,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.0 + request.frequency_penalty
        )

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop"
            )],
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(generated_text), "total_tokens": len(prompt) + len(generated_text)} # Approximated
        )

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    engine = get_engine()
    
    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = prompt[0] # Handle single prompt for now

    request_id = f"cmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if request.stream:
        # Streaming for completions not fully implemented to match OpenAI exactly in this demo, 
        # but logic is similar to chat.
        # For simplicity, returning non-streaming for now or basic stream.
        pass # TODO: Implement streaming for completions

    generated_text = engine.generate_full(
        prompt=prompt,
        max_new_tokens=request.max_tokens or 16,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=1.0 + request.frequency_penalty
    )

    return CompletionResponse(
        id=request_id,
        created=created_time,
        model=request.model,
        choices=[CompletionChoice(
            text=generated_text,
            index=0,
            logprobs=None,
            finish_reason="length" # or stop
        )],
        usage={"prompt_tokens": len(prompt), "completion_tokens": len(generated_text), "total_tokens": len(prompt) + len(generated_text)}
    )
