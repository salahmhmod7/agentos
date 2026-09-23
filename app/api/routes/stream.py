"""Streaming chat endpoint using Server-Sent Events (SSE).

The client connects with GET and receives a stream of events as
the agent runs. Each event is a JSON line prefixed with "data: ".

Event types (see app.agent.observability.EventType):
  - run_start
  - iteration_start
  - llm_call_start / llm_call_end
  - tool_call_start / tool_call_end
  - run_end
"""

import asyncio
import json
import queue

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.agent.loop import run_agent
from app.agent.observability import Event, EventType, Tracer, make_sse_handler


router = APIRouter(tags=["chat"])


def _event_to_dict(event: Event) -> dict:
    """Convert an Event to a JSON-safe dict."""
    return {
        "type": event.type.value,
        "timestamp": event.timestamp.isoformat(),
        "data": event.data,
    }


@router.get("/chat/stream")
async def chat_stream(
    message: str = Query(..., min_length=1, max_length=5000),
    conversation_id: int | None = Query(default=None),
) -> StreamingResponse:
    """Stream agent events as they happen (SSE)."""

    # Queue bridges the sync agent thread → async response generator
    q: "queue.Queue[Event | None]" = queue.Queue()
    tracer = Tracer([make_sse_handler(q)])

    # Send "run_end" event, then a sentinel (None) to close the stream
    def _run_agent_in_thread() -> None:
        try:
            run_agent(
                user_message=message,
                conversation_id=conversation_id,
                tracer=tracer,
            )
        except Exception as e:
            # Push an error event and close
            from datetime import datetime
            q.put(Event(type=EventType.ERROR, data={"message": str(e)}))
        finally:
            q.put(None)  # sentinel = end of stream

    async def event_generator():
        # Start the agent in a worker thread so we don't block the event loop
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _run_agent_in_thread)

        while True:
            # Poll the queue without blocking the event loop
            try:
                event = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if event is None:
                # End of stream
                yield "event: end\ndata: {}\n\n"
                break

            payload = json.dumps(_event_to_dict(event), ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )