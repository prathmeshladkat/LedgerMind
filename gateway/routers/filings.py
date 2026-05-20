import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from gateway.outbox import create_job_with_outbox, get_job_status
from infra.redis.client import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()

VALID_FILING_TYPES = ["10-K", "10-Q", "8-k"]

class AnalyzeRequest(BaseModel):
    """request body for POST /analyze"""
    ticker: str 
    filing_type: str = "10-K"

class AnalyzeResponse(BaseModel):
    """response body for POST /analyze"""
    job_id: str
    thread_id: str
    ticker: str
    filing_type: str
    status: str
    message: str

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_filing(request: AnalyzeRequest):
    """
    creates a filing analysis job.

    PATTERN: Outbox
    writes Job + OutboxEvent atomically to Postgres.
    CDC relay picks up the OutboxEvent and publishes to Kafka.
    fetch worker receives Kafka message and starts pipeline.

    returns immediately with thread_id.
    client uses thread_id to subscribe to WebSocket for live updates.
    """
    ticker =request.ticker.upper().strip()
    filing_type = request.filing_type.upper().strip()

    if filing_type not in VALID_FILING_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filing_type. Must be one of {VALID_FILING_TYPES}"
        )
    
    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="ticker cannot be empty"
        )
    
    # PATTERN: Outbox - atomic write to Postgres
    job_id, thread_id = await create_job_with_outbox(
        ticker=ticker,
        filing_type=filing_type,
    )

    logger.info(f"Analysis requested: {ticker} {filing_type} → {thread_id}")

    return AnalyzeResponse(
        job_id=job_id,
        thread_id=thread_id,
        ticker=ticker,
        filing_type=filing_type,
        status="pending",
        message=f"Analysis started. Connect to /api/ws/{thread_id} for live updates.",
    )


@router.get("/status/{thread_id}")
async def get_status(thread_id: str):
    """
    returns current job status from Postgres.
    PATTERN: CQRS read side
    reads from Postgres (source of record) not Redis cache.
    use this for persistent status, use WebSocket for live updates.
    """

    job = await get_job_status(thread_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found for thread_id={thread_id}"
        )

    return job

@router.websocket("/ws/{thread_id}")
async def websocket_status(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for live pipeline progress updates.

    PATTERN: Redis pub/sub as WebSocket bridge
    1. client connects to this WebSocket
    2. we subscribe to Redis channel job:{thread_id}
    3. when any agent calls publish_status(thread_id, ...) in Redis,
       we receive it here and forward to browser
    4. browser updates progress bar in real time

    flow:
    agent → Redis pub/sub → this handler → WebSocket → browser
    """
    await websocket.accept()
    redis = get_redis()

    pubsub = redis.pubsub()
    await pubsub.subscribe(f"job:{thread_id}")

    logger.info(f"WebSocket connected: thread_id={thread_id}")

    try:
        # listen for messages until client disconnects or job completes
        async for message in pubsub.listen():

            # pubsub.listen() also sends subscription confirmations
            # skip those, only forward actual messages
            if message["type"] != "message":
                continue

            # forward Redis message to WebSocket client
            await websocket.send_text(message["data"])

            # parse to check if job is done
            try:
                data = json.loads(message["data"])
                status = data.get("status", "")

                # close WebSocket when job reaches terminal state
                if status in ["complete", "emit_failed", "fetch_failed"]:
                    logger.info(
                        f"Job terminal state={status}, "
                        f"closing WebSocket: {thread_id}"
                    )
                    break

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: thread_id={thread_id}")
    finally:
        await pubsub.unsubscribe(f"job:{thread_id}")
        await pubsub.close()




    