import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langgraph.types import Command
from agents.filing_pipeline.graph import get_filing_graph_with_redis

logger = logging.getLogger(__name__)

router = APIRouter()


class ReviewRequest(BaseModel):
    """
    analyst submits corrections for low-confidence signals.
    corrections is a partial dict - only fields they want to change.
    e.g. {"revenue_growth_yoy": 0.18, "guidance_sentiment": "positive"}
    """
    corrections: dict


class ReviewResponse(BaseModel):
    thread_id: str
    status: str
    message: str


@router.post("/review/{thread_id}", response_model=ReviewResponse)
async def submit_review(thread_id: str, request: ReviewRequest):
    """
    resumes a paused LangGraph graph with analyst corrections.

    PATTERN: HITL resume
    1. analyst opens review UI, sees low-confidence signals
    2. they correct wrong fields and submit this endpoint
    3. we call graph.ainvoke with Command(resume=corrections)
    4. LangGraph unpauses from interrupt() in validate_output_node
    5. validate node merges corrections into signals
    6. pipeline continues to emit node
    7. signals saved to Postgres and published to Redis
    """
    if not request.corrections:
        raise HTTPException(
            status_code=400,
            detail="corrections cannot be empty"
        )

    logger.info(
        f"Review submitted: thread_id={thread_id} "
        f"corrections={request.corrections}"
    )

    try:
        # get graph with Redis checkpointer
        # same thread_id = finds the paused graph state in Redis
        graph = get_filing_graph_with_redis()

        config = {"configurable": {"thread_id": thread_id}}

        # Command(resume=...) tells LangGraph to unpause
        # and pass corrections back to interrupt() caller
        await graph.ainvoke(
            Command(resume={"corrections": request.corrections}),
            config=config
        )

        return ReviewResponse(
            thread_id=thread_id,
            status="resumed",
            message="Graph resumed with corrections. "
                    "Check WebSocket for completion status.",
        )

    except Exception as e:
        logger.error(f"Failed to resume graph: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume graph: {str(e)}"
        )