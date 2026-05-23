# agents/filing_pipeline/nodes/validate.py
# validates signal quality and routes accordingly
# HITL is handled by publishing to a special Redis channel
# instead of using LangGraph interrupt() which requires graph context

import logging
from agents.state import FilingState
from infra.redis.client import publish_status, get_redis
import json

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.85
MAX_RETRIES = 2


async def validate_output_node(state: FilingState) -> dict:
    """
    checks signal quality and routes accordingly.
    three outcomes:
    1. high confidence → approve
    2. low confidence + retries left → increment retry_count
    3. low confidence + no retries → publish to human_review channel
    """
    thread_id = state["thread_id"]
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    signals = state.get("signals")
    errors = state.get("errors", [])

    logger.info(
        f"Validating signals: confidence={confidence} "
        f"retry_count={retry_count}"
    )

    # case 1: no signals at all
    if not signals:
        error_msg = "No signals to validate — upstream agent failed"
        logger.error(error_msg)

        await publish_status(thread_id, "validation_failed", {
            "message": error_msg,
            "errors": errors
        })

        # publish to human review channel instead of interrupt()
        await _publish_human_review(thread_id, state, "no_signals", error_msg)

        return {
            "signals": signals,
            "confidence": confidence,
            "human_feedback": "escalated_no_signals",
        }

    # case 2: low confidence but retries remaining
    if confidence < CONFIDENCE_THRESHOLD and retry_count < MAX_RETRIES:
        logger.info(
            f"Low confidence ({confidence}), "
            f"retrying ({retry_count + 1}/{MAX_RETRIES})"
        )

        await publish_status(thread_id, "retrying", {
            "message": f"Low confidence, retrying "
                      f"({retry_count + 1}/{MAX_RETRIES})",
            "confidence": confidence
        })

        return {"retry_count": retry_count + 1}

    # case 3: low confidence + no retries left → human review
    if confidence < CONFIDENCE_THRESHOLD and retry_count >= MAX_RETRIES:
        logger.warning(f"Max retries reached, escalating to human review")

        await publish_status(thread_id, "human_review_required", {
            "message": "Low confidence after retries — analyst review needed",
            "confidence": confidence,
            "signals": signals,
        })

        # PATTERN: HITL without interrupt()
        # publish to Redis channel so review UI can show it
        await _publish_human_review(
            thread_id, state, "low_confidence",
            "Please review and correct these signals"
        )

        return {
            "signals": signals,
            "confidence": confidence,
            "human_feedback": "pending_review",
        }

    # case 4: high confidence — approve
    logger.info(f"Signals approved: confidence={confidence}")
    await publish_status(thread_id, "validated", {
        "message": "Signals validated successfully",
        "confidence": confidence
    })

    return {}


async def _publish_human_review(
    thread_id: str,
    state: FilingState,
    reason: str,
    message: str
):
    """
    publishes to human_review Redis channel.
    Review UI subscribes to this and shows the job to analysts.
    """
    try:
        redis = get_redis()
        payload = {
            "thread_id": thread_id,
            "ticker": state.get("ticker"),
            "filing_type": state.get("filing_type"),
            "job_id": state.get("job_id"),
            "reason": reason,
            "message": message,
            "signals": state.get("signals"),
            "confidence": state.get("confidence"),
        }
        await redis.publish("human_review", json.dumps(payload))
        # also store in Redis so review page can fetch it
        await redis.setex(
            f"review:{thread_id}",
            86400,  # 24 hours
            json.dumps(payload)
        )
        logger.info(f"Published to human_review channel: {thread_id}")
    except Exception as e:
        logger.error(f"Failed to publish human review: {e}")


def route_after_validation(state: FilingState) -> str:
    """
    routing function for conditional edges.
    returns emit or retry.
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    signals = state.get("signals")
    human_feedback = state.get("human_feedback")

    # if escalated to human review, move to emit
    # emit will save whatever signals we have
    if human_feedback in ("escalated_no_signals", "pending_review"):
        return "emit"

    # if just incremented retry_count, loop back
    if confidence < CONFIDENCE_THRESHOLD and retry_count <= MAX_RETRIES and signals:
        return "retry"

    return "emit"