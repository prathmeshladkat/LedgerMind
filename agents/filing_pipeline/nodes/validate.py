
import logging
from langgraph.types import interrupt
from agents.state import FilingState
from infra.redis.client import publish_status

logger = logging.getLogger(__name__)

# confidence threshold - above this, signals are published automatically
CONFIDENCE_THRESHOLD = 0.85

# max retries before escalating to human
# PATTERN: Circuit Breaker - stops retry loop after N attempts
MAX_RETRIES = 2


async def validate_output_node(state: FilingState) -> dict:
    """
    checks signal quality and routes accordingly.
    this node does NOT call an LLM - it's pure Python logic.

    three possible outcomes:
    1. confidence >= 0.85 → approve, move to emit
    2. confidence < 0.85 AND retries left → increment retry_count, loop back
    3. confidence < 0.85 AND no retries left → interrupt() for human review

    LangGraph reads the routing function (route_after_validation)
    to decide which node runs next based on what this node returns.
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

    # case 1: no signals at all (fetch or chunk failed)
    # escalate immediately without retrying
    if not signals:
        error_msg = "No signals to validate - upstream agent failed"
        logger.error(error_msg)

        await publish_status(thread_id, "validation_failed", {
            "message": error_msg,
            "errors": errors
        })

        # PATTERN: HITL - interrupt pauses graph, saves state to Redis
        # analyst sees this in the review UI and can manually fill signals
        human_input = interrupt({
            "reason": "no_signals",
            "message": error_msg,
            "errors": errors,
        })

        # when analyst resumes, human_input contains their corrections
        return {
            "signals": human_input.get("signals", {}),
            "confidence": 1.0,   # human reviewed = full confidence
            "human_feedback": str(human_input),
        }

    # case 2: low confidence but retries remaining
    # PATTERN: Circuit Breaker - only retry if under MAX_RETRIES
    if confidence < CONFIDENCE_THRESHOLD and retry_count < MAX_RETRIES:
        logger.info(
            f"Low confidence ({confidence}), "
            f"retrying ({retry_count + 1}/{MAX_RETRIES})"
        )

        await publish_status(thread_id, "retrying", {
            "message": f"Low confidence, retrying extraction "
                      f"({retry_count + 1}/{MAX_RETRIES})",
            "confidence": confidence
        })

        # increment retry_count - extract agent sees this and uses retry prompt
        return {
            "retry_count": retry_count + 1,
        }

    # case 3: low confidence AND no retries left
    # PATTERN: Circuit Breaker trips - escalate to human
    if confidence < CONFIDENCE_THRESHOLD and retry_count >= MAX_RETRIES:
        logger.warning(
            f"Max retries reached ({MAX_RETRIES}), "
            f"escalating to human review"
        )

        await publish_status(thread_id, "human_review_required", {
            "message": "Low confidence after retries - analyst review needed",
            "confidence": confidence,
            "signals": signals,
        })

        # PATTERN: HITL - graph pauses here
        # RedisSaver saves entire state to Upstash Redis
        # POST /review/{thread_id} resumes graph with analyst corrections
        human_input = interrupt({
            "reason": "low_confidence",
            "confidence": confidence,
            "current_signals": signals,
            "message": "Please review and correct these signals",
        })

        # merge analyst corrections into signals
        corrected_signals = {**signals, **human_input.get("corrections", {})}

        return {
            "signals": corrected_signals,
            "confidence": 1.0,
            "human_feedback": str(human_input),
        }

    # case 4: high confidence - approve and move to emit
    logger.info(f"Signals approved: confidence={confidence}")

    await publish_status(thread_id, "validated", {
        "message": "Signals validated successfully",
        "confidence": confidence
    })

    # return empty dict - state is already correct, nothing to change
    return {}


def route_after_validation(state: FilingState) -> str:
    """
    PATTERN: Conditional Edge (LangGraph routing)
    LangGraph calls this function after validate_output_node runs.
    the string we return tells LangGraph which node to run next.

    "emit"   → move forward to emit_node
    "retry"  → loop back to extract_signals_node
    "end"    → stop graph (only on critical failure)

    this function implements the retry loop logic.
    it reads retry_count to know if we just incremented it.
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    signals = state.get("signals")

    # if validate node just incremented retry_count,
    # and we're still under MAX_RETRIES, loop back
    if confidence < CONFIDENCE_THRESHOLD and retry_count <= MAX_RETRIES and signals:
        return "retry"

    # otherwise move forward to emit
    return "emit"