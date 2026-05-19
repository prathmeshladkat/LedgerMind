import json
import logging
from sqlalchemy import update
from agents.state import FilingState
from infra.redis.client import get_redis, cache_signals, publish_status
from infra.postgres.database import get_session
from infra.postgres.models import FilingSignal, Job

logger = logging.getLogger(__name__)


async def save_signals_to_postgres(state: FilingState) -> str:
    """
    writes final signals to Postgres as source of record.
    PATTERN: CQRS write side - Postgres is the authoritative store.
    returns the ID of the saved signal row.
    """
    signals = state["signals"]
    ticker = state["ticker"]
    filing_type = state["filing_type"]
    job_id = state["job_id"]

    # get_session() returns an async context manager
    async for session in get_session():
        signal_row = FilingSignal(
            job_id=job_id,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=state.get("filing_date"),
            revenue_growth_yoy=signals.get("revenue_growth_yoy"),
            gross_margin=signals.get("gross_margin"),
            guidance_sentiment=signals.get("guidance_sentiment"),
            key_risks=signals.get("key_risks", []),
            red_flags=signals.get("red_flags", []),
            summary=signals.get("summary"),
            confidence=state["confidence"],
            human_reviewed=bool(state.get("human_feedback")),
        )
        session.add(signal_row)

        # also update job status to done
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status="done")
        )

        await session.commit()
        logger.info(f"Signals saved to Postgres: {signal_row.id}")
        return signal_row.id


async def fanout_to_redis(state: FilingState):
    """
    PATTERN: Fanout via Redis pub/sub
    publishes signals to one channel.
    all subscribers receive the same message at the same time.

    subscribers:
    1. gateway WebSocket handler → pushes to browser
    2. voice cache updater → stores in Redis for fast reads
    3. alert evaluator → checks watchlist notification rules
    """
    ticker = state["ticker"]
    signals = state["signals"]
    thread_id = state["thread_id"]

    # PATTERN: CQRS read side - cache in Redis for fast reads
    # voice agent calls get_cached_signals(ticker) instead of Postgres
    await cache_signals(ticker, signals)

    # publish to fanout channel
    # every subscriber to "signals:new" receives this
    redis = get_redis()
    fanout_message = {
        "ticker": ticker,
        "filing_type": state["filing_type"],
        "thread_id": thread_id,
        "signals": signals,
        "confidence": state["confidence"],
        "filing_date": state.get("filing_date"),
    }
    await redis.publish("signals:new", json.dumps(fanout_message))
    logger.info(f"Published to fanout channel: signals:new")


async def emit_node(state: FilingState) -> dict:
    """
    main node function for emit agent.
    this is the last node in the filing pipeline.
    after this, the graph ends and the job is complete.
    """
    thread_id = state["thread_id"]
    ticker = state["ticker"]

    if not state.get("signals"):
        error = "emit skipped: no signals to emit"
        logger.error(error)
        return {"errors": state["errors"] + [error]}

    await publish_status(thread_id, "emitting", {
        "message": "Saving and publishing results..."
    })

    try:
        # step 1: CQRS write side - save to Postgres
        signal_id = await save_signals_to_postgres(state)

        # step 2: Fanout - publish to all subscribers via Redis
        await fanout_to_redis(state)

        # step 3: tell UI the job is complete
        await publish_status(thread_id, "complete", {
            "message": f"Analysis complete for {ticker}",
            "signal_id": signal_id,
            "signals": state["signals"],
        })

        logger.info(f"Pipeline complete for {ticker}")

        # return empty dict - graph ends after this node
        return {}

    except Exception as e:
        error_msg = f"emit failed: {str(e)}"
        logger.error(error_msg)
        await publish_status(thread_id, "emit_failed", {"message": error_msg})
        return {"errors": state["errors"] + [error_msg]}