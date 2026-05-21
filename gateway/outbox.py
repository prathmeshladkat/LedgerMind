import uuid
import logging
import asyncio
import json

from datetime import datetime, timezone
from sqlalchemy import select, update
from infra.postgres.database import get_session
from infra.postgres.models import Job, OutboxEvent
from infra.kafka.topics import FILING_INGEST
from infra.kafka.producer import get_producer

logger = logging.getLogger(__name__)

async def create_job_with_outbox(
        ticker: str,
        filing_type: str,
) -> tuple[str, str]:
    """
    PATTERN: Outbox
    writes Job + OutboxEvent in a single atomic transaction.
    returns (job_id, thread_id) so gateway can return them to client.

    atomic = both succeed or both fail together
    never partial: job without event, or event without job
    """
    job_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    kafka_payload = {
        "ticker": ticker,
        "filing_type": filing_type,
        "thread_id": thread_id,
        "job_id": job_id
    }
    
    async for session in get_session():
        job = Job(
            id=job_id,
            ticker=ticker,
            filing_type=filing_type,
            status ="pending",
            thread_id=thread_id,
            retry_count=0,
        )

        outbox_event = OutboxEvent(
            topic=FILING_INGEST,
            payload=kafka_payload,
            sent=False,
        )

        session.add(job)
        session.add(outbox_event)

        await session.commit()

        logger.info(
            f"Job created: job_id={job_id}"
            f"ticker={ticker} thread_id={thread_id}"
        )

    return job_id, thread_id


async def get_job_status(thread_id: str) -> dict | None:
    """
    reads job status from postgres.
    called by GET /status/{status_id} endpoint.
    """
    async for session in get_session():
        result = await session.execute(
            select(Job).where(Job.thread_id == thread_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            return None
        
        return {
            "job_id": job.id,
            "ticker": job.ticker,
            "filing_type": job.filing_type,
            "status": job.status,
            "thread_id": job.thread_id,
            "retry_count": job.retry_count,
            "created_at": str(job.created_at),
        }
    

async def run_cdc_relay():
    """
    CDC (Change Data Capture) relay.
    polls Postgres outbox_events table every second.
    publishes unsent events to Kafka.
    marks them as sent after successful publish.

    PATTERN: Outbox CDC Relay
    this is the second half of the Outbox pattern.
    runs as a background task inside the gateway process.
    in production this would be Debezium, but for our project
    a simple polling loop works perfectly and is easier to demo.

    why polling instead of Debezium?
    - Debezium requires Kafka Connect setup (complex)
    - polling every 1 second is fast enough for our use case
    - same guarantee: events are never lost
    """
    logger.info("CDC relay started - polling outbox table")

    while True:
        try:
            await _publish_pending_outbox_events()
        except Exception as e:
            logger.error(f"CDC relay error: {e}")

        await asyncio.sleep(1)

async def _publish_pending_outbox_events():
    """
    finds all unsent outbox events and publishes them to Kafka.
    marks each as sent after successful publish.
    """
    async for session in get_session():
        result = await session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.sent == False)  # noqa
            .limit(10)   # process max 10 at a time
        )
        events = result.scalars().all()

        if events:
            logger.info(f"CDC relay found {len(events)} unsent events")

        if not events:
            return

        producer = get_producer()

        for event in events:
            try:
                # publish to Kafka
                await producer.send_and_wait(
                    event.topic,
                    value=event.payload,
                )

                # mark as sent
                await session.execute(
                    update(OutboxEvent)
                    .where(OutboxEvent.id == event.id)
                    .values(
                        sent=True,
                        sent_at=datetime.now(timezone.utc)
                    )
                )
                logger.info(
                    f"CDC relay published: topic={event.topic} "
                    f"id={event.id}"
                )

            except Exception as e:
                logger.error(
                    f"CDC relay failed to publish event {event.id}: {e}"
                )
                # don't mark as sent — will retry next poll

        await session.commit()