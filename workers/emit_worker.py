import asyncio
import logging
from workers.base_worker import BaseWorker
from agents.filing_pipeline.nodes.emit import emit_node
from agents.state import create_initial_filing_state
from infra.kafka.topics import SIGNALS_VALIDATED, EMIT_GROUP
from infra.redis.client import init_redis
from infra.postgres.database import init_db
from infra.qdrant.client import init_qdrant

logger = logging.getLogger(__name__)


class EmitWorker(BaseWorker):
    def __init__(self):
        super().__init__(topic=SIGNALS_VALIDATED, group_id=EMIT_GROUP)

    async def process_message(self, message: dict) -> None:
        """
        final step in pipeline.
        saves validated signals to Postgres (CQRS write side).
        publishes to Redis pub/sub (Fanout pattern).
        updates job status to done.
        """
        logger.info(f"Emitting: ticker={message['ticker']}")

        state = create_initial_filing_state(
            ticker=message["ticker"],
            filing_type=message["filing_type"],
            thread_id=message["thread_id"],
            job_id=message["job_id"],
        )
        state["signals"] = message["signals"]
        state["confidence"] = message["confidence"]
        state["filing_date"] = message.get("filing_date")
        state["human_feedback"] = message.get("human_feedback")

        result = await emit_node(state)
        state.update(result)

        if state["errors"]:
            raise Exception(state["errors"][-1])

        logger.info(f"Pipeline complete: ticker={message['ticker']}")


async def main():
    logging.basicConfig(level="INFO")

    # emit worker needs Postgres and Redis
    init_db()
    init_redis()
    init_qdrant()
    from infra.kafka.producer import init_producer
    await init_producer()

    worker = EmitWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())