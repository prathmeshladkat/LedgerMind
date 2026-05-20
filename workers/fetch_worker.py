import asyncio
import logging
from workers.base_worker import BaseWorker
from agents.filing_pipeline.nodes.fetch import fetch_filing_node
from agents.state import create_initial_filing_state
from infra.kafka.topics import FILING_INGEST, FILING_RAW, FETCH_GROUP
from infra.kafka.producer import get_producer
from infra.redis.client import init_redis
from infra.qdrant.client import init_qdrant

logger = logging.getLogger(__name__)

class FetchWorker(BaseWorker):
    def __init__(self):
       super().__init__(topic=FILING_INGEST, group_id=FETCH_GROUP)

    async def process_message(self, message: dict) -> None:
        """
        called by base class for each Kafka message.
        message contains: ticker, filing_type, thread_id, job_id

        flow:
        1. build initial state from message
        2. run fetch agent
        3. if successful, publish to next topic (filing.raw)
        4. if failed, base class sends to DLQ after MAX_ATTEMPTS
        """
        logger.info(
            f"Processing: ticker={message['ticker']}"
            f"filing_type={message['filing_type']}"
        )

        state = create_initial_filing_state(
            ticker=message["ticker"],
            filing_type=message["filing_type"],
            thread_id=message["thread_id"],
            job_id=message["job_id"],
        )
        
        result = await fetch_filing_node(state)

        # merge result back into state
        state.update(result)

        # if fetch failed (errors list not empty), raise so base retries
        if state["errors"]:
            raise Exception(state["errors"][-1])

        # fetch succeeded — publish to next topic in pipeline
        # chunk_worker is listening to filing.raw
        producer = get_producer()
        await producer.send_and_wait(
            FILING_RAW,
            value={
                "ticker": state["ticker"],
                "filing_type": state["filing_type"],
                "thread_id": state["thread_id"],
                "job_id": state["job_id"],
                "gcs_blob_path": state["gcs_blob_path"],
                "filing_date": state["filing_date"],
                "accession_number": state["accession_number"],
            }
        )
        logger.info(f"Published to {FILING_RAW}: {state['ticker']}")


async def main():
    """entry point — starts the worker."""
    logging.basicConfig(level="INFO")

    # initialize shared clients before starting worker
    init_redis()
    init_qdrant()
    from infra.kafka.producer import init_producer
    await init_producer()

    worker = FetchWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())

        
 