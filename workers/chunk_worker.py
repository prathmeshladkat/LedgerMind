import asyncio
import logging
from workers.base_worker import BaseWorker
from agents.filing_pipeline.nodes.chunk import chunk_document_node
from agents.state import create_initial_filing_state
from infra.kafka.topics import FILING_RAW, FILING_CHUNKED, CHUNK_GROUP
from infra.kafka.producer import get_producer
from infra.gcs.client import download_filing
from infra.redis.client import init_redis
from infra.qdrant.client import init_qdrant, ensure_collection

logger = logging.getLogger(__name__)

class ChunkWorker(BaseWorker):
    def __init__(self):
        super().__init__(topic=FILING_RAW, group_id=CHUNK_GROUP)

    async def process_message(self, message: dict) -> None:
        """
        downloads raw HTML from GCS path in message,
        runs chunk agent, publishes to filing.chunked.
        """
        logger.info(f"Chunking: ticker={message['ticker']}")

        raw_html = await download_filing(message["gcs_blob_path"])

        state = create_initial_filing_state(
            ticker=message["ticker"],
            filing_type=message["filing_type"],
            thread_id=message["thread_id"],
            job_id=message["job_id"],
        )
        state["raw_html"] = raw_html
        state["filing_date"] = message.get("filing_date")
        state["gcs_blob_path"] = message["gcs_blob_path"]

        result = await chunk_document_node(state)
        state.update(result)

        if state["errors"]:
            raise Exception(state["errors"][-1])
        
        producer = get_producer()
        await producer.send_and_wait(
            FILING_CHUNKED,
            value={
                "ticker": state["ticker"],
                "filing_type": state["filing_type"],
                "thread_id": state["thread_id"],
                "job_id": state["job_id"],
                "filing_date": state.get("filing_date"),
                "chunk_count": len(state["chunks"]),
            }
        )
        logger.info(
            f"Published to {FILING_CHUNKED}: "
            f"{len(state['chunks'])} chunks"
        )

async def main():
    logging.basicConfig(level="INFO")
    init_redis()
    init_qdrant()
    await ensure_collection()    # creates Qdrant collection if not exists
    from infra.kafka.producer import init_producer
    await init_producer()

    worker = ChunkWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())