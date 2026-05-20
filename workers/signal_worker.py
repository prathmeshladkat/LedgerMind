import asyncio
import logging
from workers.base_worker import BaseWorker
from agents.filing_pipeline.nodes.extract import extract_signals_node
from agents.state import create_initial_filing_state
from infra.kafka.topics import FILING_CHUNKED, SIGNALS_RAW, SIGNAL_GROUP
from infra.kafka.producer import get_producer
from infra.redis.client import init_redis
from infra.qdrant.client import init_qdrant

logger = logging.getLogger(__name__)

class SignalWorker(BaseWorker):
    def __init__(self):
        super().__init__(topic=FILING_CHUNKED, group_id=SIGNAL_GROUP)

    async def process_message(self, message: dict) -> None:
        """
        runs RAG + LLM extraction.
        chunks are already in Qdrant from chunk worker.
        we just need ticker and filing_type to search them.
        """
        logger.info(f"Extracting signals: ticker={message['ticker']}")

        state = create_initial_filing_state(
            ticker=message["ticker"],
            filing_type=message["filing_type"],
            thread_id=message["thread_id"],
            job_id=message["job_id"],
        )
        state["filing_date"] = message.get("filing_date")

        chunk_count = message.get("chunk_count", 0)
        if chunk_count == 0:
            raise Exception("No chunks found - chunk worker may have failed")
        
        state["chunks"] = ["placeholder"] * chunk_count

        result = await extract_signals_node(state)
        state.update(result)

        if state["errors"] and not state.get("signals"):
            raise Exception(state["errors"][-1])
        
        producer = get_producer()
        await producer.send_and_wait(
            SIGNALS_RAW,
            value={
                "ticker": state["ticker"],
                "filing_type": state["filing_type"],
                "thread_id": state["thread_id"],
                "job_id": state["job_id"],
                "filing_date": state.get("filing_date"),
                "signals": state.get("signals"),
                "confidence": state.get("confidence", 0.0),
                "retry_count": 0,
            }
        )
        logger.info(
            f"Published to {SIGNALS_RAW}: "
            f"confidence={state.get('confidence', 0)}"
        )


async def main():
    logging.basicConfig(level="INFO")
    init_redis()
    init_qdrant()
    from infra.kafka.producer import init_producer
    await init_producer()

    worker = SignalWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
         