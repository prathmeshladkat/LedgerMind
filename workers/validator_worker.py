import asyncio
import logging
from workers.base_worker import BaseWorker
from agents.filing_pipeline.graph import get_filing_graph_with_redis
from agents.state import create_initial_filing_state
from infra.kafka.topics import SIGNALS_RAW, SIGNALS_VALIDATED, VALIDATOR_GROUP
from infra.kafka.producer import get_producer
from infra.redis.client import init_redis
from infra.qdrant.client import init_qdrant

logger = logging.getLogger(__name__)

class ValidatorWorker(BaseWorker):
    def __init__(self):
        super().__init__(topic=SIGNALS_RAW, group_id=VALIDATOR_GROUP)

    async def process_message(self, message: dict) -> None:
        """
        runs validate node via the full LangGraph graph.
        uses thread_id as the graph checkpoint key in Redis.

        three outcomes:
        1. high confidence → publishes to signals.validated
        2. low confidence, retries left → graph loops back internally
        3. low confidence, no retries → graph pauses (interrupt)
           POST /review/{thread_id} resumes it later
        """
        thread_id = message["thread_id"]
        ticker = message["ticker"]

        logger.info(f"Validating: ticker={ticker} thread_id={thread_id}")

        state = create_initial_filing_state(
            ticker=ticker,
            filing_type=message["filing_type"],
            thread_id=thread_id,
            job_id=message["job_id"],
        )
        state["signals"] = message.get("signals")
        state["confidence"] = message.get("confidence", 0.0)
        state["retry_count"] = message.get("retry_count", 0)
        state["filing_date"] = message.get("filing_date")

        config = {"configurable": {"thread_id": thread_id}}

        graph = get_filing_graph_with_redis()
        result = await graph.ainvoke(state, config=config)

        if result and result.get("signals"):
            producer = get_producer()
            await producer.send_and_wait(
                SIGNALS_VALIDATED,
                value={
                    "ticker": result["ticker"],
                    "filing_type": result["filing_type"],
                    "thread_id": thread_id,
                    "job_id": result["job_id"],
                    "filing_date": result.get("filing_date"),
                    "signals": result["signals"],
                    "confidence": result["confidence"],
                    "human_feedback": result.get("human_feedback"),
                }
            )
            logger.info(f"Published to {SIGNALS_VALIDATED}: {ticker}")
        else:

            logger.info(
                f"Graph paused for human review: thread_id={thread_id}"
            )


async def main():
    logging.basicConfig(level="INFO")
    init_redis()
    init_qdrant()
    from infra.kafka.producer import init_producer
    await init_producer()

    worker = ValidatorWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())