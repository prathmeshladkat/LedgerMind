import asyncio
import logging
from workers.base_worker import BaseWorker
from agents.filing_pipeline.nodes.validate import (
    validate_output_node,
    route_after_validation
)
from agents.filing_pipeline.nodes.emit import emit_node
from agents.state import create_initial_filing_state
from infra.kafka.topics import SIGNALS_RAW, SIGNALS_VALIDATED, VALIDATOR_GROUP
from infra.kafka.producer import get_producer
from infra.redis.client import init_redis
from infra.qdrant.client import init_qdrant
from infra.postgres.database import init_db

logger = logging.getLogger(__name__)


class ValidatorWorker(BaseWorker):
    def __init__(self):
        super().__init__(topic=SIGNALS_RAW, group_id=VALIDATOR_GROUP)

    async def process_message(self, message: dict) -> None:
        thread_id = message["thread_id"]
        ticker = message["ticker"]

        logger.info(f"Validating: ticker={ticker} thread_id={thread_id}")

        # build state from message
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

        try:
            # run validate node directly - no full graph needed
            result = await validate_output_node(state)
            state.update(result)

            # check routing decision
            route = route_after_validation(state)

            if route == "retry":
                # republish to signals.raw with incremented retry_count
                # signal worker will re-extract with retry prompt
                logger.info(
                    f"Low confidence ({state['confidence']}), "
                    f"routing back to extract"
                )
                producer = get_producer()
                await producer.send_and_wait(
                    SIGNALS_RAW,
                    value={
                        "ticker": ticker,
                        "filing_type": state["filing_type"],
                        "thread_id": thread_id,
                        "job_id": state["job_id"],
                        "filing_date": state.get("filing_date"),
                        "signals": state.get("signals"),
                        "confidence": state.get("confidence", 0.0),
                        "retry_count": state["retry_count"],
                    }
                )

            elif route == "emit":
                # run emit node directly
                emit_result = await emit_node(state)
                state.update(emit_result)

                if state.get("errors"):
                    raise Exception(state["errors"][-1])

                logger.info(f"Pipeline complete: ticker={ticker}")

        except Exception as e:
            logger.error(f"Validator failed: {e}")
            raise


async def main():
    logging.basicConfig(level="INFO")
    init_db()
    init_redis()
    init_qdrant()
    from infra.kafka.producer import init_producer
    await init_producer()

    worker = ValidatorWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())