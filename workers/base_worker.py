import json
import logging
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from infra.kafka.topics import FILING_DLQ
from infra.kafka.producer import get_consumer_config
from infra.settings import get_settings

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

class BaseWorker:
    """
    base class all workers inherit from.
    handles the consumer loop, error handling and DLQ.
    subclasses only need to implement process_message().
    """

    def __init__(self, topic:str, group_id: str):
        """
        topic - which kafka topic this worker listens to
        group_id - consumer group name
        """
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.producer = None
        self.running = False

    async def process_message(self, message: dict) -> None:
        """
        override this in each worker subclass.
        receives the deserialized message dict.
        raise an exception to trigger retry + DLQ logic.
        """
        raise NotImplementedError("Each worker must implement process_message()")

    async def send_to_dlq(self, message: dict, error: str) -> None:
        """
        PATTERN: Dead Letter Queue
        sends failed message to DLQ topic with error details attached.
        allows manual inspection and reprocessing without data loss.
        """
        dlq_payload = {
            "original_message": message,
            "error": error,
            "source_topic": self.topic,
            "group_id": self.group_id,
        }
        await self.producer.send_and_wait(
            FILING_DLQ,
            value=json.dumps(dlq_payload).encode("utf-8")
        )
        logger.error(f"Sent to DLQ: {error}")

    async def start(self) -> None:
        """
        starts the consumer and producer then enters the main loop.
        call this to run the worker — it runs forever until stopped.
        """
        settings = get_settings()

        # create consumer with shared config from producer.py
        self.consumer = AIOKafkaConsumer(
            self.topic,
            **get_consumer_config(self.group_id)
        )

        # each worker needs its own producer for DLQ publishing
        self.producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: v if isinstance(v, bytes)
                             else json.dumps(v).encode("utf-8")
        )

        await self.consumer.start()
        await self.producer.start()
        self.running = True

        logger.info(
            f"Worker started: topic={self.topic} group={self.group_id}"
        )

        try:
            await self._consume_loop()
        finally:
            await self.consumer.stop()
            await self.producer.stop()
            logger.info(f"Worker stopped: topic={self.topic}")

    async def _consume_loop(self) -> None:
        """
        main consumer loop.
        runs forever, processing one message at a time.

        PATTERN: Competing Consumers
        multiple instances of the same worker can run simultaneously.
        Kafka assigns different partitions to each instance.
        they never process the same message twice.

        PATTERN: Manual Commit
        we only commit the offset AFTER successfully processing.
        if worker crashes before commit, Kafka resends the message.
        this gives us at-least-once delivery guarantee.
        """
        async for kafka_message in self.consumer:
            if not self.running:
                break
            
            message = kafka_message.value
            attempt = 0
            success = False
            last_error = None    # add this line
    
            while attempt < MAX_ATTEMPTS and not success:
                try:
                    attempt += 1
                    logger.info(
                        f"Processing message attempt {attempt}/{MAX_ATTEMPTS} "
                        f"topic={self.topic}"
                    )
                    await self.process_message(message)
                    success = True
    
                except Exception as e:
                    last_error = e    # save error here
                    logger.warning(f"Attempt {attempt} failed: {str(e)}")
                    if attempt < MAX_ATTEMPTS:
                        wait_seconds = attempt * 1
                        logger.info(f"Retrying in {wait_seconds}s...")
                        await asyncio.sleep(wait_seconds)
    
            if not success:
                # use last_error instead of e
                await self.send_to_dlq(message, str(last_error))
    
            await self.consumer.commit()

    async def stop(self) -> None:
        """gracefully stops the worker loop."""
        self.running = False



