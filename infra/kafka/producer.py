import json
import logging
from aiokafka import AIOKafkaProducer
from infra.settings import get_settings

logger = logging.getLogger(__name__)

_producer = None

async def init_producer():
    """
    creates and starts kafka producer.
    local kafka needs no auth - just bootstrap server.
    value_serializer converts dict to bytes automatically.
    call this once at app startup.
    """

    global _producer
    settings = get_settings()

    _producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,

        # converts dict → json string → bytes before sending
        # so we can do: publish("topic", {"key": "value"})
        # instead of:   publish("topic", json.dumps({}).encode())
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),

        # converts string key to bytes
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )

    await _producer.start()
    logger.info("Kafka producer started")


def get_producer() -> AIOKafkaProducer:
    """returns producer. raises if not initialized."""
    if _producer is None:
        raise RuntimeError("Producer not initialized. Call init_producer() at startup.")
    return _producer


async def close_producer():
    """call this when app shuts down."""
    global _producer
    if _producer:
        await _producer.stop()
        logger.info("Kafka producer stopped")


async def publish(topic: str, payload: dict, key: str = None):
    """
    sends one message to a kafka topic.
    key is optional - used to ensure related messages
    go to the same partition (same ticker = same partition).
    raises exception if send fails - caller handles retry.
    """
    producer = get_producer()
    await producer.send_and_wait(topic, value=payload, key=key)
    logger.debug(f"Published to {topic} key={key}")


def get_consumer_config(group_id: str) -> dict:
    """
    returns config dict for AIOKafkaConsumer.
    every worker calls this instead of repeating config.
    enable_auto_commit=False means worker manually commits
    after successfully processing each message.
    this prevents losing messages if worker crashes mid-processing.
    """
    settings = get_settings()
    return {
        "bootstrap_servers": settings.kafka_bootstrap_servers,
        "group_id": group_id,

        # earliest = read from beginning if no offset saved
        # useful when starting fresh or debugging
        "auto_offset_reset": "earliest",

        # manual commit = we tell kafka we processed the message
        # only after our agent successfully finishes
        # if we crash before committing, kafka resends the message
        "enable_auto_commit": False,
        "max_poll_interval_ms": 300000,   # 5 minutes
        "session_timeout_ms": 30000,
        "heartbeat_interval_ms": 10000,

        # converts bytes → json string → dict automatically
        "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
    }