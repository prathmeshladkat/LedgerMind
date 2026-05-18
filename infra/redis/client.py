# 1. LangGraph checkpointer (saves graph state between steps)
# 2. pub/sub (emit agent publishes, dashboard/voice subscribes)
# 3. cache (stores signals so voice agent reads fast)

import redis.asyncio as aioredis
from infra.settings import get_settings
import logging
import json

logger = logging.getLogger(__name__)

_redis_client = None

def init_redis():
    """
    creates Redis connection to Upstash
    rediss:// means SSL connection (Upstash requires this).
    decode_responses=True means we get strings back, not bytes.
    """
    global _redis_client
    settings = get_settings()

    _redis_client = aioredis.from_url(
        settings.upstash_redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )

    logger.info("Redis client initialized")

def get_redis() -> aioredis.Redis:
    """
    returns the Redis client.
    raises error if init_Redis() was not called first.
    import this wherever you need Redis.
    """

    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_Redis() at startup.")
    return _redis_client

async def close_redis():
    """call this when app shuts down."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        logger.info("Redis connection closed")


async def publish_status(thread_id: str, status: str, data: dict = None):
    """
    publishes job progress update to a Redis channel.
    the FastAPI WebSocket handler subscribes to this channel
    and forwards updates to the browser in real time.
    so when fetch agent finishes, browser shows 'fetching done'.
    """
    redis =  get_redis()
    payload = {
        "thread_id": thread_id,
        "status": status,
        **(data or {})

    }
    await redis.publish(f"job:{thread_id}", json.dumps(payload))

async def cache_signals(ticker: str, signals: dict, ttl: int = 3600):
    """
    stores extracted signals in Redis for fast reads.
    this is the CQRS read side - writes go to Postgres,
    reads come from Redis cache.
    ttl=3600 means cache expires after 1 hour.
    voice agent calls get_cached_signals() instead of
    hitting Postgres every time.
    """
    redis = get_redis()
    await redis.setex(
        f"signals:{ticker}",
        ttl,
        json.dumps(signals)
    )


async def get_cached_signals(ticker: str) -> dict | None:
    """
    reads signals from Redis cache.
    returns None if not cached (cache miss).
    caller then falls back to Postgres.
    """
    redis = get_redis()
    data = await redis.get(f"signals:{ticker}")
    return json.loads(data) if data else None