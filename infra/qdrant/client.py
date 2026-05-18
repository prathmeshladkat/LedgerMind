from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from infra.settings import get_settings
import logging

logger = logging.getLogger(__name__)

_client = None

COLLECTION_NAME = "filing_chunks"

# 384 is the output size of sentence-transformers all-MiniLM-L6-v2
# every chunk gets converted to a list of 384 numbers
VECTOR_SIZE = 384

def init_qdrant():
    """
    creates Qdrant client pointing to Qdrant Cloud.
    api_key is required for cloud, optional for local.
    """
    global _client
    settings = get_settings()

    _client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None
    )
    logger.info("Qdrant client initialized")


def get_qdrant() -> AsyncQdrantClient:
    """returns client. raises if not initialized."""
    if _client is None:
        raise RuntimeError("Qdrant not initialized. Call init_qdrant() at startup.")
    return _client


async def ensure_collection():
    """
    creates the filing_chunks collection if it doesn't exist.
    safe to call multiple times - checks before creating.
    Distance.COSINE means we measure similarity by angle between vectors
    which works well for text similarity.
    """
    client = get_qdrant()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]

    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Qdrant collection already exists: {COLLECTION_NAME}")


async def close_qdrant():
    """call this when app shuts down."""
    global _client
    if _client:
        await _client.close()
        logger.info("Qdrant client closed")
