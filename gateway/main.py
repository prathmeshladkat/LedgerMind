import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.routers import filings, review, voice
from infra.postgres.database import init_db, close_db
from infra.redis.client import init_redis, close_redis
from infra.kafka.producer import init_producer, close_producer
from infra.qdrant.client import init_qdrant, ensure_collection
from gateway.routers import filings, review, voice, watchlist
import asyncio
from gateway.outbox import run_cdc_relay

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    runs startup code before yield
    runs shutdown code after yield
    FastAPI calls this automatically when app starts and stops
    """

    logger.info("Starting LeadgerMind gateway...")

    init_db()
    init_redis()
    try:
        init_qdrant()
        await ensure_collection()
    except Exception as e:
        logger.warning(f"Qdrant skipped at startup: {e}")
    await init_producer()

    # this polls outbox table and publishes to Kafka
    cdc_task = asyncio.create_task(run_cdc_relay())
    logger.info("All clients initalized. Gateway ready.")

    yield 

    logger.info("Shutting down gateway...")
    await close_db()
    await close_redis()
    await close_producer()
    logger.info("Gateway shutdown complete.")

app = FastAPI(
    title="LeadgerMind",
    description="AI-native investment research infrastructure",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(filings.router, prefix="/api", tags=["filings"])
app.include_router(review.router, prefix="/api", tags=["review"])
app.include_router(voice.router, prefix="/api", tags=["voice"])
app.include_router(watchlist.router, prefix="/api", tags=["watchlist"])

@app.get("/health")
async def health():
    """simple health check endpoint for load balancers."""
    return {"status": "ok", "service": "capitalsense-gateway"}

