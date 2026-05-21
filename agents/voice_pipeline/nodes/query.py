import logging
from sqlalchemy import select
from agents.state import VoiceState
from infra.redis.client import get_cached_signals, cache_signals
from infra.postgres.database import get_session
from infra.postgres.models import FilingSignal

logger = logging.getLogger(__name__)

async def fetch_from_postgres(ticker: str) -> dict | None:
    """
    reads latest signals for a ticker from postgres.
    called only on cache miss.
    returns None if no signals exist yet for this ticker.
    """
    async for session in get_session():
        result = await session.execute(
            select(FilingSignal)
            .where(FilingSignal.ticker == ticker)
            .order_by(FilingSignal.created_at.desc())
            .limit(1)
        )
        signal = result.scalar_one_or_none()

        if not signal:
            return None
        
        return {
            "ticker": signal.ticker,
            "filing_type": signal.filing_type,
            "filing_date": signal.filing_date,
            "revenue_growth_yoy": signal.revenue_growth_yoy,
            "gross_margin": signal.gross_margin,
            "guidance_sentiment": signal.guidance_sentiment,
            "key_risks": signal.key_risks,
            "red_flags": signal.red_flags,
            "summary": signal.summary,
            "confidence": signal.confidence,
        }
    
async def query_agent_node(state: VoiceState) -> dict:
    """
    fetches signals for all tickers mentioned by user.
    uses Redis cache first, falls back to Postgres.

    PATTERN: Cache-Aside
    1. check Redis cache for ticker
    2. cache hit  → use cached data instantly
    3. cache miss → query Postgres → store in Redis → return

    if no signals found for a ticker, returns a helpful message
    instead of crashing - user might ask about a company we
    haven't analyzed yet.
    """
    tickers = state.get("tickers", [])
    intent = state.get("intent", "lookup")

    if not tickers:
        logger.info("No tickers in state - returning generic response")
        return{
            "fetched_data": {
                "found": False,
                "message": "No company mentioned. Please say a ticker symbol.",
            }    

        }
    
    logger.info(f"Fetching data for tickers={tickers} intent={intent}")

    results = {}

    for ticker in tickers:
        # Cache-Aside
        cached = await get_cached_signals(ticker)

        if cached:
            #cache hit
            logger.info(f"Cache hit for {ticker}")
            results[ticker] = cached

        else:
            #cache miss
            logger.info(f"Cache miss for {ticker}, querying Postgres...")
            db_data = await fetch_from_postgres(ticker)

            if db_data:
                #populate cache fir next time
                await cache_signals(ticker, db_data)
                results[ticker] = db_data
                logger.info(f"Cached signals for {ticker}")
            else : 
                # no data at all - company not analyzed yet
                results[ticker] = {
                    "found": False,
                    "message": (
                        f"No analysis found for {ticker}. "
                        f"Try analyzing it first."
                    )
                }
                logger.info(f"No signals found for {ticker}")

    return {
        "fetched_data": {
            "found": True,
            "results": results,
            "intent": intent,
        }
    }



