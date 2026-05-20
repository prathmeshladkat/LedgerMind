
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, delete
from infra.postgres.database import get_session
from infra.postgres.models import WatchlistItem

logger = logging.getLogger(__name__)

router = APIRouter()


class AddTickerRequest(BaseModel):
    ticker: str
    filing_types: list[str] = ["10-K", "10-Q", "8-K"]


@router.get("/watchlist")
async def list_watchlist():
    """returns all tickers currently being monitored."""
    async for session in get_session():
        result = await session.execute(select(WatchlistItem))
        items = result.scalars().all()

        return {
            "watchlist": [
                {
                    "ticker": item.ticker,
                    "filing_types": item.filing_types,
                    "last_checked_at": str(item.last_checked_at)
                    if item.last_checked_at else None,
                }
                for item in items
            ]
        }


@router.post("/watchlist")
async def add_to_watchlist(request: AddTickerRequest):
    """
    adds a ticker to the watchlist.
    watchlist monitor will start checking it on next poll cycle.
    """
    ticker = request.ticker.upper().strip()

    if not ticker:
        raise HTTPException(status_code=400, detail="ticker cannot be empty")

    async for session in get_session():
        # check if already exists
        result = await session.execute(
            select(WatchlistItem).where(WatchlistItem.ticker == ticker)
        )
        existing = result.scalar_one_or_none()

        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"{ticker} is already in watchlist"
            )

        item = WatchlistItem(
            ticker=ticker,
            filing_types=request.filing_types,
        )
        session.add(item)
        await session.commit()

    logger.info(f"Added to watchlist: {ticker}")
    return {"message": f"{ticker} added to watchlist", "ticker": ticker}


@router.delete("/watchlist/{ticker}")
async def remove_from_watchlist(ticker: str):
    """removes a ticker from the watchlist."""
    ticker = ticker.upper().strip()

    async for session in get_session():
        result = await session.execute(
            delete(WatchlistItem).where(WatchlistItem.ticker == ticker)
        )
        await session.commit()

        if result.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=f"{ticker} not found in watchlist"
            )

    logger.info(f"Removed from watchlist: {ticker}")
    return {"message": f"{ticker} removed from watchlist", "ticker": ticker}