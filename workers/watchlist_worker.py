import asyncio
import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy import select, update
import httpx

from infra.postgres.database import init_db, get_session
from infra.postgres.models import WatchlistItem, OutboxEvent, Job
from infra.kafka.topics import FILING_INGEST
from infra.settings import get_settings

logger = logging.getLogger(__name__)

# how long to wait between polls
POLL_INTERVAL_SECONDS = 60 * 15   # 15 minutes

# SEC EDGAR headers - same as fetch agent
EDGAR_HEADERS = {
    "User-Agent": "CapitalSense research@capitalsense.ai",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_BASE_URL = "https://data.sec.gov"


async def get_cik_for_ticker(ticker: str) -> str | None:
    """
    converts ticker to SEC CIK number.
    returns None if ticker not found instead of raising.
    watchlist worker should skip unknown tickers gracefully.
    """
    try:
        url = f"{EDGAR_BASE_URL}/files/company_tickers.json"
        async with httpx.AsyncClient(headers=EDGAR_HEADERS) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)

        logger.warning(f"Ticker not found in EDGAR: {ticker}")
        return None

    except Exception as e:
        logger.error(f"Failed to get CIK for {ticker}: {e}")
        return None


async def get_latest_filing_date(
    cik: str,
    filing_type: str
) -> str | None:
    """
    returns the date of the most recent filing of given type.
    returns None if no filing found.
    """
    try:
        url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"
        async with httpx.AsyncClient(headers=EDGAR_HEADERS) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        filings = data["filings"]["recent"]
        forms = filings["form"]
        dates = filings["filingDate"]

        for form, date in zip(forms, dates):
            if form == filing_type:
                return date

        return None

    except Exception as e:
        logger.error(f"Failed to get filing date: {e}")
        return None


async def check_ticker_for_new_filings(
    item: WatchlistItem
) -> list[dict]:
    """
    checks if ticker has any new filings since last_checked_at.
    returns list of new filings found.
    empty list means no new filings.
    """
    new_filings = []
    ticker = item.ticker
    filing_types = item.filing_types or ["10-K", "10-Q", "8-K"]
    last_checked = item.last_checked_at

    logger.info(f"Checking {ticker} for new filings...")

    cik = await get_cik_for_ticker(ticker)
    if not cik:
        return []

    for filing_type in filing_types:
        latest_date = await get_latest_filing_date(cik, filing_type)

        if not latest_date:
            continue

        # convert string date to datetime for comparison
        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

        # PATTERN: Idempotency check
        # only trigger pipeline if filing is newer than last check
        # prevents same filing from entering pipeline twice
        if last_checked is None or latest_dt > last_checked:
            logger.info(
                f"New {filing_type} found for {ticker}: {latest_date}"
            )
            new_filings.append({
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": latest_date,
            })

    return new_filings


async def trigger_pipeline_for_filing(filing: dict) -> None:
    """
    creates Job + OutboxEvent for a newly detected filing.
    PATTERN: Outbox - same pattern as gateway uses
    ensures filing enters pipeline exactly once even if worker restarts.
    """
    job_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    kafka_payload = {
        "ticker": filing["ticker"],
        "filing_type": filing["filing_type"],
        "thread_id": thread_id,
        "job_id": job_id,
    }

    async for session in get_session():
        job = Job(
            id=job_id,
            ticker=filing["ticker"],
            filing_type=filing["filing_type"],
            status="pending",
            thread_id=thread_id,
            retry_count=0,
        )

        outbox_event = OutboxEvent(
            topic=FILING_INGEST,
            payload=kafka_payload,
            sent=False,
        )

        session.add(job)
        session.add(outbox_event)
        await session.commit()

    logger.info(
        f"Triggered pipeline: {filing['ticker']} "
        f"{filing['filing_type']} → thread_id={thread_id}"
    )


async def update_last_checked(ticker: str) -> None:
    """updates last_checked_at for a ticker after checking."""
    async for session in get_session():
        await session.execute(
            update(WatchlistItem)
            .where(WatchlistItem.ticker == ticker)
            .values(last_checked_at=datetime.now(timezone.utc))
        )
        await session.commit()


async def run_watchlist_check() -> None:
    """
    main check function - runs once per poll cycle.
    reads all watchlist tickers from Postgres.
    checks each for new filings.
    triggers pipeline for any new ones found.
    """
    logger.info("Running watchlist check...")

    async for session in get_session():
        result = await session.execute(select(WatchlistItem))
        items = result.scalars().all()

    if not items:
        logger.info("Watchlist is empty - nothing to check")
        return

    logger.info(f"Checking {len(items)} tickers...")

    for item in items:
        try:
            new_filings = await check_ticker_for_new_filings(item)

            for filing in new_filings:
                await trigger_pipeline_for_filing(filing)

            # update last_checked even if no new filings
            # so next check doesn't re-scan old dates
            await update_last_checked(item.ticker)

        except Exception as e:
            logger.error(f"Error checking {item.ticker}: {e}")
            # continue checking other tickers even if one fails

    logger.info("Watchlist check complete")


async def main():
    """
    entry point - runs watchlist check on a timer.
    loops forever: check → sleep 15 min → check → sleep → ...
    """
    logging.basicConfig(level="INFO")
    init_db()

    logger.info(
        f"Watchlist monitor started. "
        f"Polling every {POLL_INTERVAL_SECONDS}s"
    )

    while True:
        await run_watchlist_check()
        logger.info(f"Sleeping {POLL_INTERVAL_SECONDS}s until next check...")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())