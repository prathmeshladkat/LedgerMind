# tests/unit/test_step5.py
# tests for watchlist monitor and HITL flow
# run with: pytest test/unit/test_step5.py -v

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


# ── Watchlist monitor tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_run_watchlist_check_skips_empty_watchlist():
    """
    watchlist monitor must handle empty watchlist gracefully.
    no errors, no Kafka messages published.
    """
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def mock_get_session():
        yield mock_session

    with patch("workers.watchlist_worker.get_session", mock_get_session):
        from workers.watchlist_worker import run_watchlist_check
        # must not raise
        await run_watchlist_check()


@pytest.mark.asyncio
async def test_get_cik_for_ticker_returns_none_on_error():
    """
    if SEC EDGAR is unreachable, must return None not raise.
    watchlist monitor continues checking other tickers.
    """
    with patch("workers.watchlist_worker.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("network error")
        )

        from workers.watchlist_worker import get_cik_for_ticker
        result = await get_cik_for_ticker("AAPL")

    assert result is None


@pytest.mark.asyncio
async def test_check_ticker_returns_empty_when_no_new_filings():
    """
    PATTERN: Idempotency
    if last_checked_at is after filing date, no new filings returned.
    prevents same filing entering pipeline twice.
    """
    from infra.postgres.models import WatchlistItem

    # mock watchlist item - last checked today
    mock_item = MagicMock(spec=WatchlistItem)
    mock_item.ticker = "AAPL"
    mock_item.filing_types = ["10-K"]
    # last checked in the future = nothing will be "new"
    mock_item.last_checked_at = datetime(2099, 1, 1, tzinfo=timezone.utc)

    with patch(
        "workers.watchlist_worker.get_cik_for_ticker",
        new=AsyncMock(return_value="0000320193")
    ), patch(
        "workers.watchlist_worker.get_latest_filing_date",
        new=AsyncMock(return_value="2024-01-15")  # older than last_checked
    ):
        from workers.watchlist_worker import check_ticker_for_new_filings
        result = await check_ticker_for_new_filings(mock_item)

    # no new filings because filing_date < last_checked_at
    assert result == []


@pytest.mark.asyncio
async def test_check_ticker_returns_filing_when_new():
    """
    if filing date is after last_checked_at, it must be returned.
    this triggers pipeline for the new filing.
    """
    from infra.postgres.models import WatchlistItem

    mock_item = MagicMock(spec=WatchlistItem)
    mock_item.ticker = "AAPL"
    mock_item.filing_types = ["10-K"]
    mock_item.last_checked_at = None   # never checked before

    with patch(
        "workers.watchlist_worker.get_cik_for_ticker",
        new=AsyncMock(return_value="0000320193")
    ), patch(
        "workers.watchlist_worker.get_latest_filing_date",
        new=AsyncMock(return_value="2024-01-15")
    ):
        from workers.watchlist_worker import check_ticker_for_new_filings
        result = await check_ticker_for_new_filings(mock_item)

    assert len(result) == 1
    assert result[0]["ticker"] == "AAPL"
    assert result[0]["filing_type"] == "10-K"
    assert result[0]["filing_date"] == "2024-01-15"


@pytest.mark.asyncio
async def test_trigger_pipeline_writes_job_and_outbox():
    """
    PATTERN: Outbox
    trigger_pipeline_for_filing must write Job + OutboxEvent.
    same outbox pattern as gateway - atomic transaction.
    """
    added_objects = []
    mock_session = AsyncMock()
    mock_session.add = MagicMock(
        side_effect=lambda obj: added_objects.append(obj)
    )
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("workers.watchlist_worker.get_session", mock_get_session):
        from workers.watchlist_worker import trigger_pipeline_for_filing
        await trigger_pipeline_for_filing({
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-01-15",
        })

    from infra.postgres.models import Job, OutboxEvent

    jobs = [o for o in added_objects if isinstance(o, Job)]
    events = [o for o in added_objects if isinstance(o, OutboxEvent)]

    # must write exactly one Job and one OutboxEvent
    assert len(jobs) == 1
    assert len(events) == 1
    assert jobs[0].ticker == "AAPL"
    assert events[0].topic == "filing.ingest"
    assert events[0].sent is False


# ── Watchlist router tests ────────────────────────────────────

@pytest.fixture
def client():
    """FastAPI test client with mocked startup."""
    from gateway.main import app
    from fastapi.testclient import TestClient

    with patch("gateway.main.init_db"), \
         patch("gateway.main.init_redis"), \
         patch("gateway.main.init_qdrant"), \
         patch("gateway.main.ensure_collection", new=AsyncMock()), \
         patch("gateway.main.init_producer", new=AsyncMock()):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_add_ticker_to_watchlist(client):
    """POST /watchlist must add ticker and return 200."""
    mock_session = AsyncMock()
    # simulate ticker not already existing
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.routers.watchlist.get_session", mock_get_session):
        response = client.post("/api/watchlist", json={"ticker": "aapl"})

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"   # must uppercase


def test_add_duplicate_ticker_returns_409(client):
    """adding ticker already in watchlist must return 409 conflict."""
    from infra.postgres.models import WatchlistItem

    mock_session = AsyncMock()
    mock_result = MagicMock()
    # simulate ticker already exists
    mock_result.scalar_one_or_none.return_value = MagicMock(
        spec=WatchlistItem
    )
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def mock_get_session():
        yield mock_session

    with patch("gateway.routers.watchlist.get_session", mock_get_session):
        response = client.post("/api/watchlist", json={"ticker": "AAPL"})

    assert response.status_code == 409


def test_remove_ticker_from_watchlist(client):
    """DELETE /watchlist/{ticker} must return 200 on success."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.rowcount = 1    # one row deleted
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.routers.watchlist.get_session", mock_get_session):
        response = client.delete("/api/watchlist/AAPL")

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"


def test_remove_nonexistent_ticker_returns_404(client):
    """removing ticker not in watchlist must return 404."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.rowcount = 0    # nothing deleted
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.routers.watchlist.get_session", mock_get_session):
        response = client.delete("/api/watchlist/UNKNOWN")

    assert response.status_code == 404


def test_list_watchlist_returns_tickers(client):
    """GET /watchlist must return all tracked tickers."""
    from infra.postgres.models import WatchlistItem

    mock_item = MagicMock(spec=WatchlistItem)
    mock_item.ticker = "AAPL"
    mock_item.filing_types = ["10-K", "10-Q"]
    mock_item.last_checked_at = None

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_item]
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def mock_get_session():
        yield mock_session

    with patch("gateway.routers.watchlist.get_session", mock_get_session):
        response = client.get("/api/watchlist")

    assert response.status_code == 200
    data = response.json()
    assert len(data["watchlist"]) == 1
    assert data["watchlist"][0]["ticker"] == "AAPL"


# ── HITL flow tests ───────────────────────────────────────────

def test_review_endpoint_passes_corrections_to_graph(client):
    """
    PATTERN: HITL
    corrections must be passed to graph.ainvoke via Command(resume=...).
    wrong corrections structure = graph never resumes correctly.
    """
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"status": "done"})

    with patch(
        "gateway.routers.review.get_filing_graph_with_redis",
        return_value=mock_graph
    ):
        response = client.post(
            "/api/review/thread-999",
            json={"corrections": {
                "revenue_growth_yoy": 0.22,
                "guidance_sentiment": "positive"
            }}
        )

    assert response.status_code == 200
    assert response.json()["thread_id"] == "thread-999"
    assert response.json()["status"] == "resumed"

    # verify ainvoke was called - graph was actually resumed
    mock_graph.ainvoke.assert_called_once()