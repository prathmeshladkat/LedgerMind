# tests/unit/test_step4.py
# tests for gateway endpoints and outbox pattern
# uses FastAPI TestClient - no real Kafka or Postgres needed
# run with: pytest test/unit/test_step4.py -v

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


# ── Outbox pattern tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_create_job_with_outbox_returns_ids():
    """
    PATTERN: Outbox
    create_job_with_outbox must return valid job_id and thread_id.
    both are UUIDs - we just check they are non-empty strings.
    """
    # mock the database session so no real DB needed
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.outbox.get_session", mock_get_session):
        from gateway.outbox import create_job_with_outbox
        job_id, thread_id = await create_job_with_outbox(
            ticker="AAPL",
            filing_type="10-K",
        )

    assert isinstance(job_id, str)
    assert len(job_id) > 0
    assert isinstance(thread_id, str)
    assert len(thread_id) > 0
    # both must be different UUIDs
    assert job_id != thread_id


@pytest.mark.asyncio
async def test_create_job_writes_both_rows_in_one_session():
    """
    PATTERN: Outbox atomicity
    both Job and OutboxEvent must be added to same session.
    session.add must be called exactly twice.
    """
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.outbox.get_session", mock_get_session):
        from gateway.outbox import create_job_with_outbox
        await create_job_with_outbox("GOOGL", "10-Q")

    # must add exactly 2 rows: Job + OutboxEvent
    assert mock_session.add.call_count == 2
    # must commit once (atomic transaction)
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_outbox_event_has_correct_topic():
    """
    OutboxEvent must use FILING_INGEST topic.
    wrong topic = message never reaches fetch worker.
    """
    from infra.kafka.topics import FILING_INGEST
    added_objects = []

    mock_session = AsyncMock()
    mock_session.add = MagicMock(
        side_effect=lambda obj: added_objects.append(obj)
    )
    mock_session.commit = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.outbox.get_session", mock_get_session):
        from gateway.outbox import create_job_with_outbox
        await create_job_with_outbox("AAPL", "10-K")

    # find the OutboxEvent in added objects
    from infra.postgres.models import OutboxEvent
    outbox_events = [
        o for o in added_objects
        if isinstance(o, OutboxEvent)
    ]
    assert len(outbox_events) == 1
    assert outbox_events[0].topic == FILING_INGEST
    assert outbox_events[0].sent is False


# ── Filing router tests ───────────────────────────────────────

@pytest.fixture
def client():
    """
    creates a FastAPI TestClient.
    lifespan is disabled so we don't need real DB/Redis/Kafka.
    """
    from gateway.main import app
    # override lifespan to skip real client initialization
    with patch("gateway.main.init_db"), \
         patch("gateway.main.init_redis"), \
         patch("gateway.main.init_qdrant"), \
         patch("gateway.main.ensure_collection",
               new=AsyncMock()), \
         patch("gateway.main.init_producer",
               new=AsyncMock()):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def test_health_endpoint_returns_ok(client):
    """health endpoint must return 200 with ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_returns_thread_id(client):
    """
    POST /analyze must return job_id and thread_id.
    client uses thread_id to connect to WebSocket.
    """
    with patch(
        "gateway.routers.filings.create_job_with_outbox",
        new=AsyncMock(return_value=("job-123", "thread-456"))
    ):
        response = client.post("/api/analyze", json={
            "ticker": "AAPL",
            "filing_type": "10-K"
        })

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job-123"
    assert data["thread_id"] == "thread-456"
    assert data["ticker"] == "AAPL"
    assert data["status"] == "pending"


def test_analyze_endpoint_uppercases_ticker(client):
    """ticker must be uppercased regardless of input."""
    with patch(
        "gateway.routers.filings.create_job_with_outbox",
        new=AsyncMock(return_value=("job-1", "thread-1"))
    ):
        response = client.post("/api/analyze", json={
            "ticker": "aapl",    # lowercase input
            "filing_type": "10-K"
        })

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"   # must be uppercase


def test_analyze_rejects_invalid_filing_type(client):
    """invalid filing type must return 400 error."""
    response = client.post("/api/analyze", json={
        "ticker": "AAPL",
        "filing_type": "INVALID"   # not 10-K, 10-Q, or 8-K
    })
    assert response.status_code == 400
    assert "Invalid filing_type" in response.json()["detail"]


def test_analyze_rejects_empty_ticker(client):
    """empty ticker must return 400 error."""
    response = client.post("/api/analyze", json={
        "ticker": "",
        "filing_type": "10-K"
    })
    assert response.status_code == 400


def test_status_endpoint_returns_404_for_unknown_thread(client):
    """unknown thread_id must return 404."""
    with patch(
        "gateway.routers.filings.get_job_status",
        new=AsyncMock(return_value=None)
    ):
        response = client.get("/api/status/unknown-thread-id")
    assert response.status_code == 404


def test_status_endpoint_returns_job_data(client):
    """status endpoint must return job fields."""
    mock_job = {
        "job_id": "job-123",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "status": "running",
        "thread_id": "thread-456",
        "retry_count": 0,
        "created_at": "2024-01-01",
    }
    with patch(
        "gateway.routers.filings.get_job_status",
        new=AsyncMock(return_value=mock_job)
    ):
        response = client.get("/api/status/thread-456")

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert response.json()["status"] == "running"


# ── Review router tests ───────────────────────────────────────

def test_review_endpoint_rejects_empty_corrections(client):
    """
    PATTERN: HITL
    empty corrections must return 400.
    analyst must provide at least one correction.
    """
    response = client.post(
        "/api/review/thread-123",
        json={"corrections": {}}
    )
    assert response.status_code == 400


def test_review_endpoint_accepts_valid_corrections(client):
    """
    valid corrections must resume the graph and return 200.
    """
    with patch(
        "gateway.routers.review.get_filing_graph_with_redis",
        return_value=MagicMock(
            ainvoke=AsyncMock(return_value={"status": "done"})
        )
    ):
        response = client.post(
            "/api/review/thread-123",
            json={"corrections": {"revenue_growth_yoy": 0.22}}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "resumed"
    assert data["thread_id"] == "thread-123"


# ── CDC relay tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_cdc_relay_publishes_unsent_events():
    """
    PATTERN: Outbox CDC relay
    relay must find unsent events and publish them to Kafka.
    must mark events as sent after publishing.
    """
    from infra.postgres.models import OutboxEvent

    # create a fake unsent outbox event
    mock_event = MagicMock(spec=OutboxEvent)
    mock_event.id = "event-1"
    mock_event.topic = "filing.ingest"
    mock_event.payload = {"ticker": "AAPL"}

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_event]
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.commit = AsyncMock()

    mock_producer = AsyncMock()
    mock_producer.send_and_wait = AsyncMock()

    async def mock_get_session():
        yield mock_session

    with patch("gateway.outbox.get_session", mock_get_session), \
         patch("gateway.outbox.get_producer", return_value=mock_producer):

        from gateway.outbox import _publish_pending_outbox_events
        await _publish_pending_outbox_events()

    # Kafka publish must have been called
    mock_producer.send_and_wait.assert_called_once()
    call_args = mock_producer.send_and_wait.call_args
    assert call_args[0][0] == "filing.ingest"