# tests/unit/test_step3.py
# tests for all Kafka workers
# we test worker logic without real Kafka connection
# run with: pytest test/unit/test_step3.py -v

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── BaseWorker tests ──────────────────────────────────────────

def test_base_worker_requires_process_message():
    """
    BaseWorker.process_message() must raise NotImplementedError.
    forces every subclass to implement its own logic.
    """
    from workers.base_worker import BaseWorker
    worker = BaseWorker(topic="test.topic", group_id="test-group")

    with pytest.raises(NotImplementedError):
        import asyncio
        asyncio.run(worker.process_message({}))


def test_base_worker_stores_topic_and_group():
    """worker must store topic and group_id for consumer creation."""
    from workers.base_worker import BaseWorker
    worker = BaseWorker(topic="filing.ingest", group_id="fetch-workers")
    assert worker.topic == "filing.ingest"
    assert worker.group_id == "fetch-workers"


def test_base_worker_starts_not_running():
    """worker must not be running before start() is called."""
    from workers.base_worker import BaseWorker
    worker = BaseWorker(topic="test", group_id="test")
    assert worker.running is False


# ── FetchWorker tests ─────────────────────────────────────────

def test_fetch_worker_listens_to_correct_topic():
    """fetch worker must consume from filing.ingest."""
    from workers.fetch_worker import FetchWorker
    from infra.kafka.topics import FILING_INGEST
    worker = FetchWorker()
    assert worker.topic == FILING_INGEST


def test_fetch_worker_uses_correct_group():
    """fetch worker must use fetch-workers consumer group."""
    from workers.fetch_worker import FetchWorker
    from infra.kafka.topics import FETCH_GROUP
    worker = FetchWorker()
    assert worker.group_id == FETCH_GROUP


@pytest.mark.asyncio
async def test_fetch_worker_raises_on_agent_error():
    """
    if fetch agent returns errors, worker must raise.
    base class catches this and retries, then sends to DLQ.
    this prevents silently swallowing failures.
    """
    from workers.fetch_worker import FetchWorker

    worker = FetchWorker()
    message = {
        "ticker": "INVALID",
        "filing_type": "10-K",
        "thread_id": "t1",
        "job_id": "j1"
    }

    # mock fetch node to return an error
    with patch(
        "workers.fetch_worker.fetch_filing_node",
        new=AsyncMock(return_value={
            "errors": ["fetch failed: ticker not found"],
            "raw_html": None
        })
    ), patch(
        "workers.fetch_worker.init_redis"
    ), patch(
        "workers.fetch_worker.get_producer",
        return_value=AsyncMock()
    ):
        with pytest.raises(Exception, match="fetch failed"):
            await worker.process_message(message)


# ── ChunkWorker tests ─────────────────────────────────────────

def test_chunk_worker_listens_to_correct_topic():
    """chunk worker must consume from filing.raw."""
    from workers.chunk_worker import ChunkWorker
    from infra.kafka.topics import FILING_RAW
    worker = ChunkWorker()
    assert worker.topic == FILING_RAW


def test_chunk_worker_uses_correct_group():
    from workers.chunk_worker import ChunkWorker
    from infra.kafka.topics import CHUNK_GROUP
    worker = ChunkWorker()
    assert worker.group_id == CHUNK_GROUP


# ── SignalWorker tests ────────────────────────────────────────

def test_signal_worker_listens_to_correct_topic():
    """signal worker must consume from filing.chunked."""
    from workers.signal_worker import SignalWorker
    from infra.kafka.topics import FILING_CHUNKED
    worker = SignalWorker()
    assert worker.topic == FILING_CHUNKED


@pytest.mark.asyncio
async def test_signal_worker_raises_when_no_chunks():
    """
    signal worker must raise if chunk_count is 0.
    means chunk worker failed — no point calling LLM.
    """
    from workers.signal_worker import SignalWorker

    worker = SignalWorker()
    message = {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "thread_id": "t1",
        "job_id": "j1",
        "chunk_count": 0,    # chunk worker produced nothing
    }

    with patch("workers.signal_worker.extract_signals_node",
               new=AsyncMock()):
        with pytest.raises(Exception, match="No chunks found"):
            await worker.process_message(message)


# ── ValidatorWorker tests ─────────────────────────────────────

def test_validator_worker_listens_to_correct_topic():
    """validator worker must consume from signals.raw."""
    from workers.validator_worker import ValidatorWorker
    from infra.kafka.topics import SIGNALS_RAW
    worker = ValidatorWorker()
    assert worker.topic == SIGNALS_RAW


# ── EmitWorker tests ──────────────────────────────────────────

def test_emit_worker_listens_to_correct_topic():
    """emit worker must consume from signals.validated."""
    from workers.emit_worker import EmitWorker
    from infra.kafka.topics import SIGNALS_VALIDATED
    worker = EmitWorker()
    assert worker.topic == SIGNALS_VALIDATED


@pytest.mark.asyncio
async def test_emit_worker_raises_on_agent_error():
    """
    if emit agent returns errors, worker must raise.
    ensures failed emissions go to DLQ not silently dropped.
    """
    from workers.emit_worker import EmitWorker

    worker = EmitWorker()
    message = {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "thread_id": "t1",
        "job_id": "j1",
        "signals": None,      # no signals — emit will fail
        "confidence": 0.0,
    }

    with patch(
        "workers.emit_worker.emit_node",
        new=AsyncMock(return_value={
            "errors": ["emit skipped: no signals to emit"]
        })
    ):
        with pytest.raises(Exception, match="no signals"):
            await worker.process_message(message)


# ── DLQ tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_base_worker_sends_to_dlq_on_failure():
    """
    PATTERN: Dead Letter Queue
    after MAX_ATTEMPTS failures, message must go to DLQ.
    original message and error are preserved in DLQ payload.
    """
    from workers.base_worker import BaseWorker

    worker = BaseWorker(topic="test.topic", group_id="test-group")

    # mock producer
    mock_producer = AsyncMock()
    mock_producer.send_and_wait = AsyncMock()
    worker.producer = mock_producer

    await worker.send_to_dlq(
        message={"ticker": "AAPL"},
        error="something went wrong"
    )

    # verify DLQ publish was called
    mock_producer.send_and_wait.assert_called_once()

    # verify the call was to DLQ topic
    call_args = mock_producer.send_and_wait.call_args
    assert call_args[0][0] == "filing.dlq"