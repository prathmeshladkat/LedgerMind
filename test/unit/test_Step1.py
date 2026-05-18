

import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Settings tests ────────────────────────────────────────────

def test_settings_loads_without_env_file():
    """
    settings must load even if .env is empty or missing.
    all fields have default values for this reason.
    """
    from infra.settings import Settings
    s = Settings(
        neon_database_url="postgresql+asyncpg://test@localhost/test",
        upstash_redis_url="redis://localhost:6379",
    )
    assert s.neon_database_url.startswith("postgresql")
    assert s.environment == "development"
    assert s.is_dev is True


def test_settings_is_singleton():
    """
    get_settings() must return same object every call.
    lru_cache ensures .env is only read once.
    """
    from infra.settings import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ── Kafka topics tests ────────────────────────────────────────

def test_topic_names_are_correct():
    """
    topic names must never change once workers are running.
    this test catches accidental renames.
    """
    from infra.kafka.topics import (
        FILING_INGEST, FILING_RAW, FILING_CHUNKED,
        SIGNALS_RAW, SIGNALS_VALIDATED, FILING_DLQ
    )
    assert FILING_INGEST == "filing.ingest"
    assert FILING_RAW == "filing.raw"
    assert FILING_CHUNKED == "filing.chunked"
    assert SIGNALS_RAW == "signals.raw"
    assert SIGNALS_VALIDATED == "signals.validated"
    assert FILING_DLQ == "filing.dlq"


def test_all_topics_list_has_seven_items():
    """all topics must be registered in ALL_TOPICS list."""
    from infra.kafka.topics import ALL_TOPICS
    assert len(ALL_TOPICS) == 7


# ── Postgres model tests ──────────────────────────────────────

def test_all_models_import_correctly():
    """all models must be importable with correct table names."""
    from infra.postgres.models import (
        Job, FilingSignal, OutboxEvent, WatchlistItem
    )
    assert Job.__tablename__ == "jobs"
    assert FilingSignal.__tablename__ == "filing_signals"
    assert OutboxEvent.__tablename__ == "outbox_events"
    assert WatchlistItem.__tablename__ == "watchlist"


def test_job_has_required_columns():
    """Job table must have all columns pipeline depends on."""
    from infra.postgres.models import Job
    cols = [c.name for c in Job.__table__.columns]
    assert "ticker" in cols
    assert "status" in cols
    assert "thread_id" in cols
    assert "retry_count" in cols


def test_outbox_has_sent_column():
    """
    OutboxEvent must have sent column.
    CDC relay queries WHERE sent=false to find pending events.
    """
    from infra.postgres.models import OutboxEvent
    cols = [c.name for c in OutboxEvent.__table__.columns]
    assert "sent" in cols
    assert "topic" in cols
    assert "payload" in cols


def test_filing_signal_has_confidence_column():
    """
    confidence column drives the validator routing decision.
    must exist on the model.
    """
    from infra.postgres.models import FilingSignal
    cols = [c.name for c in FilingSignal.__table__.columns]
    assert "confidence" in cols
    assert "key_risks" in cols
    assert "red_flags" in cols


# ── Redis tests ───────────────────────────────────────────────

def test_get_redis_before_init_raises_error():
    """
    calling get_redis() before init_redis() must raise.
    prevents silent failures where app runs without Redis.
    """
    import infra.redis.client as rc
    rc._redis_client = None
    with pytest.raises(RuntimeError, match="not initialized"):
        rc.get_redis()


async def test_cache_signals_stores_correctly():
    """
    cache_signals must call setex with correct key and value.
    we mock Redis so no real connection needed.
    """
    import infra.redis.client as rc
    import json

    mock_redis = AsyncMock()
    rc._redis_client = mock_redis

    signals = {"revenue_growth_yoy": 0.22, "gross_margin": 0.68}
    await rc.cache_signals("AAPL", signals, ttl=3600)

    # verify setex was called with correct arguments
    mock_redis.setex.assert_called_once_with(
        "signals:AAPL",
        3600,
        json.dumps(signals)
    )
    rc._redis_client = None  # cleanup


async def test_get_cached_signals_returns_dict_on_hit():
    """cache hit must return parsed dict, not raw string."""
    import infra.redis.client as rc

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(
        return_value='{"revenue_growth_yoy": 0.22}'
    )
    rc._redis_client = mock_redis

    result = await rc.get_cached_signals("AAPL")
    assert result["revenue_growth_yoy"] == 0.22
    rc._redis_client = None


async def test_get_cached_signals_returns_none_on_miss():
    """cache miss must return None, not raise exception."""
    import infra.redis.client as rc

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    rc._redis_client = mock_redis

    result = await rc.get_cached_signals("UNKNOWN_TICKER")
    assert result is None
    rc._redis_client = None


# ── Qdrant tests ──────────────────────────────────────────────

def test_get_qdrant_before_init_raises_error():
    """same pattern as Redis - must raise before init."""
    import infra.qdrant.client as qc
    qc._client = None
    with pytest.raises(RuntimeError, match="not initialized"):
        qc.get_qdrant()


def test_qdrant_vector_size_is_correct():
    """
    vector size must match sentence-transformer model output.
    all-MiniLM-L6-v2 outputs 384 dimensions.
    wrong size = all embeddings fail to store.
    """
    from infra.qdrant.client import VECTOR_SIZE, COLLECTION_NAME
    assert VECTOR_SIZE == 384
    assert COLLECTION_NAME == "filing_chunks"


# ── GCS tests ─────────────────────────────────────────────────

def test_get_bucket_before_init_raises_error():
    """must raise before init_gcs() is called."""
    import infra.gcs.client as gc
    gc._bucket = None
    with pytest.raises(RuntimeError, match="not initialized"):
        gc.get_bucket()


async def test_upload_filing_returns_correct_blob_path():
    """
    upload must return the correct GCS path.
    voice agent and chunk agent use this path to find the file.
    """
    import infra.gcs.client as gc

    # mock the GCS bucket and blob
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_string = MagicMock()
    gc._bucket = mock_bucket

    path = await gc.upload_filing("AAPL", "10-K", "<html>filing</html>")

    # path must follow our naming convention
    assert path == "filings/AAPL/10-K/raw.html"
    mock_bucket.blob.assert_called_with("filings/AAPL/10-K/raw.html")
    gc._bucket = None  # cleanup

    