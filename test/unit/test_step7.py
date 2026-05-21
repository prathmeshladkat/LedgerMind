# tests/unit/test_step7.py
# tests for observability - tracing and metrics
# run with: pytest test/unit/test_step7.py -v

import pytest
from unittest.mock import patch, MagicMock


# ── Tracing tests ─────────────────────────────────────────────

def test_get_tracer_returns_noop_before_init():
    """
    get_tracer() must return a valid tracer even before init_tracing().
    prevents crashes in tests that don't initialize tracing.
    noop tracer silently discards all spans.
    """
    import observability.tracing as t
    t._tracer = None   # reset to uninitialized state

    tracer = t.get_tracer()
    # must return something - not None
    assert tracer is not None


def test_init_tracing_sets_global_tracer():
    """
    init_tracing() must set the module level _tracer.
    subsequent calls to get_tracer() return the same tracer.
    """
    import observability.tracing as t

    # mock the OTel exporter so no real connection needed
    with patch(
        "observability.tracing.OTLPSpanExporter"
    ) as mock_exporter, patch(
        "observability.tracing.TracerProvider"
    ) as mock_provider_class, patch(
        "observability.tracing.trace.set_tracer_provider"
    ), patch(
        "observability.tracing.trace.get_tracer",
        return_value=MagicMock()
    ):
        tracer = t.init_tracing(service_name="test-service")

    assert t._tracer is not None
    t._tracer = None   # cleanup


@pytest.mark.asyncio
async def test_trace_agent_node_decorator_calls_function():
    """
    decorator must call the wrapped function and return its result.
    tracing must not change the function's return value.
    """
    from observability.tracing import trace_agent_node

    @trace_agent_node("test_node")
    async def fake_node(state):
        return {"result": "done"}

    state = {"ticker": "AAPL", "filing_type": "10-K", "thread_id": "t1"}
    result = await fake_node(state)

    assert result == {"result": "done"}


@pytest.mark.asyncio
async def test_trace_agent_node_preserves_function_name():
    """
    decorator must preserve original function name.
    LangGraph uses __name__ to register nodes.
    wrong name = node not found error.
    """
    from observability.tracing import trace_agent_node

    @trace_agent_node("fetch_filing")
    async def fetch_filing_node(state):
        return {}

    assert fetch_filing_node.__name__ == "fetch_filing_node"


@pytest.mark.asyncio
async def test_trace_agent_node_propagates_exceptions():
    """
    if wrapped function raises, decorator must re-raise.
    exceptions must not be swallowed by tracing code.
    """
    from observability.tracing import trace_agent_node

    @trace_agent_node("failing_node")
    async def failing_node(state):
        raise ValueError("something went wrong")

    with pytest.raises(ValueError, match="something went wrong"):
        await failing_node({"ticker": "AAPL"})


# ── Metrics tests ─────────────────────────────────────────────

def test_all_metrics_are_importable():
    """
    all metric objects must import without errors.
    wrong metric type or duplicate name causes import error.
    """
    from observability.metrics import (
        FILINGS_PROCESSED,
        AGENT_DURATION,
        LLM_LATENCY,
        EXTRACTION_CONFIDENCE,
        HUMAN_REVIEWS_TRIGGERED,
        DLQ_EVENTS,
        VOICE_LATENCY,
        VOICE_COMPONENT_LATENCY,
        ACTIVE_WEBSOCKETS,
        WATCHLIST_SIZE,
    )
    # all must be non-None
    assert FILINGS_PROCESSED is not None
    assert AGENT_DURATION is not None
    assert LLM_LATENCY is not None
    assert VOICE_LATENCY is not None
    assert ACTIVE_WEBSOCKETS is not None


def test_record_filing_processed_does_not_raise():
    """
    recording a filing metric must never raise.
    metrics must not crash the pipeline if they fail.
    """
    from observability.metrics import record_filing_processed
    # must not raise for any valid input
    record_filing_processed("AAPL", "10-K", "success")
    record_filing_processed("GOOGL", "10-Q", "failed")


def test_record_agent_duration_does_not_raise():
    """agent duration recording must never raise."""
    from observability.metrics import record_agent_duration
    record_agent_duration("fetch_filing", 2.5)
    record_agent_duration("extract_signals", 15.3)


def test_record_llm_call_does_not_raise():
    """LLM call metric recording must never raise."""
    from observability.metrics import record_llm_call
    record_llm_call("llama-3.3-70b", "signal_extractor", 4.2)
    record_llm_call("gpt-4o-mini", "spoken_response", 1.1)


def test_record_confidence_does_not_raise():
    """confidence score recording must never raise."""
    from observability.metrics import record_confidence
    record_confidence(0.92)
    record_confidence(0.45)


def test_record_human_review_does_not_raise():
    """human review trigger recording must never raise."""
    from observability.metrics import record_human_review
    record_human_review("low_confidence")
    record_human_review("no_signals")


def test_record_dlq_event_does_not_raise():
    """DLQ event recording must never raise."""
    from observability.metrics import record_dlq_event
    record_dlq_event("filing.ingest")
    record_dlq_event("signals.raw")


def test_voice_metrics_do_not_raise():
    """voice metrics recording must never raise."""
    from observability.metrics import (
        record_voice_latency,
        record_voice_component,
        websocket_connected,
        websocket_disconnected,
    )
    record_voice_latency(1.2)
    record_voice_component("stt", 0.4)
    record_voice_component("tts", 0.6)
    websocket_connected()
    websocket_disconnected()


def test_websocket_gauge_increments_and_decrements():
    """
    ACTIVE_WEBSOCKETS gauge must go up on connect and down on disconnect.
    gauge tracks current count not cumulative total.
    """
    from observability.metrics import (
        ACTIVE_WEBSOCKETS,
        websocket_connected,
        websocket_disconnected,
    )

    before = ACTIVE_WEBSOCKETS._value.get()
    websocket_connected()
    assert ACTIVE_WEBSOCKETS._value.get() == before + 1

    websocket_disconnected()
    assert ACTIVE_WEBSOCKETS._value.get() == before


def test_metrics_have_correct_label_names():
    """
    label names must match what Grafana queries use.
    wrong label name = metric exists but Grafana can't filter it.
    """
    from observability.metrics import (
        FILINGS_PROCESSED,
        AGENT_DURATION,
        LLM_LATENCY,
    )

    # check label names on each metric
    assert "ticker" in FILINGS_PROCESSED._labelnames
    assert "filing_type" in FILINGS_PROCESSED._labelnames
    assert "status" in FILINGS_PROCESSED._labelnames

    assert "agent_name" in AGENT_DURATION._labelnames

    assert "model" in LLM_LATENCY._labelnames
    assert "agent" in LLM_LATENCY._labelnames