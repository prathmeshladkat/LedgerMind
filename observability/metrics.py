# observability/metrics.py
# Prometheus metrics for the whole system
#
# what is Prometheus?
# Prometheus scrapes metrics from our app every 15 seconds
# stores them as time series data
# Grafana reads Prometheus to show dashboards
#
# PATTERN: RED metrics (Rate, Errors, Duration)
# industry standard for measuring service health:
# Rate     = how many requests per second
# Errors   = how many are failing
# Duration = how long they take
# we apply this to every agent and the overall pipeline

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
)
from infra.settings import get_settings
import logging

logger = logging.getLogger(__name__)

# ── Filing pipeline metrics ────────────────────────────────────

# counts total filings processed
# labels let you filter by ticker, filing_type, status in Grafana
FILINGS_PROCESSED = Counter(
    "capitalsense_filings_processed_total",
    "Total number of filings processed",
    ["ticker", "filing_type", "status"]  # success or failed
)

# measures how long each agent node takes
# Histogram gives you p50, p95, p99 latency in Grafana
# buckets = time ranges in seconds to measure against
AGENT_DURATION = Histogram(
    "capitalsense_agent_duration_seconds",
    "Time spent in each agent node",
    ["agent_name"],   # fetch, chunk, extract, validate, emit
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# tracks LLM API call latency specifically
# useful to see if Groq or OpenAI is the bottleneck
LLM_LATENCY = Histogram(
    "capitalsense_llm_latency_seconds",
    "Time spent calling LLM APIs",
    ["model", "agent"],   # e.g. llama-3.3-70b, signal_extractor
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# tracks confidence score distribution
# if confidence is usually low, prompts need improvement
EXTRACTION_CONFIDENCE = Histogram(
    "capitalsense_extraction_confidence",
    "Distribution of LLM extraction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# counts human review triggers
# high count = LLM confidence is too low, prompts need work
HUMAN_REVIEWS_TRIGGERED = Counter(
    "capitalsense_human_reviews_total",
    "Number of times human review was triggered",
    ["reason"]   # low_confidence, no_signals
)

# counts DLQ events
# high count = workers are failing, needs investigation
DLQ_EVENTS = Counter(
    "capitalsense_dlq_events_total",
    "Number of messages sent to dead letter queue",
    ["source_topic"]
)

# ── Voice pipeline metrics ─────────────────────────────────────

# measures end-to-end voice response latency
# target: under 1.5 seconds
VOICE_LATENCY = Histogram(
    "capitalsense_voice_latency_seconds",
    "End-to-end voice response latency",
    buckets=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
)

# breaks down voice latency by component
# tells you if STT, LLM, or TTS is the bottleneck
VOICE_COMPONENT_LATENCY = Histogram(
    "capitalsense_voice_component_latency_seconds",
    "Latency of each voice pipeline component",
    ["component"],   # stt, intent, query, tts
    buckets=[0.1, 0.3, 0.5, 1.0, 2.0]
)

# ── System metrics ─────────────────────────────────────────────

# tracks currently active WebSocket connections
# sudden drop = connection issue
ACTIVE_WEBSOCKETS = Gauge(
    "capitalsense_active_websockets",
    "Number of active WebSocket connections"
)

# tracks watchlist size
WATCHLIST_SIZE = Gauge(
    "capitalsense_watchlist_tickers",
    "Number of tickers in watchlist"
)


def init_metrics(port: int = None):
    """
    starts the Prometheus metrics HTTP server.
    Prometheus scrapes metrics from this endpoint every 15 seconds.
    gateway runs on port 8001, workers on port 8002.
    configured in deploy/prometheus.yml.
    """
    settings = get_settings()
    metrics_port = port or settings.prometheus_port

    try:
        start_http_server(metrics_port)
        logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except OSError as e:
        # port already in use - happens when running multiple workers
        logger.warning(f"Metrics server already running: {e}")


def record_filing_processed(
    ticker: str,
    filing_type: str,
    status: str
):
    """
    call this when a filing completes (success or failed).
    status should be "success" or "failed".
    """
    FILINGS_PROCESSED.labels(
        ticker=ticker,
        filing_type=filing_type,
        status=status
    ).inc()


def record_agent_duration(agent_name: str, duration_seconds: float):
    """
    call this after each agent node finishes.
    duration_seconds = time the node took to run.
    """
    AGENT_DURATION.labels(agent_name=agent_name).observe(duration_seconds)


def record_llm_call(model: str, agent: str, duration_seconds: float):
    """call this after every LLM API call."""
    LLM_LATENCY.labels(model=model, agent=agent).observe(duration_seconds)


def record_confidence(confidence: float):
    """call this after signal extraction to track confidence distribution."""
    EXTRACTION_CONFIDENCE.observe(confidence)


def record_human_review(reason: str):
    """call this when interrupt() triggers human review."""
    HUMAN_REVIEWS_TRIGGERED.labels(reason=reason).inc()


def record_dlq_event(source_topic: str):
    """call this when a message goes to DLQ."""
    DLQ_EVENTS.labels(source_topic=source_topic).inc()


def record_voice_latency(total_seconds: float):
    """call this after complete voice response sent to browser."""
    VOICE_LATENCY.observe(total_seconds)


def record_voice_component(component: str, duration_seconds: float):
    """call this after each voice component (stt, intent, query, tts)."""
    VOICE_COMPONENT_LATENCY.labels(component=component).observe(duration_seconds)


def websocket_connected():
    """call when WebSocket client connects."""
    ACTIVE_WEBSOCKETS.inc()


def websocket_disconnected():
    """call when WebSocket client disconnects."""
    ACTIVE_WEBSOCKETS.dec()