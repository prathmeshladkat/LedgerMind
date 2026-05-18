# ── Filing pipeline topics ─────────────────────────────────────
# these follow the order of the pipeline
# each worker reads from one topic and publishes to the next

FILING_INGEST     = "filing.ingest"      # step 1: gateway publishes here
FILING_RAW        = "filing.raw"         # step 2: after fetch agent downloads
FILING_CHUNKED    = "filing.chunked"     # step 3: after chunk agent splits
SIGNALS_RAW       = "signals.raw"        # step 4: after extract agent runs
SIGNALS_VALIDATED = "signals.validated"  # step 5: after validator approves

# ── Support topics ─────────────────────────────────────────────
FILING_DLQ = "filing.dlq"   # dead letter queue - failed messages go here
ALERTS     = "alerts"        # watchlist monitor publishes new filing alerts

# ── Consumer group names ───────────────────────────────────────

FETCH_GROUP     = "fetch-workers"
CHUNK_GROUP     = "chunk-workers"
SIGNAL_GROUP    = "signal-workers"
VALIDATOR_GROUP = "validator-workers"
EMIT_GROUP      = "emit-workers"

# list of all topics - used to verify topics exist at startup
ALL_TOPICS = [
    FILING_INGEST,
    FILING_RAW,
    FILING_CHUNKED,
    SIGNALS_RAW,
    SIGNALS_VALIDATED,
    FILING_DLQ,
    ALERTS,
]