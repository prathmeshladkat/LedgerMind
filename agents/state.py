from typing import TypedDict, Optional
from datetime import datetime

class FilingState(TypedDict):

    """
    shared memory for the entire filing pipeline.
    every agent reads from this and writes back to this.
    only write the fields your agent is responsible for.
    never delete fields - other agents downstream may need them.
    """

    ticker: str            # e.g. "AAPL", "GOOGL"
    filing_type: str       # "10-K", "10-Q", or "8-K"
    thread_id: str         # unique id linking this run to Redis state
    job_id: str            # links to Job row in Postgres

    # --fetch agent write these ------
    raw_html: Optional[str]        # full filing HTML downloaded from SEC
    gcs_blob_path: Optional[str]   # where we stored it in GCS
    filing_Date: Optional[str]     # date of the filing
    accession_number: Optional[str]  # SEC's unique filing identifier

    # ── chunk agent writes these ───────────
    chunks: list[str]
    chunk_ids: list[str]   # matching Qdrant point IDs for each chunk

    signals: Optional[dict]

    # ── validate agent writes these ────────────────────────────
    confidence: float      # 0.0 to 1.0 - how sure LLM was
    retry_count: int       # how many times extract has been retried
    human_feedback: Optional[str]  # correction from analyst if HITL triggered

    # ── error tracking ─────────────────────────────────────────
    # list of error messages from any agent
    # we collect all errors instead of stopping on first one
    errors: list[str]


class VoiceState(TypedDict):
    """
    shared memory for the voice pipeline.
    smaller than FilingState - voice agents are lightweight.
    """

    # ---input ------
    transcript: str     # what user said (from Deepgram STT)
    session_id: str     # unique id for this voice conversation

    # ── intent agent writes this ───────────────────────────────
    # what type of query the user made
    # one of: "lookup", "brief", "compare", "drilldown"
    intent: Optional[str]
    # which ticker(s) the user mentioned
    tickers: list[str]


    # ---- query agent write this ---
    fetched_data: Optional[dict]

    # ----speak agent writes this -----
    spokent_response: Optional[str]

     # ── error tracking ───────
    errors: list[str]

def create_initial_filing_state(
    ticker: str,
    filing_type: str,
    thread_id: str,
    job_id: str
) -> FilingState:
    """
    creates a fresh FilingState with all optional fields set to None.
    call this in the worker when a new Kafka message arrives.
    passing incomplete state to LangGraph causes KeyError - always use this.
    """
    return FilingState(
        ticker=ticker,
        filing_type=filing_type,
        thread_id=thread_id,
        job_id=job_id,
        raw_html=None,
        gcs_blob_path=None,
        filing_date=None,
        accession_number=None,
        chunks=[],
        chunk_ids=[],
        signals=None,
        confidence=0.0,
        retry_count=0,
        human_feedback=None,
        errors=[],
    )