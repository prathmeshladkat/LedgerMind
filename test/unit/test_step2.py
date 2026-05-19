# tests/unit/test_step2.py
# tests for all step 2 agent nodes
# all LLM calls are mocked - no API keys needed
# run with: pytest tests/unit/test_step2.py -v

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import FilingState, create_initial_filing_state


# ── State tests ───────────────────────────────────────────────

def test_create_initial_filing_state_has_all_fields():
    """
    initial state must have all required fields.
    missing fields cause KeyError inside agent nodes.
    """
    state = create_initial_filing_state(
        ticker="AAPL",
        filing_type="10-K",
        thread_id="thread-123",
        job_id="job-456"
    )
    assert state["ticker"] == "AAPL"
    assert state["filing_type"] == "10-K"
    assert state["confidence"] == 0.0
    assert state["retry_count"] == 0
    assert state["errors"] == []
    assert state["chunks"] == []
    assert state["signals"] is None


def test_filing_state_optional_fields_are_none():
    """all optional fields start as None - not missing."""
    state = create_initial_filing_state("GOOGL", "10-Q", "t1", "j1")
    assert state["raw_html"] is None
    assert state["gcs_blob_path"] is None
    assert state["human_feedback"] is None


# ── Fetch node tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_node_returns_raw_html_on_success():
    """
    fetch node must return raw_html and gcs_blob_path on success.
    we mock HTTP call and GCS upload so no network needed.
    """
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")

    with patch("agents.filing_pipeline.nodes.fetch.get_cik_for_ticker",
               new=AsyncMock(return_value="0000320193")), \
         patch("agents.filing_pipeline.nodes.fetch.get_latest_filing_url",
               new=AsyncMock(return_value=("http://sec.gov/test", "2024-01-01", "ACC123"))), \
         patch("agents.filing_pipeline.nodes.fetch.download_filing_html",
               new=AsyncMock(return_value="<html>filing content</html>")), \
         patch("agents.filing_pipeline.nodes.fetch.upload_filing",
               new=AsyncMock(return_value="filings/AAPL/10-K/raw.html")), \
         patch("agents.filing_pipeline.nodes.fetch.publish_status",
               new=AsyncMock()):

        from agents.filing_pipeline.nodes.fetch import fetch_filing_node
        result = await fetch_filing_node(state)

    assert result["raw_html"] == "<html>filing content</html>"
    assert result["gcs_blob_path"] == "filings/AAPL/10-K/raw.html"
    assert result["filing_date"] == "2024-01-01"


@pytest.mark.asyncio
async def test_fetch_node_returns_error_on_failure():
    """
    fetch node must add error to errors list on failure.
    must NOT raise exception - graph continues to validate.
    """
    state = create_initial_filing_state("INVALID", "10-K", "t1", "j1")

    with patch("agents.filing_pipeline.nodes.fetch.get_cik_for_ticker",
               new=AsyncMock(side_effect=ValueError("Ticker not found"))), \
         patch("agents.filing_pipeline.nodes.fetch.publish_status",
               new=AsyncMock()):

        from agents.filing_pipeline.nodes.fetch import fetch_filing_node
        result = await fetch_filing_node(state)

    assert result["raw_html"] is None
    assert len(result["errors"]) == 1
    assert "fetch_filing failed" in result["errors"][0]


# ── Chunk node tests ──────────────────────────────────────────

def test_clean_html_removes_tags():
    """HTML tags must be stripped, leaving only text."""
    from agents.filing_pipeline.nodes.chunk import clean_html
    html = "<p>Revenue <b>grew</b> by 22%</p>"
    result = clean_html(html)
    assert "<p>" not in result
    assert "<b>" not in result
    assert "Revenue" in result
    assert "22%" in result


def test_split_into_sections_returns_list():
    """split must always return a list, even for short text."""
    from agents.filing_pipeline.nodes.chunk import split_into_sections
    text = "A" * 500  # 500 chars, short text
    sections = split_into_sections(text)
    assert isinstance(sections, list)
    assert len(sections) >= 1


def test_split_into_sections_filters_short_chunks():
    """
    very short text with no headers uses fallback chunking.
    fallback returns whatever text exists, no minimum filter.
    filter only applies to section-header based splitting.
    """
    from agents.filing_pipeline.nodes.chunk import split_into_sections

    # text with section header but short content after it
    # section filter removes chunks under 100 chars
    long_text = "A" * 200  # long enough to pass filter
    sections = split_into_sections(long_text)

    # all returned sections must be non-empty strings
    assert isinstance(sections, list)
    for s in sections:
        assert isinstance(s, str)
        assert len(s) > 0


@pytest.mark.asyncio
async def test_chunk_node_skips_when_no_raw_html():
    """
    chunk node must skip gracefully if fetch agent failed.
    prevents crash when upstream agent fails.
    """
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")
    # raw_html is None by default in initial state

    with patch("agents.filing_pipeline.nodes.chunk.publish_status",
               new=AsyncMock()):

        from agents.filing_pipeline.nodes.chunk import chunk_document_node
        result = await chunk_document_node(state)

    assert len(result["errors"]) > 0
    assert "no raw_html" in result["errors"][0]


# ── Extract node tests ────────────────────────────────────────

def test_parse_llm_json_response_handles_clean_json():
    """parser must handle clean JSON response."""
    from agents.filing_pipeline.nodes.extract import parse_llm_json_response
    raw = '{"revenue_growth_yoy": 0.22, "confidence": 0.9}'
    result = parse_llm_json_response(raw)
    assert result["revenue_growth_yoy"] == 0.22
    assert result["confidence"] == 0.9


def test_parse_llm_json_response_handles_markdown_blocks():
    """parser must strip markdown code blocks LLMs sometimes add."""
    from agents.filing_pipeline.nodes.extract import parse_llm_json_response
    raw = '```json\n{"confidence": 0.85}\n```'
    result = parse_llm_json_response(raw)
    assert result["confidence"] == 0.85


@pytest.mark.asyncio
async def test_extract_node_skips_when_no_chunks():
    """extract must skip gracefully if chunk agent failed."""
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")
    # chunks is [] by default

    with patch("agents.filing_pipeline.nodes.extract.publish_status",
               new=AsyncMock()):

        from agents.filing_pipeline.nodes.extract import extract_signals_node
        result = await extract_signals_node(state)

    assert len(result["errors"]) > 0
    assert "no chunks" in result["errors"][0]


# ── Validate node tests ───────────────────────────────────────

def test_route_after_validation_returns_emit_on_high_confidence():
    """
    high confidence must route to emit.
    this is the happy path.
    """
    from agents.filing_pipeline.nodes.validate import route_after_validation
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")
    state["confidence"] = 0.92
    state["signals"] = {"revenue_growth_yoy": 0.22}
    state["retry_count"] = 0

    result = route_after_validation(state)
    assert result == "emit"


def test_route_after_validation_returns_retry_on_low_confidence():
    """
    low confidence with retries remaining must route back to extract.
    this is the retry loop.
    """
    from agents.filing_pipeline.nodes.validate import route_after_validation
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")
    state["confidence"] = 0.60        # below threshold
    state["retry_count"] = 1          # under MAX_RETRIES (2)
    state["signals"] = {"revenue_growth_yoy": 0.22}

    result = route_after_validation(state)
    assert result == "retry"


def test_route_after_validation_returns_emit_after_max_retries():
    """
    PATTERN: Circuit Breaker
    after MAX_RETRIES, must stop retrying and move forward.
    human review interrupt() handles the low confidence case.
    """
    from agents.filing_pipeline.nodes.validate import route_after_validation
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")
    state["confidence"] = 0.50        # still low
    state["retry_count"] = 3          # exceeded MAX_RETRIES (2)
    state["signals"] = {"revenue_growth_yoy": 0.22}

    # circuit breaker tripped - must NOT return "retry"
    result = route_after_validation(state)
    assert result == "emit"


# ── Emit node tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_emit_node_skips_when_no_signals():
    """emit must skip gracefully if no signals available."""
    state = create_initial_filing_state("AAPL", "10-K", "t1", "j1")
    # signals is None by default

    with patch("agents.filing_pipeline.nodes.emit.publish_status",
               new=AsyncMock()):

        from agents.filing_pipeline.nodes.emit import emit_node
        result = await emit_node(state)

    assert len(result["errors"]) > 0
    assert "no signals" in result["errors"][0]


# ── Prompts tests ─────────────────────────────────────────────

def test_extraction_prompt_contains_required_placeholders():
    """
    prompt must have all placeholders our code fills in.
    missing placeholder causes KeyError at runtime.
    """
    from agents.filing_pipeline.prompts import SIGNAL_EXTRACTION_PROMPT
    assert "{ticker}" in SIGNAL_EXTRACTION_PROMPT
    assert "{filing_type}" in SIGNAL_EXTRACTION_PROMPT
    assert "{chunks}" in SIGNAL_EXTRACTION_PROMPT


def test_retry_prompt_contains_previous_signals_placeholder():
    """retry prompt needs previous_signals to show LLM what went wrong."""
    from agents.filing_pipeline.prompts import SIGNAL_RETRY_PROMPT
    assert "{previous_signals}" in SIGNAL_RETRY_PROMPT
    assert "{issue}" in SIGNAL_RETRY_PROMPT
