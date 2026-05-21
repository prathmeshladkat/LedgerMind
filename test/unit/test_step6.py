# tests/unit/test_step6.py
# tests for voice pipeline agents
# all API calls mocked - no Deepgram/ElevenLabs/OpenAI needed
# run with: pytest test/unit/test_step6.py -v

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.state import VoiceState


def make_voice_state(**kwargs) -> VoiceState:
    """helper to create VoiceState with defaults."""
    defaults = VoiceState(
        transcript="",
        session_id="session-1",
        intent=None,
        tickers=[],
        fetched_data=None,
        spoken_response=None,
        errors=[],
    )
    defaults.update(kwargs)
    return defaults


# ── Intent parser tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_parser_classifies_lookup():
    """
    lookup intent must be classified correctly.
    simple fact question about one company.
    """
    state = make_voice_state(
        transcript="What was Stripe's revenue last quarter?"
    )

    mock_response = MagicMock()
    mock_response.content = '{"intent": "lookup", "tickers": ["STRP"], "confidence": 0.95}'

    with patch(
        "agents.voice_pipeline.nodes.intent.ChatGroq"
    ) as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        from agents.voice_pipeline.nodes.intent import intent_parser_node
        result = await intent_parser_node(state)

    assert result["intent"] == "lookup"
    assert "STRP" in result["tickers"]


@pytest.mark.asyncio
async def test_intent_parser_handles_empty_transcript():
    """
    empty transcript must return default intent without crashing.
    prevents errors when browser sends empty audio.
    """
    state = make_voice_state(transcript="")

    from agents.voice_pipeline.nodes.intent import intent_parser_node
    result = await intent_parser_node(state)

    assert result["intent"] == "lookup"
    assert result["tickers"] == []
    assert len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_intent_parser_defaults_unknown_intent():
    """
    unknown intent from LLM must default to lookup.
    prevents KeyError downstream.
    """
    state = make_voice_state(transcript="something unclear")

    mock_response = MagicMock()
    # LLM returns invalid intent
    mock_response.content = '{"intent": "INVALID", "tickers": [], "confidence": 0.5}'

    with patch("agents.voice_pipeline.nodes.intent.ChatGroq") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_class.return_value = mock_llm

        from agents.voice_pipeline.nodes.intent import intent_parser_node
        result = await intent_parser_node(state)

    assert result["intent"] == "lookup"


def test_extract_tickers_from_text_finds_uppercase():
    """fallback ticker extraction must find uppercase words."""
    from agents.voice_pipeline.nodes.intent import extract_tickers_from_text
    result = extract_tickers_from_text("Tell me about AAPL and GOOGL")
    assert "AAPL" in result
    assert "GOOGL" in result


def test_extract_tickers_excludes_common_words():
    """common words like I, A, AT must not be returned as tickers."""
    from agents.voice_pipeline.nodes.intent import extract_tickers_from_text
    result = extract_tickers_from_text("I want to know about AT&T")
    assert "I" not in result
    assert "A" not in result


# ── Query agent tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_agent_returns_message_when_no_tickers():
    """
    if no tickers in state, must return helpful message not crash.
    user might say something that doesn't mention a company.
    """
    state = make_voice_state(tickers=[])

    from agents.voice_pipeline.nodes.query import query_agent_node
    result = await query_agent_node(state)

    assert result["fetched_data"]["found"] is False
    assert "message" in result["fetched_data"]


@pytest.mark.asyncio
async def test_query_agent_uses_cache_on_hit():
    """
    PATTERN: Cache-Aside
    if signals in Redis cache, must use them without hitting Postgres.
    Postgres must not be called on cache hit.
    """
    state = make_voice_state(tickers=["AAPL"], intent="lookup")

    cached_signals = {
        "ticker": "AAPL",
        "revenue_growth_yoy": 0.22,
        "confidence": 0.95
    }

    with patch(
        "agents.voice_pipeline.nodes.query.get_cached_signals",
        new=AsyncMock(return_value=cached_signals)
    ), patch(
        "agents.voice_pipeline.nodes.query.fetch_from_postgres",
        new=AsyncMock(return_value=None)
    ) as mock_db:

        from agents.voice_pipeline.nodes.query import query_agent_node
        result = await query_agent_node(state)

    # Postgres must NOT have been called
    mock_db.assert_not_called()
    assert result["fetched_data"]["found"] is True
    assert result["fetched_data"]["results"]["AAPL"]["revenue_growth_yoy"] == 0.22


@pytest.mark.asyncio
async def test_query_agent_falls_back_to_postgres_on_cache_miss():
    """
    PATTERN: Cache-Aside
    cache miss must trigger Postgres query.
    result must also be stored back in cache.
    """
    state = make_voice_state(tickers=["AAPL"], intent="lookup")

    db_signals = {
        "ticker": "AAPL",
        "revenue_growth_yoy": 0.18,
        "confidence": 0.90
    }

    with patch(
        "agents.voice_pipeline.nodes.query.get_cached_signals",
        new=AsyncMock(return_value=None)    # cache miss
    ), patch(
        "agents.voice_pipeline.nodes.query.fetch_from_postgres",
        new=AsyncMock(return_value=db_signals)
    ), patch(
        "agents.voice_pipeline.nodes.query.cache_signals",
        new=AsyncMock()
    ) as mock_cache:

        from agents.voice_pipeline.nodes.query import query_agent_node
        result = await query_agent_node(state)

    # must have populated cache after DB read
    mock_cache.assert_called_once()
    assert result["fetched_data"]["results"]["AAPL"]["revenue_growth_yoy"] == 0.18


@pytest.mark.asyncio
async def test_query_agent_handles_unknown_ticker():
    """
    ticker with no data must return found=False message.
    not an error - just means we haven't analyzed it yet.
    """
    state = make_voice_state(tickers=["UNKNOWN"], intent="lookup")

    with patch(
        "agents.voice_pipeline.nodes.query.get_cached_signals",
        new=AsyncMock(return_value=None)
    ), patch(
        "agents.voice_pipeline.nodes.query.fetch_from_postgres",
        new=AsyncMock(return_value=None)
    ):
        from agents.voice_pipeline.nodes.query import query_agent_node
        result = await query_agent_node(state)

    assert result["fetched_data"]["results"]["UNKNOWN"]["found"] is False


# ── Speak node tests ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_speak_node_returns_spoken_response():
    """speak node must return both text and audio bytes."""
    state = make_voice_state(
        intent="lookup",
        tickers=["AAPL"],
        fetched_data={
            "found": True,
            "results": {"AAPL": {"revenue_growth_yoy": 0.22}},
            "intent": "lookup"
        }
    )

    with patch(
        "agents.voice_pipeline.nodes.speak.format_for_voice",
        new=AsyncMock(return_value="Stripe grew 22 percent year over year.")
    ), patch(
        "agents.voice_pipeline.nodes.speak.text_to_speech",
        return_value=b"fake_audio_bytes"
    ):
        from agents.voice_pipeline.nodes.speak import spoken_response_node
        result = await spoken_response_node(state)

    assert result["spoken_response"] == "Stripe grew 22 percent year over year."
    assert result["audio_bytes"] == b"fake_audio_bytes"


@pytest.mark.asyncio
async def test_speak_node_handles_no_data_gracefully():
    """
    if no data found, must return helpful message not crash.
    user still hears a response.
    """
    state = make_voice_state(
        intent="lookup",
        tickers=["UNKNOWN"],
        fetched_data={"found": False, "message": "No data found."}
    )

    with patch(
        "agents.voice_pipeline.nodes.speak.text_to_speech",
        return_value=b"fallback_audio"
    ):
        from agents.voice_pipeline.nodes.speak import spoken_response_node
        result = await spoken_response_node(state)

    assert result["spoken_response"] == "No data found."
    assert result["audio_bytes"] == b"fallback_audio"


# ── Voice graph tests ─────────────────────────────────────────

def test_voice_graph_compiles_successfully():
    """voice graph must compile without errors."""
    from agents.voice_pipeline.graph import voice_graph
    assert voice_graph is not None


def test_voice_graph_has_correct_nodes():
    """voice graph must contain all 3 required nodes."""
    from agents.voice_pipeline.graph import voice_graph
    node_names = list(voice_graph.nodes.keys())
    assert "intent_parser" in node_names
    assert "query_agent" in node_names
    assert "spoken_response" in node_names


# ── Voice worker tests ────────────────────────────────────────

def test_voice_worker_stream_name_is_correct():
    """Redis Stream name must match what gateway publishes to."""
    from workers.voice_worker import VOICE_STREAM, VOICE_GROUP
    assert VOICE_STREAM == "voice:events"
    assert VOICE_GROUP == "voice-workers"


@pytest.mark.asyncio
async def test_run_voice_graph_builds_correct_initial_state():
    """
    run_voice_graph must pass correct VoiceState to graph.
    wrong state structure causes KeyError inside nodes.
    """
    mock_result = {
        "spoken_response": "test response",
        "audio_bytes": b"audio",
        "intent": "lookup",
        "tickers": ["AAPL"],
    }

    with patch(
        "agents.voice_pipeline.graph.voice_graph.ainvoke",
        new=AsyncMock(return_value=mock_result)
    ) as mock_invoke:
    
        from workers.voice_worker import run_voice_graph
        result = await run_voice_graph("What is Stripe's revenue?", "session-1")
    
    # also fix the assertion below it
    call_args = mock_invoke.call_args[0][0]


    assert call_args["transcript"] == "What is Stripe's revenue?"
    assert call_args["session_id"] == "session-1"
    assert call_args["errors"] == []