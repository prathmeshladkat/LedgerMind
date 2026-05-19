# agents/filing_pipeline/nodes/extract.py
# agent 3 in the filing pipeline
# job: use RAG to find relevant chunks, then LLM extracts signals
#
# PATTERN: RAG (Retrieval Augmented Generation)
# instead of sending all chunks to LLM, we:
# 1. convert the query to a vector
# 2. search Qdrant for most similar chunks
# 3. send only top 5 chunks to LLM
# this is faster, cheaper, and more accurate than sending everything
#
# MODEL CHOICE: Groq (llama-3.3-70b) - free tier, fast
# we use Groq here because signal extraction is called often
# OpenAI is saved for voice responses only

import json
import logging
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from agents.state import FilingState
from agents.filing_pipeline.prompts import (
    SIGNAL_EXTRACTION_PROMPT,
    SIGNAL_RETRY_PROMPT
)
from infra.qdrant.client import get_qdrant, COLLECTION_NAME
from infra.redis.client import publish_status
from infra.settings import get_settings

logger = logging.getLogger(__name__)


async def retrieve_relevant_chunks(
    ticker: str,
    filing_type: str,
    query: str,
    top_k: int = 5
) -> list[str]:
    """
    searches Qdrant for chunks most relevant to the query.
    this is the R in RAG - Retrieval.

    how it works:
    1. convert query text to a vector (same model as chunk.py used)
    2. Qdrant finds the stored vectors most similar to query vector
    3. returns the original text of those chunks

    filter by ticker and filing_type so we only search
    this company's filing, not all filings in the collection.
    """
    qdrant = get_qdrant()

    # embed the search query using same model as chunk agent
    # IMPORTANT: must use same model or vectors are incompatible
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(query).tolist()

    # search Qdrant
    # filter ensures we only get chunks from this specific filing
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    results = await qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="ticker",
                    match=MatchValue(value=ticker)
                ),
                FieldCondition(
                    key="filing_type",
                    match=MatchValue(value=filing_type)
                )
            ]
        )
    )

    # extract just the text from results
    return [r.payload["text"] for r in results]


def parse_llm_json_response(response_text: str) -> dict:
    """
    parses JSON from LLM response.
    LLMs sometimes add markdown code blocks or extra text.
    this function handles those cases gracefully.
    """
    text = response_text.strip()

    # remove markdown code blocks if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


async def extract_signals_node(state: FilingState) -> dict:
    """
    main node function for extract agent.
    uses RAG to find relevant chunks, then LLM extracts signals.

    PATTERN: RAG (Retrieval Augmented Generation)
    we search for chunks relevant to financial signals
    not all chunks - just the ones that likely contain numbers and risks
    """
    thread_id = state["thread_id"]
    ticker = state["ticker"]
    filing_type = state["filing_type"]

    # skip if chunking failed
    if not state.get("chunks"):
        error = "extract_signals skipped: no chunks in state"
        logger.warning(error)
        return {"errors": state["errors"] + [error]}

    await publish_status(thread_id, "extracting", {
        "message": "AI is reading the filing..."
    })

    try:
        # PATTERN: RAG - retrieve relevant chunks before calling LLM
        # we search for different aspects of the filing
        # then combine results for a comprehensive extraction
        search_query = (
            f"revenue growth gross margin financial results "
            f"risk factors guidance outlook red flags {ticker}"
        )

        relevant_chunks = await retrieve_relevant_chunks(
            ticker=ticker,
            filing_type=filing_type,
            query=search_query,
            top_k=5
        )

        # join chunks into one string for the prompt
        chunks_text = "\n\n---\n\n".join(relevant_chunks)

        # decide which prompt to use
        # if this is a retry, use retry prompt with previous attempt
        is_retry = state["retry_count"] > 0

        if is_retry:
            prompt = SIGNAL_RETRY_PROMPT.format(
                ticker=ticker,
                filing_type=filing_type,
                chunks=chunks_text,
                previous_signals=json.dumps(state.get("signals", {})),
                issue="Low confidence in previous extraction"
            )
        else:
            prompt = SIGNAL_EXTRACTION_PROMPT.format(
                ticker=ticker,
                filing_type=filing_type,
                chunks=chunks_text
            )

        # call Groq LLM
        # temperature=0 means deterministic output - good for extraction
        settings = get_settings()
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

        logger.info(f"Calling Groq LLM (retry={is_retry})...")
        response = await llm.ainvoke(prompt)
        response_text = response.content

        # parse the JSON response
        signals = parse_llm_json_response(response_text)
        confidence = float(signals.get("confidence", 0.0))

        logger.info(f"Extracted signals with confidence={confidence}")

        await publish_status(thread_id, "extraction_complete", {
            "message": "Signals extracted",
            "confidence": confidence
        })

        return {
            "signals": signals,
            "confidence": confidence,
        }

    except json.JSONDecodeError as e:
        # LLM returned invalid JSON - this is a known failure mode
        error_msg = f"LLM returned invalid JSON: {str(e)}"
        logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "confidence": 0.0,
        }

    except Exception as e:
        error_msg = f"extract_signals failed: {str(e)}"
        logger.error(error_msg)
        await publish_status(thread_id, "extraction_failed", {
            "message": error_msg
        })
        return {
            "errors": state["errors"] + [error_msg],
            "confidence": 0.0,
        }