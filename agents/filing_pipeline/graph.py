import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from agents.state import FilingState
from agents.filing_pipeline.nodes.fetch import fetch_filing_node
from agents.filing_pipeline.nodes.chunk import chunk_document_node
from agents.filing_pipeline.nodes.extract import extract_signals_node
from agents.filing_pipeline.nodes.validate import (
    validate_output_node,
    route_after_validation
)
from agents.filing_pipeline.nodes.emit import emit_node
from infra.settings import get_settings

logger = logging.getLogger(__name__)


def build_filing_graph():
    """
    builds and compiles the filing pipeline graph.
    call this once at worker startup.
    returns compiled graph ready to invoke.

    graph structure:
    START → fetch → chunk → extract → validate → emit → END
                                ↑          |
                                └── retry ─┘
                                (when confidence low)
    """
    settings = get_settings()

    checkpointer = RedisSaver.from_conn_string(settings.upstash_redis_url)

    # create the graph with our state type
    graph = StateGraph(FilingState)


    graph.add_node("fetch_filing", fetch_filing_node)
    graph.add_node("chunk_document", chunk_document_node)
    graph.add_node("extract_signals", extract_signals_node)
    graph.add_node("validate_output", validate_output_node)
    graph.add_node("emit", emit_node)

    # linear edges - these always go in one direction
    graph.set_entry_point("fetch_filing")
    graph.add_edge("fetch_filing", "chunk_document")
    graph.add_edge("chunk_document", "extract_signals")
    graph.add_edge("extract_signals", "validate_output")


    graph.add_conditional_edges(
        "validate_output",          # after this node runs...
        route_after_validation,     # call this function to decide next node
        {
            "retry": "extract_signals",  # "retry" → go back to extract
            "emit": "emit",              # "emit"  → go to emit
        }
    )

    graph.add_edge("emit", END)


    compiled = graph.compile(
        checkpointer=checkpointer,
    )

    logger.info("Filing pipeline graph compiled successfully")
    return compiled



filing_graph = build_filing_graph()