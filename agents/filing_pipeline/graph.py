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


def build_filing_graph(checkpointer=None):
    """
    builds and compiles the filing pipeline graph.
    checkpointer is passed in instead of created here.
    this makes testing easier - tests pass InMemorySaver,
    production passes RedisSaver.
    
    PATTERN: Dependency Injection
    instead of creating Redis connection inside this function,
    we accept it as a parameter.
    test passes a fake checkpointer, worker passes real Redis one.
    """
    graph = StateGraph(FilingState)

    graph.add_node("fetch_filing", fetch_filing_node)
    graph.add_node("chunk_document", chunk_document_node)
    graph.add_node("extract_signals", extract_signals_node)
    graph.add_node("validate_output", validate_output_node)
    graph.add_node("emit", emit_node)

    graph.set_entry_point("fetch_filing")
    graph.add_edge("fetch_filing", "chunk_document")
    graph.add_edge("chunk_document", "extract_signals")
    graph.add_edge("extract_signals", "validate_output")

    graph.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {
            "retry": "extract_signals",
            "emit": "emit",
        }
    )

    graph.add_edge("emit", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Filing pipeline graph compiled")
    return compiled


def get_filing_graph_with_redis():
    """
    creates graph with real Redis checkpointer.
    called by workers in production.
    RedisSaver must be used as context manager.
    """
    settings = get_settings()
    with RedisSaver.from_conn_string(settings.upstash_redis_url) as checkpointer:
        return build_filing_graph(checkpointer=checkpointer)


# default instance with no checkpointer for testing
# workers call get_filing_graph_with_redis() instead
filing_graph = build_filing_graph(checkpointer=None)