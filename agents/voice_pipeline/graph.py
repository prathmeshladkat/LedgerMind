import logging
from langgraph.graph import StateGraph, END
from agents.state import VoiceState
from agents.voice_pipeline.nodes.intent import intent_parser_node
from agents.voice_pipeline.nodes.query import query_agent_node
from agents.voice_pipeline.nodes.speak import spoken_response_node

logger = logging.getLogger(__name__)

def build_voice_graph():
    """
    builds the voice pipeline graph.
    linear flow: intent → query → speak → END
    no conditional edges needed - always runs all 3 nodes.
    no checkpointer - voice is stateless per utterance.
    """
    graph = StateGraph(VoiceState)

    graph.add_node("intent_parser", intent_parser_node)
    graph.add_node("query_agent", query_agent_node)
    graph.add_node("spoken_response", spoken_response_node)

    # linear edges - no branching in voice pipeline
    graph.set_entry_point("intent_parser")
    graph.add_edge("intent_parser", "query_agent")
    graph.add_edge("query_agent", "spoken_response")
    graph.add_edge("spoken_response", END)

    compiled = graph.compile()
    logger.info("Voice pipeline graph compiled")
    return compiled

voice_graph = build_voice_graph()


