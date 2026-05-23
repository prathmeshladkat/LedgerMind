# agents/voice_pipeline/graph.py
# runs voice pipeline nodes sequentially
# direct execution instead of LangGraph StateGraph
# reason: LangGraph drops bytes fields during state merging

import logging
from agents.state import VoiceState
from agents.voice_pipeline.nodes.intent import intent_parser_node
from agents.voice_pipeline.nodes.query import query_agent_node
from agents.voice_pipeline.nodes.speak import spoken_response_node

logger = logging.getLogger(__name__)


async def run_voice_pipeline_direct(state: VoiceState) -> dict:
    """
    runs all 3 voice nodes sequentially.
    manually merges state between nodes to avoid LangGraph bytes issue.
    """
    # step 1: intent parser
    intent_result = await intent_parser_node(state)
    state = {**state, **intent_result}
    logger.info(
        f"Intent: {state.get('intent')} "
        f"tickers: {state.get('tickers')}"
    )

    # step 2: query agent
    query_result = await query_agent_node(state)
    state = {**state, **query_result}
    logger.info(f"Query done")

    # step 3: spoken response
    speak_result = await spoken_response_node(state)
    state = {**state, **speak_result}
    logger.info(
        f"Speak done. "
        f"spoken={len(state.get('spoken_response') or '')} chars "
        f"audio={len(state.get('audio_bytes') or b'')} bytes"
    )

    return state


class VoiceGraphWrapper:
    """
    drop-in replacement for compiled LangGraph graph.
    exposes same ainvoke() interface so gateway code unchanged.
    """
    async def ainvoke(self, state: VoiceState) -> dict:
        return await run_voice_pipeline_direct(state)


# module level instance - imported by gateway and tests
voice_graph = VoiceGraphWrapper()