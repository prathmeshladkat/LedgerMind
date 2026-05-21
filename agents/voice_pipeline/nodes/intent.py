# this node only classifies intent - it does not fetch any data
# keeps the node small, testable, and replaceable

import json
import logging
import re
from langchain_groq import ChatGroq
from agents.state import VoiceState
from infra.settings import get_settings

logger = logging.getLogger(__name__)

# intent types the user can express
# lookup   = "what was stripe's revenue last quarter"
# brief    = "brief me on my watchlist" / "brief me on AAPL"
# compare  = "compare stripe and brex on risk"
# drilldown= "tell me more about the EU regulation risk"
VALID_INTENTS = ["lookup", "brief", "compare", "drilldown"]

INTENT_PROMPT = """
You are a financial research assistant.
Classify the user's voice query into one intent type and extract tickers.

User said: "{transcript}"

Intent types:
- lookup: asking for a specific fact or number about one company
- brief: asking for a summary or overview of one or more companies
- compare: asking to compare two or more companies
- drilldown: asking for more detail on something mentioned earlier

Return ONLY valid JSON with this exact structure:
{{
    "intent": "<lookup|brief|compare|drilldown>",
    "tickers": ["<TICKER1>", "<TICKER2>"],
    "confidence": <float 0.0 to 1.0>
}}

Rules:
- tickers must be uppercase stock symbols e.g. ["AAPL", "GOOGL"]
- if no ticker mentioned, return empty list []
- if unsure of intent, default to "lookup"
- return ONLY the JSON, nothing else
"""

async def intent_parser_node(state: VoiceState) -> dict:
    """
    classifies user's voice query into intent + extract tickers.
    reads transcript from state, writes intent and tickers back.
    """
    transcript = state.get("transcript", "")

    if not transcript:
        logger.warning("Intent parser received empty transcript")
        return {
            "intent": "lookup",
            "tickers": [],
            "errors": state.get("errors", []) + ["empty transcript"],
        }
    
    logger.info(f"Parsing intent for: '{transcript}")

    try:
        settings = get_settings()
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0,
        )

        prompt = INTENT_PROMPT.format(transcript=transcript)
        response = await llm.ainvoke(prompt)
        response_text = response.content.strip()

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        parsed = json.loads(response_text.strip())

        intent = parsed.get("intent", "lookup")
        tickers = parsed.get("tickers", [])

        if intent not in VALID_INTENTS:
            logger.warning(f"Unknown intent '{intent}', defaulting to lookup")
            intent = "lookup"

        tickers = [t.upper() for t in tickers]

        logger.info(f"Intent={intent} tickers={tickers}")

        return {
            "intent": intent,
            "tickers": tickers,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Intent parser returned invalid JSON: {e}")
        # fallback - try to extract ticker from transcript directly
        tickers = extract_tickers_from_text(transcript)
        return {
            "intent": "lookup",
            "tickers": tickers,
            "errors": state.get("errors", []) + [f"JSON parse error: {e}"],
        }

    except Exception as e:
        error_msg = f"intent_parser failed: {str(e)}"
        logger.error(error_msg)
        return {
            "intent": "lookup",
            "tickers": [],
            "errors": state.get("errors", []) + [error_msg],
        }
    
def extract_tickers_from_text(text: str) -> list[str]:
    """
    fallback ticker extraction using regex.
    looks for 1-5 upperccase letters that could be tickers.
    used whrn LLM fails to parse intent.
    """
    # match 1-5 uppercase letters as potential tickers
    exclude = {"I", "A", "AT", "IT", "IS", "BE", "ON", "IN", "OF"}
    matches = re.findall(r'\b[A-Z]{1,5}\b', text.upper())
    return [m for m in matches if m not in exclude]

