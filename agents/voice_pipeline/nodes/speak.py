# MODEL CHOICE: gpt-4o-mini via OpenAI
# voice formatting needs higher quality than Groq
# spoken text must sound natural - "grew 22 percent" not "0.22"
# gpt-4o-mini is cheap and much better at conversational formatting
#
# this node does two things:
# 1. calls OpenAI to format data into spoken sentences
# 2. calls ElevenLabs to convert text to audio

import logging
from langchain_openai import ChatOpenAI
from elevenlabs.client import ElevenLabs
from agents.state import VoiceState
from agents.filing_pipeline.prompts import VOICE_RESPONSE_PROMPT
from infra.settings import get_settings

logger = logging.getLogger(__name__)

# ElevenLabs voice ID - "Rachel" is clear and professional
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

async def format_for_voice(
        data: dict,
        intent: str,
        ticker: str
) -> str:
    """
    calls OpenAI gpt-4o-mini to format signal data as spoken sentences.
    rules enforced by prompt:
    - no bullet points, no tables, no markdown
    - numbers as words: "22 percent" not "0.22"
    - under 100 words unless full brief requested
    - ends with one follow-up offer
    """
    settings = get_settings()
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model = "gpt-4o-mini",
        temperature=0.3,
    )

    prompt = VOICE_RESPONSE_PROMPT.format(
        data=str(data),
        intent=intent,
        ticker=ticker,
    )

    response = await llm.ainvoke(prompt)
    return response.content.strip()

def text_to_speech(text: str) -> bytes:
    """
    converts text to audio bytes using ElevenLabs.
    returns raw audio bytes that WebSocket sends to browser.
    ElevenLabs client is synchronous so we call it directly.
    """
    settings = get_settings()
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    audio = client.generate(
        text=text,
        voide=DEFAULT_VOICE_ID,
        model="eleven_monolingual_v1"
    )

    audio_bytes = b"".join(audio)
    logger.info(f"Generated {len(audio_bytes)} bytes of audio")
    return audio_bytes

async def spoken_response_node(state: VoiceState) -> dict:
    """
    main node function for speak agent.
    formats data as spoken text then converts to audio.

    two step process:
    1. OpenAI formats raw signals into natural spoken sentences
    2. ElevenLabs converts those sentences to audio bytes
    """
    fetched_data = state.get("fetched_data")
    intent = state.get("intent", "lookup")
    tickers = state.get("tickers", [])
    ticker = tickers[0] if tickers else "unknown"

    if not fetched_data or not fetched_data.get("found"):
        message = fetched_data.get(
            "message",
            "I couldn't find any data for that company."

        ) if fetched_data else "I couldn't find any data."

        try:
            audio_bytes = text_to_speech(message)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            audio_bytes = b""

        return {
            "spoken_response": message,
            "audio_bytes": audio_bytes,
        }

    try:
        # step 1: format data as spoken text using openai
        logger.info(f"Formatting voice response for {ticker}...")
        spoken_text = await format_for_voice(
            data=fetched_data,
            intent=intent,
            ticker=ticker,
        )
        logger.info(f"Spoken text: '{spoken_text[:100]}...'")

        # step 2: convert to audio using ElevenLabs
        logger.info("Converting to audio via ElevenLabs...")
        audio_bytes = text_to_speech(spoken_text)

        return {
            "spoken_response": spoken_text,
            "audio_bytes": audio_bytes,
        }
    
    except Exception as e:
        error_msg = f"spoken_response failed: {str(e)}"
        logger.error(error_msg)

        # return error message as audio so user hears something
        fallback = "Sorry, I had trouble generating that response."
        try:
            audio_bytes = text_to_speech(fallback)
        except Exception:
            audio_bytes = b""

        return {
            "spoken_response": fallback,
            "audio_bytes": audio_bytes,
            "errors": state.get("errors", []) + [error_msg],
        }


