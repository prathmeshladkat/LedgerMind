# gateway/routers/voice.py
# WebSocket endpoint for voice agent
# handles full voice pipeline directly in the gateway

import logging
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agents.voice_pipeline.graph import voice_graph
from agents.state import VoiceState
from infra.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

MIN_AUDIO_BYTES = 1000


async def transcribe_audio_deepgram(audio_bytes: bytes) -> str:
    """transcribes audio using Deepgram REST API."""
    import httpx
    settings = get_settings()

    if not settings.deepgram_api_key:
        logger.warning("No Deepgram key - using mock transcript")
        return "What are NVDA's biggest risks?"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&language=en-US",
            headers={
                "Authorization": f"Token {settings.deepgram_api_key}",
                "Content-Type": "audio/webm",
            },
            content=audio_bytes,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

    transcript = (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )
    return transcript.strip()


async def run_voice_pipeline(transcript: str, session_id: str) -> dict:
    """runs the voice graph and returns result."""
    state = VoiceState(
        transcript=transcript,
        session_id=session_id,
        intent=None,
        tickers=[],
        fetched_data=None,
        spoken_response=None,
        audio_bytes=None,
        errors=[],
    )
    result = await voice_graph.ainvoke(state)
    logger.info(f"Raw result keys: {list(result.keys())}")
    logger.info(f"spoken_response: {result.get('spoken_response', 'MISSING')[:50] if result.get('spoken_response') else 'NONE'}")
    logger.info(f"audio_bytes: {len(result.get('audio_bytes') or b'')} bytes")
    return result


async def safe_send_text(websocket: WebSocket, data: dict) -> bool:
    """
    sends text message safely.
    returns False if connection is closed.
    """
    try:
        await websocket.send_text(json.dumps(data))
        return True
    except Exception as e:
        logger.error(f"Failed to send text: {e}")
        return False


async def safe_send_bytes(websocket: WebSocket, data: bytes) -> bool:
    """
    sends binary message safely.
    returns False if connection is closed.
    """
    try:
        await websocket.send_bytes(data)
        return True
    except Exception as e:
        logger.error(f"Failed to send bytes: {e}")
        return False


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for voice agent.
    browser sends audio chunks → transcribe → voice graph → send audio back
    """
    await websocket.accept()
    session_id = str(id(websocket))
    logger.info(f"Voice WebSocket connected: session={session_id}")

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            # ── binary message = audio chunk from browser ──────────
            if "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])
                continue

            # ── text message = control signal ──────────────────────
            if "text" not in message:
                continue

            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            # ── ping keepalive ─────────────────────────────────────
            if msg_type == "ping":
                await safe_send_text(websocket, {"type": "pong"})
                continue

            # ── end of speech - process the audio ──────────────────
            if msg_type != "end_of_speech":
                continue

            # check buffer has enough audio
            if len(audio_buffer) < MIN_AUDIO_BYTES:
                await safe_send_text(websocket, {
                    "type": "error",
                    "message": "Audio too short. Please speak longer."
                })
                audio_buffer.clear()
                continue

            # snapshot and clear buffer
            audio_bytes = bytes(audio_buffer)
            audio_buffer.clear()

            # tell browser we are processing
            ok = await safe_send_text(websocket, {
                "type": "processing",
                "message": "Processing your question..."
            })
            if not ok:
                break

            # ── step 1: transcribe ─────────────────────────────────
            try:
                transcript = await transcribe_audio_deepgram(audio_bytes)
                logger.info(f"Transcript: '{transcript}'")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                await safe_send_text(websocket, {
                    "type": "error",
                    "message": "Could not transcribe audio. Please try again."
                })
                continue

            if not transcript:
                await safe_send_text(websocket, {
                    "type": "error",
                    "message": "Could not understand audio. Please speak more clearly."
                })
                continue

            # send transcript to browser so user sees what we heard
            ok = await safe_send_text(websocket, {
                "type": "transcript",
                "text": transcript
            })
            if not ok:
                break

            # ── step 2: run voice pipeline ─────────────────────────
            try:
                logger.info("Running voice pipeline...")
                result = await run_voice_pipeline(transcript, session_id)
                logger.info(
                    f"Pipeline complete. "
                    f"spoken={len(result.get('spoken_response') or '')} chars "
                    f"audio={len(result.get('audio_bytes') or b'')} bytes"
                )
            except Exception as e:
                logger.error(f"Voice pipeline failed: {e}")
                await safe_send_text(websocket, {
                    "type": "error",
                    "message": "Could not generate response. Please try again."
                })
                continue

            # ── step 3: send spoken text response ──────────────────
            spoken = result.get("spoken_response") or ""
            audio_out = result.get("audio_bytes") or b""

            logger.info(f"spoken_response present: {bool(spoken)}")
            logger.info(f"audio_bytes present: {len(audio_out)} bytes")
            if spoken:
                ok = await safe_send_text(websocket, {
                    "type": "response",
                    "text": spoken
                })
                if not ok:
                    break
                logger.info("Text response sent")

            # ── step 4: send audio bytes ───────────────────────────
            audio_out = result.get("audio_bytes") or b""
            if len(audio_out) > 0:
                ok = await safe_send_bytes(websocket, audio_out)
                if ok:
                    logger.info(f"Audio sent: {len(audio_out)} bytes")
                else:
                    break

    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
    finally:
        audio_buffer.clear()
        logger.info(f"Voice WebSocket cleanup done: session={session_id}")