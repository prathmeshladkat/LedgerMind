import asyncio
import logging
import json
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agents.voice_pipeline.graph import voice_graph
from agents.state import VoiceState
from infra.redis.client import get_redis
from workers.voice_worker import transcribe_audio, run_voice_graph
from infra.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# how many seconds of silence triggers end of utterance
SILENCE_THRESHOLD_SECONDS = 0.8

# minimum audio size to attempt transcription (bytes)
# prevents transcribing empty or too-short clips
MIN_AUDIO_BYTES = 1000


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for voice agent.

    message flow:
    browser mic → WebRTC → WebSocket binary message → VAD check
    → accumulate speech chunks → silence detected
    → flush to Deepgram STT → voice graph
    → ElevenLabs TTS → binary audio → WebSocket → browser speaker

    client sends:
    - binary messages = audio chunks (webm/opus from MediaRecorder)
    - text message {"type": "end"} = user explicitly stopped

    server sends:
    - text message {"type": "transcript", "text": "..."} = what we heard
    - text message {"type": "response", "text": "..."} = spoken response text
    - binary message = audio bytes to play in browser
    """
    await websocket.accept()
    session_id = id(websocket)   # unique per connection

    logger.info(f"Voice WebSocket connected: session={session_id}")

    # buffer accumulates audio chunks during speech
    audio_buffer = bytearray()
    is_speaking = False

    try:
        while True:
            # receive next message from browser
            message = await websocket.receive()

            # binary message = audio chunk from MediaRecorder
            if "bytes" in message:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)

                # simple VAD: we accumulate until client signals end
                # full Silero VAD would run here in production
                # for demo, we process when buffer reaches threshold
                is_speaking = True

            # text message = control signal from browser
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    # client signals end of utterance
                    if msg_type == "end_of_speech":
                        if len(audio_buffer) > MIN_AUDIO_BYTES:
                            logger.info(
                                f"Processing utterance: "
                                f"{len(audio_buffer)} bytes"
                            )

                            # send acknowledgement to browser
                            await websocket.send_text(json.dumps({
                                "type": "processing",
                                "message": "Processing your question..."
                            }))

                            # run voice pipeline
                            result = await process_utterance(
                                bytes(audio_buffer),
                                session_id,
                                websocket
                            )

                            # clear buffer for next utterance
                            audio_buffer.clear()
                            is_speaking = False

                    elif msg_type == "ping":
                        await websocket.send_text(
                            json.dumps({"type": "pong"})
                        )

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
    finally:
        audio_buffer.clear()


async def process_utterance(
    audio_bytes: bytes,
    session_id: int,
    websocket: WebSocket
) -> dict:
    """
    processes one complete utterance end to end.
    sends progress updates to browser via WebSocket.
    returns final result dict.
    """
    try:
        # step 1: speech to text
        transcript = await transcribe_audio(audio_bytes)

        if not transcript:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Could not understand audio. Please try again."
            }))
            return {}

        # send transcript to browser so user sees what we heard
        await websocket.send_text(json.dumps({
            "type": "transcript",
            "text": transcript
        }))
        logger.info(f"Transcript: '{transcript}'")

        # step 2: run voice graph (intent → query → speak)
        result = await run_voice_graph(transcript, str(session_id))

        # send spoken response text to browser
        if result.get("spoken_response"):
            await websocket.send_text(json.dumps({
                "type": "response",
                "text": result["spoken_response"]
            }))

        # step 3: send audio bytes to browser
        audio_out = result.get("audio_bytes", b"")
        if audio_out:
            await websocket.send_bytes(audio_out)
            logger.info(f"Sent {len(audio_out)} audio bytes to browser")

        return result

    except Exception as e:
        logger.error(f"Utterance processing failed: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Something went wrong. Please try again."
        }))
        return {}