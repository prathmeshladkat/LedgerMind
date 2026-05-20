
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket for voice agent.
    receives audio chunks from browser via WebRTC.
    returns spoken response audio.
    full implementation in Step 6.
    """
    await websocket.accept()
    logger.info("Voice WebSocket connected (stub)")

    try:
        while True:
            # receive audio chunk from browser
            data = await websocket.receive_bytes()
            # TODO Step 6: pipe to Deepgram STT → voice graph → ElevenLabs TTS
            logger.debug(f"Received audio chunk: {len(data)} bytes")

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")