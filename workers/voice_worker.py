# VAD (Voice Activity Detection) via Silero
# we don't send audio to Deepgram immediately
# Silero VAD detects when user stops speaking
# only then do we send the complete utterance to Deepgram
# prevents sending incomplete sentences to the LLM

import asyncio
import logging
import json
from deepgram import DeepgramClient
from agents.voice_pipeline.graph import voice_graph
from agents.state import VoiceState
from infra.redis.client import get_redis
from infra.settings import get_settings

logger = logging.getLogger(__name__)

# Redis Stream name for voice events
VOICE_STREAM = "voice:events"

# consumer group for voice workers
VOICE_GROUP = "voice-workers"

async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    sends audio bytes to Deepgram for speech-to-text.
    returns transcript text.
    uses pre-recorded mode (not live streaming) for simplicity.
    live streaming is handled at the WebSocket level.
    """
    settings = get_settings()
    deepgram = DeepgramClient(settings.deepgram_api_key)

    # pre-recorded transcription
    response = await asyncio.to_thread(
        deepgram.listen.prerecorded.v("1").transcribe_file,
        {"buffer": audio_bytes, "mimetype": "audio/webm"},
        {"model": "nova-2", "language": "en-US"}
    )

    # extract transcript from response
    transcript = (
        response.results.channels[0]
        .alternatives[0]
        .transcript
    )
    return transcript.strip()

async def run_voice_graph(transcript: str, session_id: str) -> dict:
    """
    runs the full voice pipeline for one utterance.
    returns dict with spoken_response and audio_bytes.
    """
    initial_state = VoiceState(
        transcript=transcript,
        session_id=session_id,
        intent=None,
        tickers=[],
        fetched_data=None,
        spoken_response=None,
        errors=[],
    )

    result = await voice_graph.ainvoke(initial_state)
    return result


async def process_voice_message(message: dict) -> dict:
    """
    processes one voice message from Redis Stream.
    message contains: audio_bytes (base64), session_id.
    returns result with audio to send back to browser.
    """
    import base64

    session_id = message.get("session_id", "unknown")
    audio_b64 = message.get("audio_bytes", "")

    if not audio_b64:
        return {"error": "no audio received"}

    # decode base64 audio
    audio_bytes = base64.b64decode(audio_b64)

    logger.info(f"Processing voice: session={session_id} "
                f"audio={len(audio_bytes)} bytes")

    # step 1: speech to text
    transcript = await transcribe_audio(audio_bytes)

    if not transcript:
        return {"error": "could not transcribe audio"}

    logger.info(f"Transcript: '{transcript}'")

    # step 2: run voice graph
    result = await run_voice_graph(transcript, session_id)

    return {
        "session_id": session_id,
        "transcript": transcript,
        "spoken_response": result.get("spoken_response"),
        "audio_bytes": result.get("audio_bytes", b""),
    }


async def consume_voice_stream():
    """
    consumes voice events from Redis Stream.
    Redis Streams give us ordered, persistent message delivery
    with consumer group semantics - same as Kafka but lighter weight.

    PATTERN: Redis Streams consumer group
    multiple voice workers can run in parallel
    Redis assigns different messages to different workers
    prevents duplicate processing
    """
    redis = get_redis()

    # create consumer group if not exists
    try:
        await redis.xgroup_create(
            VOICE_STREAM,
            VOICE_GROUP,
            id="0",
            mkstream=True   # create stream if not exists
        )
        logger.info(f"Created consumer group: {VOICE_GROUP}")
    except Exception:
        # group already exists - that's fine
        pass

    logger.info(f"Voice worker consuming from {VOICE_STREAM}")

    while True:
        try:
            # read next message from stream
            # block=1000 means wait up to 1 second for new messages
            messages = await redis.xreadgroup(
                groupname=VOICE_GROUP,
                consumername="voice-worker-1",
                streams={VOICE_STREAM: ">"},  # > means unread messages
                count=1,
                block=1000,
            )

            if not messages:
                continue

            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    try:
                        payload = json.loads(fields.get("data", "{}"))
                        await process_voice_message(payload)

                        # acknowledge message after processing
                        await redis.xack(
                            VOICE_STREAM,
                            VOICE_GROUP,
                            message_id
                        )

                    except Exception as e:
                        logger.error(
                            f"Voice message processing failed: {e}"
                        )

        except Exception as e:
            logger.error(f"Voice stream consumer error: {e}")
            await asyncio.sleep(1)


async def main():
    logging.basicConfig(level="INFO")
    from infra.redis.client import init_redis
    from infra.postgres.database import init_db
    init_redis()
    init_db()
    await consume_voice_stream()


if __name__ == "__main__":
    asyncio.run(main())
