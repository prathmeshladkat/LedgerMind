from elevenlabs.client import ElevenLabs
from infra.settings import get_settings


settings = get_settings()
print(settings.elevenlabs_api_key[:15])

client = ElevenLabs(api_key=settings.elevenlabs_api_key)

voices = client.voices.get_all()

for voice in voices.voices:
    print(voice.name, voice.voice_id)