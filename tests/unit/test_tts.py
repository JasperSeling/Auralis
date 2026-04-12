import base64

import numpy as np

from auralis.common.definitions.openai import AudioSpeechGenerationRequest
from auralis.common.definitions.output import TTSOutput


def test_audio_speech_generation_request_speed_is_float():
    request = AudioSpeechGenerationRequest(
        input="hello",
        model="xttsv2",
        voice=[base64.b64encode(b"fake wav").decode("utf-8")],
        speed=1.25,
    )

    assert request.speed == 1.25
    assert isinstance(request.speed, float)


def test_audio_speech_generation_request_maps_voice_to_speaker_files():
    audio_bytes = b"fake wav"
    request = AudioSpeechGenerationRequest(
        input="hello",
        model="xttsv2",
        voice=[base64.b64encode(audio_bytes).decode("utf-8")],
    )

    tts_request = request.to_tts_request()

    assert tts_request.speaker_files == [audio_bytes]


def test_tts_output_change_speed_preserves_sample_rate_and_changes_length():
    output = TTSOutput(
        array=np.sin(np.linspace(0, 2 * np.pi, 24000, dtype=np.float32)),
        sample_rate=24000,
    )
    original_length = len(output.array)

    result = output.change_speed(1.5)

    assert result is output
    assert output.sample_rate == 24000
    assert len(output.array) != original_length
