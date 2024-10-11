import numpy as np
from passive_sound_localization.vad import VoiceActivityDetector
from passive_sound_localization.config.vad_config import VADConfig
import pytest


@pytest.fixture
def vad_config():
    return VADConfig(aggressiveness=2, frame_duration_ms=30, enabled=True)


@pytest.fixture
def vad(vad_config):
    return VoiceActivityDetector(vad_config)


def test_is_speaking(vad):
    # Create dummy audio data (16000 samples of silence)
    sample_rate = 16000
    silence_audio = np.zeros(sample_rate, dtype=np.int16)

    # Check if the VAD detects speech (it should not for silence)
    result = vad.is_speaking(silence_audio)
    assert not result, "VAD incorrectly detected speech in silence"
