import pytest
from passive_sound_localization.transcriber import Transcriber
from passive_sound_localization.config.transcriber_config import TranscriberConfig
from unittest.mock import patch, MagicMock
import os

@pytest.fixture
def transcriber():
    # Use the environment variable for the API key, fallback to 'fake_api_key'
    api_key = os.getenv('OPENAI_API_KEY', 'fake_api_key')
    config = TranscriberConfig(api_key=api_key, model_name="whisper-1", language="en")
    return Transcriber(config)

@pytest.mark.unit
@patch("os.path.isfile", return_value=True)
@patch("passive_sound_localization.transcriber.OpenAI")
@patch("mimetypes.guess_type")
def test_transcriber_transcribe(mock_guess_type, mock_openai, mock_isfile, transcriber):
    # Ensure mimetypes correctly returns "audio/wav"
    mock_guess_type.return_value = ("audio/wav", None)

    # Mock the OpenAI response
    mock_response = MagicMock()
    mock_response.text = "Test transcription"
    mock_openai.return_value.audio.transcriptions.create.return_value = mock_response

    # Open the actual .wav file from your resources folder
    with open("tests/resources/test_audio.wav", "rb") as real_audio:
        transcription = transcriber.transcribe("tests/resources/test_audio.wav")

    # Validate that the transcription result is correct
    assert transcription == "9"

    
