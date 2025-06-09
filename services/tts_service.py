"""Text-to-speech service using OpenAI TTS API."""
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Final

from openai import OpenAI

from core.config import config


class TTSService:
    """Service for text-to-speech conversion using OpenAI TTS API."""

    def __init__(self) -> None:
        """Initialize the TTS service."""
        self.__logger = logging.getLogger(self.__class__.__name__)
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.__client = OpenAI(api_key=config.openai_api_key)
        self.__model: Final[str] = config.openai_tts_model
        self.__voice: Final[str] = config.openai_tts_voice
        self.__format: Final[str] = config.openai_tts_output_format

    async def generate_audio(self, text: str, output_path: Path) -> bool:
        """Generate audio from text using OpenAI TTS API.

        Args:
            text: Text to convert to speech.
            output_path: Path where to save the generated audio.

        Returns:
            True if cached audio was used, False if new audio was generated.

        Raises:
            Exception: If audio generation fails.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # noinspection PyTypeChecker
        prompt_hash: str = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_file: Path = (output_path.parent /
                            f"jailbreak_prompt_{prompt_hash}.{self.__format}")

        if cache_file.exists():
            shutil.copy(cache_file, output_path)
            return True

        try:
            with self.__client.audio.speech.with_streaming_response.create(
                    model=self.__model,
                    voice=self.__voice,
                    input=text,
                    response_format=self.__format,
            ) as response:
                response.stream_to_file(output_path)
            shutil.copy(output_path, cache_file)
            return False
        except Exception:
            self.__logger.exception("Failed to generate audio with OpenAI TTS.")
            raise
