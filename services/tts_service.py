import hashlib
import logging
from pathlib import Path
import shutil
from typing import Final

from openai import OpenAI

from core.config import config


class TTSService:
    def __init__(self) -> None:
        self.__logger = logging.getLogger(self.__class__.__name__)
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.__client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.__model: Final[str] = config.OPENAI_TTS_MODEL
        self.__voice: Final[str] = config.OPENAI_TTS_VOICE
        self.__format: Final[str] = config.OPENAI_TTS_OUTPUT_FORMAT

    async def generate_audio(self, text: str, output_path: Path) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # noinspection PyTypeChecker
        prompt_hash: str = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_file: Path = output_path.parent / f"jailbreak_prompt_{prompt_hash}.{self.__format}"

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
