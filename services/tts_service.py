"""Text-to-speech service using the OpenAI TTS API.

This module provides a service to convert long texts into audio files
by chunking the text, generating audio for each chunk, and concatenating
the results.
"""
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Final, List

from openai import OpenAI
from pydub import AudioSegment

from core.config import config


class TTSService:
    """Manages text-to-speech conversion, handling API limits gracefully."""

    def __init__(self) -> None:
        """Initializes the TTS service and the OpenAI client.

        Raises:
            ValueError: If the OPENAI_API_KEY is not set in the environment.
        """
        self.__logger = logging.getLogger(self.__class__.__name__)
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.__client = OpenAI(api_key=config.openai_api_key)
        self.__model: Final[str] = config.openai_tts_model
        self.__voice: Final[str] = config.openai_tts_voice
        self.__format: Final[str] = config.openai_tts_output_format
        self.__api_char_limit: Final[int] = config.openai_tts_char_limit

    def __chunk_text(self, text: str) -> List[str]:
        """Splits a long text into chunks that respect the API character limit.

        The method splits text primarily by sentences, then falls back to new
        lines or spaces to ensure no chunk exceeds the limit.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks, each smaller than the API limit.
        """
        chunks = []
        current_chunk = ""
        sentences = text.replace("!", "!.").replace("?", "?. ").split(". ")

        for sentence in sentences:
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 > self.__api_char_limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.__api_char_limit:
                while len(chunk) > self.__api_char_limit:
                    split_pos = chunk.rfind(" ", 0, self.__api_char_limit)
                    if split_pos == -1:
                        split_pos = self.__api_char_limit
                    final_chunks.append(chunk[:split_pos])
                    chunk = chunk[split_pos:]
            final_chunks.append(chunk)

        return [c for c in final_chunks if c]

    def __generate_chunk_audio(self, text_chunk: str, file_path: Path) -> None:
        """Generates an audio file for a single text chunk via OpenAI API.

        Args:
            text_chunk (str): The text chunk to convert to speech.
            file_path (Path): The path to save the generated audio file.

        Raises:
            Exception: Propagates exceptions from the OpenAI API client.
        """
        with self.__client.audio.speech.with_streaming_response.create(
                model=self.__model,
                voice=self.__voice,
                input=text_chunk,
                response_format=self.__format,
        ) as response:
            response.stream_to_file(file_path)

    def __process_chunks(self, text_chunks: List[str]) -> AudioSegment:
        """Generates and concatenates audio for a list of text chunks.

        This method iterates through text chunks, generating audio for each
        in a temporary directory, and then combines them into a single
        AudioSegment.

        Args:
            text_chunks (List[str]): A list of text chunks to process.

        Returns:
            AudioSegment: A pydub AudioSegment with the combined audio.

        Raises:
            RuntimeError: If audio generation results in no processable
                segments.
        """
        audio_segments = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, chunk in enumerate(text_chunks):
                chunk_file_path = temp_path / f"chunk_{i}.{self.__format}"
                log_msg = f"Generating audio for chunk {i+1}/{len(text_chunks)}"
                self.__logger.info(log_msg)

                self.__generate_chunk_audio(chunk, chunk_file_path)
                segment = AudioSegment.from_file(chunk_file_path,
                                                 format=self.__format)
                audio_segments.append(segment)

        if not audio_segments:
            raise RuntimeError("Audio generation resulted in no segments.")

        self.__logger.info("Concatenating audio chunks...")
        return sum(audio_segments)

    async def generate_audio(self, text: str, output_path: Path) -> bool:
        """Generates an audio file from text, using cache if available.

        This is the main public method. It checks for a cached version of the
        audio first. If not found, it chunks the text, generates audio
        for each part, combines them, saves the final file, and caches it.

        Args:
            text (str): The full text to be converted to speech.
            output_path (Path): The path to save the final audio file.

        Returns:
            bool: True if a cached audio file was used, False otherwise.

        Raises:
            Exception: If any part of the audio generation or file handling
                fails.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_hash: str = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_file_name = f"jailbreak_prompt_{prompt_hash}.{self.__format}"
        cache_file: Path = output_path.parent / cache_file_name

        if cache_file.exists():
            shutil.copy(cache_file, output_path)
            return True

        text_chunks = self.__chunk_text(text)
        try:
            combined_audio = self.__process_chunks(text_chunks)
            combined_audio.export(output_path, format=self.__format)
        except Exception:
            self.__logger.exception("Failed to generate audio with OpenAI TTS.")
            raise

        shutil.copy(output_path, cache_file)
        return False
