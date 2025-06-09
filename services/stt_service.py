"""Speech-to-text service using OpenAI Whisper API."""
import logging
import time
from pathlib import Path
from typing import Final, List

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from scipy.io.wavfile import write as wav_write

from core.config import config


class STTService:
    """Service for speech-to-text transcription using OpenAI Whisper API."""

    def __init__(self) -> None:
        """Initialize the STT service."""
        self.__logger = logging.getLogger(self.__class__.__name__)

        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.__client = AsyncOpenAI(api_key=config.openai_api_key)
        self.__model: Final[str] = config.transcription_model
        self.__input_device_index: Final[
            int] = self.__resolve_input_device_index()

    async def record_audio(
        self,
        output_path: Path,
        max_duration: int = 10,
        sample_rate: int = 44100,
    ) -> None:
        """Record audio with silence detection.

        Args:
            output_path: Path where to save the recorded audio.
            max_duration: Maximum recording duration in seconds.
            sample_rate: Audio sample rate.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            recorded_data = self.__capture_with_silence_detection(
                max_duration=max_duration,
                sample_rate=sample_rate,
            )
            self.__save_recording_to_file(
                recorded_data,
                output_path,
                sample_rate,
            )
        except Exception:
            self.__logger.exception(
                "Audio recording with silence detection failed",)
            raise

    async def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file to transcribe.

        Returns:
            Transcribed text.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            Exception: If transcription fails.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        try:
            with audio_path.open("rb") as file_handle:
                response = await self.__client.audio.transcriptions.create(
                    file=file_handle,
                    model=self.__model,
                )
            return response.text.strip()
        except Exception:
            self.__logger.exception("Transcription failed")
            raise

    def __capture_with_silence_detection(
        self,
        max_duration: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Capture audio with automatic silence detection.

        Args:
            max_duration: Maximum recording duration in seconds.
            sample_rate: Audio sample rate.

        Returns:
            Recorded audio data as numpy array.
        """
        silence_threshold = 500
        silence_duration_limit = 1.5
        frame_duration = 0.1
        frame_size = int(sample_rate * frame_duration)
        silence_frame_count = int(silence_duration_limit / frame_duration)

        recorded: List[np.ndarray] = []
        silent_chunks = 0
        start_time = time.time()
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_size,
            device=self.__input_device_index,
        )
        with stream:
            while True:
                data, _ = stream.read(frame_size)
                recorded.append(data.copy())
                amplitude = np.abs(data).mean()
                is_silent = amplitude < silence_threshold
                if is_silent:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                if silent_chunks >= silence_frame_count or (time.time() -
                                                            start_time
                                                            > max_duration):
                    break
        return np.concatenate(recorded, axis=0)

    @staticmethod
    def __resolve_input_device_index() -> int:
        """Resolve input device index from configuration.

        Returns:
            Input device index.

        Raises:
            RuntimeError: If device is not found.
        """
        device_name = config.virtual_input_name
        for idx, device in enumerate(sd.query_devices()):
            if device_name and device_name.lower() in device["name"].lower(
            ) and device["max_input_channels"] > 0:
                return idx
        raise RuntimeError(f"Device '{device_name}' not found.")

    @staticmethod
    def __save_recording_to_file(
        data: np.ndarray,
        path: Path,
        sample_rate: int,
    ) -> None:
        """Save recorded audio data to file.

        Args:
            data: Audio data to save.
            path: Path where to save the file.
            sample_rate: Audio sample rate.
        """
        wav_write(path, sample_rate, data)
