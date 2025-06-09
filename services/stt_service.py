import logging
from pathlib import Path
import time
from typing import (
    Final,
    List,
)

import numpy as np
from openai import AsyncOpenAI
from scipy.io.wavfile import write as wav_write
import sounddevice as sd

from core.config import config


class STTService:
    def __init__(self) -> None:
        self.__logger = logging.getLogger(self.__class__.__name__)

        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.__client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.__model: Final[str] = config.TRANSCRIPTION_MODEL
        self.__input_device_index: Final[int] = self.__resolve_input_device_index()

    async def record_audio(self, output_path: Path, max_duration: int = 10, sample_rate: int = 44100) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            recorded_data = self.__capture_with_silence_detection(max_duration=max_duration, sample_rate=sample_rate)
            self.__save_recording_to_file(recorded_data, output_path, sample_rate)
        except Exception:
            self.__logger.exception("Audio recording with silence detection failed")
            raise

    async def transcribe_audio(self, audio_path: Path) -> str:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        try:
            with audio_path.open("rb") as f:
                response = await self.__client.audio.transcriptions.create(
                    file=f,
                    model=self.__model,
                )
            return response.text.strip()
        except Exception:
            self.__logger.exception("Transcription failed")
            raise

    def __capture_with_silence_detection(self, max_duration: int, sample_rate: int) -> np.ndarray:
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
                if silent_chunks >= silence_frame_count or (time.time() - start_time > max_duration):
                    break
        return np.concatenate(recorded, axis=0)

    @staticmethod
    def __resolve_input_device_index() -> int:
        device_name = config.VIRTUAL_INPUT_NAME
        for idx, device in enumerate(sd.query_devices()):
            if device_name and device_name.lower() in device["name"].lower() and device["max_input_channels"] > 0:
                return idx
        raise RuntimeError(f"Device '{device_name}' not found.")

    @staticmethod
    def __save_recording_to_file(data: np.ndarray, path: Path, sample_rate: int) -> None:
        wav_write(path, sample_rate, data)
