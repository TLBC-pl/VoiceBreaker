"""Application configuration management."""
import os
from pathlib import Path
from typing import Final, Optional

import sounddevice as sd
from dotenv import load_dotenv

from utils.audio_device_utils import AudioDeviceUtils


class AppConfig:
    """Initialize application configuration from environment variables."""

    def __init__(self) -> None:
        """Initialize application configuration from environment variables."""
        load_dotenv()
        self.openai_api_key: Final[str] = os.getenv("OPENAI_API_KEY", "")
        self.base_dir: Final[Path] = Path(__file__).resolve().parent.parent
        self.audio_output_dir: Final[Path] = self.base_dir / "recorded_audio"
        self.openai_tts_voice: Final[str] = os.getenv(
            "OPENAI_TTS_VOICE",
            "alloy",
        )
        self.openai_tts_model: Final[str] = os.getenv(
            "OPENAI_TTS_MODEL",
            "tts-1",
        )
        self.openai_tts_output_format: Final[str] = os.getenv(
            "OPENAI_TTS_OUTPUT_FORMAT",
            "mp3",
        )
        self.transcription_model: Final[str] = os.getenv(
            "TRANSCRIPTION_MODEL",
            "whisper-1",
        )
        self.gpt_evaluation_model: Final[str] = os.getenv(
            "GPT_EVALUATION_MODEL",
            "gpt-4o-mini",
        )

        self.microphone_name: Final[
            Optional[str]] = self.__resolve_microphone_name()

        self.virtual_output_name: Final[Optional[str]] = os.getenv(
            "VIRTUAL_OUTPUT_NAME",
            "CABLE Input (VB-Audio Virtual Cable)",
        )
        self.virtual_input_name: Final[Optional[str]] = os.getenv(
            "VIRTUAL_INPUT_NAME",
            "CABLE Output (VB-Audio Virtual Cable)",
        )
        self.microphone_sample_rate: Final[
            int] = AudioDeviceUtils.get_input_device_samplerate(
                self.microphone_name,)

        self.bot_output_device: Final[Optional[str]] = os.getenv(
            "BOT_OUTPUT_DEVICE",)

        _debug_env = os.getenv("DEBUG", "0").lower()
        self.debug: Final[bool] = _debug_env in {"1", "true", "yes", "on"}

    @staticmethod
    def __resolve_microphone_name() -> Optional[str]:
        """Resolve microphone name from environment or default device.

        Returns:
            Microphone device name if found, None otherwise.
        """
        mic_name = os.getenv("MICROPHONE_NAME")
        if mic_name:
            return mic_name

        # noinspection PyBroadException
        try:
            default_input_index = sd.default.device[0]
            if default_input_index == -1:
                return None
            device_info = sd.query_devices(default_input_index)
            return device_info.get("name")
        except Exception:  # pylint: disable=broad-exception-caught
            return None


config = AppConfig()
