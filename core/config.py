import os
from pathlib import Path
from typing import (
    Final,
    Optional,
)

from dotenv import load_dotenv
import sounddevice as sd

from utils.audio_device_utils import AudioDeviceUtils


class AppConfig:
    def __init__(self) -> None:
        load_dotenv()
        self.OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY", "")
        self.BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
        self.AUDIO_OUTPUT_DIR: Final[Path] = self.BASE_DIR / "recorded_audio"
        self.OPENAI_TTS_VOICE: Final[str] = os.getenv("OPENAI_TTS_VOICE", "alloy")
        self.OPENAI_TTS_MODEL: Final[str] = os.getenv("OPENAI_TTS_MODEL", "tts-1")
        self.OPENAI_TTS_OUTPUT_FORMAT: Final[str] = os.getenv("OPENAI_TTS_OUTPUT_FORMAT", "mp3")
        self.TRANSCRIPTION_MODEL: Final[str] = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
        self.GPT_EVALUATION_MODEL: Final[str] = os.getenv("GPT_EVALUATION_MODEL", "gpt-4o-mini")

        self.MICROPHONE_NAME: Final[Optional[str]] = self.__resolve_microphone_name()

        self.VIRTUAL_OUTPUT_NAME: Final[Optional[str]] = os.getenv(
            "VIRTUAL_OUTPUT_NAME",
            "CABLE Input (VB-Audio Virtual Cable)",
        )
        self.VIRTUAL_INPUT_NAME: Final[Optional[str]] = os.getenv(
            "VIRTUAL_INPUT_NAME",
            "CABLE Output (VB-Audio Virtual Cable)",
        )
        self.MICROPHONE_SAMPLE_RATE: Final[int] = AudioDeviceUtils.get_input_device_samplerate(self.MICROPHONE_NAME)

        self.BOT_OUTPUT_DEVICE: Final[Optional[str]] = os.getenv("BOT_OUTPUT_DEVICE")

        _debug_env = os.getenv("DEBUG", "0").lower()
        self.DEBUG: Final[bool] = _debug_env in {"1", "true", "yes", "on"}

    @staticmethod
    def __resolve_microphone_name() -> Optional[str]:
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
        except Exception: # pylint: disable=broad-exception-caught
            return None


config = AppConfig()
