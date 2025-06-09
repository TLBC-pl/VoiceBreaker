"""Audio player adapter for playing audio files through specified devices."""
import logging
from pathlib import Path
from typing import Final, Optional

import sounddevice as sd
import soundfile as sf

from core.config import config


class AudioPlayerAdapter:
    """Adapter for playing audio files through specified audio devices."""

    def __init__(self) -> None:
        """Initialize the audio player adapter."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._device_name: Final[Optional[str]] = config.virtual_output_name

    def play_audio(self, file_path: Path) -> None:
        """Play audio file through the configured output device.

        Args:
            file_path: Path to the audio file to play.
        """
        if not self.__validate_file(file_path):
            return

        device_index = self.__find_output_device_index()
        if device_index is None:
            self._logger.error("Audio output device was not found.")
            return

        self.__play(file_path, device_index)

    def __validate_file(self, file_path: Path) -> bool:
        """Validate that the audio file exists.

        Args:
            file_path: Path to the audio file to validate.

        Returns:
            True if file exists, False otherwise.
        """
        if not file_path.exists():
            self._logger.error(f"Audio file does not exist: {file_path}")
            return False
        return True

    def __find_output_device_index(self) -> Optional[int]:
        """Find the index of the configured output device.

        Returns:
            Device index if found, None otherwise.
        """
        if self._device_name is None:
            self._logger.error("Audio output device name is not set in config.")
            return None
        for idx, device in enumerate(sd.query_devices()):
            if self._device_name.lower() in device["name"].lower(
            ) and device["max_output_channels"] > 0:
                return idx
        self._logger.error(
            f"Audio output device '{self._device_name}' not found.",)
        return None

    def __play(self, file_path: Path, device_index: int) -> None:
        """Play the audio file through the specified device.

        Args:
            file_path: Path to the audio file to play.
            device_index: Index of the audio device to use.
        """
        try:
            data, samplerate = sf.read(str(file_path), dtype="float32")
            sd.play(data, samplerate=samplerate, device=device_index)
            sd.wait()
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self._logger.exception(
                f"Error occurred while playing audio: {exception}",)
