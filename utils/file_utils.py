"""File and audio utilities for validation and loading."""
import logging
from pathlib import Path
from typing import List

import sounddevice as sd


class FileAndAudioUtils:
    """Utility class for file operations and audio device validation."""

    def __init__(self) -> None:
        """Initialize the file and audio utilities."""
        self.__logger = logging.getLogger(self.__class__.__name__)

    def load_prompt_from_file(self, file_path: Path) -> str:
        """Load prompt text from file.

        Args:
            file_path: Path to the prompt file.

        Returns:
            Content of the prompt file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            Exception: If file reading fails.
        """
        if not file_path.exists():
            self.__logger.error(f"Prompt file does not exist: {file_path}")
            raise FileNotFoundError(f"Prompt file does not exist: {file_path}")
        try:
            text = file_path.read_text(encoding="utf-8")
            return text
        except Exception as exception:
            self.__logger.error(
                f"Failed to load prompt from file {file_path}: {exception}",)
            raise

    def validate_audio_devices(self, device_names: List[str]) -> bool:
        """Validate that all specified audio devices are available.

        Args:
            device_names: List of device names to validate.

        Returns:
            True if all devices are found, False otherwise.
        """
        available_devices = sd.query_devices()
        device_names_lower = [d["name"].lower() for d in available_devices]
        all_found = True
        for name in device_names:
            if not name:
                self.__logger.error("Device name is None or empty.")
                all_found = False
                continue
            if name.lower() not in device_names_lower:
                self.__logger.error(f"Audio device not found: {name}")
                all_found = False
        return all_found
