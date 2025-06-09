"""Audio device utilities for querying device properties."""
from typing import Optional

import sounddevice as sd


class AudioDeviceUtils:
    """Utility class for audio device operations."""

    @staticmethod
    def get_input_device_samplerate(device_name: Optional[str]) -> int:
        """Get sample rate for the specified input device.

        Args:
            device_name: Name of the input device.

        Returns:
            Sample rate of the device, defaults to 44100 if not found.
        """
        if not device_name:
            return 44100
        # noinspection PyBroadException
        try:
            for device in sd.query_devices():
                if device_name.lower().strip() == device["name"].lower().strip(
                ):
                    return int(device.get("default_samplerate", 44100))
        except Exception:  # pylint: disable=broad-exception-caught  # nosec
            pass
        return 44100
