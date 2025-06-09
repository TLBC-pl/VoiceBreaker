from typing import Optional

import sounddevice as sd


class AudioDeviceUtils:
    @staticmethod
    def get_input_device_samplerate(device_name: Optional[str]) -> int:
        if not device_name:
            return 44100
        # noinspection PyBroadException
        try:
            for device in sd.query_devices():
                if device_name.lower().strip() == device["name"].lower().strip():
                    return int(device.get("default_samplerate", 44100))
        except Exception: # pylint: disable=broad-exception-caught
            pass
        return 44100
