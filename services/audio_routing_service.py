import logging
from typing import Optional

import sounddevice as sd


class AudioRoutingService:
    def __init__(self) -> None:
        self.__logger = logging.getLogger(self.__class__.__name__)


    def route_audio_input(self, source_name: str) -> None:
        idx = self.__find_device_index_by_name(source_name, is_input=True)
        if idx is None:
            self.__logger.error(f"Failed to find input device: {source_name}")
            raise RuntimeError(f"Audio input device not found: {source_name}")
        sd.default.device = (idx, sd.default.device[1])

    def route_audio_output(self, output_name: str) -> None:
        idx = self.__find_device_index_by_name(output_name, is_input=False)
        if idx is None:
            self.__logger.error(f"Failed to find output device: {output_name}")
            raise RuntimeError(f"Audio output device not found: {output_name}")
        sd.default.device = (sd.default.device[0], idx)

    @staticmethod
    def __find_device_index_by_name(name: str, is_input: bool) -> Optional[int]:
        if not name:
            return None
        for idx, info in enumerate(sd.query_devices()):
            if name.lower() in info["name"].lower() and (
                info["max_input_channels"] > 0 if is_input else info["max_output_channels"] > 0
            ):
                return idx
        return None
