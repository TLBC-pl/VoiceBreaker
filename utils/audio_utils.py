"""Audio utilities for silence detection and waiting."""
import asyncio
import logging
from typing import Optional

import numpy as np
import sounddevice as sd


class SilenceWaiter:
    """Utility class for waiting for silence on audio devices."""

    def __init__(
        self,
        bot_output_device: Optional[str] = None,
        required_silence: float = 2.0,
        threshold: float = 500,
        sample_rate: int = 44100,
        frame_duration: float = 0.1,
        required: bool = False,
    ) -> None:
        """Initialize the silence waiter.

        Args:
            bot_output_device: Name of the bot output device to monitor.
            required_silence: Required silence duration in seconds.
            threshold: Amplitude threshold for silence detection.
            sample_rate: Audio sample rate.
            frame_duration: Duration of each audio frame in seconds.
            required: Whether silence detection is required.
        """
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__bot_output_device = bot_output_device
        self.__required_silence = required_silence
        self.__threshold = threshold
        self.__sample_rate = sample_rate
        self.__frame_duration = frame_duration
        self.__required = required

    async def wait_for_silence(self) -> None:
        """Wait for silence on the configured output device.

        Raises:
            RuntimeError: If bot output device is required but not set.
        """
        if not self.__bot_output_device:
            if self.__required:
                raise RuntimeError(
                    "BOT_OUTPUT_DEVICE must be set when running with --verify.",
                )
            return

        input_index = self.__find_device_index()
        device_info = sd.query_devices(input_index)
        self.__logger.info(
            f"Waiting for silence on device: {device_info['name']} "
            f"(index {input_index})",)
        frame_size = int(self.__sample_rate * self.__frame_duration)
        silent_frames_required = int(
            self.__required_silence / self.__frame_duration,)
        silent_counter = 0
        stream = sd.InputStream(
            samplerate=self.__sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_size,
            device=input_index,
        )
        with stream:
            while True:
                data, _ = stream.read(frame_size)
                amplitude = np.abs(data).mean()
                self.__logger.debug(f"Frame amplitude: {amplitude:.2f}")
                if amplitude < self.__threshold:
                    silent_counter += 1
                    if silent_counter >= silent_frames_required:
                        self.__logger.info("Silence detected.")
                        break
                else:
                    silent_counter = 0
                await asyncio.sleep(self.__frame_duration)

    def __find_device_index(self) -> Optional[int]:
        """Find device index for the bot output device.

        Returns:
            Device index if found.

        Raises:
            RuntimeError: If device is not found.
        """
        if not self.__bot_output_device:
            return None
        for idx, dev in enumerate(sd.query_devices()):
            if self.__bot_output_device.lower() in dev["name"].lower(
            ) and dev["max_output_channels"] > 0:
                return idx
        devices = [
            f"{idx}: {dev['name']}"
            for idx, dev in enumerate(sd.query_devices())
        ]
        error_msg = (
            f"BOT_OUTPUT_DEVICE '{self.__bot_output_device}' not found!\n"
            f"Available devices:\n" + "\n".join(devices))
        self.__logger.error(error_msg)
        raise RuntimeError(error_msg)
