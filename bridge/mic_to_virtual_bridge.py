"""Bridge for forwarding microphone audio to virtual cable."""
import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from core.config import config


class MicrophoneToVirtualCableBridge:
    """Bridge that forwards microphone audio to a virtual cable output."""

    def __init__(self, mic_name: str, virtual_output_name: str) -> None:
        """Initialize the microphone to virtual cable bridge.

        Args:
            mic_name: Name of the microphone input device.
            virtual_output_name: Name of the virtual output device.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._mic_name = mic_name
        self._virtual_output_name = virtual_output_name

        self._sample_rate: int = config.microphone_sample_rate
        self._input_index: Optional[int] = None
        self._output_index: Optional[int] = None
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._frames_forwarded: int = 0
        self._running: bool = False

    async def start(self) -> None:
        """Start the audio bridge between microphone and virtual cable."""
        self._input_index = self.__find_device_index(
            self._mic_name,
            is_input=True,
        )
        self._output_index = self.__find_device_index(
            self._virtual_output_name,
            is_input=False,
        )
        if self._input_index is None or self._output_index is None:
            self._logger.error(
                f"Could not initialize audio bridge. "
                f"Input index: {self._input_index}, "
                f"Output index: {self._output_index}",
            )
            return
        self.__setup_streams()
        try:
            self._input_stream.start()
            self._output_stream.start()
            self._running = True
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self._logger.exception(
                f"Failed to start microphone bridge: {exception}",
            )

    async def stop(self) -> None:
        """Stop the audio bridge and clean up resources."""
        try:
            if self._input_stream:
                self._input_stream.stop()
                self._input_stream.close()
                self._input_stream = None
            if self._output_stream:
                self._output_stream.stop()
                self._output_stream.close()
                self._output_stream = None
            self._running = False
        except Exception as exception:  # pylint: disable=broad-exception-caught
            self._logger.exception(
                f"Error occurred while stopping microphone bridge: {exception}",
            )

    def __setup_streams(self) -> None:
        """Set up input and output audio streams with callbacks."""

        def input_callback(indata, _frames, _time, status):
            if status:
                self._logger.warning(f"Mic input stream warning: {status}")
            try:
                self._audio_queue.put_nowait(indata.copy())
            except queue.Full:
                self._logger.warning(
                    "Microphone audio queue overflow — dropping frames!",
                )

        def output_callback(outdata, frames, _time, _status):
            chunk = np.zeros((frames, 1), dtype=np.float32)
            filled = 0
            try:
                while filled < frames:
                    data = self._audio_queue.get_nowait()
                    samples_available = data.shape[0]
                    samples_needed = frames - filled
                    if samples_available > samples_needed:
                        chunk[filled:] = data[:samples_needed]
                        remaining = data[samples_needed:]
                        self._audio_queue.queue.appendleft(remaining)
                        filled = frames
                    else:
                        chunk[filled:filled + samples_available] = data
                        filled += samples_available
            except queue.Empty:
                pass
            outdata[:] = chunk

        self._input_stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            blocksize=2048,
            device=self._input_index,
            callback=input_callback,
        )
        self._output_stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            blocksize=2048,
            device=self._output_index,
            callback=output_callback,
        )

    @staticmethod
    def __find_device_index(name: str, is_input: bool) -> Optional[int]:
        """Find device index by name and type.

        Args:
            name: Name of the device to find.
            is_input: True for input devices, False for output devices.

        Returns:
            Device index if found, None otherwise.
        """
        for i, device in enumerate(sd.query_devices()):
            if name.lower().strip() == device["name"].lower().strip():
                if is_input and device["max_input_channels"] > 0:
                    return i
                if not is_input and device["max_output_channels"] > 0:
                    return i
        return None
