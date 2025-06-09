"""Conversation service for managing jailbreak flow and audio interactions."""
import asyncio
import logging

from adapters.audio_player_adapter import AudioPlayerAdapter
from bridge.mic_to_virtual_bridge import MicrophoneToVirtualCableBridge
from core.config import config
from services.audio_routing_service import AudioRoutingService
from services.jailbreak_evaluation_service import (
    JailbreakEvaluationService,
    JailbreakPromptResult,
)
from services.stt_service import STTService
from services.tts_service import TTSService
from utils.audio_utils import SilenceWaiter
from utils.file_utils import FileAndAudioUtils


class ConversationService:
    """Service for managing conversation flow and jailbreak attempts."""

    def __init__(self, bypass_jailbreak_result: bool = True) -> None:
        """Initialize the conversation service.

        Args:
            bypass_jailbreak_result: Whether to bypass jailbreak evaluation.
        """
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__audio_router = AudioRoutingService()
        self.__tts_service = TTSService()
        self.__stt_service = STTService()
        self.__evaluation_service = JailbreakEvaluationService()
        self.__player = AudioPlayerAdapter()
        self.__mic_bridge = MicrophoneToVirtualCableBridge(
            mic_name=config.microphone_name,
            virtual_output_name=config.virtual_output_name,
        )
        self.__bypass_jailbreak_result = bypass_jailbreak_result
        self.__utils = FileAndAudioUtils()
        self.__silence_waiter = SilenceWaiter(
            bot_output_device=getattr(config, "bot_output_device", None),
            required_silence=2.0,
            required=not bypass_jailbreak_result,
        )

    async def run_jailbreak_flow(self, prompt_text: str) -> None:
        """Run the complete jailbreak flow.

        Args:
            prompt_text: Text of the jailbreak prompt to execute.
        """
        if not self.__validate_audio_setup():
            self.__logger.error("❌ Audio device validation failed. Exiting.")
            return

        self.__logger.info("🗣️ Generating jailbreak prompt audio (TTS)...")
        await self.__generate_and_play_prompt(prompt_text)
        self.__logger.info("▶️ Playing jailbreak prompt audio...")
        self.__logger.info("⏳ Starting microphone-to-virtual-cable bridge...")
        await self.__mic_bridge.start()

        if self.__bypass_jailbreak_result:
            self.__logger.info(
                "🎤 Microphone is now live. You can talk to the bot.",)
            await self.__maintain_mic_forwarding()
            return

        self.__logger.info("🔴 Recording model's response...")
        transcript = await self.__record_and_transcribe_response()
        self.__logger.info("🧠 Evaluating jailbreak attempt...")
        jailbreak_result = await self.__evaluate_jailbreak(transcript)
        await self.__handle_jailbreak_result(jailbreak_result)

    def __validate_audio_setup(self) -> bool:
        """Validate that all required audio devices are available.

        Returns:
            True if all devices are available, False otherwise.
        """
        devices = [
            config.microphone_name,
            config.virtual_output_name,
            config.virtual_input_name,
        ]
        if not self.__utils.validate_audio_devices(devices):
            self.__logger.error(
                "❌ One or more audio devices are missing or misconfigured.",)
            return False
        return True

    async def __generate_and_play_prompt(self, prompt_text: str) -> None:
        """Generate and play the jailbreak prompt audio.

        Args:
            prompt_text: Text to convert to speech and play.
        """
        audio_path = config.audio_output_dir / "jailbreak_prompt.wav"
        cache_used = await self.__tts_service.generate_audio(
            prompt_text,
            audio_path,
        )
        if cache_used:
            self.__logger.info("💾 Using cached prompt audio.")
        else:
            self.__logger.info("🆕 Prompt audio generated via TTS.")
        self.__logger.info("🔈 Routing audio devices and waiting for silence...")
        self.__audio_router.route_audio_output(config.virtual_output_name)
        self.__audio_router.route_audio_input(config.virtual_input_name)
        self.__logger.info("🤫 Waiting for silence on virtual input...")
        await self.__silence_waiter.wait_for_silence()
        self.__logger.info("▶️ Playing jailbreak prompt audio...")
        self.__player.play_audio(audio_path)

    async def __record_and_transcribe_response(self) -> str:
        """Record and transcribe the model's response.

        Returns:
            Transcribed text of the model's response.
        """
        response_path = config.audio_output_dir / "model_response.wav"
        await self.__stt_service.record_audio(response_path, max_duration=10)
        self.__logger.info("🔎 Transcribing model response via Whisper API...")
        return await self.__stt_service.transcribe_audio(response_path)

    async def __evaluate_jailbreak(
        self,
        transcript: str,
    ) -> JailbreakPromptResult:
        """Evaluate the jailbreak attempt.

        Args:
            transcript: Transcript of the model's response.

        Returns:
            Result of the jailbreak evaluation.
        """
        return await self.__evaluation_service.evaluate_jailbreak(transcript)

    async def __handle_jailbreak_result(
        self,
        result: JailbreakPromptResult,
    ) -> None:
        """Handle the result of jailbreak evaluation.

        Args:
            result: Result of the jailbreak evaluation.
        """
        if result.success or self.__bypass_jailbreak_result:
            self.__logger.info(
                "🎤 Microphone is now live. You can talk to the bot.",)
            await self.__maintain_mic_forwarding()
        else:
            self.__logger.info(
                "⛔️ Jailbreak attempt failed or was rejected. "
                "Stopping audio bridge.",)
            await self.__mic_bridge.stop()

    async def __maintain_mic_forwarding(self) -> None:
        """Maintain microphone forwarding until interrupted."""
        self.__logger.info(
            "🟢 Microphone forwarding is active. Press Ctrl+C to stop.",)
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.__logger.info("🛑 Microphone forwarding stopped (Ctrl+C).")
        finally:
            await self.__mic_bridge.stop()
