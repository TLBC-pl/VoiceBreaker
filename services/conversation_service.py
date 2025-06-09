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
    def __init__(self, bypass_jailbreak_result: bool = True) -> None:
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__audio_router = AudioRoutingService()
        self.__tts_service = TTSService()
        self.__stt_service = STTService()
        self.__evaluation_service = JailbreakEvaluationService()
        self.__player = AudioPlayerAdapter()
        self.__mic_bridge = MicrophoneToVirtualCableBridge(
            mic_name=config.MICROPHONE_NAME,
            virtual_output_name=config.VIRTUAL_OUTPUT_NAME,
        )
        self.__bypass_jailbreak_result = bypass_jailbreak_result
        self.__utils = FileAndAudioUtils()
        self.__silence_waiter = SilenceWaiter(
            bot_output_device=getattr(config, "BOT_OUTPUT_DEVICE", None),
            required_silence=2.0,
            required=not bypass_jailbreak_result,
        )

    async def run_jailbreak_flow(self, prompt_text: str) -> None:
        if not self.__validate_audio_setup():
            self.__logger.error("❌ Audio device validation failed. Exiting.")
            return

        self.__logger.info("🗣️ Generating jailbreak prompt audio (TTS)...")
        await self.__generate_and_play_prompt(prompt_text)
        self.__logger.info("▶️ Playing jailbreak prompt audio...")
        self.__logger.info("⏳ Starting microphone-to-virtual-cable bridge...")
        await self.__mic_bridge.start()

        if self.__bypass_jailbreak_result:
            self.__logger.info("🎤 Microphone is now live. You can talk to the bot.")
            await self.__maintain_mic_forwarding()
            return

        self.__logger.info("🔴 Recording model's response...")
        transcript = await self.__record_and_transcribe_response()
        self.__logger.info("🧠 Evaluating jailbreak attempt...")
        jailbreak_result = await self.__evaluate_jailbreak(transcript)
        await self.__handle_jailbreak_result(jailbreak_result)

    def __validate_audio_setup(self) -> bool:
        devices = [config.MICROPHONE_NAME, config.VIRTUAL_OUTPUT_NAME, config.VIRTUAL_INPUT_NAME]
        if not self.__utils.validate_audio_devices(devices):
            self.__logger.error("❌ One or more audio devices are missing or misconfigured.")
            return False
        return True

    async def __generate_and_play_prompt(self, prompt_text: str) -> None:
        audio_path = config.AUDIO_OUTPUT_DIR / "jailbreak_prompt.wav"
        cache_used = await self.__tts_service.generate_audio(prompt_text, audio_path)
        if cache_used:
            self.__logger.info("💾 Using cached prompt audio.")
        else:
            self.__logger.info("🆕 Prompt audio generated via TTS.")
        self.__logger.info("🔈 Routing audio devices and waiting for silence...")
        self.__audio_router.route_audio_output(config.VIRTUAL_OUTPUT_NAME)
        self.__audio_router.route_audio_input(config.VIRTUAL_INPUT_NAME)
        self.__logger.info("🤫 Waiting for silence on virtual input...")
        await self.__silence_waiter.wait_for_silence()
        self.__logger.info("▶️ Playing jailbreak prompt audio...")
        self.__player.play_audio(audio_path)

    async def __record_and_transcribe_response(self) -> str:
        response_path = config.AUDIO_OUTPUT_DIR / "model_response.wav"
        await self.__stt_service.record_audio(response_path, max_duration=10)
        self.__logger.info("🔎 Transcribing model response via Whisper API...")
        return await self.__stt_service.transcribe_audio(response_path)

    async def __evaluate_jailbreak(self, transcript: str) -> JailbreakPromptResult:
        return await self.__evaluation_service.evaluate_jailbreak(transcript)

    async def __handle_jailbreak_result(self, result: JailbreakPromptResult) -> None:
        if result.success or self.__bypass_jailbreak_result:
            self.__logger.info("🎤 Microphone is now live. You can talk to the bot.")
            await self.__maintain_mic_forwarding()
        else:
            self.__logger.info("⛔️ Jailbreak attempt failed or was rejected. Stopping audio bridge.")
            await self.__mic_bridge.stop()

    async def __maintain_mic_forwarding(self) -> None:
        self.__logger.info("🟢 Microphone forwarding is active. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.__logger.info("🛑 Microphone forwarding stopped (Ctrl+C).")
        finally:
            await self.__mic_bridge.stop()
