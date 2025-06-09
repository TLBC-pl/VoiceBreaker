import asyncio
import logging
from pathlib import Path

import click

from services.conversation_service import ConversationService
from utils.file_utils import FileAndAudioUtils

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--prompt-file", "-p",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the jailbreak prompt file (.txt)",
)
@click.option(
    "--verify/--no-verify",
    default=False,
    help="Enable jailbreak result verification before forwarding the microphone.",
)
def main(prompt_file: Path, verify: bool) -> None:
    utils = FileAndAudioUtils()
    prompt_text: str = utils.load_prompt_from_file(prompt_file)
    conversation_service = ConversationService(bypass_jailbreak_result=not verify)
    asyncio.run(conversation_service.run_jailbreak_flow(prompt_text))

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
