"""Main entry point for the VoiceBreaker application."""
import logging

from core.logging_config import setup_logging
from interfaces.cli_interface import main as cli_entrypoint


def run() -> None:
    """Run the VoiceBreaker application."""
    setup_logging()
    logging.getLogger(__name__).info("Initializing VoiceBreaker application...")
    cli_entrypoint()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    run()
