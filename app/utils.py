import logging
from pathlib import Path

import yaml

from scripts.predict_single_transformer import TransformerTranslator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_server_config() -> dict:
    config_path = BASE_DIR / "config" / "server_config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelRegistry:
    """
    Loads and holds all TransformerTranslator instances at application startup.
    Each direction maps to one translator, initialized once and reused across requests.
    """

    def __init__(self):
        self._translators: dict[str, TransformerTranslator] = {}
        self._config = _load_server_config()

    def load_all(self):
        """
        Loads all models defined in server_config.yml.
        Raises FileNotFoundError at startup if any model or tokenizer file is missing.
        """
        model_configs = self._config.get("models", {})

        for name, cfg in model_configs.items():
            model_path = BASE_DIR / cfg["model_path"]
            tokenizer_path = BASE_DIR / cfg["tokenizer_path"]
            direction = cfg["direction"]

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found for '{direction}': {model_path}\n"
                    "Ensure the model weights are present before starting the server."
                )

            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer file not found for '{direction}': {tokenizer_path}\n"
                    "Ensure the tokenizer model is present before starting the server."
                )

            logger.info(f"Loading model for direction '{direction}' from {model_path}...")
            self._translators[direction] = TransformerTranslator(
                model_path=str(model_path),
                tokenizer_path=str(tokenizer_path),
            )
            logger.info(f"Model for '{direction}' loaded successfully.")

    def get_translator(self, direction: str) -> TransformerTranslator:
        """Returns the translator for the given direction."""
        return self._translators[direction]

    @property
    def config(self) -> dict:
        return self._config


# Single shared instance used across the application
registry = ModelRegistry()