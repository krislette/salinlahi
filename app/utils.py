import logging
import os
import yaml

from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download

from scripts.predict_single_transformer import TransformerTranslator
from scripts.predict_single_seq2seq import Seq2SeqTranslator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# Type alias for either translator
AnyTranslator = Union[TransformerTranslator, Seq2SeqTranslator]

# Maps model_type string from config to the corresponding translator class
_TRANSLATOR_CLASSES = {
    "transformer": TransformerTranslator,
    "recurrent": Seq2SeqTranslator,
}


def _load_server_config() -> dict:
    config_path = BASE_DIR / "config" / "server_config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelRegistry:
    """
    Loads and holds all translator instances at application startup.

    Each translator is keyed by (direction, model_type), e.g.:
        ("tgl-war", "transformer") -> TransformerTranslator
        ("war-tgl", "recurrent")   -> Seq2SeqTranslator

    Instances are initialized once at startup and reused across requests.
    """

    def __init__(self):
        self._translators: dict[tuple[str, str], AnyTranslator] = {}
        self._config = _load_server_config()

    def load_all(self):
        """
        Loads all models defined in server_config.yml.
        Raises FileNotFoundError at startup if any model or tokenizer file is missing.
        Raises ValueError if a model_type in config is not recognized.
        """
        model_configs = self._config.get("models", {})

        for name, cfg in model_configs.items():
            model_type = cfg["model_type"]
            direction = cfg["direction"]
            model_path = BASE_DIR / cfg["model_path"]
            tokenizer_path = BASE_DIR / cfg["tokenizer_path"]

            if model_type not in _TRANSLATOR_CLASSES:
                raise ValueError(
                    f"Unknown model_type '{model_type}' for '{name}'. "
                    f"Must be one of: {list(_TRANSLATOR_CLASSES.keys())}"
                )

            if not model_path.exists():
                os.makedirs(model_path.parent, exist_ok=True)
                hf_hub_download(
                    repo_id="krislette/salinlahi-models",
                    filename=f"{model_path.relative_to(BASE_DIR)}",
                    local_dir=BASE_DIR,
                )

            if not tokenizer_path.exists():
                os.makedirs(tokenizer_path.parent, exist_ok=True)
                hf_hub_download(
                    repo_id="krislette/salinlahi-models",
                    filename=f"{tokenizer_path.relative_to(BASE_DIR)}",
                    local_dir=BASE_DIR,
                )

            logger.info(
                f"Loading '{model_type}' model for direction '{direction}' from {model_path}..."
            )

            translator_class = _TRANSLATOR_CLASSES[model_type]
            self._translators[(direction, model_type)] = translator_class(
                model_path=str(model_path),
                tokenizer_path=str(tokenizer_path),
            )

            logger.info(f"'{model_type}' model for '{direction}' loaded successfully.")

    def get_translator(self, direction: str, model_type: str) -> AnyTranslator:
        """Returns the translator for the given direction and model type."""
        key = (direction, model_type)
        if key not in self._translators:
            raise KeyError(
                f"No translator found for direction='{direction}', model='{model_type}'."
            )
        return self._translators[key]

    @property
    def config(self) -> dict:
        return self._config


# Single shared instance used across the application
registry = ModelRegistry()
