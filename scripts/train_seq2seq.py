import sys
import yaml
import logging
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocessor import prepare_data_from_csv
from src.models.recurrent.seq2seq import build_model
from src.models.recurrent.trainer import build_trainer
from src.models.recurrent.dataset import (
    build_vocabularies_from_jsonl,
    build_dataloaders,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

LANGUAGE_PAIRS = {
    "fil_war": "fil -> war",
    "war_fil": "war -> fil",
}


def load_config(path: str = "config/model_config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_for_language(lang_dir: str, config: dict, device: torch.device) -> None:
    data_dir = Path(config["paths"]["processed_data_dir"]) / lang_dir
    train_jsonl = str(data_dir / "train.jsonl")
    val_jsonl = str(data_dir / "val.jsonl")

    logger.info(f"Building vocabularies from {train_jsonl}")
    source_vocab, target_vocab = build_vocabularies_from_jsonl(train_jsonl)

    artifact_dir = Path("models/recurrent") / lang_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_vocab.save(str(artifact_dir / "source_vocab.json"))
    target_vocab.save(str(artifact_dir / "target_vocab.json"))

    train_loader, val_loader = build_dataloaders(
        train_jsonl,
        val_jsonl,
        batch_size=config["training"]["batch_size"],
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )

    model = build_model(config, len(source_vocab), len(target_vocab), device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    config_copy = {
        **config,
        "paths": {**config["paths"], "model_checkpoint_dir": str(artifact_dir)},
    }
    trainer = build_trainer(model, config_copy, device)
    trainer.train(train_loader, val_loader, num_epochs=config["training"]["num_epochs"])
    logger.info(f"Done — artifacts at {artifact_dir}")


if __name__ == "__main__":
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    prepare_data_from_csv(
        csv_dir="data/external",
        output_dir=config["paths"]["processed_data_dir"],
    )

    for lang_dir, label in LANGUAGE_PAIRS.items():
        logger.info(f"=== Training {label} ===")
        train_for_language(lang_dir, config, device)
