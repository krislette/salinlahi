import sys
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.recurrent.seq2seq import build_model
from src.models.recurrent.dataset import Vocabulary, SOS_IDX, EOS_IDX

LANGUAGE_PAIRS = {
    "fil_war": {"name": "Filipino -> Waray", "dir": "fil_war"},
    "war_fil": {"name": "Waray -> Filipino", "dir": "war_fil"},
}


def load_config(path: str = "config/model_config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_translator(lang_key: str, config: dict, device: torch.device) -> tuple:
    if lang_key not in LANGUAGE_PAIRS:
        raise ValueError(
            f"Unknown key '{lang_key}'. Choose from: {list(LANGUAGE_PAIRS)}"
        )

    artifact_dir = Path("models/recurrent") / LANGUAGE_PAIRS[lang_key]["dir"]
    checkpoint_path = artifact_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run train_seq2seq.py first."
        )

    source_vocab = Vocabulary.load(str(artifact_dir / "source_vocab.json"))
    target_vocab = Vocabulary.load(str(artifact_dir / "target_vocab.json"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("config", config)
    model = build_model(train_config, len(source_vocab), len(target_vocab), device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, source_vocab, target_vocab


def translate(
    text: str,
    lang_key: str,
    config: dict,
    device: torch.device,
    max_output_length: int = 50,
) -> str:
    model, source_vocab, target_vocab = load_translator(lang_key, config, device)
    tokens = text.lower().strip().split()
    source_ids = source_vocab.encode(tokens)
    source_tensor = torch.tensor(source_ids, dtype=torch.long).unsqueeze(0).to(device)
    source_lengths = torch.tensor([len(source_ids)], dtype=torch.long)
    predicted_ids, _ = model.translate(
        source_tensor, source_lengths, SOS_IDX, EOS_IDX, max_output_length
    )
    return " ".join(target_vocab.decode(predicted_ids, skip_special_tokens=True))


def translate_fil_to_war(text: str, config: dict, device: torch.device) -> str:
    return translate(text, "fil_war", config, device)


def translate_war_to_fil(text: str, config: dict, device: torch.device) -> str:
    return translate(text, "war_fil", config, device)


if __name__ == "__main__":
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tests = {
        "fil_war": "saan ang pinakamalapit na ospital",
        "war_fil": "hain an pinakaharani nga ospital",
    }

    for key, sentence in tests.items():
        info = LANGUAGE_PAIRS[key]
        try:
            result = translate(sentence, key, config, device)
            print(f"{info['name']}: {sentence!r} -> {result!r}")
        except FileNotFoundError as e:
            print(f"[{info['name']}] Skipped — {e}")
