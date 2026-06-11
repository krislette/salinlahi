import sys
import json
import yaml
import torch
import logging
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.recurrent.seq2seq import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

LANGUAGE_PAIRS = {
    "fil_war": "fil_war",
    "war_fil": "war_fil",
}


def load_config(path: str = "config/model_config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# BLEU implementation (corpus-level, no external deps)


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(
    hypotheses: list[list[str]], references: list[list[str]], n: int
) -> tuple[int, int]:
    total_clipped, total_count = 0, 0
    for hyp, ref in zip(hypotheses, references):
        hyp_counts = _ngram_counts(hyp, n)
        ref_counts = _ngram_counts(ref, n)
        clipped = {ng: min(c, ref_counts[ng]) for ng, c in hyp_counts.items()}
        total_clipped += sum(clipped.values())
        total_count += sum(hyp_counts.values())
    return total_clipped, total_count


def corpus_bleu(hypotheses: list[list[str]], references: list[list[str]]) -> float:
    """Computes corpus-level BLEU-4."""
    import math

    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(len(r) for r in references)

    # Brevity penalty
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    log_avg = 0.0
    for n in range(1, 5):
        clipped, total = _clipped_precision(hypotheses, references, n)
        if total == 0 or clipped == 0:
            return 0.0
        log_avg += math.log(clipped / total)

    return bp * math.exp(log_avg / 4) * 100


# Evaluation


def evaluate_language(lang_code: str, config: dict, device: torch.device) -> dict:
    from src.models.recurrent.tokenizer import SPMTokenizer, BOS_IDX, EOS_IDX

    artifact_dir = Path("models/recurrent") / LANGUAGE_PAIRS[lang_code]
    checkpoint_path = artifact_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run train_seq2seq.py first."
        )

    spm_path = str(
        Path(config["paths"]["processed_data_dir"])
        / LANGUAGE_PAIRS[lang_code]
        / "spm.model"
    )
    source_tokenizer = SPMTokenizer(spm_path)
    target_tokenizer = SPMTokenizer(spm_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint.get("config", config)
    model = build_model(
        train_config, len(source_tokenizer), len(target_tokenizer), device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_jsonl = (
        Path(config["paths"]["processed_data_dir"])
        / LANGUAGE_PAIRS[lang_code]
        / "test.jsonl"
    )
    if not test_jsonl.exists():
        raise FileNotFoundError(f"Test file not found: {test_jsonl}")

    hypotheses, references = [], []

    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            source_ids = source_tokenizer.encode(record["source_text"])
            source_tensor = (
                torch.tensor(source_ids, dtype=torch.long).unsqueeze(0).to(device)
            )
            source_lengths = torch.tensor([len(source_ids)], dtype=torch.long)

            predicted_ids, _ = model.translate(
                source_tensor, source_lengths, BOS_IDX, EOS_IDX
            )
            hyp_tokens = target_tokenizer.decode(
                predicted_ids, skip_special_tokens=True
            ).split()
            ref_tokens = record["target_text"].split()

            hypotheses.append(hyp_tokens)
            references.append(ref_tokens)

    bleu = corpus_bleu(hypotheses, references)
    results = {
        "language_pair": lang_code,
        "num_sentences": len(hypotheses),
        "bleu": round(bleu, 4),
    }
    logger.info(f"{lang_code} | Sentences: {len(hypotheses)} | BLEU: {bleu:.2f}")
    return results


if __name__ == "__main__":
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    for lang_code in LANGUAGE_PAIRS:
        try:
            result = evaluate_language(lang_code, config, device)
            all_results.append(result)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {lang_code}: {e}")

    results_path = Path("results/rnn_evaluation.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
