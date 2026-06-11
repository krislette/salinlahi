import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.models.recurrent.tokenizer import SPMTokenizer, PAD_IDX


class TranslationDataset(Dataset):
    """
    PyTorch Dataset that wraps the JSONL files produced by the preprocessor.

    Each JSONL record looks like this:
        {
          "source_text": "saan ang ospital",
          "target_text": "hain an ospital"
        }

    Raw text is encoded on-the-fly using SPMTokenizer (SentencePiece BPE).
    """

    def __init__(
        self,
        jsonl_path: str,
        source_tokenizer: SPMTokenizer,
        target_tokenizer: SPMTokenizer,
    ) -> None:
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.records = self._load_jsonl(jsonl_path)

    def _load_jsonl(self, path: str) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        # Encode raw text into subword piece IDs, wrapped with BOS and EOS
        source_ids = self.source_tokenizer.encode(record["source_text"])
        target_ids = self.target_tokenizer.encode(record["target_text"])
        return {
            "source_tokens": torch.tensor(source_ids, dtype=torch.long),
            "target_tokens": torch.tensor(target_ids, dtype=torch.long),
        }


def collate_translation_batch(batch: list[dict]) -> dict:
    """
    Collate function passed to DataLoader.

    Sequences in a batch have different lengths, so we need to:
      1. Record the real (pre-padding) length of each source sequence
         so the encoder can use pack_padded_sequence.
      2. Pad all source sequences to the length of the longest one in the batch.
      3. Pad all target sequences to the length of the longest one in the batch.

    Padding is always done with PAD_IDX (0), which the model ignores.
    """
    source_sequences = [item["source_tokens"] for item in batch]
    target_sequences = [item["target_tokens"] for item in batch]

    # Record real lengths before padding (needed by pack_padded_sequence in encoder)
    source_lengths = torch.tensor(
        [seq.shape[0] for seq in source_sequences], dtype=torch.long
    )

    # pad_sequence stacks a list of tensors and pads the shorter ones with PAD_IDX
    # batch_first=True gives shape (batch_size, max_len) which is what the model expects
    padded_source = pad_sequence(
        source_sequences, batch_first=True, padding_value=PAD_IDX
    )
    padded_target = pad_sequence(
        target_sequences, batch_first=True, padding_value=PAD_IDX
    )

    return {
        "source_tokens": padded_source,  # (batch_size, max_source_len)
        "source_lengths": source_lengths,  # (batch_size,)
        "target_tokens": padded_target,  # (batch_size, max_target_len)
    }


def load_tokenizers(artifact_dir: str) -> tuple[SPMTokenizer, SPMTokenizer]:
    """
    Load the shared SentencePiece model for a language pair.
    Both source and target use the same SPM model since they share a vocabulary.
    """
    spm_path = str(Path(artifact_dir) / "spm.model")
    source_tokenizer = SPMTokenizer(spm_path)
    target_tokenizer = SPMTokenizer(spm_path)
    print(f"SPM vocab size: {len(source_tokenizer)}")
    return source_tokenizer, target_tokenizer


def build_dataloaders(
    train_jsonl_path: str,
    val_jsonl_path: str,
    batch_size: int,
    source_tokenizer: SPMTokenizer,
    target_tokenizer: SPMTokenizer,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience function that wraps the train and validation JSONL files into
    ready-to-use DataLoaders.

    The vocabularies must already be built (from the training set) before calling this.

    Args:
        train_jsonl_path: path to the training JSONL (e.g. waray_train.jsonl)
        val_jsonl_path:   path to the validation JSONL (e.g. waray_val.jsonl)
        batch_size:       number of sentence pairs per batch
        source_vocab:     built source (Filipino) vocabulary
        target_vocab:     built target (regional language) vocabulary
        num_workers:      number of worker processes for data loading (0 = main process)

    Returns:
        train_loader, val_loader
    """
    train_dataset = TranslationDataset(
        train_jsonl_path, source_tokenizer, target_tokenizer
    )
    val_dataset = TranslationDataset(val_jsonl_path, source_tokenizer, target_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_translation_batch,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
        num_workers=num_workers,
    )

    return train_loader, val_loader
