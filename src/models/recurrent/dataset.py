import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Special tokens that every vocabulary must have
PAD_TOKEN = "<pad>"  # used to fill shorter sequences in a batch to equal length
SOS_TOKEN = "<sos>"  # start-of-sequence — the first input token fed to the decoder
EOS_TOKEN = "<eos>"  # end-of-sequence — signals the decoder to stop generating
UNK_TOKEN = "<unk>"  # unknown token — replaces any word not seen during training

# Their fixed integer IDs (we assign them manually so they're always consistent)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """
    Builds and maintains a two-way mapping between string tokens and integer ids.

    The data team's tokenizer (tokenizer.py) produces string token lists like:
        ["istasyon", "ng", "pulisya"]

    The model needs integer ids:
        [14, 3, 27]

    This class handles that conversion. It also wraps sequences with <sos> and <eos>
    and replaces unseen tokens with <unk> at inference time.

    Usage:
        vocab = Vocabulary()
        vocab.build(list_of_token_lists)
        ids = vocab.encode(["istasyon", "ng", "pulisya"])
        tokens = vocab.decode(ids)
    """

    def __init__(self) -> None:
        # token string -> integer id
        self.token_to_id: dict[str, int] = {}

        # integer id -> token string (for decoding predictions back to text)
        self.id_to_token: dict[int, str] = {}

        # Register the four special tokens at fixed, reserved positions
        for idx, token in enumerate([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def build(self, token_lists: list[list[str]], min_frequency: int = 1) -> None:
        """
        Build the vocabulary from a list of already-tokenized sentences.

        Args:
            token_lists:   each inner list is one tokenized sentence
            min_frequency: tokens appearing fewer times than this are excluded
                           (helps reduce noise from typos in low-resource data)
        """
        # Count how often each token appears across the entire dataset
        token_counts = Counter(token for tokens in token_lists for token in tokens)

        # Assign an integer id to every token that meets the frequency threshold
        # We start from 4 because 0-3 are already taken by the special tokens
        next_id = len(self.token_to_id)
        for token, count in sorted(token_counts.items()):
            if count >= min_frequency and token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

    def encode(self, tokens: list[str]) -> list[int]:
        """
        Convert a list of string tokens to a list of integer ids.
        Wraps the sequence with <sos> and <eos> automatically.
        Tokens not in the vocabulary are mapped to <unk>.
        """
        ids = [SOS_IDX]
        for token in tokens:
            ids.append(self.token_to_id.get(token, UNK_IDX))
        ids.append(EOS_IDX)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> list[str]:
        """
        Convert a list of integer ids back to string tokens.

        Args:
            skip_special_tokens: if True, removes <pad>, <sos>, <eos>, <unk> from output
        """
        special = {PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX}
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special:
                continue
            tokens.append(self.id_to_token.get(idx, UNK_TOKEN))
        return tokens

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str) -> None:
        """Persist the vocabulary to a JSON file so it can be reloaded for inference."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Reload a vocabulary that was previously saved with .save()."""
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for token, idx in loaded.items():
            vocab.token_to_id[token] = idx
            vocab.id_to_token[idx] = token
        return vocab


class TranslationDataset(Dataset):
    """
    PyTorch Dataset that wraps the JSONL files produced by the data team's tokenizer.py.

    Each JSONL record looks like this (from translation_pairs.csv):
        {
          "language_pair": "Tagalog-Waray",
          "split": "train",
          "source_lang": "Tagalog",
          "target_lang": "Waray",
          "source_text": "Istasyon ng Pulisya",
          "target_text": "Istasyon Hab Pulis",
          "source_tokens": ["istasyon", "ng", "pulisya"],
          "target_tokens": ["istasyon", "hab", "pulis"]
        }

    This dataset reads those string token lists and converts them to integer id tensors
    using the provided source and target vocabularies.
    """

    def __init__(
        self,
        jsonl_path: str,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
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

        # Convert string token lists to integer id tensors
        # encode() wraps each sequence with <sos> and <eos> automatically
        source_ids = self.source_vocab.encode(record["source_tokens"])
        target_ids = self.target_vocab.encode(record["target_tokens"])

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


def build_vocabularies_from_jsonl(
    jsonl_path: str,
    min_frequency: int = 1,
) -> tuple[Vocabulary, Vocabulary]:
    """
    Read a JSONL file (produced by the data team's tokenizer.py) and build
    source and target vocabularies from the token lists in it.

    This should be called once on the training split only. The resulting
    vocabularies are then used to encode train, validation, and test sets.

    Args:
        jsonl_path:    path to the JSONL file (e.g. data/processed/waray_train.jsonl)
        min_frequency: minimum token count to be included in the vocabulary

    Returns:
        source_vocab, target_vocab
    """
    source_token_lists = []
    target_token_lists = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_token_lists.append(record["source_tokens"])
            target_token_lists.append(record["target_tokens"])

    source_vocab = Vocabulary()
    source_vocab.build(source_token_lists, min_frequency=min_frequency)

    target_vocab = Vocabulary()
    target_vocab.build(target_token_lists, min_frequency=min_frequency)

    print(f"Source vocabulary size: {len(source_vocab)}")
    print(f"Target vocabulary size: {len(target_vocab)}")

    return source_vocab, target_vocab


def build_dataloaders(
    train_jsonl_path: str,
    val_jsonl_path: str,
    batch_size: int,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
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
    train_dataset = TranslationDataset(train_jsonl_path, source_vocab, target_vocab)
    val_dataset = TranslationDataset(val_jsonl_path, source_vocab, target_vocab)

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
