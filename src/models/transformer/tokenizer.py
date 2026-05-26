import torch
import json
import sentencepiece as spm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List, Dict

# Initialization of indexing for special tokens
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for loading and tokenizing translation pairs.
    Uses a SentencePiece (BPE) model to build a shared vocabulary, 
    dynamically encoding text into sub-word integer IDs.
    """
    def __init__(self, jsonl_path: str, spm_model_path: str = 'spm.model', max_seq_len: int = 128):
        self.data: List[dict] = []
        self.max_seq_len = max_seq_len

        # Load the JSONL dataset
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        # Initialize and load the trained SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

        # Create a shared vocabulary dictionary to interface seamlessly 
        # with standard PyTorch Embedding layers that expect len(vocab).
        shared_vocab: Dict[str, int] = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.src_vocab = shared_vocab
        self.tgt_vocab = shared_vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        # Extract raw text, falling back to joined tokens if necessary
        src_text = item.get('source_text') or ' '.join(item.get('source_tokens', []))
        tgt_text = item.get('target_text') or ' '.join(item.get('target_tokens', []))

        # Encode text to raw SPM integer IDs
        src_encoded = self.sp.encode(src_text, out_type=int)
        tgt_encoded = self.sp.encode(tgt_text, out_type=int)

        # TRUNCATE to (max_seq_len - 2) to leave room for <SOS> and <EOS>
        src_encoded = src_encoded[:self.max_seq_len - 2]
        tgt_encoded = tgt_encoded[:self.max_seq_len - 2]

        # Wrap with sequence boundary tokens
        src_ids = [SOS_IDX] + src_encoded + [EOS_IDX]
        tgt_ids = [SOS_IDX] + tgt_encoded + [EOS_IDX]

        return (torch.tensor(src_ids, dtype=torch.long), 
                torch.tensor(tgt_ids, dtype=torch.long))


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates a batch of tokenized sequences, padding them to the length of the 
    longest sequence in the batch.
    
    Returns tensors in [seq_len, batch_size] format (batch_first=False) 
    to align with PyTorch's native nn.Transformer.
    """
    src_batch, tgt_batch = zip(*batch)
    
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)

    return src_batch, tgt_batch