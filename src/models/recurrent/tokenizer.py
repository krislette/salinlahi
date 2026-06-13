import sentencepiece as spm

PAD_IDX = 0  # SentencePiece reserves 0 for <pad> when we configure it that way
BOS_IDX = 1  # <s>  — equivalent to old <sos>
EOS_IDX = 2  # </s> — equivalent to old <eos>
UNK_IDX = 3  # <unk>


class SPMTokenizer:
    """
    Thin wrapper around a trained SentencePiece model.
    Replaces the old Vocabulary class — same interface: encode(), decode(), __len__().
    """

    def __init__(self, model_path: str) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text: str) -> list[int]:
        # Encode raw text -> list of integer piece IDs, wrapped with BOS and EOS
        return [BOS_IDX] + self.sp.Encode(text, out_type=int) + [EOS_IDX]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # Filter out BOS/EOS/PAD before decoding back to text
        if skip_special_tokens:
            ids = [i for i in ids if i not in (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX)]
        return self.sp.Decode(ids)

    def __len__(self) -> int:
        return self.sp.GetPieceSize()

    @staticmethod
    def train(
        texts: list[str],
        model_prefix: str,
        vocab_size: int = 8000,
    ) -> None:
        """
        Train a SentencePiece BPE model on the given list of raw text strings.
        Saves <model_prefix>.model and <model_prefix>.vocab to disk.
        """
        # Write texts to a temp file since SPM needs a file input
        tmp_path = model_prefix + "_tmp_corpus.txt"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in texts:
                f.write(line.strip() + "\n")

        spm.SentencePieceTrainer.Train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=PAD_IDX,
            bos_id=BOS_IDX,
            eos_id=EOS_IDX,
            unk_id=UNK_IDX,
            character_coverage=0.9995,  # high coverage for Filipino/Waray scripts
            minloglevel=2,
        )

        import os

        os.remove(tmp_path)
