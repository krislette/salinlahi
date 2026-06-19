import json
import sentencepiece as spm

from src.utils.helpers import JSON_DATA, TOKENIZER_MODEL


def train_sentencepiece_model():
    """
    Reads the tokenized JSONL training dataset, extracts the raw text,
    and trains a Shared-Vocabulary SentencePiece (BPE) model.
    """
    # Define working paths based on your config
    jsonl_path = JSON_DATA

    # Create a temporary text file in the same directory as the tokenizer model
    tokenizer_dir = TOKENIZER_MODEL.parent
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = tokenizer_dir / "spm_training_corpus.txt"

    # SentencePiece requires a prefix without the '.model' extension
    spm_prefix = str(TOKENIZER_MODEL.with_suffix(""))
    vocab_size = 4000

    print(f"Building corpus from {jsonl_path}...")
    line_count = 0

    # Extract raw source + target text into a flat corpus file
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(
        corpus_path, "w", encoding="utf-8"
    ) as f_out:

        for line in f_in:
            item = json.loads(line)

            # Prefer raw text; fall back to joined tokens
            src = item.get("source_text") or " ".join(item.get("source_tokens", []))
            tgt = item.get("target_text") or " ".join(item.get("target_tokens", []))

            if src.strip():
                f_out.write(src.strip() + "\n")
                line_count += 1
            if tgt.strip():
                f_out.write(tgt.strip() + "\n")
                line_count += 1

    print(f"Corpus written: {line_count:,} lines  →  {corpus_path}")

    # Train the SentencePiece BPE model
    print(f"\nTraining SentencePiece BPE model (vocab_size={vocab_size})...")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=spm_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        # Match existing special token index layout exactly
        pad_id=0,
        pad_piece="<PAD>",
        bos_id=1,
        bos_piece="<SOS>",
        eos_id=2,
        eos_piece="<EOS>",
        unk_id=3,
        unk_piece="<UNK>",
        character_coverage=1.0,
        add_dummy_prefix=True,
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )

    print("\nDone! Saved:")
    print(f"  {spm_prefix}.model  ← load this at inference time")
    print(f"  {spm_prefix}.vocab  ← human-readable token list")

    # Cleanup the temporary corpus text file to save space
    if corpus_path.exists():
        corpus_path.unlink()
        print(f"Cleaned up temporary corpus file: {corpus_path}")

    # Quick sanity check
    sp = spm.SentencePieceProcessor()
    sp.load(f"{spm_prefix}.model")

    print(f"\nVocab Size: {sp.get_piece_size()}")


if __name__ == "__main__":
    train_sentencepiece_model()

"""
To run:

python scripts/train_tokenizer.py

This is used for the transformer models to improve
performance by grouping tokens together.
"""
