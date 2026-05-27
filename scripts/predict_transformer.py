import argparse
import time

import pandas as pd
import torch

from src.utils.helpers import TRANSFORMER_MODEL, TOKENIZER_MODEL, JSON_DATA, CSV_DATA
from src.models.transformer.seq2seq import BaselineSeq2SeqTransformer
from src.models.transformer.tokenizer import TranslationDataset, PAD_IDX, SOS_IDX, EOS_IDX
from src.models.transformer.helpers import generate_square_subsequent_mask

SAMPLE_SIZE = 1500
MAX_SEQ_LEN = 128
BEAM_SIZE = 3
MAX_DECODING_LEN = 40
ALPHA = 0.7


def configure_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_vram = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"Initial GPU Memory Usage: {initial_vram:.2f} MB")

    return device


def beam_search_decode(
    model: torch.nn.Module,
    src_tensor: torch.Tensor,
    device: torch.device,
    beam_size: int = BEAM_SIZE,
    max_len: int = MAX_DECODING_LEN,
    alpha: float = ALPHA,
):
    model.eval()
    src_tensor = src_tensor.to(device)

    with torch.no_grad():
        src_mask = torch.zeros((src_tensor.shape[0], src_tensor.shape[0]), device=device, dtype=torch.bool)

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            memory = model.encode(src_tensor, src_mask)

        beams = [([SOS_IDX], 0.0)]
        completed = []

        for _ in range(max_len):
            candidates = []

            for seq, score in beams:
                if seq[-1] == EOS_IDX:
                    completed.append((seq, score))
                    continue

                trg_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1)
                trg_mask = generate_square_subsequent_mask(trg_tensor.size(0)).to(device)

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    out = model.decode(trg_tensor, memory, trg_mask)
                    logits = model.generator(out[-1])

                probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                top_probs, top_idxs = torch.topk(probs, beam_size)

                for p, idx in zip(top_probs, top_idxs):
                    idx = idx.item()
                    p = p.item()

                    if len(seq) > 2 and idx == seq[-1] and idx == seq[-2]:
                        continue

                    candidates.append((seq + [idx], score + p))

            beams = sorted(
                candidates,
                key=lambda x: x[1] / ((len(x[0]) ** alpha) + 1e-6),
                reverse=True,
            )[:beam_size]

            if beams and all(seq[-1] == EOS_IDX for seq, _ in beams):
                break

        completed.extend(beams)
        if not completed:
            return beams[0][0]

        best_seq = sorted(
            completed,
            key=lambda x: x[1] / ((len(x[0]) ** alpha) + 1e-6),
            reverse=True,
        )[0][0]

    return best_seq


def translate_sentence(sentence: str, model: torch.nn.Module, sp, device: torch.device) -> str:
    src_tokens = [SOS_IDX] + sp.encode(str(sentence), out_type=int) + [EOS_IDX]
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1)

    predicted_ids = beam_search_decode(model, src_tensor, device, beam_size=BEAM_SIZE)
    clean_ids = [idx for idx in predicted_ids if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
    return sp.decode(clean_ids)


def run_transformer_batch_inference(num_samples: int = SAMPLE_SIZE):
    device = configure_device()

    print("\nLoading tokenizer and dataset...")
    train_dataset = TranslationDataset(
        str(JSON_DATA),
        spm_model_path=str(TOKENIZER_MODEL),
        max_seq_len=MAX_SEQ_LEN,
    )
    val_dataset = TranslationDataset(
        str(JSON_DATA),
        spm_model_path=str(TOKENIZER_MODEL),
        max_seq_len=MAX_SEQ_LEN,
    )
    sp = train_dataset.sp
    vocab_size = sp.get_piece_size()

    model = BaselineSeq2SeqTransformer(
        num_encoder_layers=6,
        num_decoder_layers=6,
        emb_size=512,
        nhead=8,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim_feedforward=2048,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location=device))
    model.eval()
    print("Model loaded successfully!")

    print("\nLoading test dataset...")
    df = pd.read_csv(CSV_DATA)
    if 'split' in df.columns:
        df = df[df['split'] == 'test']

    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"Evaluation samples: {len(sample_df)}")

    print("\n" + "=" * 60)
    print(f"FINAL INFERENCE PIPELINE ({len(sample_df)} SAMPLES)")
    print("=" * 60)

    for idx, row in sample_df.iterrows():
        source_sentence = str(row.get("source_text", ""))
        reference_sentence = str(row.get("target_text", ""))

        if not source_sentence:
            continue

        translation = translate_sentence(source_sentence, model, sp, device)

        print(f"\nSample {idx + 1}")
        print(f"Source     : {source_sentence}")
        print(f"Reference  : {reference_sentence}")
        print(f"Prediction : {translation}")
        print("-" * 60)

    print("\nInference pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer inference on the test split")
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_SIZE,
        help="Number of random samples to translate from the test set",
    )
    args = parser.parse_args()

    run_transformer_batch_inference(num_samples=args.samples)
