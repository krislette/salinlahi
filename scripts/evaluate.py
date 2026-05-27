import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

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


def run_transformer_evaluation(num_samples: int = SAMPLE_SIZE):
    device = configure_device()

    print("\nLoading tokenizer and dataset...")
    train_dataset = TranslationDataset(
        str(JSON_DATA),
        spm_model_path=str(TOKENIZER_MODEL),
        max_seq_len=MAX_SEQ_LEN,
    )
    sp = train_dataset.sp
    vocab_size = sp.get_piece_size()

    print("\nBuilding transformer model...")
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

    print("\nLoading model checkpoint...")
    model.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location=device))
    model.eval()
    print("Model loaded successfully!")

    print("\nLoading test dataset...")
    df = pd.read_csv(CSV_DATA)
    if 'split' in df.columns:
        df = df[df['split'] == 'test']

    test_df = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"Evaluation samples: {len(test_df)}")

    references = []
    predictions = []
    results = []
    total_inference_time = 0.0
    smoothie = SmoothingFunction().method4
    chrf_metric = CHRF()

    evaluation_start = time.time()
    print(f"\nStarting evaluation on {len(test_df)} samples...\n")

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            source_sentence = str(row["source_text"])
            reference_sentence = str(row["target_text"])

            start_time = time.time()
            prediction = translate_sentence(source_sentence, model, sp, device)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            references.append([reference_sentence.split()])
            predictions.append(prediction.split())

            sentence_bleu1 = sentence_bleu(
                [reference_sentence.split()],
                prediction.split(),
                weights=(1, 0, 0, 0),
                smoothing_function=smoothie,
            )
            sentence_bleu2 = sentence_bleu(
                [reference_sentence.split()],
                prediction.split(),
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoothie,
            )
            sentence_bleu3 = sentence_bleu(
                [reference_sentence.split()],
                prediction.split(),
                weights=(1/3, 1/3, 1/3, 0),
                smoothing_function=smoothie,
            )
            sentence_bleu4 = sentence_bleu(
                [reference_sentence.split()],
                prediction.split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie,
            )
            sentence_chrf = chrf_metric.sentence_score(prediction, [reference_sentence]).score

            results.append({
                "source": source_sentence,
                "reference": reference_sentence,
                "prediction": prediction,
                "bleu1": sentence_bleu1,
                "bleu2": sentence_bleu2,
                "bleu3": sentence_bleu3,
                "bleu4": sentence_bleu4,
                "chrf": sentence_chrf,
                "inference_time_sec": inference_time,
            })

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples")

    total_evaluation_time = time.time() - evaluation_start
    results_df = pd.DataFrame(results)

    bleu1 = corpus_bleu(
        references,
        predictions,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothie,
    )
    bleu2 = corpus_bleu(
        references,
        predictions,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothie,
    )
    bleu3 = corpus_bleu(
        references,
        predictions,
        weights=(1/3, 1/3, 1/3, 0),
        smoothing_function=smoothie,
    )
    bleu4 = corpus_bleu(
        references,
        predictions,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    pred_texts = [" ".join(pred) for pred in predictions]
    ref_texts = [" ".join(ref[0]) for ref in references]
    chrf_score = chrf_metric.corpus_score(pred_texts, [ref_texts])

    avg_inference_time = total_inference_time / len(test_df)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print("\n================================================")
    print("TRANSFORMER MACHINE TRANSLATION EVALUATION")
    print("================================================")
    print(f"Evaluated Samples       : {len(results_df)}")
    print(f"BLEU-1 Score            : {bleu1:.4f}")
    print(f"BLEU-2 Score            : {bleu2:.4f}")
    print(f"BLEU-3 Score            : {bleu3:.4f}")
    print(f"BLEU-4 Score            : {bleu4:.4f}")
    print(f"chrF (F-score)          : {chrf_score.score:.4f}")
    print(f"Average Inference Time  : {avg_inference_time:.4f} sec/sample")
    print(f"Total Evaluation Time   : {total_evaluation_time:.2f} sec")
    print(f"Peak VRAM Usage         : {peak_vram:.2f} MB")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_output_path = results_dir / "transformer_mt_evaluation_results_BPE.csv"
    results_df.to_csv(results_output_path, index=False)
    print(f"\nDetailed results saved to: {results_output_path}")

    summary_metrics = {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "chrF": chrf_score.score,
        "Average_Inference_Time_sec": avg_inference_time,
        "Total_Evaluation_Time_sec": total_evaluation_time,
        "Peak_VRAM_MB": peak_vram,
    }

    summary_df = pd.DataFrame([summary_metrics])
    summary_output_path = results_dir / "transformer_evaluation_summary.csv"
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Summary metrics saved to: {summary_output_path}")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer evaluation on the test split")
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_SIZE,
        help="Number of random samples to translate from the test set",
    )
    args = parser.parse_args()

    run_transformer_evaluation(num_samples=args.samples)
