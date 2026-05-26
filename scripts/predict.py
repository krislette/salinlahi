import torch
import sentencepiece as spm
import pandas as pd
import argparse

# Imports mainly for the transformer model
from src.utils.helpers import TRANSFORMER_MODEL, TOKENIZER_MODEL, TEST_DATA_PATH
from models.transformer.seq2seq import BaselineSeq2SeqTransformer
from models.transformer.tokenizer import PAD_IDX, SOS_IDX, EOS_IDX
from models.transformer.helpers import generate_square_subsequent_mask

def beam_search_decode(model: torch.nn.Module, src_tensor: torch.Tensor, device: torch.device, beam_size: int = 3, max_len: int = 100, alpha: float = 0.7):
    """
    Executes Beam Search decoding to find the most probable translation sequence.
    """
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
                    prob = model.generator(out[-1, :]) 
                
                probs = torch.log_softmax(prob, dim=-1).squeeze(0)
                top_probs, top_idxs = torch.topk(probs, beam_size)

                for p, idx in zip(top_probs, top_idxs):
                    idx = idx.item()
                    p = p.item()

                    if len(seq) > 2 and idx == seq[-1] and idx == seq[-2]:
                        continue

                    new_seq = seq + [idx]
                    new_score = score + p 
                    candidates.append((new_seq, new_score))

            beams = sorted(
                candidates,
                key=lambda x: x[1] / ((len(x[0]) ** alpha) + 1e-6),
                reverse=True
            )[:beam_size]

            if all(seq[-1] == EOS_IDX for seq, _ in beams):
                break

        completed.extend(beams)
        if not completed:
            return beams[0][0]

        best_seq = sorted(
            completed,
            key=lambda x: x[1] / ((len(x[0]) ** alpha) + 1e-6),
            reverse=True
        )[0][0]

    return best_seq


def run_transformer_batch_inference(num_samples: int = 5):
    """
    Initializes the Transformer model and runs automated batch inference 
    on randomly sampled rows from the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dependencies on {device.type.upper()}...")

    # Load Tokenizer
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(str(TOKENIZER_MODEL))
        vocab_size = sp.get_piece_size()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load Model Architecture
    model = BaselineSeq2SeqTransformer(
        num_encoder_layers=6,
        num_decoder_layers=6,
        emb_size=512,
        nhead=8,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim_feedforward=2048,
        dropout=0.3
    ).to(device)

    # Load Golden Weights
    try:
        model.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Load Test Dataset
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        # Assuming the CSV has a 'split' column, filter for test if it exists
        if 'split' in df.columns:
            df = df[df['split'] == 'test']
            
        sample_df = df.sample(n=min(num_samples, len(df))).reset_index(drop=True)
    except Exception as e:
        print(f"Error loading test data at {TEST_DATA_PATH}: {e}")
        return

    print("\n" + "="*60)
    print(f"AUTOMATED BATCH INFERENCE ({len(sample_df)} SAMPLES)")
    print("="*60)

    # Automated Inference Loop
    for idx, row in sample_df.iterrows():
        source_sentence = str(row.get("source_text", ""))
        reference_sentence = str(row.get("target_text", ""))

        if not source_sentence:
            continue

        # Encode
        src_tokens = [SOS_IDX] + sp.encode(source_sentence, out_type=int) + [EOS_IDX]
        src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1)

        # Predict
        predicted_ids = beam_search_decode(model, src_tensor, device, beam_size=3)

        # Decode
        clean_ids = [i for i in predicted_ids if i not in [SOS_IDX, EOS_IDX, PAD_IDX]]
        translation = sp.decode(clean_ids)

        print(f"\nSample {idx + 1}")
        print(f"Source     : {source_sentence}")
        print(f"Reference  : {reference_sentence}")
        print(f"Prediction : {translation}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Automated Batch Inference")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "recurrent"], 
                        help="Choose which architecture to test")
    parser.add_argument("--samples", type=int, default=5, 
                        help="Number of random samples to translate from the test set")
    
    args = parser.parse_args()

    if args.model == "transformer":
        run_transformer_batch_inference(num_samples=args.samples)
    elif args.model == "recurrent":
        print("Recurrent inference engine is to be implemented.")