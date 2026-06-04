import argparse
import torch
import sentencepiece as spm

from src.utils.helpers import TRANSFORMER_MODEL, TOKENIZER_MODEL
from src.models.transformer.seq2seq import BaselineSeq2SeqTransformer
from src.models.transformer.tokenizer import PAD_IDX, SOS_IDX, EOS_IDX
from src.models.transformer.helpers import generate_square_subsequent_mask

# Hyperparameters for inference
BEAM_SIZE = 5
MAX_DECODING_LEN = 40
ALPHA = 0.7


def beam_search_decode(
    model: torch.nn.Module,
    src_tensor: torch.Tensor,
    device: torch.device,
    beam_size: int = BEAM_SIZE,
    max_len: int = MAX_DECODING_LEN,
    alpha: float = ALPHA,
):
    """Executes Beam Search decoding to find the most probable translation sequence."""
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

                    # N-Gram Repetition Blocking
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


class TransformerTranslator:
    """
    A unified wrapper class for the Transformer model. 
    Designed specifically to be instantiated once at the start of an API server.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Translator API on: {self.device}")

        # Load Tokenizer directly from the model file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(TOKENIZER_MODEL))
        vocab_size = self.sp.get_piece_size()

        # Initialize Architecture
        self.model = BaselineSeq2SeqTransformer(
            num_encoder_layers=6,
            num_decoder_layers=6,
            emb_size=512,
            nhead=8,
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            dim_feedforward=2048,
            dropout=0.3,
        ).to(self.device)

        # Load Weights
        self.model.load_state_dict(torch.load(TRANSFORMER_MODEL, map_location=self.device, weights_only=True))
        self.model.eval()
        print("Model weights loaded and ready for inference.")

    def predict(self, sentence: str) -> str:
        """Takes a single raw string and returns the translated string."""
        if not sentence.strip():
            return ""

        # Encode
        src_tokens = [SOS_IDX] + self.sp.encode(str(sentence), out_type=int) + [EOS_IDX]
        src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1)

        # Decode using Beam Search
        predicted_ids = beam_search_decode(self.model, src_tensor, self.device, beam_size=BEAM_SIZE)
        
        # Clean up special tokens and detokenize back to text
        clean_ids = [idx for idx in predicted_ids if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]]
        translation = self.sp.decode(clean_ids)
        
        return translation


if __name__ == "__main__":
    # CLI setup for manual testing from the terminal
    parser = argparse.ArgumentParser(description="Single-Input Transformer Inference")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The source sentence you want to translate",
    )
    args = parser.parse_args()

    # Initialize the translator and run the prediction
    translator = TransformerTranslator()
    result = translator.predict(args.text)
    
    print("\n" + "=" * 50)
    print(f"Source     : {args.text}")
    print(f"Translation: {result}")
    print("=" * 50 + "\n")