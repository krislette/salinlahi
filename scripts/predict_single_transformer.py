import argparse
import torch
import sentencepiece as spm
import logging

from src.models.transformer.seq2seq import BaselineSeq2SeqTransformer
from src.models.transformer.tokenizer import PAD_IDX, SOS_IDX, EOS_IDX
from src.models.transformer.helpers import generate_square_subsequent_mask

from scripts.preprocess import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def beam_search_decode(
    model: torch.nn.Module,
    src_tensor: torch.Tensor,
    device: torch.device,
    beam_size: int = 5,
    max_len: int = 40,
    alpha: float = 0.7,
):
    """Executes Beam Search decoding to find the most probable translation sequence."""
    model.eval()
    src_tensor = src_tensor.to(device)

    with torch.no_grad():
        src_mask = torch.zeros(
            (src_tensor.shape[0], src_tensor.shape[0]), device=device, dtype=torch.bool
        )

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            memory = model.encode(src_tensor, src_mask)

        beams = [([SOS_IDX], 0.0)]
        completed = []

        for _ in range(max_len):
            candidates = []

            for seq, score in beams:
                if seq[-1] == EOS_IDX:
                    completed.append((seq, score))
                    continue

                trg_tensor = torch.tensor(
                    seq, dtype=torch.long, device=device
                ).unsqueeze(1)
                trg_mask = generate_square_subsequent_mask(trg_tensor.size(0)).to(
                    device
                )

                with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
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
    Instantiate this twice in your API server (once for WAR-TGL, once for TGL-WAR).
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Translator API on: {self.device} using {model_path}")

        try:
            # Load Tokenizer
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(str(tokenizer_path))
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
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            logger.info("Model weights loaded and ready for inference.")

        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise e

    def predict(self, sentence: str, beam_size: int = 3) -> str:
        """Takes a single raw string and returns the translated string."""
        if not sentence or not isinstance(sentence, str) or not sentence.strip():
            return ""

        try:
            # Limits the amount of input to max of 100
            cleaned_sentence = preprocess_text(sentence.lower())
            raw_tokens = self.sp.encode(str(cleaned_sentence), out_type=int)[:100]
            src_tokens = [SOS_IDX] + raw_tokens + [EOS_IDX]
            src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1)

            # Decode
            predicted_ids = beam_search_decode(
                self.model, src_tensor, self.device, beam_size=beam_size
            )

            clean_ids = [
                idx for idx in predicted_ids if idx not in [SOS_IDX, EOS_IDX, PAD_IDX]
            ]
            translation = self.sp.decode(clean_ids)

            return translation

        except Exception as e:
            logger.error(f"Translation generation failed: {str(e)}")
            return "[Error: Unable to process translation]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Input Transformer Inference")
    parser.add_argument("--text", type=str, required=True, help="The source sentence")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["war-tgl", "tgl-war"],
        required=True,
        help="Translation direction",
    )
    args = parser.parse_args()

    # Determine paths based on direction argument
    if args.direction == "war-tgl":
        m_path = "models/transformer/waray_tagalog_transformer_model.pt"
        t_path = "models/transformer/tokenizer/waray_tagalog_bpe.model"
    else:
        m_path = "models/transformer/tagalog_waray_transformer_model.pt"
        t_path = "models/transformer/tokenizer/tagalog_waray_bpe.model"

    # Initialize the specific translator
    translator = TransformerTranslator(model_path=m_path, tokenizer_path=t_path)
    result = translator.predict(args.text)

    print("\n" + "=" * 50)
    print(f"Source ({args.direction.upper()}) : {args.text}")
    print(f"Translation: {result}")
    print("=" * 50 + "\n")
