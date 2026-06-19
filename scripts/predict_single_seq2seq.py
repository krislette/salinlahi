import argparse
import torch
import yaml
import logging

from src.models.recurrent.seq2seq import build_model
from src.models.recurrent.tokenizer import SPMTokenizer, BOS_IDX, EOS_IDX

from scripts.preprocess import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Seq2SeqTranslator:
    """
    A unified wrapper class for the Seq2Seq model.
    Instantiate this twice in your API server (once for TGL-WAR, once for WAR-TGL).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config_path: str = "config/model_config.yml",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Initializing Seq2Seq Translator on: {self.device} using {model_path}"
        )

        try:
            # Load config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Load Tokenizer
            self.tokenizer = SPMTokenizer(str(tokenizer_path))
            vocab_size = len(self.tokenizer)

            # Load checkpoint FIRST to get the model's saved config
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = checkpoint.get("config", config)

            # Initialize Architecture with checkpoint's config
            self.model = build_model(
                self.config,
                source_vocab_size=vocab_size,
                target_vocab_size=vocab_size,
                device=self.device,
            )

            # Load Weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            logger.info("Model weights loaded and ready for inference.")

        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise e

    def predict(self, sentence: str) -> str:
        """Takes a single raw string and returns the translated string."""
        if not sentence or not isinstance(sentence, str) or not sentence.strip():
            return ""

        try:
            # Encode source sentence (limits to max 100 tokens)
            cleaned_sentence = preprocess_text(sentence)
            source_ids = self.tokenizer.encode(cleaned_sentence.lower())[:100]
            source_tensor = (
                torch.tensor(source_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            )
            source_lengths = torch.tensor([len(source_ids)], dtype=torch.long)

            # Decode (greedy decoding)
            predicted_ids, _ = self.model.translate(
                source_tensor,
                source_lengths,
                sos_idx=BOS_IDX,
                eos_idx=EOS_IDX,
                max_output_length=50,
            )

            # Decode tokens back to text
            translation = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)

            return translation

        except Exception as e:
            logger.error(f"Translation generation failed: {str(e)}")
            return "[Error: Unable to process translation]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Input Seq2Seq Inference")
    parser.add_argument("--text", type=str, required=True, help="The source sentence")
    parser.add_argument(
        "--direction",
        type=str,
        choices=["tgl-war", "war-tgl"],
        required=True,
        help="Translation direction",
    )
    args = parser.parse_args()

    # Determine paths based on direction argument
    if args.direction == "war-tgl":
        m_path = "models/recurrent/war_fil/best_model.pt"
        t_path = "data/processed/war_fil/spm.model"
    else:  # tgl-war
        m_path = "models/recurrent/fil_war/best_model.pt"
        t_path = "data/processed/fil_war/spm.model"

    # Initialize the specific translator
    translator = Seq2SeqTranslator(model_path=m_path, tokenizer_path=t_path)
    result = translator.predict(args.text)

    print("\n" + "=" * 50)
    print(f"Source ({args.direction.upper()}) : {args.text}")
    print(f"Translation: {result}")
    print("=" * 50 + "\n")
