import random

import torch
import torch.nn as nn

from src.models.recurrent.encoder import Encoder
from src.models.recurrent.decoder import Decoder


class Seq2Seq(nn.Module):
    """
    Full sequence-to-sequence model that wires the Encoder and Decoder together.

    During training, teacher forcing is used — with some probability the model is
    fed the ground-truth target token at each step instead of its own prediction.
    This helps stabilize training early on. As training progresses, the teacher
    forcing ratio can be annealed downward so the model learns to rely on itself.

    During inference (translate()), no teacher forcing is used. The model generates
    one token at a time using its own output, stopping when it produces <eos>.

    Usage:
        encoder = Encoder(...)
        decoder = Decoder(...)
        model = Seq2Seq(encoder, decoder, pad_token_idx=0, device=device)

        # Training
        logits = model(source_tokens, source_lengths, target_tokens, teacher_forcing_ratio=0.5)

        # Inference
        predicted_ids, attention_maps = model.translate(source_tokens, source_lengths, sos_idx, eos_idx)
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_token_idx: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_idx = pad_token_idx
        self.device = device

    def build_source_padding_mask(self, source_tokens: torch.Tensor) -> torch.Tensor:
        # Returns a boolean mask: 1 where a token is real, 0 where it's padding
        # The attention module uses this to zero out padding positions
        return source_tokens != self.pad_token_idx
        # shape: (batch_size, source_len)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_lengths: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Forward pass for training. Runs the encoder over the full source sentence, then
        unrolls the decoder step-by-step over the target sequence.

        Args:
            source_tokens:          (batch_size, source_len) — tokenized Filipino input
            source_lengths:         (batch_size,) — real (non-padded) lengths
            target_tokens:          (batch_size, target_len) — tokenized target including <sos> and <eos>
            teacher_forcing_ratio:  probability of feeding the gold token instead of model prediction

        Returns:
            all_logits: (batch_size, target_len, target_vocab_size)
                        Unnormalized scores at every target position. Position 0 is skipped
                        because there's no prediction before the <sos> token.
        """
        batch_size = source_tokens.shape[0]
        target_len = target_tokens.shape[1]
        target_vocab_size = self.decoder.output_projection.out_features

        # Pre-allocate the output tensor for all timesteps
        all_logits = torch.zeros(
            batch_size, target_len, target_vocab_size, device=self.device
        )

        source_padding_mask = self.build_source_padding_mask(source_tokens)

        # Run the encoder once to get hidden states for every source position
        encoder_outputs, decoder_hidden, decoder_cell = self.encoder(
            source_tokens, source_lengths
        )
        # encoder_outputs:  (batch_size, source_len, hidden_size * 2)
        # decoder_hidden:   (batch_size, hidden_size) — summary of the source sentence

        # The RNN expects the hidden state to have a leading num_layers dimension
        decoder_hidden = decoder_hidden.unsqueeze(0)
        # decoder_hidden: (1, batch_size, hidden_size)

        if decoder_cell is not None:
            decoder_cell = decoder_cell.unsqueeze(0)

        # Kick off decoding with the <sos> token (first column of target_tokens)
        current_input_token = target_tokens[:, 0]
        # current_input_token: (batch_size,)

        for timestep in range(1, target_len):
            step_logits, decoder_hidden, decoder_cell, _ = self.decoder.forward_step(
                current_input_token,
                decoder_hidden,
                decoder_cell,
                encoder_outputs,
                source_padding_mask,
            )
            all_logits[:, timestep] = step_logits

            # Decide whether to use teacher forcing for the next input token
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                # Feed the ground-truth token — the model is "taught" the correct sequence
                current_input_token = target_tokens[:, timestep]
            else:
                # Feed the model's own prediction — forces it to learn to recover from mistakes
                current_input_token = step_logits.argmax(dim=1)

        return all_logits

    def translate(
        self,
        source_tokens: torch.Tensor,
        source_lengths: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_output_length: int = 50,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """
        Greedy inference — generates a translation one token at a time.

        The model always feeds its own previous prediction as input (no teacher forcing).
        Decoding stops when the model outputs <eos> or when max_output_length is reached.

        Args:
            source_tokens:     (1, source_len) — a single source sentence (batch size 1)
            source_lengths:    (1,) — the real length of the source sentence
            sos_idx:           integer id of the <sos> token in the target vocabulary
            eos_idx:           integer id of the <eos> token in the target vocabulary
            max_output_length: hard cap on how many tokens to generate

        Returns:
            predicted_token_ids: list of integer token ids (excluding <sos>, including <eos>)
            attention_maps:      list of (source_len,) tensors, one per generated token
        """
        self.eval()
        with torch.no_grad():
            source_padding_mask = self.build_source_padding_mask(source_tokens)

            encoder_outputs, decoder_hidden, decoder_cell = self.encoder(
                source_tokens, source_lengths
            )

            decoder_hidden = decoder_hidden.unsqueeze(0)
            if decoder_cell is not None:
                decoder_cell = decoder_cell.unsqueeze(0)

            # Start with the <sos> token
            current_input_token = torch.tensor([sos_idx], device=self.device)

            predicted_token_ids = []
            attention_maps = []

            for _ in range(max_output_length):
                step_logits, decoder_hidden, decoder_cell, attention_weights = (
                    self.decoder.forward_step(
                        current_input_token,
                        decoder_hidden,
                        decoder_cell,
                        encoder_outputs,
                        source_padding_mask,
                    )
                )

                predicted_token = step_logits.argmax(dim=1)
                predicted_token_ids.append(predicted_token.item())
                attention_maps.append(attention_weights.squeeze(0).cpu())

                # Stop once the model generates the end-of-sequence token
                if predicted_token.item() == eos_idx:
                    break

                current_input_token = predicted_token

        return predicted_token_ids, attention_maps


def build_model(
    config: dict, source_vocab_size: int, target_vocab_size: int, device: torch.device
) -> Seq2Seq:
    """
    Convenience factory that reads the model config dict and returns a ready-to-train Seq2Seq model.

    Expects the config to have a 'model' key with the following structure (matching model_config.yml):
        model:
            embedding_dim: 256
            hidden_size: 512
            num_encoder_layers: 2
            dropout: 0.3
            rnn_type: "gru"
            pad_token_idx: 0

    Args:
        config:            parsed YAML config dict
        source_vocab_size: size of the Filipino vocabulary
        target_vocab_size: size of the target language vocabulary
        device:            torch.device to place the model on

    Returns:
        A Seq2Seq model with all parameters initialized on the given device.
    """
    model_cfg = config["model"]

    encoder = Encoder(
        source_vocab_size=source_vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_encoder_layers"],
        dropout=model_cfg["dropout"],
        rnn_type=model_cfg["rnn_type"],
    )

    decoder = Decoder(
        target_vocab_size=target_vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        encoder_hidden_size=model_cfg["hidden_size"],
        decoder_hidden_size=model_cfg["hidden_size"],
        dropout=model_cfg["dropout"],
        rnn_type=model_cfg["rnn_type"],
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        pad_token_idx=model_cfg["pad_token_idx"],
        device=device,
    ).to(device)

    return model
