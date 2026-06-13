import torch
import torch.nn as nn
import math

from src.models.transformer.positional import PositionalEncoding

class BaselineSeq2SeqTransformer(nn.Module):
    """
    A custom Sequence-to-Sequence Transformer architecture optimized for 
    translation tasks. Includes Weight Tying and Xavier initialization.
    """
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(BaselineSeq2SeqTransformer, self).__init__()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # Tensors expected in shape: [seq_len, batch_size, emb_size]
        )

        # Word Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)

        # Final linear layer to project to the target vocabulary
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        # WEIGHT TYING
        # Ties the decoder's output projection weights to the target embedding weights
        self.generator.weight = self.tgt_tok_emb.weight

        # Use the positional encoding setup
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        # Initialize parameters for better convergence
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the model parameters using Xavier Uniform.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src: torch.Tensor, 
                trg: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor, 
                src_padding_mask: torch.Tensor, 
                tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training.
        """
        # Embed and add positional encoding (scaled by sqrt of embedding size)
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.transformer.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg) * math.sqrt(self.transformer.d_model))

        # Pass through the Transformer
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask)

        # Generate probabilities for the vocabulary
        return self.generator(outs)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Passes the source sequence through the Encoder. Used during inference.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.transformer.d_model))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Passes the target sequence and Encoder memory through the Decoder. Used during inference.
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.transformer.d_model))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)