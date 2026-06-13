import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the 
    tokens in the sequence. The positional encodings have the same dimension 
    as the embeddings, so that the two can be summed.
    """
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Calculate positional encodings
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # Unsqueeze at dim 1 to create shape: [maxlen, 1, emb_size]
        # This allows perfect broadcasting over [seq_len, batch_size, emb_size] 
        # when the Transformer is set to batch_first=False.
        pos_embedding = pos_embedding.unsqueeze(1)

        self.dropout = nn.Dropout(dropout)

        # register_buffer ensures this tensor is saved with the model state 
        # but is NOT updated by the optimizer gradients.
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        Adds the positional encoding to the token embedding and applies dropout.
        
        Args:
            token_embedding: Tensor of shape [seq_len, batch_size, emb_size]
            
        Returns:
            Tensor of shape [seq_len, batch_size, emb_size]
        """
        # Slice pos_embedding up to the sequence length of the input.
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])