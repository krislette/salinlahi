import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Bidirectional RNN/GRU/LSTM encoder for the Filipino source sentence.

    The encoder processes the entire source sequence and produces two things:
      1. Hidden states for every input token — these are used by the attention mechanism
         in the decoder to figure out which parts of the source to focus on.
      2. A single summary hidden state — this initializes the decoder so it has a
         starting sense of what the source sentence said.

    We use a bidirectional RNN so each token sees both its left and right context,
    which generally improves translation quality.
    """

    def __init__(
        self,
        source_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Embedding layer turns integer token ids into dense float vectors
        # padding_idx=0 means the padding token contributes zero gradient
        self.embedding = nn.Embedding(source_vocab_size, embedding_dim, padding_idx=0)

        # Dropout is applied to the embeddings before they enter the RNN
        self.dropout = nn.Dropout(dropout)

        # Choose the RNN variant based on the config value
        rnn_options = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        if self.rnn_type not in rnn_options:
            raise ValueError(
                f"rnn_type must be one of {list(rnn_options.keys())}, got '{rnn_type}'"
            )

        rnn_class = rnn_options[self.rnn_type]

        # bidirectional=True makes the RNN read the sequence both forward and backward
        # The output at each position is the concatenation of both directions, so the
        # effective output size doubles to hidden_size * 2
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # The decoder is unidirectional (hidden_size), but the encoder produces a
        # bidirectional hidden state (hidden_size * 2). This linear layer bridges the gap
        # by projecting the concatenated forward+backward final state into hidden_size
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # source_tokens:  (batch_size, source_len) — integer token ids for the Filipino input
        # source_lengths: (batch_size,) — the real (non-padded) length of each sentence

        embedded = self.dropout(self.embedding(source_tokens))
        # embedded: (batch_size, source_len, embedding_dim)

        # Packing tells the RNN to skip padding positions, which saves compute
        # and prevents padding from contaminating the hidden state
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        if self.rnn_type == "lstm":
            packed_outputs, (hidden, cell) = self.rnn(packed_input)
        else:
            packed_outputs, hidden = self.rnn(packed_input)
            cell = None  # RNN and GRU don't have a cell state

        # Unpack back to a padded tensor so we can run attention over all positions
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )
        # encoder_outputs: (batch_size, source_len, hidden_size * 2)

        # hidden has shape (num_layers * num_directions, batch_size, hidden_size)
        # For a 2-layer bidirectional model the layout is:
        #   index 0 — layer 1 forward
        #   index 1 — layer 1 backward
        #   index 2 — layer 2 forward  (top forward layer)
        #   index 3 — layer 2 backward (top backward layer)
        # We only want the top layer, which is the most abstract representation
        top_forward_hidden = hidden[-2]  # (batch_size, hidden_size)
        top_backward_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Concatenate then project into the decoder's expected hidden size
        decoder_initial_hidden = torch.tanh(
            self.hidden_projection(
                torch.cat([top_forward_hidden, top_backward_hidden], dim=1)
            )
        )
        # decoder_initial_hidden: (batch_size, hidden_size)

        # Do the same projection for the cell state if we're using LSTM
        decoder_initial_cell = None
        if self.rnn_type == "lstm":
            top_forward_cell = cell[-2]
            top_backward_cell = cell[-1]
            decoder_initial_cell = torch.tanh(
                self.hidden_projection(
                    torch.cat([top_forward_cell, top_backward_cell], dim=1)
                )
            )
            # decoder_initial_cell: (batch_size, hidden_size)

        return encoder_outputs, decoder_initial_hidden, decoder_initial_cell
