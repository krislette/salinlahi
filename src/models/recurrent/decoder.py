import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention from Bahdanau et al. (2015).

    At each decoder step, the attention module compares the current decoder hidden
    state against every encoder output position and produces a context vector —
    a weighted sum of encoder outputs — that tells the decoder where to look in the
    source sentence. This is especially useful for translation because word order
    between Filipino and the regional languages isn't always 1-to-1.

    The energy score for each source position i is computed as:
        score(decoder_hidden, encoder_output_i)
            = score_vector( tanh( W_enc * encoder_output_i
                                + W_dec * decoder_hidden ) )

    Scores are then passed through softmax to get normalized attention weights.
    """

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int) -> None:
        super().__init__()

        # Projects encoder outputs (bidirectional, so * 2) into the attention space
        self.encoder_projection = nn.Linear(
            encoder_hidden_size * 2, decoder_hidden_size
        )

        # Projects the decoder hidden state into the same attention space
        self.decoder_projection = nn.Linear(decoder_hidden_size, decoder_hidden_size)

        # Collapses the attention space down to a single scalar score per source position
        self.score_vector = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden:  (batch_size, decoder_hidden_size)
        # encoder_outputs: (batch_size, source_len, encoder_hidden_size * 2)

        # Expand the decoder hidden state so it can be broadcast across all source positions
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        # decoder_hidden_expanded: (batch_size, 1, decoder_hidden_size)

        # Add the two projections and apply tanh — this is the "energy" function
        energy = torch.tanh(
            self.encoder_projection(encoder_outputs)
            + self.decoder_projection(decoder_hidden_expanded)
        )
        # energy: (batch_size, source_len, decoder_hidden_size)

        # Reduce to one score per source position
        scores = self.score_vector(energy).squeeze(2)
        # scores: (batch_size, source_len)

        # Padding positions should never get attention, so we set their scores to -inf
        # before softmax so they come out as zero weight
        if source_padding_mask is not None:
            scores = scores.masked_fill(source_padding_mask == 0, float("-inf"))

        # Softmax turns the raw scores into a proper probability distribution
        attention_weights = F.softmax(scores, dim=1)
        # attention_weights: (batch_size, source_len)

        # Build the context vector as a weighted sum of the encoder outputs
        # bmm performs batch matrix multiplication: (batch, 1, source_len) x (batch, source_len, hidden*2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context_vector: (batch_size, 1, encoder_hidden_size * 2)

        return context_vector.squeeze(1), attention_weights


class Decoder(nn.Module):
    """
    Autoregressive GRU/LSTM decoder with Bahdanau attention.

    At each decoding step the decoder:
      1. Embeds the previous output token (or <sos> at the first step)
      2. Attends over all encoder outputs to build a context vector
      3. Concatenates the embedding and context vector, then runs one RNN step
      4. Projects the RNN output to a distribution over the target vocabulary

    The final prediction combines the RNN output, context vector, and embedding
    (a technique called "input feeding" / deep output) so the model has direct
    access to what it just generated and what it attended to.
    """

    def __init__(
        self,
        target_vocab_size: int,
        embedding_dim: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        dropout: float,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.encoder_hidden_size = encoder_hidden_size

        # Embedding for the target language (Waray / Kapampangan / Hiligaynon)
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.attention = BahdanauAttention(encoder_hidden_size, decoder_hidden_size)

        # The RNN input at each step is the concatenation of:
        #   - the embedded previous token  (embedding_dim)
        #   - the attention context vector (encoder_hidden_size * 2)
        rnn_input_dim = embedding_dim + encoder_hidden_size * 2

        rnn_options = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        if self.rnn_type not in rnn_options:
            raise ValueError(
                f"rnn_type must be one of {list(rnn_options.keys())}, got '{rnn_type}'"
            )

        self.rnn = rnn_options[self.rnn_type](
            input_size=rnn_input_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # The output projection combines three sources of information for a richer prediction:
        #   - the new RNN hidden state       (decoder_hidden_size)
        #   - the attention context vector   (encoder_hidden_size * 2)
        #   - the embedded input token       (embedding_dim)
        self.output_projection = nn.Linear(
            decoder_hidden_size + encoder_hidden_size * 2 + embedding_dim,
            target_vocab_size,
        )

    def forward_step(
        self,
        input_token: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_cell: torch.Tensor | None,
        encoder_outputs: torch.Tensor,
        source_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Performs a single decoder step and returns the vocabulary logits for that step.

        Args:
            input_token:        (batch_size,) — token id fed as input at this step
            decoder_hidden:     (1, batch_size, decoder_hidden_size) — current hidden state
            decoder_cell:       (1, batch_size, decoder_hidden_size) — current cell (LSTM only)
            encoder_outputs:    (batch_size, source_len, encoder_hidden_size * 2)
            source_padding_mask:(batch_size, source_len) — 1 for real tokens, 0 for padding

        Returns:
            vocab_logits:       (batch_size, target_vocab_size) — unnormalized scores
            decoder_hidden:     (1, batch_size, decoder_hidden_size) — updated hidden state
            decoder_cell:       (1, batch_size, decoder_hidden_size) or None
            attention_weights:  (batch_size, source_len) — where the model looked
        """

        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        # embedded: (batch_size, 1, embedding_dim)

        # Squeeze out the leading num_layers dimension (always 1 here) for the attention call
        context_vector, attention_weights = self.attention(
            decoder_hidden.squeeze(0), encoder_outputs, source_padding_mask
        )
        # context_vector:    (batch_size, encoder_hidden_size * 2)
        # attention_weights: (batch_size, source_len)

        # Concatenate the embedding and context vector, then step the RNN
        rnn_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)
        # rnn_input: (batch_size, 1, embedding_dim + encoder_hidden_size * 2)

        if self.rnn_type == "lstm":
            rnn_output, (decoder_hidden, decoder_cell) = self.rnn(
                rnn_input, (decoder_hidden, decoder_cell)
            )
        else:
            rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
            decoder_cell = None
        # rnn_output: (batch_size, 1, decoder_hidden_size)

        # Combine rnn output, context, and embedding to form the final prediction input
        # This gives the model a direct view of what it attended to and its last token
        prediction_input = torch.cat(
            [rnn_output.squeeze(1), context_vector, embedded.squeeze(1)], dim=1
        )
        # prediction_input: (batch_size, decoder_hidden_size + encoder_hidden_size * 2 + embedding_dim)

        vocab_logits = self.output_projection(prediction_input)
        # vocab_logits: (batch_size, target_vocab_size)

        return vocab_logits, decoder_hidden, decoder_cell, attention_weights
