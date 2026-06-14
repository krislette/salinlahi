from typing import Literal
from pydantic import BaseModel


class TranslationRequest(BaseModel):
    text: str
    direction: Literal["tgl-war", "war-tgl"]
    model: Literal["transformer", "recurrent"]


class TranslationResponse(BaseModel):
    translation: str
    direction: str
    source_language: str
    target_language: str
    model: str


class LanguagePair(BaseModel):
    direction: str
    source_language: str
    target_language: str


class LanguagesResponse(BaseModel):
    supported_pairs: list[LanguagePair]


class BaseModelInfoResponse(BaseModel):
    architecture: str
    supported_directions: list[str]


class TransformerInfoResponse(BaseModelInfoResponse):
    num_encoder_layers: int
    num_decoder_layers: int
    embedding_size: int
    attention_heads: int
    feedforward_dim: int


class RecurrentInfoResponse(BaseModelInfoResponse):
    rnn_type: str
    embedding_dim: int
    hidden_size: int
    num_encoder_layers: int
    dropout: float