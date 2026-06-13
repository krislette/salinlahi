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


class ModelInfoResponse(BaseModel):
    architecture: str
    num_encoder_layers: int
    num_decoder_layers: int
    embedding_size: int
    attention_heads: int
    feedforward_dim: int
    supported_directions: list[str]