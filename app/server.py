import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    TranslationRequest,
    TranslationResponse,
    LanguagePair,
    LanguagesResponse,
    ModelInfoResponse,
)
from app.utils import registry
from app.validators import validate_translation_request

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all models on startup. Fails fast if any model file is missing."""
    logger.info("Starting Salinlahi API — loading translation models...")
    registry.load_all()
    logger.info("All models loaded. Server is ready.")
    yield
    logger.info("Shutting down Salinlahi API.")


app = FastAPI(
    title="Salinlahi API",
    description="Neural machine translation between Tagalog and Waray.",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_config = registry.config.get("cors", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_config.get("allowed_origins", []),
    allow_credentials=False,
    allow_methods=_cors_config.get("allowed_methods", ["GET", "POST"]),
    allow_headers=_cors_config.get("allowed_headers", ["*"]),
)

_max_characters: int = registry.config.get("validation", {}).get("max_characters", 250)
_model_configs: dict = registry.config.get("models", {})


@app.get("/", tags=["Health"])
def health_check():
    """Returns the API status and a list of available endpoints."""
    return {
        "status": "ok",
        "endpoints": [
            "POST /api/v1/translate",
            "GET  /api/v1/languages",
            "GET  /api/v1/model/info",
        ],
    }


@app.post("/api/v1/translate", response_model=TranslationResponse, tags=["Translation"])
def translate(request: TranslationRequest):
    """
    Translates text between Tagalog and Waray.

    - text: The source sentence to translate (max 250 characters).
    - direction: Either "tgl-war" (Tagalog → Waray) or "war-tgl" (Waray → Tagalog).
    """
    text = validate_translation_request(request, max_characters=_max_characters)

    translator = registry.get_translator(request.direction)
    translation = translator.predict(text)

    if not translation:
        raise HTTPException(
            status_code=500,
            detail="The model returned an empty translation. Please try again.",
        )

    cfg = next(c for c in _model_configs.values() if c["direction"] == request.direction)

    return TranslationResponse(
        translation=translation,
        direction=request.direction,
        source_language=cfg["source_language"],
        target_language=cfg["target_language"],
        model="BaselineSeq2SeqTransformer",
    )


@app.get("/api/v1/languages", response_model=LanguagesResponse, tags=["Translation"])
def get_languages():
    """Returns the list of supported translation directions."""
    pairs = [
        LanguagePair(
            direction=cfg["direction"],
            source_language=cfg["source_language"],
            target_language=cfg["target_language"],
        )
        for cfg in _model_configs.values()
    ]
    return LanguagesResponse(supported_pairs=pairs)


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Returns the architecture details of the deployed translation model."""
    return ModelInfoResponse(
        architecture="BaselineSeq2SeqTransformer",
        num_encoder_layers=6,
        num_decoder_layers=6,
        embedding_size=512,
        attention_heads=8,
        feedforward_dim=2048,
        supported_directions=[cfg["direction"] for cfg in _model_configs.values()],
    )