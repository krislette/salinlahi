import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import Annotated, Union

from app.schemas import (
    TranslationRequest,
    TranslationResponse,
    LanguagePair,
    LanguagesResponse,
    TransformerInfoResponse,
    RecurrentInfoResponse,
)
from app.utils import registry
from app.validators import validate_translation_request

logger = logging.getLogger(__name__)

_MODEL_DISPLAY_NAMES = {
    "transformer": "BaselineSeq2SeqTransformer",
    "recurrent": "Seq2SeqGRU",
}


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
_architecture_configs: dict = registry.config.get("architecture", {})

# Pre-compute the set of valid (direction, model_type) pairs from config
_supported_keys: set[tuple[str, str]] = {
    (cfg["direction"], cfg["model_type"])
    for cfg in _model_configs.values()
}

# Pre-compute supported directions per model type
_supported_directions_by_model: dict[str, list[str]] = {}
for cfg in _model_configs.values():
    _supported_directions_by_model.setdefault(cfg["model_type"], [])
    direction = cfg["direction"]
    if direction not in _supported_directions_by_model[cfg["model_type"]]:
        _supported_directions_by_model[cfg["model_type"]].append(direction)

# Maps model type to its response schema constructor
_MODEL_INFO_SCHEMAS = {
    "transformer": TransformerInfoResponse,
    "recurrent": RecurrentInfoResponse,
}


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

    - **text**: The source sentence to translate (max 250 characters).
    - **direction**: Either "tgl-war" (Tagalog → Waray) or "war-tgl" (Waray → Tagalog).
    - **model**: Either "transformer" or "recurrent".
    """
    text = validate_translation_request(
        request,
        max_characters=_max_characters,
        supported_keys=_supported_keys,
    )

    translator = registry.get_translator(request.direction, request.model)
    translation = translator.predict(text)

    if not translation:
        raise HTTPException(
            status_code=500,
            detail="The model returned an empty translation. Please try again.",
        )

    cfg = next(
        c for c in _model_configs.values()
        if c["direction"] == request.direction and c["model_type"] == request.model
    )

    return TranslationResponse(
        translation=translation,
        direction=request.direction,
        source_language=cfg["source_language"],
        target_language=cfg["target_language"],
        model=_MODEL_DISPLAY_NAMES.get(request.model, request.model),
    )


@app.get("/api/v1/languages", response_model=LanguagesResponse, tags=["Translation"])
def get_languages():
    """Returns the list of supported translation directions."""
    seen_directions = set()
    pairs = []

    for cfg in _model_configs.values():
        if cfg["direction"] not in seen_directions:
            pairs.append(
                LanguagePair(
                    direction=cfg["direction"],
                    source_language=cfg["source_language"],
                    target_language=cfg["target_language"],
                )
            )
            seen_directions.add(cfg["direction"])

    return LanguagesResponse(supported_pairs=pairs)


@app.get("/api/v1/model/info", tags=["Model"])
def get_model_info(
    model: Annotated[str, "The model architecture to query: 'transformer' or 'recurrent'"],
) -> Union[TransformerInfoResponse, RecurrentInfoResponse]:
    """
    Returns the architecture details of the specified translation model.

    - **model**: Either "transformer" or "recurrent".
    """
    if model not in _architecture_configs:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Must be one of: {list(_architecture_configs.keys())}",
        )

    arch_cfg = _architecture_configs[model]
    supported_directions = _supported_directions_by_model.get(model, [])
    schema_class = _MODEL_INFO_SCHEMAS[model]

    return schema_class(
        **arch_cfg,
        supported_directions=supported_directions,
    )