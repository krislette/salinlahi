from fastapi import HTTPException

from app.schemas import TranslationRequest


def validate_translation_request(
    request: TranslationRequest,
    max_characters: int,
    supported_keys: set[tuple[str, str]],
) -> str:
    """
    Validates and sanitizes a translation request.

    Returns the cleaned input text if valid.
    Raises HTTPException with status 400 if:
      - The input text is blank or whitespace.
      - The input text exceeds the character limit.
      - The (direction, model) combination is not supported.
    """
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text must not be empty or whitespace.",
        )

    if len(text) > max_characters:
        raise HTTPException(
            status_code=400,
            detail=f"Input text exceeds the maximum allowed length of {max_characters} characters.",
        )

    if (request.direction, request.model) not in supported_keys:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The combination of direction='{request.direction}' and "
                f"model='{request.model}' is not supported."
            ),
        )

    return text