from fastapi import HTTPException

from app.schemas import TranslationRequest


def validate_translation_request(request: TranslationRequest, max_characters: int) -> str:
    """
    Validates and sanitizes a translation request.

    Returns the cleaned input text if valid.
    Raises HTTPException with status 400 if the input is blank or exceeds the character limit.
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

    return text