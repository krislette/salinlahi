"""
Rules that we have set for preprocessing Tagalog and Waray:

[1] Remove from dataset:
. (period)
, (comma)
! (exclamation)
? (question)
; (semicolon)
: (colon)
" (double quote)
– (en dash)
— (em dash)
( ) [ ] { } (paranthesis, brackets, curly braces)

[2] Keep on the dataset:
' (single quote)
’ (apostrophe)
- (hyphen)
"""

import re


def preprocess_text(text: str) -> str:
    """
    Preprocess text to match training data rules.

    Remove: . , ! ? ; : " – — ( ) [ ] { }
    Keep:   ' (apostrophe) - (hyphen)
    """
    # Characters to remove
    chars_to_remove = r"[\.\,\!\?\;\:\"\–\—\(\)\[\]\{\}]"

    # Remove the specified characters
    text = re.sub(chars_to_remove, "", text)

    return text.strip()
