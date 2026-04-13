from __future__ import annotations

import re

COMMON_PLATE_PATTERNS = (
    r"^[A-Z]{2,3}\d{3,4}[A-Z]{0,2}$",
    r"^[A-Z]{1,3}\d{1,4}[A-Z]{1,3}$",
    r"^\d{1,4}[A-Z]{1,3}\d{1,4}$",
)


def clean_plate_text(raw_text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text.upper())
    return cleaned.strip()


def _matches_common_plate_patterns(text: str) -> bool:
    return any(re.fullmatch(pattern, text) for pattern in COMMON_PLATE_PATTERNS)


def is_plausible_plate(text: str, min_length: int = 5, max_length: int = 10) -> bool:
    if not min_length <= len(text) <= max_length:
        return False
    if not any(char.isdigit() for char in text) or not any(char.isalpha() for char in text):
        return False
    if len(set(text)) <= 2 and len(text) >= 6:
        return False

    if _matches_common_plate_patterns(text):
        return True

    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    return 1 <= letters <= 5 and 1 <= digits <= 6
