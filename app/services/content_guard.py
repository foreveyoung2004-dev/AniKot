from __future__ import annotations

import re

ADULT_TEXT_PATTERNS = [
    r"\bхентай\b",
    r"\bhentai\b",
    r"\b18\+\b",
    r"\bporn(?:o)?\b",
    r"\bпорно\b",
    r"\beroge\b",
    r"\bэроге\b",
    r"\bsex anime\b",
    r"\bаниме секс\b",
]


def explicit_text(text: str) -> bool:
    value = (text or "").lower()
    return any(re.search(pattern, value, flags=re.I) for pattern in ADULT_TEXT_PATTERNS)
