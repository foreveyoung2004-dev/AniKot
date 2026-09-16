from __future__ import annotations

import re

ADULT_TEXT_PATTERNS = [
    r"\bхентай\b", r"\bhentai\b", r"\b18\+\b", r"\bporn(?:o)?\b", r"\bпорно\b",
    r"\beroge\b", r"\bэроге\b", r"\bsex anime\b", r"\bаниме секс\b",
]

ADULT_TAG_MARKERS = {
    "rating:explicit", "explicit", "sex", "sexual_intercourse", "penis", "pussy", "vagina",
    "nipples", "nude", "naked", "cum", "fellatio", "paizuri", "masturbation",
}

# If any of these appear with meaningful confidence, Pro+ refuses to identify the title.
# This is deliberately conservative because anime character age can be ambiguous.
MINOR_RISK_MARKERS = {
    "loli", "lolicon", "shota", "shotacon", "child", "young_girl", "young_boy",
    "underage", "minor", "preteen", "elementary_school", "middle_school",
}


def explicit_text(text: str) -> bool:
    value = (text or "").lower()
    return any(re.search(pattern, value, flags=re.I) for pattern in ADULT_TEXT_PATTERNS)


def classify_tags(
    grouped: dict[str, list[tuple[str, float]]],
    adult_threshold: float = 0.55,
    minor_threshold: float = 0.45,
) -> tuple[bool, list[str], bool, list[str]]:
    adult_hits: list[str] = []
    minor_hits: list[str] = []
    for items in grouped.values():
        for tag, score in items:
            normalized = tag.lower().replace(" ", "_")
            if score >= adult_threshold and (
                normalized in ADULT_TAG_MARKERS or any(m in normalized for m in ADULT_TAG_MARKERS)
            ):
                adult_hits.append(tag)
            if score >= minor_threshold and (
                normalized in MINOR_RISK_MARKERS or any(m in normalized for m in MINOR_RISK_MARKERS)
            ):
                minor_hits.append(tag)
    return bool(adult_hits), adult_hits[:10], bool(minor_hits), minor_hits[:10]


def explicit_tags(
    grouped: dict[str, list[tuple[str, float]]],
    threshold: float = 0.55,
) -> tuple[bool, list[str]]:
    adult, hits, _, _ = classify_tags(grouped, adult_threshold=threshold)
    return adult, hits
