from __future__ import annotations

import html
import re
from typing import Any

import httpx

QUERY = """
query ($search: String) {
  Media(search: $search, type: ANIME) {
    id
    title { romaji english native }
    format status episodes seasonYear averageScore genres description(asHtml: false) siteUrl
    isAdult
  }
}
"""

class AniListClient:
    def __init__(self, timeout: float = 20):
        self.timeout = timeout

    async def search(self, title: str) -> dict[str, Any] | None:
        if not title.strip():
            return None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post("https://graphql.anilist.co", json={"query": QUERY, "variables": {"search": title}})
            if response.status_code >= 400:
                return None
            return response.json().get("data", {}).get("Media")

    @staticmethod
    def clean_description(text: str | None, limit: int = 500) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit] + ("…" if len(text) > limit else "")
