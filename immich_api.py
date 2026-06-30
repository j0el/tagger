"""Immich HTTP API client — tags, descriptions, asset search, thumbnails."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (skip if already set)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


@dataclass
class AssetInfo:
    id: str
    created_at: datetime
    file_name: str
    description: Optional[str]
    people_names: list[str]


@dataclass
class TagInfo:
    id: str
    name: str
    value: str
    parent_id: Optional[str]


class ImmichClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        # value -> TagInfo, populated by load_all_tags()
        self._tag_cache: dict[str, TagInfo] = {}

    # ------------------------------------------------------------------ #
    # Internal HTTP                                                         #
    # ------------------------------------------------------------------ #

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        binary: bool = False,
        retries: int = 3,
    ) -> bytes | dict | list:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers: dict[str, str] = {"x-api-key": self._api_key}
        if not binary:
            headers["Content-Type"] = "application/json"

        last_exc: Exception = RuntimeError("no attempt made")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if binary:
                        return raw
                    return json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in (429, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                body_text = exc.read().decode(errors="replace") if exc.fp else ""
                raise RuntimeError(f"HTTP {exc.code} {method} {path}: {body_text}") from exc
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        raise last_exc

    # ------------------------------------------------------------------ #
    # Tag cache                                                             #
    # ------------------------------------------------------------------ #

    def load_all_tags(self) -> None:
        """Fetch all tags from Immich and populate the local value→TagInfo cache."""
        data = self._request("GET", "/api/tags")
        assert isinstance(data, list)
        self._tag_cache.clear()
        for t in data:
            info = TagInfo(
                id=t["id"],
                name=t["name"],
                value=t["value"],
                parent_id=t.get("parentId"),
            )
            self._tag_cache[info.value] = info

    def ensure_tag(self, value: str) -> str:
        """Return the tag ID for *value*, creating the full hierarchy if needed.

        For "ai:nature/wildlife/birds" this guarantees:
          ai:nature → ai:nature/wildlife → ai:nature/wildlife/birds
        all exist as Immich tags before returning the leaf ID.
        """
        if value in self._tag_cache:
            return self._tag_cache[value].id

        parts = value.split("/")
        parent_id: Optional[str] = None

        for depth in range(1, len(parts) + 1):
            ancestor_value = "/".join(parts[:depth])

            if ancestor_value in self._tag_cache:
                parent_id = self._tag_cache[ancestor_value].id
                continue

            leaf_name = parts[depth - 1]
            body: dict[str, str] = {"name": leaf_name}
            if parent_id is not None:
                body["parentId"] = parent_id

            result = self._request("POST", "/api/tags", body=body)
            assert isinstance(result, dict)
            info = TagInfo(
                id=result["id"],
                name=result["name"],
                value=result["value"],
                parent_id=result.get("parentId"),
            )
            self._tag_cache[info.value] = info
            parent_id = info.id

        return self._tag_cache[value].id

    # ------------------------------------------------------------------ #
    # Asset search                                                          #
    # ------------------------------------------------------------------ #

    def find_new_assets(
        self,
        since: Optional[datetime] = None,
        page_size: int = 100,
    ) -> Iterator[AssetInfo]:
        """Yield all assets (ordered by createdAt asc), optionally filtered to after *since*."""
        body: dict = {
            "size": page_size,
            "withExif": True,
            "withPeople": True,
            "order": "asc",
        }
        if since is not None:
            body["createdAfter"] = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        page = 1
        while True:
            body["page"] = page
            resp = self._request("POST", "/api/search/metadata", body=body)
            assert isinstance(resp, dict)
            block = resp.get("assets", {})
            for item in block.get("items", []):
                yield self._parse_asset(item)
            if not block.get("hasNextPage", False):
                break
            page += 1

    def find_all_assets(self, page_size: int = 100) -> Iterator[AssetInfo]:
        return self.find_new_assets(since=None, page_size=page_size)

    @staticmethod
    def _parse_asset(item: dict) -> AssetInfo:
        try:
            created_at = datetime.fromisoformat(
                item.get("createdAt", "").replace("Z", "+00:00")
            )
        except Exception:
            created_at = datetime.now(timezone.utc)

        description: Optional[str] = None
        if exif := item.get("exifInfo"):
            description = exif.get("description") or None

        people_names = [p["name"] for p in item.get("people", []) if p.get("name")]

        return AssetInfo(
            id=item["id"],
            created_at=created_at,
            file_name=item.get("originalFileName", ""),
            description=description,
            people_names=people_names,
        )

    # ------------------------------------------------------------------ #
    # Asset mutations                                                       #
    # ------------------------------------------------------------------ #

    def get_thumbnail(self, asset_id: str) -> bytes:
        data = self._request(
            "GET",
            f"/api/assets/{asset_id}/thumbnail?size=preview",
            binary=True,
        )
        assert isinstance(data, bytes)
        return data

    def get_asset_ai_tag_ids(self, asset_id: str) -> list[str]:
        """Return IDs of all current tags whose value starts with 'ai:'."""
        data = self._request("GET", f"/api/assets/{asset_id}")
        assert isinstance(data, dict)
        return [t["id"] for t in data.get("tags", []) if t.get("value", "").startswith("ai:")]

    def assign_tag_to_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        if asset_ids:
            self._request("PUT", f"/api/tags/{tag_id}/assets", body={"ids": asset_ids})

    def remove_tag_from_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        if asset_ids:
            self._request("DELETE", f"/api/tags/{tag_id}/assets", body={"ids": asset_ids})

    def update_description(self, asset_id: str, description: str) -> None:
        self._request("PUT", f"/api/assets/{asset_id}", body={"description": description})
