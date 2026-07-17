"""Zenodo REST API helpers: fetch community records and update record metadata."""

import json
import re
import time
from pathlib import Path

import requests

ZENODO_API_URL = "https://zenodo.org/api"
CACHE_MAX_AGE = 7 * 24 * 3600  # one week in seconds


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove HTML tags from *text* and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(community: str) -> Path:
    return Path.home() / f".zenodo_auto_meta_{community}.json"


def _load_cache(community: str) -> list | None:
    """Return cached records if they exist and are less than one week old."""
    path = _cache_path(community)
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
        age = time.time() - data.get("timestamp", 0)
        if age < CACHE_MAX_AGE:
            return data["records"]
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def _save_cache(community: str, records: list) -> None:
    path = _cache_path(community)
    with path.open("w") as fh:
        json.dump({"timestamp": time.time(), "records": records}, fh)


# ---------------------------------------------------------------------------
# Fetching records
# ---------------------------------------------------------------------------

def fetch_community_records(
    community: str,
    *,
    force_refresh: bool = False,
    zenodo_url: str = ZENODO_API_URL,
) -> list:
    """Return all records from a Zenodo community.

    Records are downloaded in pages of 10 and accumulated.  The result is
    cached in ``~/.zenodo_auto_meta_<community>.json`` and reused for up to
    one week unless *force_refresh* is ``True``.
    """
    if not force_refresh:
        cached = _load_cache(community)
        if cached is not None:
            return cached

    records: list = []
    page = 1
    while True:
        response = requests.get(
            f"{zenodo_url}/records",
            params={"communities": community, "page": page, "size": 10},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        records.extend(hits)
        if len(hits) < 10:
            break
        page += 1

    _save_cache(community, records)
    return records


# ---------------------------------------------------------------------------
# Record field accessors
# ---------------------------------------------------------------------------

def get_tags(record: dict) -> list[str]:
    """Return the keyword list (tags) for a record, or an empty list."""
    return record.get("metadata", {}).get("keywords", []) or []


def get_description(record: dict) -> str:
    """Return the plain-text description of a record (HTML stripped)."""
    raw = record.get("metadata", {}).get("description", "") or ""
    return strip_html(raw)


def get_link(record: dict) -> str:
    """Return the HTML link to the record's landing page."""
    return record.get("links", {}).get("html", "")


# ---------------------------------------------------------------------------
# Separating records
# ---------------------------------------------------------------------------

def separate_records(records: list) -> tuple[list, list]:
    """Split *records* into (with_tags, without_tags)."""
    with_tags, without_tags = [], []
    for record in records:
        (with_tags if get_tags(record) else without_tags).append(record)
    return with_tags, without_tags


# ---------------------------------------------------------------------------
# Updating records
# ---------------------------------------------------------------------------

def _record_id_from_link(link: str) -> str:
    """Extract the numeric record ID from a Zenodo HTML record link."""
    return link.rstrip("/").split("/")[-1]


def update_record_keywords(
    link: str,
    keywords: list[str],
    token: str,
    *,
    zenodo_url: str = ZENODO_API_URL,
) -> None:
    """Add *keywords* to the Zenodo record identified by *link*.

    The record must be owned by the authenticated user (identified by *token*).
    The workflow is: unlock → update metadata → re-publish.
    """
    record_id = _record_id_from_link(link)
    headers = {"Authorization": "Bearer " + token}
    base = f"{zenodo_url}/deposit/depositions/{record_id}"

    # Unlock the published record for editing
    r = requests.post(f"{base}/actions/edit", headers=headers, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Cannot unlock record {record_id} for editing: {r.status_code} {r.text}"
        )

    # Fetch the current metadata
    r = requests.get(base, headers=headers, timeout=30)
    r.raise_for_status()
    deposition = r.json()
    metadata = deposition.get("metadata", {})

    # Merge keywords (deduplicate, preserve existing)
    existing = list(metadata.get("keywords", []) or [])
    merged = existing + [kw for kw in keywords if kw not in existing]
    metadata["keywords"] = merged

    # Write the updated metadata
    r = requests.put(base, headers=headers, json={"metadata": metadata}, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Cannot update metadata for record {record_id}: {r.status_code} {r.text}"
        )

    # Re-publish
    r = requests.post(f"{base}/actions/publish", headers=headers, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Cannot re-publish record {record_id}: {r.status_code} {r.text}"
        )
