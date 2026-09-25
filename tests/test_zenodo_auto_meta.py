"""Unit tests for zenodo-auto-meta."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from zenodo_auto_meta.zenodo import (
    _cache_path,
    _load_cache,
    _save_cache,
    fetch_community_records,
    get_description,
    get_link,
    get_tags,
    get_title,
    separate_records,
    strip_html,
    update_record_keywords,
)
from zenodo_auto_meta.llm import build_messages, predict_tags
from zenodo_auto_meta.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD_WITH_TAGS = {
    "metadata": {
        "keywords": ["python", "bioimaging"],
        "title": "A great bioimaging tool",
        "description": "<p>A great tool for bioimaging.</p>",
    },
    "doi_url": "https://zenodo.org/record/111",
}

SAMPLE_RECORD_WITHOUT_TAGS = {
    "metadata": {
        "keywords": [],
        "description": "<p>Another record without tags.</p>",
    },
    "doi_url": "https://zenodo.org/record/222",
}

SAMPLE_RECORD_NO_DESC = {
    "metadata": {"keywords": [], "description": ""},
    "doi_url": "https://zenodo.org/record/333",
}


# ---------------------------------------------------------------------------
# zenodo.strip_html
# ---------------------------------------------------------------------------

def test_strip_html_basic():
    assert strip_html("<p>Hello <b>world</b>!</p>") == "Hello world !"


def test_strip_html_empty():
    assert strip_html("") == ""


def test_strip_html_none():
    assert strip_html(None) == ""


# ---------------------------------------------------------------------------
# zenodo record accessors
# ---------------------------------------------------------------------------

def test_get_tags_with_keywords():
    assert get_tags(SAMPLE_RECORD_WITH_TAGS) == ["python", "bioimaging"]


def test_get_tags_empty():
    assert get_tags(SAMPLE_RECORD_WITHOUT_TAGS) == []


def test_get_description_strips_html():
    assert get_description(SAMPLE_RECORD_WITH_TAGS) == "A great tool for bioimaging."


def test_get_link():
    assert get_link(SAMPLE_RECORD_WITH_TAGS) == "https://zenodo.org/record/111"


def test_get_title():
    assert get_title(SAMPLE_RECORD_WITH_TAGS) == "A great bioimaging tool"


# ---------------------------------------------------------------------------
# zenodo.separate_records
# ---------------------------------------------------------------------------

def test_separate_records():
    records = [SAMPLE_RECORD_WITH_TAGS, SAMPLE_RECORD_WITHOUT_TAGS, SAMPLE_RECORD_NO_DESC]
    with_tags, without_tags = separate_records(records)
    assert len(with_tags) == 1
    assert len(without_tags) == 2


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def test_save_and_load_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    community = "test-community"
    records = [SAMPLE_RECORD_WITH_TAGS]
    _save_cache(community, records)
    loaded = _load_cache(community)
    assert loaded is not None
    assert loaded == records


def test_load_cache_expired(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    community = "test-community"
    # Write a cache that is already expired (timestamp = 0)
    cache_file = _cache_path(community)
    cache_file.write_text(
        yaml.safe_dump({"timestamp": 0, "records": [SAMPLE_RECORD_WITH_TAGS]})
    )
    assert _load_cache(community) is None


def test_load_cache_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _load_cache("nonexistent-community") is None


# ---------------------------------------------------------------------------
# fetch_community_records (mocked HTTP)
# ---------------------------------------------------------------------------

def _make_hits(n: int) -> dict:
    hits = [{"metadata": {"keywords": ["tag"], "description": f"desc{i}"}, "links": {"html": f"https://zenodo.org/record/{i}"}} for i in range(n)]
    return {"hits": {"hits": hits, "total": n}}


def test_fetch_returns_all_pages(tmp_path, monkeypatch):
    monkeypatch.setattr("zenodo_auto_meta.zenodo.Path.home", lambda: tmp_path)
    responses = [_make_hits(10), _make_hits(5)]
    call_count = [0]

    def mock_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = responses[call_count[0]]
        call_count[0] += 1
        return resp

    monkeypatch.setattr("zenodo_auto_meta.zenodo.requests.get", mock_get)
    records = fetch_community_records("mycomm", force_refresh=True)
    assert len(records) == 15
    assert call_count[0] == 2


def test_fetch_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("zenodo_auto_meta.zenodo.Path.home", lambda: tmp_path)
    _save_cache("mycomm", [SAMPLE_RECORD_WITH_TAGS])
    get_mock = MagicMock()
    monkeypatch.setattr("zenodo_auto_meta.zenodo.requests.get", get_mock)
    records = fetch_community_records("mycomm", force_refresh=False)
    get_mock.assert_not_called()
    assert records == [SAMPLE_RECORD_WITH_TAGS]


# ---------------------------------------------------------------------------
# llm.build_messages
# ---------------------------------------------------------------------------

def test_build_messages_structure():
    examples = [("desc A", ["tag1", "tag2"]), ("desc B", ["tag3"])]
    messages = build_messages("new description", examples)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "tag1, tag2"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "new description"


def test_build_messages_max_examples():
    examples = [(f"desc {i}", [f"tag{i}"]) for i in range(30)]
    messages = build_messages("new", examples, max_examples=5)
    # system + 5 pairs (user+assistant) + final user = 12
    assert len(messages) == 12


def test_build_messages_truncates_long_description():
    long_desc = "a" * 3000
    messages = build_messages(long_desc, [])
    assert len(messages[-1]["content"]) < 3000


# ---------------------------------------------------------------------------
# llm.predict_tags (mocked OpenAI client)
# ---------------------------------------------------------------------------

def test_predict_tags(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "bioimaging, python, microscopy"
    mock_client.chat.completions.create.return_value = mock_response

    monkeypatch.setattr("zenodo_auto_meta.llm.OpenAI", lambda **kw: mock_client)
    tags = predict_tags("A description", [])
    assert tags == ["bioimaging", "python", "microscopy"]


# ---------------------------------------------------------------------------
# CLI integration (mocked)
# ---------------------------------------------------------------------------

def test_cli_predict_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("zenodo_auto_meta.zenodo.Path.home", lambda: tmp_path)
    records = [SAMPLE_RECORD_WITH_TAGS, SAMPLE_RECORD_WITHOUT_TAGS]
    monkeypatch.setattr("zenodo_auto_meta.cli.fetch_community_records", lambda *a, **kw: records)

    mock_predict = MagicMock(return_value=["bio", "imaging"])
    monkeypatch.setattr("zenodo_auto_meta.cli.predict_tags", mock_predict)

    output = tmp_path / "out.yml"
    ret = main(["mycomm", "--output", str(output)])
    assert ret == 0
    data = yaml.safe_load(output.read_text())
    assert len(data) == 1
    assert data[0]["title"] == "A great bioimaging tool"
    assert data[0]["proposed_tags"] == ["bio", "imaging"]


def test_cli_update_mode_missing_token(tmp_path):
    yml_file = tmp_path / "curated.yml"
    yml_file.write_text(yaml.safe_dump([{"link": "https://zenodo.org/record/1", "proposed_tags": ["x"]}]))
    ret = main(["mycomm", "--update", str(yml_file)])
    assert ret == 1


def test_cli_update_mode(tmp_path, monkeypatch):
    entry = {"link": "https://zenodo.org/record/42", "proposed_tags": ["bio"]}
    yml_file = tmp_path / "curated.yml"
    yml_file.write_text(yaml.safe_dump([entry]))

    mock_update = MagicMock()
    monkeypatch.setattr("zenodo_auto_meta.cli.update_record_keywords", mock_update)

    ret = main(["mycomm", "--update", str(yml_file), "--zenodo-token", "mytoken"])
    assert ret == 0
    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args
    assert call_kwargs[0][0] == "https://zenodo.org/record/42"
    assert call_kwargs[0][1] == ["bio"]
    assert call_kwargs[0][2] == "mytoken"
