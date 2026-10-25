"""Model files from the local cache first, the Hub only for what is missing."""

import sys
import types

import pytest

from growth_rag.config import BGE_M3_REVISION, EmbeddingSettings, Settings
from growth_rag.utils.hf import ModelFilesUnavailable, local_snapshot

REQUIRED = ["config.json", ("model.safetensors", "pytorch_model.bin")]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A fake huggingface_hub: `cache` is the snapshot folder; `online` says
    whether a real download would succeed. Records every call."""
    cache = tmp_path / "snapshots" / "abc"
    state = types.SimpleNamespace(calls=[], cached=True, online=True, cache=cache)

    def snapshot_download(repo, revision=None, allow_patterns=None, local_files_only=False):
        state.calls.append("cache" if local_files_only else "download")
        if local_files_only:
            if not state.cached:
                raise FileNotFoundError("not cached")  # LocalEntryNotFoundError is one
            return str(cache)
        if not state.online:
            raise OSError("offline")
        cache.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            (cache / name).write_text("x")
        return str(cache)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return state


def test_complete_cache_means_no_download(hub):
    hub.cache.mkdir(parents=True)
    (hub.cache / "config.json").write_text("{}")
    (hub.cache / "pytorch_model.bin").write_text("w")  # either weights file will do
    assert local_snapshot("BAAI/bge-m3", "abc", ["config.json"], REQUIRED) == hub.cache
    assert hub.calls == ["cache"]


def test_missing_files_are_downloaded(hub):
    hub.cache.mkdir(parents=True)
    (hub.cache / "config.json").write_text("{}")  # weights missing
    local_snapshot("BAAI/bge-m3", "abc", ["config.json", "pytorch_model.bin"], REQUIRED)
    assert hub.calls == ["cache", "download"]


def test_not_cached_then_downloaded(hub):
    hub.cached = False
    local_snapshot("BAAI/bge-m3", "abc", ["config.json", "model.safetensors"], REQUIRED)
    assert hub.calls == ["cache", "download"]


def test_not_cached_and_offline_is_a_clear_error(hub):
    hub.cached, hub.online = False, False
    with pytest.raises(ModelFilesUnavailable, match="not in the local cache.*HF_HUB_OFFLINE"):
        local_snapshot("BAAI/bge-m3", "abc", ["config.json"], REQUIRED)


def test_a_local_folder_is_used_as_is(hub, tmp_path):
    folder = tmp_path / "my-model"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    with pytest.raises(ModelFilesUnavailable, match="missing model.safetensors or pytorch_model.bin"):
        local_snapshot(str(folder), None, [], REQUIRED)
    (folder / "model.safetensors").write_text("w")
    assert local_snapshot(str(folder), None, [], REQUIRED) == folder
    assert hub.calls == []


def test_revision_is_pinned_and_can_be_overridden(monkeypatch):
    assert EmbeddingSettings().revision == BGE_M3_REVISION
    monkeypatch.setenv("GROWTH_RAG_EMBED_REVISION", "main")
    assert Settings.from_env().embedding.revision == "main"
    monkeypatch.setenv("GROWTH_RAG_EMBED_REVISION", "")
    assert Settings.from_env().embedding.revision is None
