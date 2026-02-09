from __future__ import annotations

import io
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import bn_ml.model_sync as model_sync_module
from bn_ml.model_sync import (
    GitSyncSettings,
    ModelSyncError,
    RunpodSyncSettings,
    next_daily_run,
    parse_daily_time,
    pull_models_from_runpod,
    publish_models_to_git,
    pull_models_from_git,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def _git_init_remote_and_clones(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)

    publisher = tmp_path / "publisher-repo"
    subprocess.run(["git", "clone", str(remote), str(publisher)], check=True, capture_output=True, text=True)
    _git(publisher, "config", "user.email", "bot@example.com")
    _git(publisher, "config", "user.name", "BN-ML Bot")
    _git(publisher, "checkout", "-b", "main")
    (publisher / "README.md").write_text("models repo\n", encoding="utf-8")
    _git(publisher, "add", "README.md")
    _git(publisher, "commit", "-m", "init")
    _git(publisher, "push", "-u", "origin", "main")

    client = tmp_path / "client-repo"
    subprocess.run(
        ["git", "clone", "--branch", "main", str(remote), str(client)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(client, "config", "user.email", "client@example.com")
    _git(client, "config", "user.name", "BN-ML Client")
    return publisher, client


def _zip_models_archive(prefix: str = "models") -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{prefix}/BTC_USDC/rf.joblib", "rf-v1")
        bundle.writestr(f"{prefix}/BTC_USDC/metadata.json", '{"symbol":"BTC/USDC"}')
    return payload.getvalue()


def test_parse_daily_time_and_next_run() -> None:
    hour, minute = parse_daily_time("06:00")
    assert (hour, minute) == (6, 0)

    now = datetime(2026, 2, 9, 5, 30, tzinfo=timezone.utc)
    nxt = next_daily_run(now, hour=6, minute=0)
    assert nxt == datetime(2026, 2, 9, 6, 0, tzinfo=timezone.utc)

    now_late = datetime(2026, 2, 9, 6, 1, tzinfo=timezone.utc)
    nxt_late = next_daily_run(now_late, hour=6, minute=0)
    assert nxt_late == datetime(2026, 2, 10, 6, 0, tzinfo=timezone.utc)


def test_parse_daily_time_invalid() -> None:
    with pytest.raises(ValueError):
        parse_daily_time("nope")
    with pytest.raises(ValueError):
        parse_daily_time("24:00")
    with pytest.raises(ValueError):
        parse_daily_time("12:60")


def test_publish_then_pull_models_roundtrip(tmp_path: Path) -> None:
    publisher_repo, client_repo = _git_init_remote_and_clones(tmp_path)

    producer_models = tmp_path / "producer-models"
    (producer_models / "BTC_USDC").mkdir(parents=True, exist_ok=True)
    (producer_models / "BTC_USDC" / "rf.joblib").write_text("v1", encoding="utf-8")
    (producer_models / "BTC_USDC" / "metadata.json").write_text('{"trained_at":"2026-02-09T00:00:00+00:00"}', encoding="utf-8")

    publish_settings = GitSyncSettings(
        repo_dir=publisher_repo,
        models_dir=producer_models,
        repo_models_subdir="models",
        remote="origin",
        branch="main",
    )
    publish_result = publish_models_to_git(settings=publish_settings, commit_message="sync models")
    assert publish_result["committed"] is True
    assert publish_result["pushed"] is True

    second_publish = publish_models_to_git(settings=publish_settings, commit_message="sync models again")
    assert second_publish["committed"] is False
    assert second_publish["pushed"] is False

    consumer_models = tmp_path / "consumer-models"
    pull_settings = GitSyncSettings(
        repo_dir=client_repo,
        models_dir=consumer_models,
        repo_models_subdir="models",
        remote="origin",
        branch="main",
    )
    pull_result = pull_models_from_git(settings=pull_settings)
    assert pull_result["pulled"] is True
    assert (consumer_models / "BTC_USDC" / "rf.joblib").read_text(encoding="utf-8") == "v1"


def test_publish_refuses_dirty_worktree(tmp_path: Path) -> None:
    publisher_repo, _ = _git_init_remote_and_clones(tmp_path)

    producer_models = tmp_path / "producer-models"
    (producer_models / "BTC_USDC").mkdir(parents=True, exist_ok=True)
    (producer_models / "BTC_USDC" / "rf.joblib").write_text("v1", encoding="utf-8")

    (publisher_repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    settings = GitSyncSettings(
        repo_dir=publisher_repo,
        models_dir=producer_models,
        repo_models_subdir="models",
        remote="origin",
        branch="main",
        allow_dirty_worktree=False,
    )
    with pytest.raises(ModelSyncError):
        publish_models_to_git(settings=settings)


def test_next_daily_run_requires_timezone() -> None:
    now_naive = datetime.now() + timedelta(seconds=1)
    with pytest.raises(ValueError):
        next_daily_run(now_naive, hour=0, minute=0)


def test_pull_models_from_runpod_direct_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models"
    settings = RunpodSyncSettings(
        trigger_url="https://runpod.example/api/run",
        models_dir=models_dir,
        trigger_payload={"symbols": ["BTC/USDC"]},
    )

    def fake_json_request(*, method, url, headers, payload, timeout_sec):  # type: ignore[no-untyped-def]
        assert method == "POST"
        assert url == "https://runpod.example/api/run"
        assert payload == {"symbols": ["BTC/USDC"]}
        assert float(timeout_sec) > 0
        assert "Accept" in headers
        return {
            "status": "COMPLETED",
            "output": {"models_archive_url": "https://cdn.example/models.zip"},
        }

    def fake_bytes_request(*, method, url, headers, timeout_sec):  # type: ignore[no-untyped-def]
        assert method == "GET"
        assert url == "https://cdn.example/models.zip"
        assert float(timeout_sec) > 0
        assert "Accept" in headers
        return _zip_models_archive(prefix="models")

    monkeypatch.setattr(model_sync_module, "_http_json_request", fake_json_request)
    monkeypatch.setattr(model_sync_module, "_http_bytes_request", fake_bytes_request)

    result = pull_models_from_runpod(settings=settings)
    assert result["pulled"] is True
    assert result["provider"] == "runpod"
    assert int(result["model_files"]) >= 2
    assert (models_dir / "BTC_USDC" / "rf.joblib").read_text(encoding="utf-8") == "rf-v1"


def test_pull_models_from_runpod_polling_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "models"
    settings = RunpodSyncSettings(
        trigger_url="https://runpod.example/api/run",
        status_url_template="https://runpod.example/api/status/{job_id}",
        models_dir=models_dir,
        poll_interval_sec=1.0,
        job_timeout_sec=30.0,
    )

    calls: list[tuple[str, str]] = []
    responses = iter(
        [
            {"id": "job-123", "status": "IN_PROGRESS"},
            {"id": "job-123", "status": "IN_PROGRESS"},
            {"id": "job-123", "status": "COMPLETED", "output": {"models_archive_url": "https://cdn.example/archive.zip"}},
        ]
    )

    def fake_json_request(*, method, url, headers, payload, timeout_sec):  # type: ignore[no-untyped-def]
        calls.append((method, url))
        assert float(timeout_sec) > 0
        return next(responses)

    def fake_bytes_request(*, method, url, headers, timeout_sec):  # type: ignore[no-untyped-def]
        assert method == "GET"
        assert url == "https://cdn.example/archive.zip"
        return _zip_models_archive(prefix="models")

    monkeypatch.setattr(model_sync_module, "_http_json_request", fake_json_request)
    monkeypatch.setattr(model_sync_module, "_http_bytes_request", fake_bytes_request)
    monkeypatch.setattr(model_sync_module.time, "sleep", lambda _: None)

    result = pull_models_from_runpod(settings=settings)
    assert result["pulled"] is True
    assert result["job_id"] == "job-123"
    assert result["status"] == "COMPLETED"
    assert calls[0] == ("POST", "https://runpod.example/api/run")
    assert calls[1] == ("GET", "https://runpod.example/api/status/job-123")


def test_pull_models_from_runpod_failed_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RunpodSyncSettings(
        trigger_url="https://runpod.example/api/run",
        status_url_template="https://runpod.example/api/status/{job_id}",
        models_dir=tmp_path / "models",
    )

    responses = iter(
        [
            {"id": "job-999", "status": "IN_PROGRESS"},
            {"id": "job-999", "status": "FAILED", "error": "gpu oom"},
        ]
    )

    def fake_json_request(*, method, url, headers, payload, timeout_sec):  # type: ignore[no-untyped-def]
        return next(responses)

    def fake_bytes_request(*, method, url, headers, timeout_sec):  # type: ignore[no-untyped-def]
        raise AssertionError("download must not be called on failed job")

    monkeypatch.setattr(model_sync_module, "_http_json_request", fake_json_request)
    monkeypatch.setattr(model_sync_module, "_http_bytes_request", fake_bytes_request)
    monkeypatch.setattr(model_sync_module.time, "sleep", lambda _: None)

    with pytest.raises(ModelSyncError, match="failed"):
        pull_models_from_runpod(settings=settings)


def test_pull_models_from_runpod_rejects_unsafe_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RunpodSyncSettings(
        trigger_url="https://runpod.example/api/run",
        models_dir=tmp_path / "models",
    )

    def fake_json_request(*, method, url, headers, payload, timeout_sec):  # type: ignore[no-untyped-def]
        return {"status": "COMPLETED", "output": {"models_archive_url": "https://cdn.example/bad.zip"}}

    def fake_bytes_request(*, method, url, headers, timeout_sec):  # type: ignore[no-untyped-def]
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("../escape.txt", "bad")
        return payload.getvalue()

    monkeypatch.setattr(model_sync_module, "_http_json_request", fake_json_request)
    monkeypatch.setattr(model_sync_module, "_http_bytes_request", fake_bytes_request)

    with pytest.raises(ModelSyncError, match="Unsafe path"):
        pull_models_from_runpod(settings=settings)
