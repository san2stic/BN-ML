from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

class ModelSyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitSyncSettings:
    repo_dir: Path
    models_dir: Path
    repo_models_subdir: str = "models"
    remote: str = "origin"
    branch: str = "main"
    allow_dirty_worktree: bool = False


@dataclass(frozen=True)
class RunpodSyncSettings:
    trigger_url: str
    models_dir: Path
    status_url_template: str | None = None
    trigger_method: str = "POST"
    trigger_payload: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    api_key: str | None = None
    request_timeout_sec: float = 20.0
    poll_interval_sec: float = 10.0
    job_timeout_sec: float = 3600.0
    download_timeout_sec: float = 300.0
    extract_subdir: str | None = "models"
    job_id_paths: tuple[str, ...] = ("id", "job_id", "requestId")
    status_paths: tuple[str, ...] = ("status", "state")
    download_url_paths: tuple[str, ...] = (
        "output.models_archive_url",
        "output.model_archive_url",
        "output.download_url",
        "models_archive_url",
        "model_archive_url",
        "download_url",
    )


RUNPOD_SUCCESS_STATES = {"COMPLETED", "SUCCEEDED", "SUCCESS"}
RUNPOD_FAILURE_STATES = {"FAILED", "ERROR", "CANCELED", "CANCELLED", "TIMED_OUT", "TIMEOUT"}


def parse_daily_time(value: str) -> tuple[int, int]:
    raw = str(value or "").strip()
    if ":" not in raw:
        raise ValueError(f"Invalid time format: {value!r}. Expected HH:MM")
    hour_raw, minute_raw = raw.split(":", 1)
    hour = int(hour_raw)
    minute = int(minute_raw)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time value: {value!r}. Expected HH:MM in 24h range")
    return hour, minute


def next_daily_run(now_local: datetime, hour: int, minute: int) -> datetime:
    if now_local.tzinfo is None:
        raise ValueError("now_local must be timezone-aware")
    candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now_local:
        candidate = candidate + timedelta(days=1)
    return candidate


def pull_models_from_runpod(settings: RunpodSyncSettings) -> dict:
    trigger_url = str(settings.trigger_url or "").strip()
    if not trigger_url:
        raise ModelSyncError("Missing RunPod trigger URL.")

    trigger_method = str(settings.trigger_method or "POST").upper()
    payload = dict(settings.trigger_payload or {})
    response = _http_json_request(
        method=trigger_method,
        url=trigger_url,
        headers=_build_api_headers(settings),
        payload=payload if trigger_method in {"POST", "PUT", "PATCH"} else None,
        timeout_sec=max(1.0, float(settings.request_timeout_sec)),
    )

    job_id = _as_str(_extract_first(response, settings.job_id_paths))
    status_value = _as_str(_extract_first(response, settings.status_paths))
    status_upper = status_value.upper() if status_value else ""
    download_url = _as_str(_extract_first(response, settings.download_url_paths))
    final_payload = response
    final_status = status_upper

    if not download_url:
        status_url_template = str(settings.status_url_template or "").strip()
        if not status_url_template:
            raise ModelSyncError(
                "RunPod response did not provide a model archive URL. "
                "Set model_sync.runpod.status_url_template or return download URL directly in endpoint output."
            )
        if not job_id:
            raise ModelSyncError("RunPod response missing job id.")

        deadline = time.monotonic() + max(5.0, float(settings.job_timeout_sec))
        poll_interval = max(1.0, float(settings.poll_interval_sec))
        while True:
            if time.monotonic() >= deadline:
                raise ModelSyncError(f"RunPod job {job_id} timed out after {settings.job_timeout_sec:.0f}s.")

            status_url = status_url_template.format(job_id=job_id, id=job_id)
            polled = _http_json_request(
                method="GET",
                url=status_url,
                headers=_build_api_headers(settings),
                payload=None,
                timeout_sec=max(1.0, float(settings.request_timeout_sec)),
            )
            final_payload = polled
            final_status = _as_str(_extract_first(polled, settings.status_paths)).upper()
            if final_status in RUNPOD_FAILURE_STATES:
                detail = _as_str(_extract_first(polled, ("error", "output.error", "message"))) or "unknown error"
                raise ModelSyncError(f"RunPod job {job_id} failed ({final_status}): {detail}")

            download_url = _as_str(_extract_first(polled, settings.download_url_paths))
            if final_status in RUNPOD_SUCCESS_STATES and download_url:
                break
            if download_url and not final_status:
                break

            time.sleep(poll_interval)

    archive_bytes = _http_bytes_request(
        method="GET",
        url=download_url,
        headers=_build_download_headers(settings),
        timeout_sec=max(1.0, float(settings.download_timeout_sec)),
    )
    file_count = _replace_models_from_zip(
        archive_bytes=archive_bytes,
        models_dir=settings.models_dir,
        extract_subdir=settings.extract_subdir,
    )

    return {
        "pulled": True,
        "provider": "runpod",
        "job_id": job_id,
        "status": final_status or _as_str(_extract_first(final_payload, settings.status_paths)),
        "download_url": download_url,
        "model_files": file_count,
        "models_dir": str(settings.models_dir),
    }


def publish_models_to_git(settings: GitSyncSettings, commit_message: str | None = None) -> dict:
    _ensure_repo(settings.repo_dir)
    _checkout_branch(settings.repo_dir, settings.branch)
    if not settings.allow_dirty_worktree:
        _ensure_clean_worktree(settings.repo_dir)

    _pull_ff_only(settings.repo_dir, settings.remote, settings.branch)
    repo_models_dir = settings.repo_dir / settings.repo_models_subdir
    _mirror_tree(src=settings.models_dir, dst=repo_models_dir)

    _git(settings.repo_dir, ["add", "--all", "--", settings.repo_models_subdir])
    changed_paths = _changed_paths(settings.repo_dir, scope=settings.repo_models_subdir)
    if not changed_paths:
        return {
            "committed": False,
            "pushed": False,
            "changed_paths": [],
            "branch": settings.branch,
            "remote": settings.remote,
        }

    message = commit_message or f"chore(models): sync {datetime.now(timezone.utc).isoformat()}"
    _git(settings.repo_dir, ["commit", "-m", message])
    _git(settings.repo_dir, ["push", settings.remote, settings.branch])
    return {
        "committed": True,
        "pushed": True,
        "changed_paths": changed_paths,
        "branch": settings.branch,
        "remote": settings.remote,
    }


def pull_models_from_git(settings: GitSyncSettings) -> dict:
    _ensure_repo(settings.repo_dir)
    _checkout_branch(settings.repo_dir, settings.branch)
    if not settings.allow_dirty_worktree:
        _ensure_clean_worktree(settings.repo_dir)

    _pull_ff_only(settings.repo_dir, settings.remote, settings.branch)
    repo_models_dir = settings.repo_dir / settings.repo_models_subdir
    if not repo_models_dir.exists():
        raise ModelSyncError(f"Model folder not found in repo: {repo_models_dir}")

    _mirror_tree(src=repo_models_dir, dst=settings.models_dir)
    file_count = _count_files(settings.models_dir)
    return {
        "pulled": True,
        "model_files": file_count,
        "models_dir": str(settings.models_dir),
        "branch": settings.branch,
        "remote": settings.remote,
    }


def _build_api_headers(settings: RunpodSyncSettings) -> dict[str, str]:
    headers = {str(k): str(v) for k, v in (settings.headers or {}).items()}
    headers.setdefault("Accept", "application/json")
    if settings.api_key:
        headers.setdefault("Authorization", f"Bearer {settings.api_key}")
    return headers


def _build_download_headers(settings: RunpodSyncSettings) -> dict[str, str]:
    headers = _build_api_headers(settings)
    headers.pop("Content-Type", None)
    headers["Accept"] = "application/octet-stream"
    return headers


def _http_json_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout_sec: float,
) -> dict[str, Any]:
    body: bytes | None = None
    req_headers = dict(headers)
    if payload is not None:
        req_headers.setdefault("Content-Type", "application/json")
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, data=body, headers=req_headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read()
    except error.HTTPError as exc:
        detail = _read_http_error_body(exc)
        raise ModelSyncError(f"HTTP {exc.code} on {method.upper()} {url}: {detail}") from exc
    except error.URLError as exc:
        raise ModelSyncError(f"Network error on {method.upper()} {url}: {exc.reason}") from exc

    try:
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded)
    except Exception as exc:
        raise ModelSyncError(f"Invalid JSON response from {url}") from exc
    if not isinstance(parsed, dict):
        raise ModelSyncError(f"Expected JSON object from {url}")
    return parsed


def _http_bytes_request(*, method: str, url: str, headers: dict[str, str], timeout_sec: float) -> bytes:
    req = request.Request(url=url, data=None, headers=headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read()
    except error.HTTPError as exc:
        detail = _read_http_error_body(exc)
        raise ModelSyncError(f"HTTP {exc.code} on {method.upper()} {url}: {detail}") from exc
    except error.URLError as exc:
        raise ModelSyncError(f"Network error on {method.upper()} {url}: {exc.reason}") from exc

    if not data:
        raise ModelSyncError(f"Empty archive payload from {url}")
    return data


def _read_http_error_body(exc: error.HTTPError) -> str:
    try:
        payload = exc.read()
        if payload:
            return payload.decode("utf-8", errors="replace").strip()[:500]
    except Exception:
        pass
    return str(exc.reason or "request failed")


def _extract_first(payload: Any, dotted_paths: tuple[str, ...]) -> Any:
    for dotted in dotted_paths:
        value = _extract_path(payload, dotted)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _extract_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        key = part.strip()
        if not key:
            return None
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
            continue
        if isinstance(current, list) and key.isdigit():
            idx = int(key)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _replace_models_from_zip(*, archive_bytes: bytes, models_dir: Path, extract_subdir: str | None) -> int:
    with tempfile.TemporaryDirectory(prefix="bnml_runpod_") as tmp:
        root = Path(tmp)
        extracted = root / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(archive_bytes=archive_bytes, destination=extracted)
        source = _resolve_models_root(extracted=extracted, extract_subdir=extract_subdir)
        if not source.exists():
            raise ModelSyncError(f"Archive models source not found: {source}")
        if _count_files(source) <= 0:
            raise ModelSyncError("Archive did not contain model files.")
        _mirror_tree(src=source, dst=models_dir)
    return _count_files(models_dir)


def _safe_extract_zip(*, archive_bytes: bytes, destination: Path) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            base = destination.resolve()
            for member in archive.infolist():
                normalized = member.filename.replace("\\", "/")
                if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
                    raise ModelSyncError(f"Unsafe path in archive: {member.filename!r}")
                target = (destination / normalized).resolve()
                if target != base and os.path.commonpath([str(base), str(target)]) != str(base):
                    raise ModelSyncError(f"Unsafe path in archive: {member.filename!r}")
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise ModelSyncError("Downloaded file is not a valid ZIP archive.") from exc


def _resolve_models_root(*, extracted: Path, extract_subdir: str | None) -> Path:
    preferred = str(extract_subdir or "").strip().strip("/\\")
    if preferred:
        direct = extracted / preferred
        if direct.exists() and direct.is_dir():
            return direct

    top_level = [item for item in extracted.iterdir() if item.name != "__MACOSX"]
    if len(top_level) == 1 and top_level[0].is_dir():
        single = top_level[0]
        if preferred:
            nested = single / preferred
            if nested.exists() and nested.is_dir():
                return nested
        return single

    return extracted


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ensure_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        raise ModelSyncError(f"Repository directory does not exist: {repo_dir}")
    if not (repo_dir / ".git").exists():
        raise ModelSyncError(f"Not a git repository: {repo_dir}")


def _git(repo_dir: Path, args: list[str]) -> str:
    cmd = ["git", "-C", str(repo_dir), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or "unknown git error"
        raise ModelSyncError(f"Git command failed ({' '.join(args)}): {detail}")
    return (proc.stdout or "").strip()


def _git_proc(repo_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "-C", str(repo_dir), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def _checkout_branch(repo_dir: Path, branch: str) -> None:
    current = _git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    if current == branch:
        return
    checkout = _git_proc(repo_dir, ["checkout", branch])
    if checkout.returncode == 0:
        return
    tracking_ref = f"origin/{branch}"
    checkout_tracking = _git_proc(repo_dir, ["checkout", "-b", branch, "--track", tracking_ref])
    if checkout_tracking.returncode == 0:
        return
    checkout_local = _git_proc(repo_dir, ["checkout", "-b", branch])
    if checkout_local.returncode == 0:
        return
    stderr = (checkout.stderr or checkout_tracking.stderr or checkout_local.stderr or "").strip()
    raise ModelSyncError(f"Unable to checkout branch {branch}: {stderr or 'unknown error'}")


def _pull_ff_only(repo_dir: Path, remote: str, branch: str) -> None:
    pull = _git_proc(repo_dir, ["pull", "--ff-only", remote, branch])
    if pull.returncode == 0:
        return
    ls_remote = _git_proc(repo_dir, ["ls-remote", "--heads", remote, branch])
    if ls_remote.returncode == 0 and not (ls_remote.stdout or "").strip():
        # first publish path: remote exists but target branch does not yet exist
        return
    stderr = (pull.stderr or pull.stdout or "").strip()
    raise ModelSyncError(f"git pull --ff-only failed for {remote}/{branch}: {stderr or 'unknown error'}")


def _ensure_clean_worktree(repo_dir: Path) -> None:
    dirty = _git(repo_dir, ["status", "--porcelain"])
    if dirty.strip():
        raise ModelSyncError(f"Working tree is dirty for repo {repo_dir}. Commit/stash changes first.")


def _changed_paths(repo_dir: Path, scope: str) -> list[str]:
    raw = _git(repo_dir, ["status", "--porcelain", "--", scope])
    out: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        out.append(line[3:].strip())
    return out


def _mirror_tree(src: Path, dst: Path) -> None:
    if dst.is_file():
        dst.unlink()
    if dst.exists():
        shutil.rmtree(dst)

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        return

    dst.mkdir(parents=True, exist_ok=True)


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())
