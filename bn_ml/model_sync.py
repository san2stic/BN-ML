from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
