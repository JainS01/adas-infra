"""Capture a snapshot of the execution environment for the run manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess  # nosec B404
import sys
from pathlib import Path


def git_sha(repo_root: Path | None = None) -> str:
    """Return the current HEAD SHA, or 'no-git' if git is unavailable."""
    try:
        result = subprocess.run(  # nosec
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root or Path.cwd(),
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "no-git"


def hydra_config_hash(cfg_dict: dict) -> str:  # type: ignore[type-arg]
    """Return a SHA-256 of the serialised Hydra config (deterministic ordering)."""
    serialised = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]


def package_versions(packages: list[str] | None = None) -> dict[str, str]:
    """Return installed versions of key packages."""
    targets = packages or ["torch", "ray", "pyarrow", "pydantic", "hydra-core"]
    versions: dict[str, str] = {}
    for pkg in targets:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def platform_info() -> dict[str, str]:
    """Return a dict of platform metadata included in env snapshots."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
    }
