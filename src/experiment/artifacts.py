"""Persist experiment logs and the JSON sidecar manifest."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.runner_log import RunnerLog

__all__ = ["write_logs", "write_manifest"]


def write_logs(pickle_path: Path, logs: list[RunnerLog]) -> None:
    with pickle_path.open("wb") as handle:
        pickle.dump(logs, handle)


def write_manifest(
    pickle_path: Path,
    config: dict[str, Any],
    models: dict[str, Any],
    datasets: dict[str, Any],
) -> Path:
    manifest_path = pickle_path.with_suffix(".json")
    payload = {
        "seed": config["seed"],
        "instances": config["instances"],
        "n_jobs": config["n_jobs"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": list(models),
        "n_datasets": len(datasets),
        "pickle": str(pickle_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
