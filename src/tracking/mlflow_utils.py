"""MLflow helpers shared by the detection, classification and pipeline tracks.

Credentials are read from environment variables (never hard-code them):
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
"""
from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import mlflow

VALID_TRACKS = {"detection", "classification", "pipeline", "test"}


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict (e.g. a Hydra config) into 'a.b.c' keys for mlflow.log_params."""
    items: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def setup_tracking(experiment_name: str, tracking_uri: Optional[str] = None) -> None:
    """Point MLflow at the shared server and select the experiment.

    experiment_name examples: 'Detection', 'Classification', 'Pipeline'.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Load your Colab secrets first "
            "(see docs/getting_started.md)."
        )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def log_metrics_safe(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log only finite numeric metrics (skips None / NaN / inf instead of crashing)."""
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            clean[k] = f
    if clean:
        mlflow.log_metrics(clean, step=step)


@contextmanager
def start_tracked_run(
    run_name: str,
    owner: str,
    track: str,
    dataset_version: str,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, str]] = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run with the team's standard tags and failure logging.

    On success the run gets tag status=finished. If your code raises, the run
    is kept, tagged status=failed, the full traceback is uploaded as an
    artifact, and the exception is re-raised.

        with start_tracked_run("DET-2026-001_YOLOv8_Baseline", owner="member_a",
                               track="detection", dataset_version="v1",
                               seed=42, config=cfg_dict) as run:
            ...
    """
    if track not in VALID_TRACKS:
        raise ValueError(f"track must be one of {sorted(VALID_TRACKS)}, got '{track}'")

    with mlflow.start_run(run_name=run_name) as run:
        tags = {
            "owner": owner,
            "track": track,
            "dataset_version": dataset_version,
            "seed": str(seed),
            "git_commit": get_git_commit(),
            "python_version": platform.python_version(),
            "status": "running",
        }
        if extra_tags:
            tags.update({k: str(v) for k, v in extra_tags.items()})
        mlflow.set_tags(tags)

        if config:
            flat = flatten_dict(config)
            # MLflow params are limited in length; stringify and truncate safely
            mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})

        try:
            yield run
        except BaseException as exc:  # includes KeyboardInterrupt (manual stop in Colab)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error_type", type(exc).__name__)
            mlflow.set_tag("error_message", str(exc)[:250])
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "error_traceback.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("".join(traceback.format_exception(*sys.exc_info())))
                mlflow.log_artifact(path)
            raise
        else:
            mlflow.set_tag("status", "finished")
