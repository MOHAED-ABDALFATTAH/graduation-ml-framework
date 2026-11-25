import math

import mlflow
import pytest

from src.tracking import (
    flatten_dict, log_metrics_safe, setup_tracking, start_tracked_run,
)


@pytest.fixture()
def local_mlflow(tmp_path, monkeypatch):
    # Newer MLflow versions require a database backend; use a throwaway SQLite file.
    monkeypatch.chdir(tmp_path)  # artifacts land in ./mlruns inside tmp_path
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    setup_tracking("Test_Experiment", tracking_uri=uri)
    yield uri


def test_flatten_dict():
    assert flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {
        "a.b": 1, "a.c.d": 2, "e": 3}


def test_successful_run_has_standard_tags(local_mlflow):
    with start_tracked_run("run1", owner="member_a", track="test",
                           dataset_version="v1", seed=42,
                           config={"training": {"lr": 0.001}}) as run:
        log_metrics_safe({"loss": 0.5, "bad": float("nan"), "none": None})
    r = mlflow.get_run(run.info.run_id)
    assert r.data.tags["owner"] == "member_a"
    assert r.data.tags["track"] == "test"
    assert r.data.tags["dataset_version"] == "v1"
    assert r.data.tags["seed"] == "42"
    assert r.data.tags["status"] == "finished"
    assert "git_commit" in r.data.tags
    assert r.data.params["training.lr"] == "0.001"
    assert r.data.metrics["loss"] == 0.5
    assert "bad" not in r.data.metrics and "none" not in r.data.metrics


def test_failed_run_is_kept_and_tagged(local_mlflow):
    run_id = None
    with pytest.raises(RuntimeError):
        with start_tracked_run("run2", owner="member_b", track="test",
                               dataset_version="v1", seed=1) as run:
            run_id = run.info.run_id
            raise RuntimeError("boom")
    r = mlflow.get_run(run_id)
    assert r.data.tags["status"] == "failed"
    assert r.data.tags["error_type"] == "RuntimeError"
    assert r.data.tags["error_message"] == "boom"
    client = mlflow.tracking.MlflowClient()
    names = [a.path for a in client.list_artifacts(run_id)]
    assert "error_traceback.txt" in names


def test_invalid_track_rejected(local_mlflow):
    with pytest.raises(ValueError):
        with start_tracked_run("x", owner="a", track="nonsense",
                               dataset_version="v1", seed=1):
            pass


def test_missing_uri_raises(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError):
        setup_tracking("Whatever")
