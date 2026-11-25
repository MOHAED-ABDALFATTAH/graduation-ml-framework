from pathlib import Path

from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_default_config_composes():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
    assert cfg.experiment.track == "detection"
    assert cfg.dataset.annotation_format == "yolo"
    assert len(cfg.dataset.class_names) >= 1


def test_model_override():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["model=faster_rcnn"])
    assert cfg.model.name == "faster_rcnn"


def test_experiment_preset():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["+experiment=baseline"])
    assert cfg.experiment.name == "baseline"
