import pandas as pd
import pytest

from src.data import assign_splits, build_manifest, validate_dataset

CLASSES = ["aphid", "whitefly"]


def make_dataset(tmp_path, n_groups=10, per_group=5):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    rows = []
    for g in range(n_groups):
        for i in range(per_group):
            name = f"g{g}_{i}.jpg"
            (tmp_path / "images" / name).write_bytes(f"img-{g}-{i}".encode())
            (tmp_path / "labels" / f"g{g}_{i}.txt").write_text(
                f"{(g + i) % 2} 0.5 0.5 0.2 0.2\n"
            )
            rows.append({"filename": name, "group_id": f"plant{g}"})
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path


def test_split_is_deterministic_and_group_aware():
    groups = {f"g{i}": 5 for i in range(20)}
    a = assign_splits(groups, seed=42)
    b = assign_splits(dict(reversed(list(groups.items()))), seed=42)
    assert a == b  # independent of input order
    assert set(a.values()) == {"train", "val", "test"}
    assert assign_splits(groups, seed=1) != a  # seed matters


def test_split_fractions_roughly_respected():
    groups = {f"g{i}": 5 for i in range(40)}
    s = assign_splits(groups, (0.7, 0.15, 0.15), seed=0)
    sizes = {k: sum(5 for g in s if s[g] == k) for k in ("train", "val", "test")}
    total = sum(sizes.values())
    assert 0.6 <= sizes["train"] / total <= 0.8
    assert sizes["val"] > 0 and sizes["test"] > 0


def test_no_group_in_two_splits(tmp_path):
    ds = make_dataset(tmp_path)
    m = build_manifest(ds, ds / "metadata.csv", "v1", seed=42)
    assert (m.groupby("group_id")["split"].nunique() == 1).all()
    assert {"checksum", "num_boxes", "split", "dataset_version"} <= set(m.columns)


def test_valid_dataset_has_no_errors(tmp_path):
    ds = make_dataset(tmp_path)
    m = build_manifest(ds, ds / "metadata.csv", "v1", seed=42)
    rep = validate_dataset(ds, m, CLASSES)
    assert rep["errors"] == []


def test_detects_bad_class_and_out_of_range_box(tmp_path):
    ds = make_dataset(tmp_path)
    m = build_manifest(ds, ds / "metadata.csv", "v1", seed=42)
    (ds / "labels" / "g0_0.txt").write_text("5 0.5 0.5 0.2 0.2\n")
    (ds / "labels" / "g0_1.txt").write_text("0 0.99 0.5 0.5 0.2\n")
    (ds / "labels" / "g0_2.txt").write_text("0 0.5 0.5\n")
    rep = validate_dataset(ds, m, CLASSES)
    text = " ".join(rep["errors"])
    assert "class id 5" in text
    assert "outside the image" in text
    assert "expected 5 values" in text


def test_detects_group_leakage(tmp_path):
    ds = make_dataset(tmp_path)
    m = build_manifest(ds, ds / "metadata.csv", "v1", seed=42)
    other = "test" if m.loc[0, "split"] != "test" else "train"
    m.loc[0, "split"] = other
    rep = validate_dataset(ds, m, CLASSES)
    assert any("LEAKAGE: groups" in e for e in rep["errors"])


def test_detects_identical_images_across_splits(tmp_path):
    ds = make_dataset(tmp_path)
    a = ds / "images" / "g0_0.jpg"
    for g in range(1, 10):
        (ds / "images" / f"g{g}_0.jpg").write_bytes(a.read_bytes())
    m = build_manifest(ds, ds / "metadata.csv", "v1", seed=42)
    rep = validate_dataset(ds, m, CLASSES)
    assert any("identical image" in e for e in rep["errors"])


def test_metadata_mismatch_raises(tmp_path):
    ds = make_dataset(tmp_path)
    (ds / "images" / "extra.jpg").write_bytes(b"x")
    with pytest.raises(ValueError):
        build_manifest(ds, ds / "metadata.csv", "v1")


def test_missing_label_is_warning_not_error(tmp_path):
    ds = make_dataset(tmp_path)
    (ds / "labels" / "g0_0.txt").unlink()
    m = build_manifest(ds, ds / "metadata.csv", "v1", seed=42)
    rep = validate_dataset(ds, m, CLASSES)
    assert rep["errors"] == []
    assert any("background" in w for w in rep["warnings"])
