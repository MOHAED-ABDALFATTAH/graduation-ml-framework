"""Create dataset_manifest.csv with group-aware splits.

Example (Colab, with Drive mounted):
    python scripts/make_manifest.py \
        --dataset-dir /content/drive/MyDrive/Graduation_Project/01_Dataset/detection_v1 \
        --metadata /content/drive/MyDrive/Graduation_Project/01_Dataset/detection_v1/metadata.csv \
        --version detection_v1 --seed 42

Run it ONCE. The split must never be regenerated after experiments start.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--metadata", required=True, help="CSV with filename,group_id")
    ap.add_argument("--version", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    out = Path(args.dataset_dir) / "dataset_manifest.csv"
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists. The split is frozen; use --force only if you "
                 f"are creating a NEW dataset version.")
    manifest = build_manifest(args.dataset_dir, args.metadata, args.version,
                              (args.train, args.val, args.test), args.seed)
    manifest.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(manifest["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
