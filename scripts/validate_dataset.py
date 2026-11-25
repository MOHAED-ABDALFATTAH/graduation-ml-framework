"""Validate a detection dataset (YOLO format) against its manifest.

Example:
    python scripts/validate_dataset.py \
        --dataset-dir /content/drive/MyDrive/Graduation_Project/01_Dataset/detection_v1 \
        --classes aphid whitefly beetle
Exit code is 1 if any error is found.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import validate_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--classes", nargs="+", required=True,
                    help="class names in the frozen order (index = class id)")
    args = ap.parse_args()

    manifest = pd.read_csv(Path(args.dataset_dir) / "dataset_manifest.csv")
    report = validate_dataset(args.dataset_dir, manifest, args.classes)
    print(json.dumps(report["stats"], indent=2))
    for w in report["warnings"][:50]:
        print("WARNING:", w)
    if len(report["warnings"]) > 50:
        print(f"... and {len(report['warnings']) - 50} more warnings")
    for e in report["errors"][:50]:
        print("ERROR:", e)
    if report["errors"]:
        print(f"\nFAILED: {len(report['errors'])} error(s)")
        sys.exit(1)
    print("\nOK: no errors")


if __name__ == "__main__":
    main()
