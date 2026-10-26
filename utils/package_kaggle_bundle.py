"""Build and verify the Kaggle upload ZIP using portable POSIX archive paths."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle_upload"
OUTPUT_PATH = ROOT / "dist" / "vi-extoxic-kaggle-upload.zip"


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Kaggle upload directory not found: {SOURCE_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in SOURCE_DIR.rglob("*") if path.is_file())

    with zipfile.ZipFile(
        OUTPUT_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            archive_name = path.relative_to(SOURCE_DIR).as_posix()
            archive.write(path, archive_name)

    with zipfile.ZipFile(OUTPUT_PATH) as archive:
        names = archive.namelist()
        invalid = [name for name in names if "\\" in name]
        if invalid:
            raise ValueError(f"ZIP contains Windows separators: {invalid[:5]}")
        required = {
            "archive.zip",
            "training_bundle/training/evaluate_sft.py",
            "training_bundle/training/prompts.py",
            "training_bundle/training/few_shot_examples.json",
        }
        missing = required.difference(names)
        if missing:
            raise ValueError(f"ZIP is missing required files: {sorted(missing)}")
        eval_source = archive.read(
            "training_bundle/training/evaluate_sft.py"
        ).decode("utf-8")
        if "cot-en" not in eval_source or "--few-shot" not in eval_source:
            raise ValueError("ZIP contains an outdated evaluate_sft.py")

    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"Created: {OUTPUT_PATH}")
    print(f"Files: {len(files)}")
    print(f"Size: {OUTPUT_PATH.stat().st_size:,} bytes")
    print(f"SHA-256: {digest}")
    print("ZIP paths: POSIX separators verified")
    print("Required prompt and few-shot files: verified")


if __name__ == "__main__":
    main()
