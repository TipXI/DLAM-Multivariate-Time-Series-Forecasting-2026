"""Download and verify project dataset files from Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import requests

HF_DATASET_BASE = "hf://datasets/AIML-TUDA/dlam-ts-project-data-2026"
RAW_BASE_URL = "https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026/raw/main"

FILES = [
    "train.csv",
    "validation_input.csv",
    "forecast_index_validation.csv",
    "metadata.json",
]


def download_dataset(output_dir: Path | str = "data") -> None:
    """Download all dataset files to output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset files to: {output_path.resolve()}")

    for filename in FILES:
        target_file = output_path / filename
        if filename.endswith(".csv"):
            hf_url = f"{HF_DATASET_BASE}/{filename}"
            print(f"  Downloading {filename} from {hf_url} ...")
            df = pd.read_csv(hf_url)
            df.to_csv(target_file, index=False)
            print(f"    Saved {filename} ({df.shape[0]} rows, {df.shape[1]} cols)")
        elif filename.endswith(".json"):
            raw_url = f"{RAW_BASE_URL}/{filename}"
            print(f"  Downloading {filename} from {raw_url} ...")
            resp = requests.get(raw_url, timeout=30)
            resp.raise_for_status()
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"    Saved {filename}")

    print("\nAll files successfully downloaded and verified!")


if __name__ == "__main__":
    download_dataset("data")
