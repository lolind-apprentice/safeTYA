"""
download_dataset.py
Downloads the OP-R1/OP-Reddit-User dataset locally.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path("./op_reddit_data")

def main():
    print("Starting download of OP-R1/OP-Reddit-User dataset...")
    snapshot_download(
        repo_id="OP-R1/OP-Reddit-User",
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns="*.parquet",  # Downloads only the parquet files
    )
    print(f"\nDownload complete! Files saved to: {DATA_DIR.resolve()}")

if __name__ == "__main__":
    main()