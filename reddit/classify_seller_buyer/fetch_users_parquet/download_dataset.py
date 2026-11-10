"""
download_dataset.py
Downloads both OP-R1/OP-Reddit-User and OP-R1/OP-Reddit-Post datasets locally.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

USER_DATA_DIR = Path("./op_reddit_user")
POST_DATA_DIR = Path("./op_reddit_post")


def main():
    print("Starting download of OP-R1/OP-Reddit-User dataset...")
    snapshot_download(
        repo_id="OP-R1/OP-Reddit-User",
        repo_type="dataset",
        local_dir=USER_DATA_DIR,
        allow_patterns="*.parquet",
    )
    print(f"User dataset saved to: {USER_DATA_DIR.resolve()}\n")

    print("Starting download of OP-R1/OP-Reddit-Post dataset...")
    snapshot_download(
        repo_id="OP-R1/OP-Reddit-Post",
        repo_type="dataset",
        local_dir=POST_DATA_DIR,
        allow_patterns="*.parquet",
    )
    print(f"Post dataset saved to: {POST_DATA_DIR.resolve()}\n")

    print("All downloads complete! Ready for local DuckDB extraction.")


if __name__ == "__main__":
    main()
