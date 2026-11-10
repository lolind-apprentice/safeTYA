"""
fetch_users_local.py
Extracts 10,000 user records from local parquet files using DuckDB.
"""
import argparse
from pathlib import Path
import duckdb

def main():
    parser = argparse.ArgumentParser(description="Extract user records locally via DuckDB.")
    parser.add_argument("--n", type=int, default=10000, help="Number of user records to extract.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--data-dir", type=str, default="./op_reddit_data", help="Directory with parquet files.")
    parser.add_argument("--out", type=str, default="10k_raw_users.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    
    # Locate all parquet files in directory
    parquet_files = list(data_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}. Run download_dataset.py first.")

    glob_pattern = str(data_path / "**" / "*.parquet")
    print(f"Connecting to DuckDB and querying local files: {glob_pattern}...")

    con = duckdb.connect()

    # Query local files directly using multithreaded DuckDB engine
    # RESERVOIR sampling ensures exact reproducibility matching your seed
    query = f"""
        COPY (
            SELECT * 
            FROM read_parquet('{glob_pattern}')
            USING SAMPLE {args.n} (reservoir, {args.seed})
        ) TO '{args.out}' (FORMAT JSON);
    """

    con.execute(query)
    print(f"\nSuccess! Sampled {args.n:,} user records to {args.out}")

if __name__ == "__main__":
    main()