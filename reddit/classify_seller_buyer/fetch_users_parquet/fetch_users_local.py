"""
py fetch_users_local.py --n [MAX number] --seed [number] --out [filename]
fetch_users_local.py
Performs a local DuckDB JOIN between OP-Reddit-User and OP-Reddit-Post
on user_id to output fully hydrated JSONL records in seconds.
"""
import argparse
from pathlib import Path
import duckdb


def main():
    parser = argparse.ArgumentParser(
        description="Locally join OP-Reddit-User and OP-Reddit-Post using DuckDB."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10000,
        help="Number of user records to extract.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "--user-dir",
        type=str,
        default="./op_reddit_user",
        help="Directory with OP-Reddit-User parquet files.",
    )
    parser.add_argument(
        "--post-dir",
        type=str,
        default="./op_reddit_post",
        help="Directory with OP-Reddit-Post parquet files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="10k_raw_users.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    user_path = Path(args.user_dir).resolve()
    post_path = Path(args.post_dir).resolve()

    if not user_path.exists() or not post_path.exists():
        raise FileNotFoundError(
            "Ensure both --user-dir and --post-dir directories exist with downloaded parquet files."
        )

    user_glob = str(user_path / "**" / "*.parquet").replace("\\", "/")
    post_glob = str(post_path / "**" / "*.parquet").replace("\\", "/")

    print("Connecting to DuckDB and querying local parquet files...")
    con = duckdb.connect()

    query = f"""
        COPY (
            WITH sampled_users AS (
                SELECT user_id 
                FROM read_parquet('{user_glob}', union_by_name=True)
                GROUP BY user_id
                USING SAMPLE {args.n} (reservoir, {args.seed})
            ),
            user_posts AS (
                SELECT 
                    p.user_id,
                    COALESCE(p.subreddit, 'unknown') AS subreddit,
                    COALESCE(p.kind, 'comment') AS kind,
                    COALESCE(p.text, '') AS text
                FROM read_parquet('{post_glob}', union_by_name=True) p
                INNER JOIN sampled_users u ON p.user_id = u.user_id
            ),
            sub_grouped AS (
                SELECT 
                    user_id,
                    subreddit,
                    list(struct_pack(kind := kind, text := text)) AS items
                FROM user_posts
                GROUP BY user_id, subreddit
            )
            SELECT 
                user_id,
                list(struct_pack(subreddit := subreddit, items := items)) AS threads,
                CAST(NULL AS VARCHAR) AS role,
                CAST(NULL AS FLOAT) AS confidence,
                CAST(NULL AS VARCHAR) AS rationale
            FROM sub_grouped
            GROUP BY user_id
        ) TO '{args.out}' (FORMAT JSON, ARRAY FALSE);
    """

    con.execute(query)
    print(
        f"\nSuccess! Joined and exported {args.n:,} hydrated user records to {args.out}"
    )


if __name__ == "__main__":
    main()
