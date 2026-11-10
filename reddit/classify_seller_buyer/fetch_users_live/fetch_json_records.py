"""Step 1: Pick N random users and save their FULL retrieve output (one JSON per line).

Optimized with multi-threading to speed up remote Parquet reads over Hugging Face HTTP streams.

Usage:
    py fetch_json_records.py --n 1000 --seed 42 --out raw_users.jsonl --no-context --workers 16
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from retrieve import OpR1Retriever, _Repository

# Cache directory listing once to avoid redundant network calls
_orig_data_files = _Repository.data_files
_file_cache = {}


def _cached_data_files(self):
    if self.value not in _file_cache:
        _file_cache[self.value] = _orig_data_files(self)
    return _file_cache[self.value]


_Repository.data_files = _cached_data_files


def json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def sample_user_ids(r, n, seed):
    files = [p for p in r._users.data_files() if p.endswith(".parquet")]
    print(f"{len(files)} parquet file(s) in {r._users.value}", file=sys.stderr)
    ids = set()
    for path in files:
        with r._parquet(r._users, path) as pf:
            ids.update(pf.read(columns=["user_id"]).column("user_id").to_pylist())
        print(f"  read {path}: {len(ids):,} users so far", file=sys.stderr)
    ids = sorted(ids)
    if len(ids) < n:
        raise SystemExit(f"Only {len(ids)} users found, fewer than --n {n}")
    return random.Random(seed).sample(ids, n)


def fetch_single_user(retriever, uid, with_context):
    try:
        user = retriever.get_user(uid, with_context=with_context)
        return uid, user, None
    except Exception as e:
        return uid, None, f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("raw_users.jsonl"))
    p.add_argument(
        "--no-context",
        action="store_true",
        default=True,
        help="Skip parent/reply context for maximum fetch speed",
    )
    p.add_argument("--with-context", action="store_false", dest="no_context")
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent worker threads for fetching",
    )
    p.add_argument("--posts-repo", default="OP-R1/OP-Reddit-Post")
    p.add_argument("--users-repo", default="OP-R1/OP-Reddit-User")
    args = p.parse_args()

    r = OpR1Retriever(posts_repo=args.posts_repo, users_repo=args.users_repo)
    ids = sample_user_ids(r, args.n, args.seed)
    Path("sampled_user_ids.txt").write_text("\n".join(ids) + "\n")

    done = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            done = {json.loads(line)["user_id"] for line in f if line.strip()}

    ids_to_fetch = [uid for uid in ids if uid not in done]
    print(
        f"Fetching {len(ids_to_fetch)} users with {args.workers} worker threads "
        f"(with_context={not args.no_context})...",
        file=sys.stderr,
    )

    start = time.time()
    with args.out.open("a", encoding="utf-8") as out, open(
        "errors.tsv", "a", encoding="utf-8"
    ) as err:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    fetch_single_user, r, uid, not args.no_context
                ): uid
                for uid in ids_to_fetch
            }

            for i, future in enumerate(as_completed(futures), 1):
                uid, user, error = future.result()
                if error:
                    err.write(f"{uid}\t{error}\n")
                    err.flush()
                    print(f"[{i}/{len(ids_to_fetch)}] ERROR {uid}: {error}", file=sys.stderr)
                elif user:
                    out.write(
                        json.dumps(user, ensure_ascii=False, default=json_default)
                        + "\n"
                    )
                    out.flush()

                if i % 25 == 0 or i == len(ids_to_fetch):
                    elapsed = time.time() - start
                    print(
                        f"[{i}/{len(ids_to_fetch)}] {elapsed:.0f}s  "
                        f"requests={r.http_range_requests}",
                        file=sys.stderr,
                    )

    print("Done ->", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
