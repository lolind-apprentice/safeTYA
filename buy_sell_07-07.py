#!/usr/bin/env python3
"""
buy_and_sell07-07.py

Scans Reddit post/comment JSONL exports for language associated with
soliciting or advertising drug sales, and separates that content from
buyer-side ("looking to buy") content.

INPUT FILES (JSONL, one JSON object per line), expected in --input-dir:
    r_post, r_comment      -> subreddit/dataset 1
    r2_post, r2_comment    -> subreddit/dataset 2

OUTPUT FILES (JSONL), written to --output-dir:
    r_out_post_sell,   r_out_comment_sell
    r_out_post_buy,    r_out_comment_buy
    r2_out_post_sell,  r2_out_comment_sell
    r2_out_post_buy,   r2_out_comment_buy

Each output row is the FULL original JSON object (post or comment),
unmodified, so no downstream fields are lost.

USAGE:
    python filter_drug_posts.py --input-dir /path/to/files --output-dir /path/to/out
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Keyword sets

# SELLER-signal terms: language typically used by someone advertising product,
# soliciting orders, or describing fulfillment logistics.
SELL_TERMS = [
    r"\bshipping\b", r"\bship(?:s|ped|ping)?\s+(worldwide|international|globally)\b",
    r"\bglobal\s*(shipping|ship)?\b", r"\bselling\b", r"\bfor\s+sale\b",
    r"\bDM\s+(me|for)\b", r"\bmenu\b", r"\brestock(ed)?\b",
    r"\bin\s+stock\b", r"\bavailable\s+now\b", r"\bpricing\b", r"\bwickr\b",
    r"\btelegram\b.{0,15}\b(order|menu|DM)\b", r"\bbulk\s+(deals?|discounts?|pricing)\b",
    r"\bdiscreet\s+packag(?:ing|e)\b", r"\bvendor\b", r"\bcop\s+(from|link)\b",
    r"\bstealth\s+ship\b", r"\bsell(?:er|ing)?\b",
]

# BUYER-signal terms: language typically used by someone seeking a source.
BUY_TERMS = [
    r"\bISO\b", r"\bWTB\b", r"\blooking\s+to\s+buy\b", r"\bneed\s+a\s+(plug|source|hookup)\b",
    r"\bwhere\s+(can|do)\s+i\s+(buy|find|get)\b", r"\banyone\s+(selling|sell|have)\b",
    r"\bsource\s+for\b", r"\bhookup\b", r"\bwhere\s+to\s+cop\b", r"\bbuy(?:ing)?\b",
    r"\bpurchase\b", r"\bplug\b",
]

SELL_RE = re.compile("|".join(SELL_TERMS), re.IGNORECASE)
BUY_RE = re.compile("|".join(BUY_TERMS), re.IGNORECASE)

def extract_text(row: dict) -> str:
    """Pull all searchable text fields out of a post or comment row."""
    parts = []
    for field in ("title", "selftext", "body"):
        val = row.get(field)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(parts)


def classify(text: str) -> str | None:
    """Return 'sell', 'buy', or None based on which signal set matches.

    Seller language takes priority: a post that mixes both ("selling X,
    also looking to buy Y in trade") is still primarily a sell listing.
    """
    has_sell = bool(SELL_RE.search(text))
    has_buy = bool(BUY_RE.search(text))
    if has_sell:
        return "sell"
    if has_buy:
        return "buy"
    return None

def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        print(f"  [!] missing file: {path}", file=sys.stderr)
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] bad JSON at {path}:{line_num}: {e}", file=sys.stderr)
    return rows

def write_jsonl(path: Path, rows: list):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def post_id_full(post: dict) -> str:
    """Return the fullname (t3_xxx) form of a post id for matching to comment link_id."""
    name = post.get("name")
    if name:
        return name
    pid = post.get("id", "")
    return f"t3_{pid}" if pid else ""

def process_dataset(label: str, post_path: Path, comment_path: Path, out_dir: Path):
    print(f"\n=== Dataset: {label} ===")
    posts = load_jsonl(post_path)
    comments = load_jsonl(comment_path)
    print(f"  loaded {len(posts)} posts, {len(comments)} comments")

    sell_posts, buy_posts = [], []
    sell_post_ids, buy_post_ids = set(), set()

    for post in posts:
        text = extract_text(post)
        cls = classify(text)
        if cls == "sell":
            sell_posts.append(post)
            sell_post_ids.add(post_id_full(post))
        elif cls == "buy":
            buy_posts.append(post)
            buy_post_ids.add(post_id_full(post))

    # Comments matched two ways:
    #   (a) comment body itself matches sell/buy language, OR
    #   (b) comment belongs to a thread (link_id) already flagged as sell/buy
    sell_comments, buy_comments = [], []
    for c in comments:
        text = extract_text(c)
        cls = classify(text)
        link = c.get("link_id", "")
        if cls == "sell" or link in sell_post_ids:
            sell_comments.append(c)
        elif cls == "buy" or link in buy_post_ids:
            buy_comments.append(c)

    write_jsonl(out_dir / f"{label}_out_post_sell.jsonl", sell_posts)
    write_jsonl(out_dir / f"{label}_out_comment_sell.jsonl", sell_comments)
    write_jsonl(out_dir / f"{label}_out_post_buy.jsonl", buy_posts)
    write_jsonl(out_dir / f"{label}_out_comment_buy.jsonl", buy_comments)

    print(f"  SELL -> {len(sell_posts)} posts, {len(sell_comments)} comments")
    print(f"  BUY  -> {len(buy_posts)} posts, {len(buy_comments)} comments")

    return {
        "sell_posts": len(sell_posts), "sell_comments": len(sell_comments),
        "buy_posts": len(buy_posts), "buy_comments": len(buy_comments),
        "total_posts": len(posts), "total_comments": len(comments),
    }


# Hardcoded paths 
DOWNLOADS_DIR = Path.home() / "Downloads"

R_POST_PATH = DOWNLOADS_DIR / "r_drugs_posts.jsonl"
R_COMMENT_PATH = DOWNLOADS_DIR / "r_drugs_comments.jsonl"      
R2_POST_PATH = DOWNLOADS_DIR / "r_opiates_posts.jsonl"
R2_COMMENT_PATH = DOWNLOADS_DIR / "r_opiates_comments.jsonl"

OUTPUT_DIR = DOWNLOADS_DIR / "reddit_filter_output"

def main():
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    stats["r"] = process_dataset("r", R_POST_PATH, R_COMMENT_PATH, out_dir)
    stats["r2"] = process_dataset("r2", R2_POST_PATH, R2_COMMENT_PATH, out_dir)

    print("\n=== Summary ===")
    for label, s in stats.items():
        print(f"{label}: {s}")


if __name__ == "__main__":
    main()