"""Step 3 (Jev version): Label each user with Jev through OpenRouter's System One API.

Usage:
    py classify_users.py --inp normalized.jsonl --out labeled.jsonl --concurrency 32 --limit 0
"""
import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/alpha/decisions"
VALID = {"buyer", "seller", "user", "none"}
MAX_ITEM_CHARS = 700    # per post/comment
MAX_TOTAL_CHARS = 6000  # per user context window

QUESTIONS = {
    "role": {
        "type": "choice",
        "instructions": "Based only on what this Reddit user wrote, what is their role regarding illicit drugs?",
        "criteria": {
            "buyer": "Seeks to acquire illicit drugs or asks for sources, vendors, or contacts to buy from.",
            "seller": "Advertises, supplies, or brokers illicit drugs: offers, prices, shipping, or telling people to DM to buy.",
            "user": "Talks about their own personal drug use or experiences, without trying to buy or sell.",
            "none": "No illicit drug use, buying, or selling: unrelated topics, recovery, news, policy, or general discussion.",
        },
    },
    "wants_to_buy": {
        "type": "noul",
        "instructions": "Does the user ask how or where to buy or obtain illegal drugs, or ask for vendors or sources?",
    },
    "wants_to_sell": {
        "type": "noul",
        "instructions": "Does the user offer, advertise, or arrange to sell or supply illegal drugs?",
    },
    "personal_use": {
        "type": "noul",
        "instructions": "Does the user describe their own use of illegal drugs or non-prescribed medication?",
    },
}
NOULS = [k for k, q in QUESTIONS.items() if q["type"] == "noul"]
FATAL = {400, 401, 402, 403, 404}


def normalize(row):
    """Convert raw user record to minimal format required by Jev labeling script."""
    threads = []
    for thread in row.get("threads", []):
        subreddit = thread.get("subreddit", "")
        items = []
        for item in thread.get("items", []):
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            kind = item.get("kind", "comment")
            items.append({"kind": kind, "text": text.strip()})
        if items:
            threads.append({"subreddit": subreddit, "items": items})

    return {"user_id": row.get("user_id"), "threads": threads}


def render(user):
    """Turn a normalized user into plain text for the model."""
    lines, total = [], 0
    for thread in user["threads"]:
        for item in thread["items"]:
            tag = "POST" if item["kind"] in ("post", "submission") else "COMMENT"
            text = item["text"].replace("\n", " ")[:MAX_ITEM_CHARS]
            line = f"[r/{thread['subreddit']}] {tag}: {text}"
            if total + len(line) > MAX_TOTAL_CHARS:
                lines.append("[... more text omitted ...]")
                return "\n".join(lines)
            lines.append(line)
            total += len(line)
    return "\n".join(lines)


async def label_one(client, args, user, sem):
    body = {"model": args.model, "state": render(user), "questions": QUESTIONS}
    last = "unknown error"
    async with sem:
        for attempt in range(5):
            try:
                r = await client.post(URL, json=body)
                if r.status_code in FATAL:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    break
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                data = r.json()
                answers = data["answers"]
                role_answer = answers["role"]
                role = role_answer["choice"]
                if role not in VALID:
                    raise ValueError(f"unexpected role {role!r}")
                return {
                    **user,
                    "role": role,
                    "confidence": role_answer.get("confidence"),
                    "probabilities": role_answer.get("probabilities"),
                    "scores": {k: answers.get(k, {}).get("noul") for k in NOULS},
                    "rationale": None,
                    "model": data.get("model"),
                }
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                await asyncio.sleep(2 ** attempt)
    return {**user, "role": "error", "confidence": None, "rationale": last, "model": None}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", type=Path, default=Path("normalized.jsonl"))
    p.add_argument("--out", type=Path, default=Path("labeled.jsonl"))
    p.add_argument("--model", default="~typesafe/jev-latest")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="only label first N users (0 = all)")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in environment variables.")

    done = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            done = {json.loads(line)["user_id"] for line in f if line.strip()}

    users = []
    with args.inp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                u = normalize(json.loads(line))
                if u["threads"] and u["user_id"] not in done:
                    users.append(u)

    if args.limit:
        users = users[: args.limit]
    print(f"Labeling {len(users)} users with {args.model} ({len(done)} already done)")

    sem = asyncio.Semaphore(args.concurrency)
    roles, builds = Counter(), Counter()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        with args.out.open("a", encoding="utf-8") as out:
            tasks = [label_one(client, args, u, sem) for u in users]
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                row = await coro
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                roles[row["role"]] += 1
                builds[row.get("model")] += 1
                if i % 25 == 0 or i == len(users):
                    print(f"  {i}/{len(users)}  {dict(roles)}")
                    
    print("Roles distribution:", dict(roles))
    print("Model builds used  :", dict(builds))


if __name__ == "__main__":
    asyncio.run(main())
