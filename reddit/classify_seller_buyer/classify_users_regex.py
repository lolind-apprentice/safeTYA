"""
classify_users_regex.py

Labels per-user Reddit records with a coarse role using weighted regex signals.

INPUT (JSONL, one user per line):
    {"user_id": "...",
     "threads": [{"subreddit": "...", "items": [{"kind": "post"|"comment", "text": "..."}]}],
     "role": null, "confidence": null, "rationale": null}

OUTPUT (JSONL): same record, with these fields filled in:
    role        "seller" | "buyer" | "none"
    confidence  float in [0, 1]
    rationale   short human-readable explanation with matched snippets

USAGE:
    python classify_users_regex.py --inp 20k_raw_users.jsonl --out 20k_regex_labeled.jsonl   
    python classify_users_regex.py --inp users.jsonl --out labeled.jsonl --limit 100

Design notes:
  * Bare words like "vendor", "buy", "telegram" are NOT signals. Consumers
    talk about vendors constantly; only first-person offers/solicitations
    and explicit source-seeking phrases count.
  * Each pattern has a weight. Scores are summed per user over de-duplicated
    items, with each pattern's contribution capped so one spammy phrase
    repeated many times can't dominate.
  * Roles are scored independently; seller needs a higher bar than buyer.
  * Regex is a cheap first pass. Review borderline cases (low confidence)
    with a human or an LLM.
"""

import argparse
import json
import re
import sys
from collections import Counter
from typing import Iterator

# (pattern, weight, name)
SELL_PATTERNS = [
    (r"\b(i'?m|i\s+am|we'?re|we\s+are)\s+(now\s+)?(selling|offering|restocking)\b", 4, "first-person selling"),
    (r"\bfor\s+sale\b", 3, "for sale"),
    (r"\b(my|our)\s+(menu|price\s*list|prices?|shop|store)\b", 3, "my menu/prices"),
    (r"\b(dm|pm|inbox)\s+(me|us)\b", 2, "DM me"),
    (r"\bships?\s+(worldwide|internationally|globally|from|to)\b", 2, "ships worldwide/from"),
    (r"\bstealth\s+ship\w*|\bdiscreet(ly)?\s+(ship|packag)\w*", 3, "stealth/discreet shipping"),
    (r"\bwickr\b|\bsignal\s+me\b|\bwhatsapp\s+me\b|\btelegram\s*[:@]", 2, "contact handle"),
    (r"\bbulk\s+(deals?|discounts?|pricing|orders?)\b", 2, "bulk pricing"),
    (r"\b(restocked|just\s+restocked|back\s+in\s+stock|in\s+stock|available\s+now)\b", 1, "stock language"),
    (r"\b(i|we)\s+(can\s+|will\s+)?(send|ship|post|supply|sell)\s+(you|u|some)\b", 2, "offer to supply"),
    (r"\bcop\s+(from|link)\b|\bvouch(es|ed)?\s+(for\s+)?(me|us)\b", 2, "cop/vouch"),
]

BUY_PATTERNS = [
    (r"\b(wtb|looking\s+to\s+buy|looking\s+for\s+a\s+(plug|source|vendor|hookup))\b", 4, "WTB/looking for plug"),
    (r"\bneed\s+a\s+(plug|source|hookup|vendor)\b", 4, "need a plug"),
    (r"\bwhere\s+(can|could|do|would)\s+(i|you|we)\s+(buy|find|get|source|order)\b", 2, "where to get"),
    (r"\b(anyone|anybody)\s+(know|got|have)\s+(a\s+)?(source|vendor|plug|link)\b", 3, "anyone know a source"),
    (r"\bsource\s+for\b", 1, "source for"),
    (r"\bwhere\s+to\s+(cop|buy|order)\b", 3, "where to cop"),
    (r"\bcan\s+(you|u|someone)\s+(send|link|dm|pm|inbox)\s+me\b", 2, "send me (link/name)"),
    (r"\b(iso|wtb)\b(?=.{0,40}\b(g|mg|oz|gram|pill|tab|bars?)\b)", 3, "ISO <qty>"),
]

SELL = [(re.compile(p, re.I | re.S), w, n) for p, w, n in SELL_PATTERNS]
BUY = [(re.compile(p, re.I | re.S), w, n) for p, w, n in BUY_PATTERNS]

PER_PATTERN_CAP = 3        # max hits per pattern that count toward score
SELL_THRESHOLD = 4         # score needed to call "seller"
BUY_THRESHOLD = 2          # score needed to call "buyer"
SKIP_TEXT = {"", "[removed]", "[deleted]"}


def iter_texts(user: dict) -> Iterator[tuple[str, str, str]]:
    """Yield (subreddit, kind, text) for each unique, non-empty item."""
    seen = set()
    for thread in user.get("threads") or []:
        sub = thread.get("subreddit", "")
        for item in thread.get("items") or []:
            text = (item.get("text") or "").strip()
            if text in SKIP_TEXT or text in seen:
                continue
            seen.add(text)
            yield sub, item.get("kind", ""), text


def snippet(text: str, m: re.Match, width: int = 40) -> str:
    s = max(0, m.start() - width)
    e = min(len(text), m.end() + width)
    return ("..." if s else "") + text[s:e].replace("\n", " ") + ("..." if e < len(text) else "")


def score(user: dict, patterns) -> tuple[int, Counter, list[str]]:
    total = 0
    hits: Counter = Counter()
    examples: list[str] = []
    for _sub, _kind, text in iter_texts(user):
        for rx, weight, name in patterns:
            m = rx.search(text)
            if not m:
                continue
            if hits[name] < PER_PATTERN_CAP:
                total += weight
                if len(examples) < 3:
                    examples.append(f'"{snippet(text, m)}"')
            hits[name] += 1
    return total, hits, examples


def label_user(user: dict) -> dict:
    n_items = sum(1 for _ in iter_texts(user))
    sell_score, sell_hits, sell_ex = score(user, SELL)
    buy_score, buy_hits, buy_ex = score(user, BUY)

    if sell_score >= SELL_THRESHOLD and sell_score >= buy_score:
        role, s, hits, ex = "seller", sell_score, sell_hits, sell_ex
        conf = min(0.95, 0.4 + 0.08 * s)
    elif buy_score >= BUY_THRESHOLD:
        role, s, hits, ex = "buyer", buy_score, buy_hits, buy_ex
        conf = min(0.9, 0.3 + 0.08 * s)
    else:
        role, s, hits, ex = "none", max(sell_score, buy_score), Counter(), []
        conf = 0.7 if n_items >= 10 and s == 0 else 0.5

    if role == "none":
        rationale = f"No strong sell/buy signals across {n_items} items (sell={sell_score}, buy={buy_score})."
    else:
        pats = ", ".join(f"{k} x{v}" for k, v in hits.most_common(4))
        rationale = (f"{role} score {s} over {n_items} items "
                     f"(other-side score {buy_score if role == 'seller' else sell_score}). "
                     f"Signals: {pats}. e.g. {' | '.join(ex)}")

    user["role"] = role
    user["confidence"] = round(conf, 2)
    user["rationale"] = rationale
    return user


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, help="input users JSONL")
    ap.add_argument("--out", required=True, help="output labeled JSONL")
    ap.add_argument("--limit", type=int, default=0, help="only process first N users (0 = all)")
    args = ap.parse_args()

    counts: Counter = Counter()
    bad = 0
    with open(args.inp, encoding="utf-8", errors="replace") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                user = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                print(f"[!] bad JSON at line {i}: {e}", file=sys.stderr)
                continue
            labeled = label_user(user)
            counts[labeled["role"]] += 1
            fout.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"  {i} users  {dict(counts)}", file=sys.stderr)
            if args.limit and i >= args.limit:
                break

    print(f"Done. {dict(counts)}  (skipped {bad} bad lines)  -> {args.out}")


if __name__ == "__main__":
    main()
