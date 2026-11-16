"""
Processes LMSYS, ShareChat, and WildChat datasets to extract English user messages.

Usage:
    python prepare_dataset.py
"""
import json
from pathlib import Path
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import fasttext

_LID = None  # loaded lazily so each worker process gets its own copy

# Output directory
OUT = Path("processed_data")
OUT.mkdir(parents=True, exist_ok=True)

# Uses your saved `hf auth login` credentials.
TOKEN = True
LIMIT = 100_000_000  # effectively "no limit"
SHARECHAT_LIMIT = 8_000  # per platform

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def _get_lid():
    global _LID
    if _LID is None:
        _LID = fasttext.load_model("lid.176.ftz")
    return _LID

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)

def looks_english(text, min_prob=0.80):
    """True if fastText thinks the text is English with enough confidence."""
    # Strip fenced code, which confuses language ID.
    text = _CODE_BLOCK.sub(" ", text)
    # fastText rejects newlines; also cap length for speed.
    text = " ".join(text.split())[:1000]
    if len(text) < 20:
        return False
    labels, probs = _get_lid().predict(text, k=1)
    return labels[0] == "__label__en" and probs[0] >= min_prob

def is_english(value):
    """Match English language labels, case-insensitively."""
    return isinstance(value, str) and value.strip().lower() in {
        "english", "en", "en-us", "en-gb"
    }

def stream_to_jsonl(path, rows, limit, label):
    """Write each row to disk as it is produced (no big in-memory list)."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
            if count % 10_000 == 0:
                print(f"{label}: {count:,}")
            if count >= limit:
                break
    print(f"{label} saved: {count:,} -> {path}")
    return count

def get_user_messages(conversation):
    """Return only user turns from a conversation."""
    if not isinstance(conversation, list):
        return []

    return [
        turn for turn in conversation
        if isinstance(turn, dict)
        and str(turn.get("role", "")).lower() == "user"
        and isinstance(turn.get("content"), str)
        and turn["content"].strip()
    ]

def load_stream(repo, config=None, split="train"):
    """Stream a dataset without downloading it all at once."""
    args = (repo, config) if config else (repo,)
    return load_dataset(*args, split=split, streaming=True, token=TOKEN)

# --------------------------------------------------
# 1. LMSYS Chat-1M
# https://huggingface.co/datasets/lmsys/lmsys-chat-1m
# Goal: 100,000 English conversation rows,
# retaining user messages only within each conversation.
# --------------------------------------------------

def iter_lmsys(ds):
    # LMSYS uses a language field and a conversation
    # containing role/content message dictionaries.
    print(f"\nLoading LMSYS: ...")
    for row in ds:
        if not is_english(row.get("language")):
            continue
        turns = get_user_messages(row.get("conversation"))
        if turns:
            yield {
                
             # Preserve the conversation structure, but remove
             # assistant and system turns.
                "conversation_id": row.get("conversation_id"),
                "conversation": turns,
            }

# --------------------------------------------------
# 2. ShareChat
# https://huggingface.co/datasets/tucnguyen/ShareChat 
# Goal: 8,000 English user messages per platform.
# Keep only plain_text in each output row.
# --------------------------------------------------

def iter_sharechat(ds):
    print(f"\nLoading ShareChat: ...")
    for row in ds:
        if not is_english(row.get("detected_language_final")):
            continue
        if str(row.get("role", "")).lower() != "user":
            continue
        text = row.get("plain_text")
        if isinstance(text, str) and text.strip():
            yield {"plain_text": text}

# --------------------------------------------------
# 3. WildChat-4.8M
# https://huggingface.co/datasets/allenai/WildChat-4.8M 
# Goal: 100,000 English conversations.
# Keep the complete conversation for each English row.
# --------------------------------------------------


def iter_wildchat(ds):
    print(f"\nLoading WildChat-4.8M ...")

    for row in ds:
        if not is_english(row.get("language")):
            continue
        turns = get_user_messages(row.get("conversation"))
        # Keep only user messages, excluding assistant responses
        if turns:
            yield {
                "conversation_id": row.get("conversation_id")
                                   or row.get("conversation_hash"),
                "language": row.get("language"),
                "prompts": [t["content"] for t in turns],
            }
 
# --------------------------------------------------
# 4. Arena Chatbot 
# https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/viewer/default/train
# Goal: 100,000 English conversations.
# Keep the user side of conversation for each English row.
# --------------------------------------------------

def iter_arena(ds):
    print(f"\nLoading Arena Chatbot ...")
    # Each row has conversation_a and conversation_b with the SAME user
    # prompts (only the model replies differ), so read conversation_a only
    # to avoid duplicates.
    for row in ds:
        if not is_english(row.get("language")):
            continue
        turns = get_user_messages(row.get("conversation_a"))
        if turns:
            yield {
                "question_id": row.get("question_id"),
                "language": row.get("language"),
                "prompts": [t["content"] for t in turns],
            }

# --------------------------------------------------
# 5. ShareGPT 
# https://huggingface.co/datasets/RyokoAI/ShareGPT52K
# Goal: 100,000 English conversations.
# Keep the user side of conversation for each English row.
# --------------------------------------------------

def iter_sharegpt(ds):
    print("\nLoading ShareGPT52K...")
    for row in ds:
        convo = row.get("conversations")
        if not isinstance(convo, list):
            continue
        prompts = [
            t["value"] for t in convo
            if isinstance(t, dict)
            and str(t.get("from", "")).lower() == "human"
            and isinstance(t.get("value"), str)
            and t["value"].strip()
        ]
        if not prompts:
            continue
        if not looks_english(" ".join(prompts)):
            continue
        yield {
            "id": row.get("id"),
            "prompts": prompts,
        }

# --------------------------------------------------
# Run all dataset jobs
# --------------------------------------------------

def process_lmsys(limit=LIMIT):
    ds = load_stream("lmsys/lmsys-chat-1m", "default")
    stream_to_jsonl(OUT / "lmsys_english_user_only.jsonl",
                    iter_lmsys(ds), limit, "LMSYS")

def process_sharechat(platform, limit=SHARECHAT_LIMIT):
    ds = load_stream("tucnguyen/ShareChat", platform)
    stream_to_jsonl(OUT / f"sharechat_{platform}_english_user.jsonl",
                    iter_sharechat(ds), limit, f"ShareChat/{platform}")


def process_wildchat(limit=LIMIT):
    ds = load_stream("allenai/WildChat-4.8M", "default")
    stream_to_jsonl(OUT / "wildchat_english_user_only.jsonl",
                    iter_wildchat(ds), limit, "WildChat")


def process_arena(limit=LIMIT):
    ds = load_stream("lmsys/chatbot_arena_conversations")
    stream_to_jsonl(OUT / "arena_english_user_only.jsonl",
                    iter_arena(ds), limit, "Arena")



def process_sharegpt(limit=LIMIT):
    ds = load_stream("RyokoAI/ShareGPT52K")
    stream_to_jsonl(OUT / "sharegpt_english_user_only.jsonl",
                    iter_sharegpt(ds), limit, "ShareGPT")
        

def run_jobs():
    jobs = [
        (process_lmsys, ()),
        (process_arena, ()),
        (process_wildchat, ()),
        (process_sharegpt, ()),
    ]
    jobs += [(process_sharechat, (p,))
             for p in ["claude", "chatgpt", "perplexity", "gemini", "grok"]]

    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn, *args): (fn.__name__, args)
                   for fn, args in jobs}
        for fut in as_completed(futures):
            name, args = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"FAILED {name}{args}: {e!r}")


if __name__ == "__main__":
    run_jobs()
    print(f"\nFinished. Output files are in {OUT}/")