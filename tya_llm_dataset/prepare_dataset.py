import json
from pathlib import Path
from datasets import load_dataset

# Output directory
OUT = Path("processed_data2")
OUT.mkdir(parents=True, exist_ok=True)

# Uses your saved `hf auth login` credentials.
TOKEN = True

# Helper functions

def is_english(value):
    """Match English language labels, case-insensitively."""
    return isinstance(value, str) and value.strip().lower() in {
        "english", "en", "en-us", "en-gb"
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str
                ) + "\n"
            )

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


def process_dataset(repo, config, split="train"):
    """Stream a dataset without downloading it all at once."""
    return load_dataset(
        repo,
        config,
        split=split,
        streaming=True,
        token=TOKEN,
    )


# 1. LMSYS Chat-1M
# Goal: 100,000 English conversation rows,
# retaining user messages only within each conversation.

def process_lmsys(limit=100_000):
    print("\nLoading LMSYS Chat-1M...")

    ds = process_dataset("lmsys/lmsys-chat-1m", "default")
    output = []
    count = 0

    for row in ds:
        # LMSYS uses a language field and a conversation
        # containing role/content message dictionaries.
        if not is_english(row.get("language")):
            continue

        user_turns = get_user_messages(row.get("conversation"))
        if not user_turns:
            continue

        # Preserve the conversation structure, but remove
        # assistant and system turns.
        output.append({
            "conversation_id": row.get("conversation_id"),
            "conversation": user_turns,
        })

        count += 1
        if count % 10_000 == 0:
            print(f"LMSYS: {count:,} conversations")

        if count >= limit:
            break

    write_jsonl(OUT / "lmsys_english_user_only.jsonl", output)
    print(f"LMSYS saved: {count:,} conversations")


# 2. ShareChat
# Goal: 8,000 English user messages per platform.
# Keep only plain_text in each output row.

def process_sharechat(platform, limit=8_000):
    print(f"\nLoading ShareChat: {platform}...")

    ds = process_dataset("tucnguyen/ShareChat", platform)
    output = []

    for row in ds:
        if not is_english(row.get("detected_language_final")):
            continue

        if str(row.get("role", "")).lower() != "user":
            continue

        text = row.get("plain_text")
        if not isinstance(text, str) or not text.strip():
            continue

        output.append({"plain_text": text})

        if len(output) >= limit:
            break

    path = OUT / f"sharechat_{platform}_english_user.jsonl"
    write_jsonl(path, output)
    print(f"{platform}: saved {len(output):,} user messages")


# 3. WildChat-4.8M
# Goal: 100,000 English conversations.
# Keep the complete conversation for each English row.

def process_wildchat(limit=100_000):
    print("\nLoading WildChat-4.8M...")

    ds = process_dataset("allenai/WildChat-4.8M", "default")
    output = []
    count = 0

    for row in ds:
        if not is_english(row.get("language")):
            continue

        conversation = row.get("conversation")
        if not isinstance(conversation, list) or not conversation:
            continue


        # Keep only user messages, excluding assistant responses
        user_prompts = get_user_messages(conversation)

        if not user_prompts:
            continue

        output.append({
            "conversation_id": row.get("conversation_id"),                
            "language": row.get("language"),
                "prompts": [turn["content"] for turn in user_prompts],
            })

        count += 1
        if count % 10_000 == 0:
            print(f"WildChat: {count:,} conversations")

        if count >= limit:
            break

    write_jsonl(OUT / "wildchat_english_100k.jsonl", output)
    print(f"WildChat saved: {count:,} conversations")


# Run all dataset jobs

if __name__ == "__main__":
    process_lmsys(limit=100_000_000)

    for platform in [
        "claude",
        "chatgpt",
        "perplexity",
        "gemini",
        "grok",
    ]:
        process_sharechat(platform, limit=100_000_000)

    process_wildchat(limit=100_000_000)

    print("\nFinished. Output files are in processed_data/")
