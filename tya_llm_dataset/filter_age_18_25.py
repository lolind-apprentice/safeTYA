"""
Filter JSONL prompt datasets for explicit textual clues that the writer may be
18-25 years old. Recursively processes every .jsonl file in an input folder.

  python filter_18_25.py --input-folder processed_data
  python filter_18_25.py --input-folder processed_data --workers 8
  python filter_18_25.py --input-folder data --include-life-stage

Keyword/regex matching only; it does NOT determine anyone's actual age.
"""

import argparse
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Iterable

SNIPPET_CHARS = 100  # context kept on each side of a match

# ---- single combined regex (one pass instead of 7) -------------------------
NUM = r"(?:1[89]|2[0-5])"
WORD = r"(?:eighteen|nineteen|twenty(?:[- ](?:one|two|three|four|five))?)"
ME = r"(?:i am|i'm|im)"
AGE_WORD = r"(?:[- ]?(?:years?|yrs?)[- ]?old|[- ]?y/?o)"

AGE_RE = re.compile(
    rf"\b(?:"
    rf"(?P<a>{ME}\s+(?:currently\s+|about\s+|aged\s+)?{NUM}\s*{AGE_WORD})"
    rf"|(?P<b>(?:i just turned|i recently turned|i'm turning|i am turning)\s+{NUM})"
    # bare "I'm 20" only if it ends the clause (avoids "I'm 20 minutes late")
    rf"|(?P<c>{ME}\s+{NUM}(?=\s*(?:[,.;:!?)]|$|and\b)))"
    rf"|(?P<d>(?:my age is|age:)\s*{NUM})"
    rf"|(?P<e>{ME}\s+{WORD}(?:{AGE_WORD})?)"
    rf"|(?P<f>{ME}\s+in\s+my\s+(?:early|mid|late)\s+twenties)"
    rf"|(?P<g>(?:i was born in|born in)\s+200[1-8])"
    rf")\b",
    re.IGNORECASE,
)

LIFE_RE = re.compile(
    r"\b(?:(?:i am|i'm|im)\s+(?:a\s+)?(?:college|university)\s+"
    r"(?:student|undergrad|freshman|sophomore|junior|senior)"
    r"|my\s+(?:freshman|sophomore|junior|senior)\s+year\s+of\s+(?:college|uni|university))\b",
    re.IGNORECASE,
)

# Cheap prefilter: most texts contain none of these, so skip the big regex.
PREFILTER = re.compile(r"i am|i'm|im |turn|age|born", re.IGNORECASE)

KIND = {"a": "explicit_age", "b": "turning_age", "c": "bare_age", "d": "age_declaration",
        "e": "spelled_age", "f": "decade", "g": "birth_year"}


def collect_texts(row: dict[str, Any], text_key: str | None = None) -> list[str]:
    values: list[str] = []
    if text_key and text_key in row:
        val = row[text_key]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        if isinstance(val, list):
            return [t.strip() for t in val if isinstance(t, str) and t.strip()]

    if isinstance(row.get("plain_text"), str):
        values.append(row["plain_text"])
    if isinstance(row.get("prompts"), list):
        values += [p.strip() for p in row["prompts"] if isinstance(p, str) and p.strip()]
    if isinstance(row.get("conversation"), list):
        for turn in row["conversation"]:
            if isinstance(turn, dict):
                c = turn.get("content")
                if isinstance(c, str) and c.strip():
                    values.append(c.strip())
    return list(dict.fromkeys(values))


def find_clues(texts: Iterable[str], include_life_stage: bool) -> list[dict[str, str]]:
    out = []
    for text in texts:
        t = text.replace("’", "'").replace("‘", "'")
        if not PREFILTER.search(t) and not (include_life_stage and "college" in t.lower()):
            continue
        for m in AGE_RE.finditer(t):
            kind = KIND[m.lastgroup]
            s, e = max(0, m.start() - SNIPPET_CHARS), min(len(t), m.end() + SNIPPET_CHARS)
            out.append({"clue_type": kind, "matched_text": m.group(0), "snippet": t[s:e]})
        if include_life_stage:
            for m in LIFE_RE.finditer(t):
                s, e = max(0, m.start() - SNIPPET_CHARS), min(len(t), m.end() + SNIPPET_CHARS)
                out.append({"clue_type": "life_stage_only", "matched_text": m.group(0),
                            "snippet": t[s:e]})
    return out


def process_batch(args):
    lines, include_life_stage, text_key = args
    out, total = [], 0
    for line in lines:
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        clues = find_clues(collect_texts(row, text_key), include_life_stage)
        if clues:
            row["_age_pattern_matches"] = clues
            row["_age_range_candidate"] = "18-25"
            out.append(json.dumps(row, ensure_ascii=False, default=str))
    return total, out


def batches(f, size, include_life_stage, text_key):
    buf = []
    for line in f:
        buf.append(line)
        if len(buf) >= size:
            yield buf, include_life_stage, text_key
            buf = []
    if buf:
        yield buf, include_life_stage, text_key


def process_file(inp: Path, outp: Path, include_life_stage, text_key, workers, pool):
    outp.parent.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    with inp.open("r", encoding="utf-8") as src, outp.open("w", encoding="utf-8") as dst:
        for t, rows in pool.imap(process_batch, batches(src, 2000, include_life_stage, text_key)):
            total += t
            kept += len(rows)
            for r in rows:
                dst.write(r + "\n")
    return total, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--output-folder")
    ap.add_argument("--text-key")
    ap.add_argument("--include-life-stage", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    root = Path(args.input_folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input folder invalid: {root}")
    out_root = (Path(args.output_folder).expanduser().resolve() if args.output_folder
                else root.parent / f"{root.name}_tya")

    files = sorted(p for p in root.rglob("*.jsonl")
                   if p.is_file() and not p.stem.endswith("_tya"))
    total_rows = kept_rows = 0

    with Pool(args.workers) as pool:
        for f in files:
            rel = f.relative_to(root)
            outf = out_root / rel.parent / f"{f.stem}_tya{f.suffix}"
            t, k = process_file(f, outf, args.include_life_stage, args.text_key,
                                args.workers, pool)
            total_rows += t
            kept_rows += k
            print(f"{rel}: scanned {t:,}, kept {k:,}")

    print(f"\nFinished. Scanned: {total_rows:,}, Kept: {kept_rows:,}")


if __name__ == "__main__":
    main()