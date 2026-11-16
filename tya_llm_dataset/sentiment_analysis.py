"""
Quick analysis of the processed JSONL files:
   - sentiment (VADER)
   - most repeated words and bigrams
   - topics (TF-IDF + NMF)

Usage:
    python sentiment_analysis.py

Requirements:
    pip install vaderSentiment scikit-learn
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = Path("processed_data")
SAMPLE_SIZE = 50_000      # max prompts analysed per file (reservoir sample)
N_TOPICS = 8
TOP_WORDS = 25
MAX_CHARS = 1_000         # truncate long prompts (speed, and VADER works best short)
SEED = 42

TOKEN = re.compile(r"[a-zA-Z]{3,}")
STOP = set(ENGLISH_STOP_WORDS)


def extract_prompts(row):
    """Normalise the different output schemas into a list of prompt strings."""
    if "plain_text" in row:                     # ShareChat
        return [row["plain_text"]]
    if "prompts" in row:                        # WildChat, Arena, ShareGPT
        return row["prompts"]
    if "conversation" in row:                   # LMSYS
        return [t["content"] for t in row["conversation"]]
    return []


def reservoir_sample(path, k, rng):
    """Uniformly sample k prompts from a file without loading it all."""
    sample, seen = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            for p in extract_prompts(json.loads(line)):
                if not isinstance(p, str) or not p.strip():
                    continue
                p = p[:MAX_CHARS]
                seen += 1
                if len(sample) < k:
                    sample.append(p)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < k:
                        sample[j] = p
    return sample, seen


def sentiment_summary(prompts):
    sia = SentimentIntensityAnalyzer()
    pos = neg = neu = 0
    total = 0.0
    for p in prompts:
        c = sia.polarity_scores(p)["compound"]
        total += c
        if c >= 0.05:
            pos += 1
        elif c <= -0.05:
            neg += 1
        else:
            neu += 1
    n = len(prompts)
    return {
        "mean_compound": round(total / n, 3),
        "positive_%": round(100 * pos / n, 1),
        "neutral_%": round(100 * neu / n, 1),
        "negative_%": round(100 * neg / n, 1),
    }


def top_words(prompts, n):
    words, bigrams = Counter(), Counter()
    for p in prompts:
        toks = [t for t in TOKEN.findall(p.lower()) if t not in STOP]
        words.update(toks)
        bigrams.update(zip(toks, toks[1:]))
    return (words.most_common(n),
            [(" ".join(b), c) for b, c in bigrams.most_common(n)])


def topics(prompts, n_topics, n_terms=10):
    vec = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        max_df=0.5, min_df=5, max_features=20_000,
    )
    X = vec.fit_transform(prompts)
    nmf = NMF(n_components=n_topics, random_state=SEED, init="nndsvd", max_iter=300)
    W = nmf.fit_transform(X)
    terms = vec.get_feature_names_out()
    share = Counter(W.argmax(axis=1))           # prompts whose strongest topic is k
    out = []
    for k, comp in enumerate(nmf.components_):
        top = [terms[i] for i in comp.argsort()[::-1][:n_terms]]
        out.append((k, round(100 * share[k] / len(prompts), 1), top))
    return sorted(out, key=lambda t: -t[1])


def analyze_file(path, rng):
    prompts, seen = reservoir_sample(path, SAMPLE_SIZE, rng)
    lines = [f"\n{'=' * 70}\n{path.name}  (analysed {len(prompts):,} of {seen:,} prompts)\n{'=' * 70}"]
    if len(prompts) < 100:
        lines.append("Too few prompts to analyse.")
        return "\n".join(lines)

    lines.append("\nSENTIMENT (VADER)")
    for k, v in sentiment_summary(prompts).items():
        lines.append(f"  {k}: {v}")

    words, bigrams = top_words(prompts, TOP_WORDS)
    lines.append("\nTOP WORDS")
    lines.append("  " + ", ".join(f"{w} ({c:,})" for w, c in words))
    lines.append("\nTOP BIGRAMS")
    lines.append("  " + ", ".join(f"{w} ({c:,})" for w, c in bigrams))

    lines.append(f"\nTOPICS (NMF, {N_TOPICS}; % = prompts whose strongest topic it is)")
    for k, pct, top in topics(prompts, N_TOPICS):
        lines.append(f"  [{pct:>5}%] " + ", ".join(top))
    return "\n".join(lines)


if __name__ == "__main__":
    rng = random.Random(SEED)
    report = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        print(f"Analysing {path.name}...")
        text = analyze_file(path, rng)
        print(text)
        report.append(text)

    Path("analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nSaved analysis_report.txt")