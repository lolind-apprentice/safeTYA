#!/usr/bin/env python3

"""
Reddit post analysis script -- r/opiates r/drugs.

Adds:
  1. Drug term co-occurrence analysis (which drugs are mentioned TOGETHER
     in the same post -- e.g. fentanyl+heroin can signal adulterant/
     contamination concerns, a known harm-reduction theme)
  2. VADER sentiment/distress scoring (tone, not just keyword presence)

Install requirements first:
    pip install pandas vaderSentiment 

Usage:
    python3 reddit_pattern_analysis.py /path/to/r_opiates_posts.jsonl
"""

import sys
import json
import re
from itertools import combinations
from collections import Counter

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BUY_SELL_TERMS = ["buy", "buying", "sell", "selling", "wts", "wtb", "plug", "source", "hook up", "hookup"]

# Opiate/opioid-relevant terms
DRUG_NAMES = [
    "fentanyl", "fent", "heroin", "oxycodone", "oxy", "oxycontin", "hydrocodone",
    "vicodin", "methadone", "suboxone", "buprenorphine", "subutex", "morphine",
    "codeine", "tramadol", "percocet", "percs", "dilaudid", "hydromorphone",
    "kratom", "opium", "naloxone", "narcan",
]

RECOVERY_TERMS = ["recover", "recovery", "clean", "relapse", "sober", "sobriety",
                  "withdrawal", "withdrawals", "quitting", "quit", "taper", "tapering"]


def word_in_text(term, text):
    """Word-boundary match so 'oxy' doesn't match inside 'toxic', 'hypoxia', etc."""
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def load_jsonl_to_df(path):
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            records.append({
                "username": post.get("author", "[deleted]"),
                "created_utc": post.get("created_utc"),
                "subreddit": post.get("subreddit", ""),
                "url": post.get("url") or post.get("permalink", ""),
                "title": post.get("title", "") or "",
                "selftext": post.get("selftext", "") or "",
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "removed_by_category": post.get("removed_by_category"),
            })
    if skipped:
        print(f"  [note] skipped {skipped} malformed lines")
    df = pd.DataFrame(records)
    df["time_iso"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    return df


def categorize(df):
    text = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.lower()

    df["has_buy_sell_language"] = text.apply(lambda t: any(word_in_text(term, t) for term in BUY_SELL_TERMS))
    df["has_recovery_language"] = text.apply(lambda t: any(word_in_text(term, t) for term in RECOVERY_TERMS))

    def matched_drugs(t):
        return [d for d in DRUG_NAMES if word_in_text(d, t)]
    df["matched_drug_terms"] = text.apply(matched_drugs)
    df["has_drug_name_mention"] = df["matched_drug_terms"].apply(lambda lst: len(lst) > 0)

    return df, text


def drug_cooccurrence(df, top_n=15):
    """
    For posts mentioning 2+ drug terms, count which PAIRS appear together.
    This is the co-occurrence signal -- e.g. fentanyl+heroin appearing
    together often signals adulterant/contamination concern threads.
    """
    pair_counter = Counter()
    for terms in df["matched_drug_terms"]:
        unique_terms = sorted(set(terms))
        if len(unique_terms) >= 2:
            for pair in combinations(unique_terms, 2):
                pair_counter[pair] += 1
    return pair_counter.most_common(top_n)


def post_length_stats(df):
    """Word count of title+selftext, compared across categories."""
    df["word_count"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.split().apply(len)
    return df


def time_of_day_stats(df):
    """Hour of day (UTC) each post was made, for late-night posting checks."""
    df["hour_utc"] = df["time_iso"].dt.hour
    return df


def posting_frequency_per_author(df, min_posts=2):
    """
    Count posts per author and, for repeat posters, list their posts
    in chronological order with category flags -- lets you eyeball
    escalating/de-escalating trajectories (e.g. 'day 1 clean' -> 'relapsed').
    """
    counts = df["username"].value_counts()
    repeat_authors = counts[counts >= min_posts].index
    repeat_df = df[df["username"].isin(repeat_authors)].sort_values(["username", "time_iso"])
    return counts, repeat_df


def sentiment_scores(text_series):
    analyzer = SentimentIntensityAnalyzer()
    compounds, negs, poss, neus = [], [], [], []
    for t in text_series:
        # VADER works best on shorter text; truncate very long posts for speed
        # without losing the overall tone signal.
        scores = analyzer.polarity_scores(t[:3000])
        compounds.append(scores["compound"])
        # get a sentiment score
        negs.append(scores["neg"])
        poss.append(scores["pos"])
        neus.append(scores["neu"])
    return compounds, negs, poss, neus


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 reddit_pattern_analysis.py file.jsonl")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path} ...")
    df = load_jsonl_to_df(path)
    print(f"Loaded {len(df)} posts.\n")

    df, text = categorize(df)
    n = len(df)

    print("=== Category counts (NOT mutually exclusive, NOT merged) ===")
    for col in ["has_buy_sell_language", "has_drug_name_mention", "has_recovery_language"]:
        count = df[col].sum()
        print(f"  {col}: {count} ({count/n*100:.1f}% of posts)")

    print("\n=== Overlap counts ===")
    print(f"  buy_sell AND recovery: {(df['has_buy_sell_language'] & df['has_recovery_language']).sum()}")
    print(f"  buy_sell AND drug_name: {(df['has_buy_sell_language'] & df['has_drug_name_mention']).sum()}")
    print(f"  recovery AND drug_name: {(df['has_recovery_language'] & df['has_drug_name_mention']).sum()}")

    print("\n=== Top individual drug terms mentioned ===")
    all_terms = df["matched_drug_terms"].explode().dropna()
    print(all_terms.value_counts().head(15).to_string())

    print("\n=== Drug term CO-OCCURRENCE (pairs mentioned in the same post)===")
    pairs = drug_cooccurrence(df)
    for (a, b), count in pairs:
        print(f"  {a} + {b}: {count}")

    print("\n=== Post length (word count) by category ===")
    df = post_length_stats(df)
    for col in ["has_buy_sell_language", "has_recovery_language", "has_drug_name_mention"]:
        avg_true = df.loc[df[col], "word_count"].mean()
        avg_false = df.loc[~df[col], "word_count"].mean()
        print(f"  {col}=True avg words: {avg_true:.1f}  |  {col}=False avg words: {avg_false:.1f}")
    print(f"  Overall median word count: {df['word_count'].median():.0f}")

    print("\n=== Time-of-day posting pattern (UTC hour) ===")
    df = time_of_day_stats(df)
    print(df["hour_utc"].value_counts().sort_index().to_string())
    late_night = df["hour_utc"].apply(lambda h: h >= 0 and h < 6)  # midnight-6am UTC as example window
    print(f"  Posts between 00:00-06:00 UTC: {late_night.sum()} ({late_night.sum()/n*100:.1f}%)")

    print("\n=== Posting frequency per author ===")
    author_counts, repeat_df = posting_frequency_per_author(df, min_posts=2)
    print(f"  Total unique authors: {df['username'].nunique()}")
    print(f"  Authors with 2+ posts: {(author_counts >= 2).sum()}")
    print(f"  Authors with 5+ posts: {(author_counts >= 5).sum()}")
    print(f"  Most active author posted {author_counts.max()} times")
    repeat_out_path = "repeat_authors_chronological.csv"
    repeat_df[["username", "time_iso", "title", "has_buy_sell_language",
               "has_recovery_language", "has_drug_name_mention"]].to_csv(repeat_out_path, index=False)
    print(f"  Wrote chronological repeat-author posts to {repeat_out_path}")
    print("  (NOTE: pseudonymize usernames before publishing any individual trajectory)")

    print("\n=== Running VADER sentiment scoring (this takes a bit on large files)===")
    compounds, negs, poss, neus = sentiment_scores(text)
    df["sentiment_compound"] = compounds
    df["sentiment_neg"] = negs
    df["sentiment_pos"] = poss
    df["sentiment_neu"] = neus

    print("\n=== Average sentiment by category ===")
    for col in ["has_buy_sell_language", "has_recovery_language", "has_drug_name_mention"]:
        avg_true = df.loc[df[col], "sentiment_compound"].mean()
        avg_false = df.loc[~df[col], "sentiment_compound"].mean()
        print(f"  {col}=True avg compound sentiment: {avg_true:.3f}  |  {col}=False: {avg_false:.3f}")

    print("\n=== Most negative posts overall (lowest compound score) ===")
    most_neg = df.nsmallest(5, "sentiment_compound")[["title", "sentiment_compound"]]
    print(most_neg.to_string(index=False))

    print("\n=== Posts per year ===")
    print(df["time_iso"].dt.year.value_counts().sort_index().to_string())

    out_cols = ["username", "time_iso", "subreddit", "url", "title", "selftext",
                "score", "num_comments", "removed_by_category",
                "has_buy_sell_language", "has_drug_name_mention", "has_recovery_language",
                "matched_drug_terms", "word_count", "hour_utc",
                "sentiment_compound", "sentiment_neg", "sentiment_pos", "sentiment_neu"]
    out_path = "tagged__posts_fin.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote full tagged dataset (with sentiment + drug terms) to {out_path}")


if __name__ == "__main__":
    main()