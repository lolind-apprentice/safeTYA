#!/usr/bin/env python3
"""
zeroshot_two.py - progress tracking
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import argparse
import time
from openai import OpenAI
from collections import defaultdict


class BuySellClassifier:
    """Zero-shot classifier using OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini", batch_size: int = 5):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print(f" OpenRouter initialized with model: {model}")
        print(f"   Batch size: {batch_size}")
    
    def classify_text(self, text: str) -> Dict:
        """Classify a single text using OpenRouter."""
        system_prompt = """You are a classifier for Reddit posts. Classify each post into one of these categories:
- SELLING DRUGS: Post is advertising, selling, or offering drugs for sale.
- BUYING DRUGS: Post is looking to buy, source, or find drugs.
- RECOVERY SUPPORT: Post is about recovery, sobriety, harm reduction.
- GENERAL DISCUSSION: Post is not about buying or selling drugs.

Respond with ONLY ONE WORD: SELL, BUY, RECOVERY, or GENERAL."""
        
        if len(text) < 10:
            return {"label": "general discussion", "confidence": 1.0}
        
        if len(text) > 4000:
            text = text[:4000]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0.1,
            )
            
            prediction = response.choices[0].message.content.strip().upper()
            
            label_map = {
                "SELL": "selling drugs",
                "BUY": "buying drugs", 
                "RECOVERY": "recovery support",
                "GENERAL": "general discussion"
            }
            
            return {
                "label": label_map.get(prediction, "general discussion"),
                "confidence": 0.9,
                "raw_output": prediction
            }
            
        except Exception as e:
            return {
                "label": "general discussion",
                "confidence": 0.0,
                "raw_output": f"ERROR: {str(e)[:50]}"
            }
    
    def classify_batch(self, texts: List[str], desc: str = "") -> List[Dict]:
        """Classify a batch of texts with progress bar."""
        results = []
        
        # Use tqdm for progress
        for text in tqdm(texts, desc=desc, leave=False):
            results.append(self.classify_text(text))
            # Small delay to avoid rate limits
            time.sleep(0.02)
        
        return results


def extract_text(row: dict) -> str:
    """Extract text from post or comment."""
    parts = []
    for field in ("title", "selftext", "body"):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts)


def load_jsonl(path: Path, max_rows: int = None):
    """Load JSONL file with progress."""
    rows = []
    if not path.exists():
        return rows
    
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        if max_rows:
            lines = lines[:max_rows]
        
        for line in tqdm(lines, desc=f"Loading {path.name}", leave=False):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def process_filtered_files(input_dir: Path, output_dir: Path, classifier, max_datasets: int = None, max_posts: int = None):
    """Process all filtered output files."""
    # Find all filtered files and group them by dataset
    all_files = list(input_dir.glob("*_out_*_*.jsonl"))
    
    if not all_files:
        print(f"\nNo filtered files found in {input_dir}\n")
        return []
    
    # Group by dataset
    datasets = defaultdict(lambda: {"posts": {"sell": [], "buy": []}, "comments": {"sell": [], "buy": []}})
    
    for file_path in all_files:
        parts = file_path.stem.split("_out_")
        if len(parts) != 2:
            continue
        
        dataset = parts[0]
        rest = parts[1]
        
        if "_" in rest:
            type_part, category = rest.split("_")
            if type_part == "post":
                datasets[dataset]["posts"][category].append(file_path)
            elif type_part == "comment":
                datasets[dataset]["comments"][category].append(file_path)
    
    print(f"\nFound {len(datasets)} datasets to process\n")
    
    # Limit datasets
    if max_datasets and max_datasets > 0:
        datasets = dict(list(datasets.items())[:max_datasets])
        print(f"\nProcessing first {len(datasets)} datasets\n")
    
    all_results = []
    
    # Main progress bar for datasets
    for dataset_name, files in tqdm(datasets.items(), desc="Processing datasets", unit="dataset"):
        print(f"\n{'='*60}")
        print(f"📂 Dataset: {dataset_name}")
        print(f"{'='*60}\n")
        
        dataset_results = []
        
        # Process sell posts
        sell_post_files = files["posts"].get("sell", [])
        for file_path in sell_post_files:
            print(f"   📄 Processing: {file_path.name}\n")
            posts = load_jsonl(file_path, max_rows=max_posts)
            print(f"      Loaded {len(posts)} posts\n")
            
            if not posts:
                continue
            
            # Extract texts
            # Batch extraction before classification
            texts = []
            valid_posts = []
            for post in posts:
                text = extract_text(post)
                if len(text) >= 10:
                    texts.append(text[:4000])
                    valid_posts.append(post)
            
            if not texts:
                continue
            
            # Classify with progress
            print(f"      Classifying {len(texts)} posts...\n")
            results = classifier.classify_batch(texts, desc=f"      {file_path.name}")
            
            for post, result in zip(valid_posts, results):
                dataset_results.append({
                    "dataset": dataset_name,
                    "file_type": "post_sell",
                    "id": post.get('id', ''),
                    "title": post.get('title', ''),
                    "text": extract_text(post)[:500],
                    "full_text": extract_text(post),
                    "original_classification": "sell",
                    "bert_prediction": result["label"],
                    "confidence": result["confidence"],
                    "raw_output": result.get("raw_output", ""),
                })
        
        # Process buy posts
        buy_post_files = files["posts"].get("buy", [])
        for file_path in buy_post_files:
            print(f"   📄 Processing: {file_path.name}\n")
            posts = load_jsonl(file_path, max_rows=max_posts)
            print(f"      Loaded {len(posts)} posts")
            
            if not posts:
                continue
            
            texts = []
            valid_posts = []
            for post in posts:
                text = extract_text(post)
                if len(text) >= 10:
                    texts.append(text[:4000])
                    valid_posts.append(post)
            
            if not texts:
                continue
            
            print(f"      Classifying {len(texts)} posts...\n")
            results = classifier.classify_batch(texts, desc=f"      {file_path.name}")
            
            for post, result in zip(valid_posts, results):
                dataset_results.append({
                    "dataset": dataset_name,
                    "file_type": "post_buy",
                    "id": post.get('id', ''),
                    "title": post.get('title', ''),
                    "text": extract_text(post)[:500],
                    "full_text": extract_text(post),
                    "original_classification": "buy",
                    "bert_prediction": result["label"],
                    "confidence": result["confidence"],
                    "raw_output": result.get("raw_output", ""),
                })
        
        # Skip comments for speed (optional)
        print(f"      Skipping comments for speed (use --process-comments to include)\n")
        
        # Save dataset results
        if dataset_results:
            df = pd.DataFrame(dataset_results)
            output_file = output_dir / f"{dataset_name}_classified.csv"
            df.to_csv(output_file, index=False)
            print(f"   Saved {len(dataset_results)} results to: {output_file}\n")
            all_results.extend(dataset_results)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot buy/sell classifier for filtered files")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with filtered files")
    parser.add_argument("--output-dir", type=Path, default=Path("./openrouter_output"), help="Output directory")
    parser.add_argument("--api-key", type=str, required=True, help="OpenRouter API key")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.2-3b-instruct:free", help="Model to use")
    parser.add_argument("--max-datasets", type=int, default=5, help="Max datasets to process")
    parser.add_argument("--max-posts", type=int, default=None, help="Max posts per file")
    parser.add_argument("--full", action="store_true", help="Process all datasets")
    parser.add_argument("--process-comments", action="store_true", help="Also process comments (slower)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    classifier = BuySellClassifier(args.api_key, args.model)
    
    # Process files
    max_datasets = None if args.full else args.max_datasets
    results = process_filtered_files(
        args.input_dir, 
        args.output_dir, 
        classifier, 
        max_datasets,
        args.max_posts
    )
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        df = pd.DataFrame(results)
        print(f"\n Total classified: {len(df)} posts/comments")
        
        print("\nOriginal classification breakdown:")
        original_stats = df['original_classification'].value_counts()
        for label, count in original_stats.items():
            print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nBERT prediction breakdown:")
        bert_stats = df['bert_prediction'].value_counts()
        for label, count in bert_stats.items():
            print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nOriginal vs BERT Prediction:")
        confusion = pd.crosstab(df['original_classification'], df['bert_prediction'])
        print(confusion)
        
        combined_path = args.output_dir / "all_classified.csv"
        df.to_csv(combined_path, index=False)
        print(f"\n Combined output saved to: {combined_path}")
    
    print(f"\nAll output saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()