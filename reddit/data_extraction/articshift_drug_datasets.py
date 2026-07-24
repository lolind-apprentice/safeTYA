import requests
import json
import time
import os 
from datetime import datetime, timedelta, timezone

BASE_URL="https://arctic-shift.photon-reddit.com"
OUTPUT_DIR ="output"
CHECKPOINT_FILE= os.path.join(OUTPUT_DIR,"checkpoint.json")

# settings
SUBREDDITS = [ "1P_LSD",
    "1V_LSD",
    "2cb",
    "4acodmt",
    "5MeODMT",
    "Ayahuasca",
    "CannabisExtracts",
    "DMT",
    "DMXE",
    "DPH",
    "DrugCombos",
    "LSA",
    "LSD",
    "MDMA",
    "MXE",
    "MemantineHCl",
    "Nootropics",
    "PCP",
    "aMT",
    "adderall",
    "afinil",
    "ambien",
    "anabolic",
    "benzodiazepines",
    "cannabis",
    "cocaine",
    "dissociatives",
    "drugs",
    "dxm",
    "fentanyl",
    "ketamine",
    "mescaline",
    "meth",
    "modafinil",
    "noids",
    "opiates",
    "treedibles"
]

YEARS_BACK = 10
PAGE_LIMIT = 100  # results per request (max ~100 for search endpoint)
REQUEST_DELAY_SECONDS = 5.0 
MAX_RETRIES = 5

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f) #json to str
        except json.JSONDecodeError:
            print("Checkpoint file corrupted.")
            return {}
    return {}
  
def save_checkpoint(checkpoint):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

def get_request(url_end: str, params: dict) -> dict:
    # get url by using artic shift api
    new_url = f"{BASE_URL}{url_end}"
    for attempt in range(1, MAX_RETRIES+1): # number of attempts
        # try and except to avoid errors
        try:
            response = requests.get(new_url, params=params, timeout=60)
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining is not None and int(remaining) < 5:
                reset = int(response.headers.get("X-RateLimit-Reset", 10))
                print(f"\tOnly {remaining} requests left, pausing {reset}s to be safe...")
                time.sleep(reset)
                
            if response.status_code == 429: # rate limiting requests
                wait = int(response.headers.get("X-RateLimit-Reset", 30))
                print(f"\tRate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            wait = min(60, 2 ** attempt)
            print(f"\tError ({e}). Retry {attempt}/{MAX_RETRIES} in {wait}s...")
            time.sleep(wait)
    print(f"\tGiving up on {new_url} with params {params} after {MAX_RETRIES} retries.")
    return None


def download_subreddit(subreddit, kind, after_ts, before_ts, checkpoint):
    """
    kind = 'posts' or 'comments'
    Paginates ascending by created_utc, appending results to a .jsonl file.
    Resumes from checkpoint if this subreddit/kind was partially done.
    """
    key = f"{subreddit}:{kind}"
    # checkpoint is dictionary, tell us last time it fetched a comment/post
    # cursor tells it where to resume from, either the last timestamp it successfully fetched, or the very start
    cursor = checkpoint.get(key, {}).get("last_created_utc", after_ts) 
    done = checkpoint.get(key, {}).get("done", False) #checkpoint to see if whole subreddit is already done
 
    if done:
        print(f"\t[{kind}] already complete, skipping.")
        return
 
    out_path = os.path.join(OUTPUT_DIR, f"{subreddit}_{kind}_ouput.jsonl")
    mode = "a" if os.path.exists(out_path) else "w"
 
    endpoint = f"/api/{kind}/search"
    total_fetched = checkpoint.get(key, {}).get("total_fetched", 0)
 
    with open(out_path, mode, encoding="utf-8") as f:
        while True:
            params = {
                "subreddit": subreddit,
                "after": cursor,
                "before": before_ts,
                "limit": PAGE_LIMIT,
                "sort": "asc",
            }
            result = get_request(endpoint, params)
            if result is None:
                # save progress and bail; re-running will resume from here.
                checkpoint[key] = {
                    "last_created_utc": cursor,
                    "done": False,
                    "total_fetched": total_fetched,
                }
                save_checkpoint(checkpoint)
                print(f"\t[{subreddit} {kind}] stopped early due to repeated errors. "                  
                      f"Progress saved — re-run to resume.")
                return
 
            items = result.get("data", [])
            if not items:
                break
 
            for item in items:
                f.write(json.dumps(item) + "\n")
            total_fetched += len(items)
            cursor = items[-1]["created_utc"] + 1  # avoid re-fetching same item
 
            print(f"  [{kind}] fetched {total_fetched} so far "
                  f"(cursor at {datetime.fromtimestamp(cursor, tz=timezone.utc).date()})")
 
            checkpoint[key] = {
                "last_created_utc": cursor,
                "done": False,
                "total_fetched": total_fetched,
            }
            save_checkpoint(checkpoint)
 
            if len(items) < PAGE_LIMIT:
                break  # last page
 
            time.sleep(REQUEST_DELAY_SECONDS)
 
    checkpoint[key] = {
        "last_created_utc": cursor,
        "done": True,
        "total_fetched": total_fetched,
    }
    save_checkpoint(checkpoint)
    print(f"[{subreddit} {kind}] done. Total: {total_fetched}")
 
def main():
    if not SUBREDDITS:
        print("Add your subreddit names to the SUBREDDITS list before running.")
        return
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = load_checkpoint()
 
    now = datetime.now(timezone.utc)
    after_ts = int((now - timedelta(days=365 * YEARS_BACK)).timestamp())
    before_ts = int(now.timestamp())
 
    for i, subreddit in enumerate(SUBREDDITS, 1):
        print(f"\n[{i}/{len(SUBREDDITS)}] r/{subreddit}")
        for kind in ("posts", "comments"):
            download_subreddit(subreddit, kind, after_ts, before_ts, checkpoint)
 
    print("\nAll done. Files are in the 'output/' folder.")
 
 
if __name__ == "__main__":
    main()


