## DATA EXTRACTION ##
## **articshift_drug_datasets.py**  <br />

A Python script that collects historical posts and comments from 37 drug-related subreddits using the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) API, which archives Reddit data going back further than Reddit's own API allows.

## Functionality

- Pulls **posts and comments** for each of the 37 target subreddits.
- Covers a **10-year** historical range per subreddit.
- Saves each subreddit's data to its own output file, so datasets can be worked with independently.

## Handling the API responsibly

Arctic Shift is a free, community-run API, and its docs note that excessive requests can trigger rate limiting. The script includes several safeguards to stay well within reasonable use:

| Safeguard | What it does |
|---|---|
| **Delay between requests** | Waits 5 seconds between each API call, rather than firing requests back-to-back. |
| **Proactive rate-limit check** | Reads the `X-RateLimit-Remaining` header on every response to see how many requests are left, and slows down as that number gets low, rather than waiting to be rate-limited before reacting. |
| **Moderate page size** (`PAGE_LIMIT = 100`) | Requests a reasonable chunk of results per call instead of the maximum allowed, reducing the times we call the API. |
| **Rate-limit handling** | If the API responds with a rate-limit error (HTTP 429), the script reads the `X-RateLimit-Reset` header to know exactly how long to wait, pauses for that duration, then automatically retries. |
| **Retry with backoff** | Failed requests (errors or timeouts) are retried up to 5 times, with increasing wait times between attempts, rather than retrying instantly. |

## Checkpointing 

The most important reliability feature is its **resumability**

- After every successful batch of data, it saves its exact progress, writing the last timestamp successfully fetched to a checkpoint file.
- If the script is interrupted for any reason (crash, rate limiting, manual stop, network issue), simply re-running it picks up **exactly where it left off**, resuming from the last saved timestamp rather than re-downloading everything from the beginning.
- Once a subreddit's full 10-year range is completely collected, it's marked as done and skipped automatically on future runs.

## Output

Each subreddit produces separate files for posts and comments, so the 37 subreddits result in a structured, per-subreddit dataset ready for downstream analysis.
