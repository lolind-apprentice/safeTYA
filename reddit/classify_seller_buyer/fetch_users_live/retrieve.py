"""Retrieve one OP-R1 user's nested record without downloading the corpus.
This final access stage joins the indexes produced by ``reshard_posts`` with
the nested records produced by ``user_threads_to_parquet``.

All credit to https://github.com/Tianyi-Billy-Ma/OP-R1

Usage:
    python retrieve.py --user-id USER_ID [--out user.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

import fsspec
import pyarrow.parquet as pq
from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfFileSystem
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_file_system import HfFileSystemFile

# --- inlined from the OP-R1 pipeline's common.py -------------------------
# The upstream Reddit dumps are not type-clean: parent_id/link_id sometimes
# arrive as ints rather than "t1_abc" strings. Coerce defensively.
def coerce_required_str_with_status(value) -> tuple[str, bool]:
    if isinstance(value, str):
        return value, False
    return str(value), True
# ------------------------------------------------------------------------


POST_COLUMNS: Final = [
    "id", #post/comment id
    "kind", #type of item
    "subreddit", #subreddit name
    "source", #where came from
    "user_id", #author id
    "created_utc", 
    "text", #content7
    "score", #reddit score
    "link_id", #id of thread
    "parent_id", #id of parent
]

#maps user 
USER_INDEX_PATH: Final = "index/user_index.parquet"
ID_INDEX_PATH: Final = "index/id_index.parquet"
REDDIT_ID_PREFIXES: Final = ("t1_", "t3_")
RANGE_BLOCK_SIZE: Final = 64 * 1024
type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]


class RetrievalError(RuntimeError):
    pass


class UnknownUserError(RetrievalError):
    pass


class UnknownItemError(RetrievalError):
    pass


@final
class _TransferStats:
    """Mutable accumulator shared by every opened file in one retriever."""

    __slots__ = ("bytes_fetched", "http_range_requests", "local_read_requests")

    def __init__(self) -> None:
        self.bytes_fetched = 0 #total bytes downloaded/read
        self.http_range_requests = 0 #requests
        self.local_read_requests = 0 #reads


@dataclass(frozen=True, slots=True)
class _UserSlice:
    subreddit: str
    path: str #parquet file cointaining the user's row
    row_start: int
    row_count: int #consecutive rows belong to user


@dataclass(frozen=True, slots=True)
class _ItemLocation:
    path: str # Parquet file containing the item
    row_offset: int


class _MeasuredHfFile(HfFileSystemFile):
    # A remote file that counts how many bytes/requests it uses. 
    def __init__(self, fs, path: str, *, stats: _TransferStats, **kwargs) -> None:
        self._stats = stats
        super().__init__(fs, path, **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        payload = super()._fetch_range(start, end)
        self._stats.bytes_fetched += len(payload)
        self._stats.http_range_requests += 1
        return payload


class _MeasuredHfFileSystem(HfFileSystem):
    # HF filesystem that opens "measured" files so downloads are counted.
    def __init__(self, *, stats: _TransferStats, **kwargs) -> None:
        self._stats = stats
        super().__init__(**kwargs)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        revision: str | None = None,
        **kwargs,
    ):
        effective_block_size = self.block_size if block_size is None else block_size
        if effective_block_size == 0 or mode != "rb":
            return super()._open(
                path,
                mode=mode,
                block_size=block_size,
                revision=revision,
                **kwargs,
            )
        if effective_block_size is not None:
            kwargs["block_size"] = effective_block_size
        return _MeasuredHfFile(
            self,
            path,
            mode=mode,
            revision=revision,
            stats=self._stats,
            **kwargs,
        )


class _MeasuredLocalFile:
    # Wraps a local file so reads are counted the same way as remote ones.
    def __init__(self, raw, stats: _TransferStats) -> None:
        self._raw = raw
        self._stats = stats

    def read(self, size: int = -1) -> bytes:
        payload = self._raw.read(size)
        self._record(payload)
        return payload

    def readline(self, size: int = -1) -> bytes:
        payload = self._raw.readline(size)
        self._record(payload)
        return payload

    def _record(self, payload: bytes) -> None:
        if payload:
            self._stats.bytes_fetched += len(payload)
            self._stats.local_read_requests += 1

    def __iter__(self) -> _MeasuredLocalFile:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self) -> _MeasuredLocalFile:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._raw.close()

    def __getattr__(self, name: str):
        return getattr(self._raw, name)


@dataclass(frozen=True, slots=True)
class _Repository:
    value: str
    revision: str
    filesystem: AbstractFileSystem
    local_root: Path | None
    stats: _TransferStats

    @classmethod
    def create(
        cls,
        value: str,
        *,
        revision: str,
        token: str | None,
        cache_dir: str | Path | None,
        stats: _TransferStats,
    ) -> _Repository:
        root = Path(value).expanduser()
        if root.is_dir():
            return cls(
                value, revision, fsspec.filesystem("file"), root.resolve(), stats
            )
        # Otherwise treat it as a Hugging Face dataset repo.
        filesystem = _MeasuredHfFileSystem(
            stats=stats,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        return cls(value, revision, filesystem, None, stats)

    @property
    def is_local(self) -> bool:
        return self.local_root is not None

    def path(self, relative: str) -> str:
        if self.local_root is not None:
            return str(self.local_root / relative)
        return f"datasets/{self.value}/{relative}"

    def open(self, relative: str):
        if self.local_root is not None:
            raw = self.filesystem.open(self.path(relative), "rb")
            return _MeasuredLocalFile(raw, self.stats)
        return self.filesystem.open(
            self.path(relative),
            "rb",
            revision=self.revision,
            block_size=RANGE_BLOCK_SIZE,
        )

    def data_files(self) -> list[str]:
        if self.local_root is not None:
            return sorted(
                str(path.relative_to(self.local_root))
                for path in self.local_root.rglob("*")
                if path.suffix in {".parquet", ".jsonl"}
            )
        root = f"datasets/{self.value}"
        paths = self.filesystem.find(root, revision=self.revision)
        return sorted(
            "/".join(path.split("/")[3:])
            for path in paths
            if Path(path).suffix in {".parquet", ".jsonl"}
        )


def _canonical_id(item_id: str) -> str:
    # Strip "t1_"/"t3_" so ids match the dataset's unprefixed form.
    return item_id[3:] if item_id.startswith(REDDIT_ID_PREFIXES) else item_id


def _stat_value(value) -> str:
    # Parquet min/max stats may be bytes; turn them into plain strings.
    return value.decode() if isinstance(value, bytes) else str(value)


def _candidate_row_groups(
    parquet: pq.ParquetFile,
    column: str,
    values: set[str],
) -> list[int]:
    # Find which row groups MIGHT contain the wanted values, using the
    # min/max statistics stored in the file footer (avoids reading all data).

    column_index = parquet.schema_arrow.get_field_index(column)
    if column_index < 0:
        raise RetrievalError(f"Parquet file is missing required column {column!r}.")
    candidates: list[int] = []
    for group_index in range(parquet.num_row_groups):
        statistics = (
            parquet.metadata.row_group(group_index).column(column_index).statistics
        )
        if statistics is None or not statistics.has_min_max:
            candidates.append(group_index)
            continue
        minimum = _stat_value(statistics.min)
        maximum = _stat_value(statistics.max)
        if any(minimum <= value <= maximum for value in values):
            candidates.append(group_index)
    return candidates


def _row_group_starts(parquet: pq.ParquetFile) -> list[int]:
    starts = [0]
    for group_index in range(parquet.num_row_groups):
        starts.append(
            starts[-1] + parquet.metadata.row_group(group_index).num_rows,
        )
    return starts


def _object_list(value: JsonValue, field: str) -> list[JsonObject]:
    # Validate that value is a list of dicts; return it typed.
    if not isinstance(value, list):
        raise RetrievalError(f"Structure field {field!r} must be a list.")
    objects: list[JsonObject] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise RetrievalError(f"Structure field {field!r} must contain objects.")
        objects.append(entry)
    return objects


def _string_list(value: JsonValue, field: str) -> list[str]:
    # Validate that value is a list of dicts; return it typed.
    if not isinstance(value, list) or not all(
        isinstance(entry, str) for entry in value
    ):
        raise RetrievalError(f"Structure field {field!r} must contain strings.")
    return [entry for entry in value if isinstance(entry, str)]


class OpR1Retriever:
    """Fetch indexed OP-R1 user records with byte-ranged Parquet reads."""

    def __init__(
        self,
        posts_repo: str = "OP-R1/OP-Reddit-Post",
        users_repo: str = "OP-R1/OP-Reddit-User",
        token: str | None = None,
        revision: str = "main",
        cache_dir: str | Path | None = None,
    ) -> None:
        self._stats = _TransferStats()
        self._posts = _Repository.create(
            posts_repo,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            stats=self._stats,
        )
        self._users = _Repository.create(
            users_repo,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            stats=self._stats,
        )
        self._token = token
        self._user_index_cache: dict[str, list[_UserSlice]] = {}
        self._missing_users: set[str] = set()
        self._id_index_cache: dict[str, _ItemLocation] = {}
        self._missing_ids: set[str] = set()
        self._item_cache: dict[str, JsonObject] = {}
        self._structure_cache: dict[str, JsonObject] = {}

    @property
    def bytes_fetched(self) -> int:
        return self._stats.bytes_fetched

    @property
    def http_range_requests(self) -> int:
        return self._stats.http_range_requests

    @property
    def local_read_requests(self) -> int:
        return self._stats.local_read_requests

    @contextmanager
    def _parquet(
        self,
        repo: _Repository,
        path: str,
        *,
        index: bool = False,
    ) -> Iterator[pq.ParquetFile]:
        # Opens a Parquet file for a "with" block and translates errors.
        try:
            with repo.open(path) as source:
                yield pq.ParquetFile(source)
        except FileNotFoundError as error:
            if index:
                raise RetrievalError(
                    f"Required index {path!r} is missing from {repo.value!r}; "
                    "publish the generated index file before retrieving records.",
                ) from error
            raise RetrievalError(
                f"Required data file {path!r} is missing from {repo.value!r}.",
            ) from error
        except HfHubHTTPError as error:
            status = getattr(error.response, "status_code", None)
            if status in {401, 403} or self._token is None:
                raise RetrievalError(
                    f"Cannot access Hugging Face dataset {repo.value!r}. "
                    "If it is private or gated, pass token=... or authenticate with "
                    "the HF_TOKEN environment variable.",
                ) from error
            raise RetrievalError(
                f"Hugging Face access failed for {repo.value!r}: {error}",
            ) from error

    def _user_slices(self, user_id: str) -> list[_UserSlice]:
        if user_id in self._user_index_cache:
            return self._user_index_cache[user_id]
        if user_id in self._missing_users:
            raise UnknownUserError(f"Unknown user_id {user_id!r}.")
        with self._parquet(self._posts, USER_INDEX_PATH, index=True) as parquet:
            groups = _candidate_row_groups(parquet, "user_id", {user_id})
            index_rows = parquet.read_row_groups(groups).to_pylist() if groups else []
        rows = [row for row in index_rows if row["user_id"] == user_id]
        if not rows:
            self._missing_users.add(user_id)
            raise UnknownUserError(
                f"Unknown user_id {user_id!r}; verify the anonymized id and posts repo.",
            )
        slices = [
            _UserSlice(
                subreddit=str(row["subreddit"]),
                path=str(row["path"]),
                row_start=int(row["row_start"]),
                row_count=int(row["row_count"]),
            )
            for row in rows
        ]
        self._user_index_cache[user_id] = slices
        return slices

    def _read_user_rows(self, user_id: str) -> list[JsonObject]:
        rows: list[JsonObject] = []
        for user_slice in self._user_slices(user_id):
            with self._parquet(self._posts, user_slice.path) as parquet:
                starts = _row_group_starts(parquet)
                row_end = user_slice.row_start + user_slice.row_count
                groups = [
                    group_index
                    for group_index in range(parquet.num_row_groups)
                    if starts[group_index] < row_end
                    and starts[group_index + 1] > user_slice.row_start
                ]
                if not groups:
                    raise RetrievalError(
                        f"Index range for {user_id!r} is outside {user_slice.path!r}.",
                    )
                table = parquet.read_row_groups(groups, columns=POST_COLUMNS)
                offset = user_slice.row_start - starts[groups[0]]
                selected = table.slice(offset, user_slice.row_count)
            if selected.num_rows != user_slice.row_count:
                raise RetrievalError(
                    f"Index range for {user_id!r} exceeds {user_slice.path!r}.",
                )
            selected_rows = selected.to_pylist()
            assert all(row["user_id"] == user_id for row in selected_rows), (
                f"Corrupt user index: {user_id!r} range includes another user"
            )
            rows.extend(selected_rows)
        return rows

    def _lookup_item_locations(self, ids: set[str]) -> None:
        missing = ids - self._id_index_cache.keys() - self._missing_ids
        if not missing:
            return
        with self._parquet(self._posts, ID_INDEX_PATH, index=True) as parquet:
            groups = _candidate_row_groups(parquet, "id", missing)
            index_rows = parquet.read_row_groups(groups).to_pylist() if groups else []
        found: set[str] = set()
        if index_rows:
            for row in index_rows:
                item_id, _ = coerce_required_str_with_status(row["id"])
                if item_id not in missing:
                    continue
                self._id_index_cache[item_id] = _ItemLocation(
                    path=str(row["path"]),
                    row_offset=int(row["row_offset"]),
                )
                found.add(item_id)
        self._missing_ids.update(missing - found)

    def get_items(self, ids: list[str]) -> dict[str, JsonObject]:
        """Fetch arbitrary records by id (used for anchors/parents/replies)."""
        requested = list(dict.fromkeys(ids))
        canonical = {_canonical_id(item_id) for item_id in requested}
        self._lookup_item_locations(canonical)
        unknown = sorted(canonical & self._missing_ids)
        if unknown:
            raise UnknownItemError(
                f"Unknown item id(s): {', '.join(unknown)}; verify id_index.parquet.",
            )

        by_path: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for item_id in canonical - self._item_cache.keys():
            location = self._id_index_cache[item_id]
            by_path[location.path].append((item_id, location.row_offset))

        for path, locations in by_path.items():
            with self._parquet(self._posts, path) as parquet:
                starts = _row_group_starts(parquet)
                group_for_id = {
                    item_id: bisect_right(starts, row_offset) - 1
                    for item_id, row_offset in locations
                }
                if any(
                    group < 0 or group >= parquet.num_row_groups
                    for group in group_for_id.values()
                ):
                    raise RetrievalError(f"id_index row offset is outside {path!r}.")
                groups = sorted(set(group_for_id.values()))
                table = parquet.read_row_groups(groups, columns=POST_COLUMNS)
                concatenated_starts: dict[int, int] = {}
                current = 0
                for group in groups:
                    concatenated_starts[group] = current
                    current += starts[group + 1] - starts[group]
                for item_id, row_offset in locations:
                    group = group_for_id[item_id]
                    local_offset = (
                        concatenated_starts[group] + row_offset - starts[group]
                    )
                    record = table.slice(local_offset, 1).to_pylist()[0]
                    if record["id"] != item_id:
                        raise RetrievalError(
                            f"Corrupt id index: expected {item_id!r}, got "
                            f"{record['id']!r} in {path!r}.",
                        )
                    self._item_cache[item_id] = record

        return {
            item_id: copy.deepcopy(self._item_cache[_canonical_id(item_id)])
            for item_id in requested
        }

    def _load_user_structure(self, user_id: str) -> JsonObject:
        if user_id in self._structure_cache:
            return copy.deepcopy(self._structure_cache[user_id])
        try:
            files = self._users.data_files()
        except (FileNotFoundError, HfHubHTTPError) as error:
            if self._token is None and not self._users.is_local:
                raise RetrievalError(
                    f"Cannot access Hugging Face dataset {self._users.value!r}; "
                    "pass token=... or authenticate with HF_TOKEN if it is private.",
                ) from error
            raise RetrievalError(
                f"Unable to list structure files in {self._users.value!r}: {error}",
            ) from error
        parquet_files = [path for path in files if path.endswith(".parquet")]
        jsonl_files = [path for path in files if path.endswith(".jsonl")]
        matches: list[JsonObject] = []
        for path in parquet_files:
            with self._parquet(self._users, path) as parquet:
                groups = _candidate_row_groups(parquet, "user_id", {user_id})
                if groups:
                    matches.extend(
                        row
                        for row in parquet.read_row_groups(groups).to_pylist()
                        if row["user_id"] == user_id
                    )
        for path in jsonl_files:
            row = self._find_jsonl_user(path, user_id)
            if row is not None:
                matches.append(row)
            if matches:
                break
        if not matches:
            raise UnknownUserError(
                f"User {user_id!r} exists in the posts index but has no structure "
                f"record in {self._users.value!r}.",
            )
        if len(matches) > 1:
            raise RetrievalError(
                f"User {user_id!r} has duplicate structure records in "
                f"{self._users.value!r}.",
            )
        self._structure_cache[user_id] = matches[0]
        return copy.deepcopy(matches[0])

    def _find_jsonl_user(self, path: str, user_id: str) -> JsonObject | None:
        with self._users.open(path) as source:
            source.seek(0, 2)
            low = 0
            high = source.tell()
            while low < high:
                middle = (low + high) // 2
                source.seek(middle)
                if middle:
                    source.readline()
                raw_line = source.readline()
                if not raw_line:
                    high = middle
                    continue
                row = json.loads(raw_line)
                current_id = row.get("user_id")
                if not isinstance(current_id, str):
                    raise RetrievalError(
                        f"Structure file {path!r} has a row without a user_id.",
                    )
                if current_id < user_id:
                    low = source.tell()
                elif current_id > user_id:
                    high = middle
                else:
                    return row
            source.seek(low)
            raw_line = source.readline()
            if not raw_line:
                return None
            row = json.loads(raw_line)
            return row if row.get("user_id") == user_id else None

    @staticmethod
    def _context_record(
        record: JsonObject,
        context_chars: int | None,
    ) -> JsonObject:
        context = copy.deepcopy(record)
        text = context.get("text")
        if context_chars is not None and isinstance(text, str):
            context["text"] = text[:context_chars]
        return context

    def get_user(
        self,
        user_id: str,
        *,
        with_context: bool = True,
        context_chars: int | None = 500,
    ) -> JsonObject:
        """Full nested structure for one user, with text filled in."""
        if context_chars is not None and context_chars < 0:
            raise RetrievalError("context_chars must be non-negative or None.")
        own_rows = self._read_user_rows(user_id)
        structure = self._load_user_structure(user_id)
        own_by_id = {
            _canonical_id(coerce_required_str_with_status(row["id"])[0]): row
            for row in own_rows
        }
        context_ids: set[str] = set()
        threads = _object_list(structure.get("threads", []), "threads")
        for thread in threads:
            anchor_id = thread.get("anchor_id")
            if with_context and isinstance(anchor_id, str):
                context_ids.add(anchor_id)
            for item in _object_list(thread.get("items", []), "threads[].items"):
                item_id = _canonical_id(
                    coerce_required_str_with_status(item["id"])[0]
                )
                if item_id not in own_by_id:
                    raise RetrievalError(
                        f"Structure item {item_id!r} is absent from {user_id!r}'s "
                        "indexed post rows.",
                    )
                item["text"] = own_by_id[item_id]["text"]
                if with_context:
                    parent_id = item.get("parent_id")
                    if isinstance(parent_id, str):
                        context_ids.add(parent_id)
                    context_ids.update(
                        _string_list(item.get("reply_ids", []), "reply_ids"),
                    )

        if not with_context:
            return structure

        contexts = self.get_items(sorted(context_ids)) if context_ids else {}
        for thread in threads:
            anchor_id = thread.get("anchor_id")
            thread["anchor"] = (
                self._context_record(contexts[anchor_id], context_chars)
                if isinstance(anchor_id, str)
                else None
            )
            for item in _object_list(thread.get("items", []), "threads[].items"):
                parent_id = item.get("parent_id")
                item["parent"] = (
                    self._context_record(contexts[parent_id], context_chars)
                    if isinstance(parent_id, str)
                    else None
                )
                item["replies"] = [
                    self._context_record(contexts[reply_id], context_chars)
                    for reply_id in _string_list(
                        item.get("reply_ids", []),
                        "reply_ids",
                    )
                ]
        return structure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--context-chars", type=int, default=500)
    parser.add_argument("--posts-repo", default="OP-R1/OP-Reddit-Post")
    parser.add_argument("--users-repo", default="OP-R1/OP-Reddit-User")
    return parser.parse_args()

def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

def main() -> int:
    args = _parse_args()
    retriever = OpR1Retriever(
        posts_repo=args.posts_repo,
        users_repo=args.users_repo,
    )
    try:
        user = retriever.get_user(
            args.user_id,
            with_context=not args.no_context,
            context_chars=args.context_chars,
        )
    except RetrievalError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output = json.dumps(user, ensure_ascii=False, indent=2, default=_json_default)
    if args.out is None:
        print(output)
    else:
        args.out.write_text(output + "\n", encoding="utf-8")
    print(
        f"bytes_fetched={retriever.bytes_fetched} "
        f"http_range_requests={retriever.http_range_requests} "
        f"local_read_requests={retriever.local_read_requests}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
