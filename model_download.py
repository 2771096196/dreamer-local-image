"""Resumable official-model downloads with immutable revisions and hash checks."""
from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import time
import threading
import urllib.parse
import urllib.request

MODEL = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
COMPONENTS = {"processor", "scheduler", "text_encoder", "transformer", "vae"}


def manifest(endpoint="https://huggingface.co", revision=REVISION):
    url = f"{endpoint.rstrip('/')}/api/models/{MODEL}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(url, timeout=45) as response:
        rows = json.load(response)
    files = []
    for row in rows:
        path = PurePosixPath(row.get("path", ""))
        if (row.get("type") != "file" or path.is_absolute() or ".." in path.parts
                or "\\" in str(path)):
            continue
        if path.parts[0] not in COMPONENTS and str(path) not in {"model_index.json", "LICENSE"}:
            continue
        lfs = row.get("lfs") or {}
        files.append({"path": str(path), "size": row["size"], "sha256": lfs.get("oid"),
                      "git_sha1": row.get("oid") if not lfs else None})
    if not files or not any(row["path"] == "model_index.json" for row in files):
        raise ValueError("Official model manifest is incomplete")
    return files


def verify(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha256() if entry.get("sha256") else hashlib.sha1()
    if not entry.get("sha256"):
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == (entry.get("sha256") or entry.get("git_sha1"))


def download_file(url: str, destination: Path, entry: dict, report=print):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verify(destination, entry):
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > entry["size"]:
        partial.replace(partial.with_suffix(".part.invalid"))
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            if offset < entry["size"]:
                headers = {"User-Agent": "Dreamer-Local-Image/0.1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Download server returned the wrong byte range")
                    elif offset:
                        offset = 0
                    last_report = time.monotonic()
                    with partial.open("ab" if offset else "wb") as output:
                        while chunk := response.read(4 * 1024 * 1024):
                            output.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last_report > 30:
                                report(f"{entry['path']}: {offset / entry['size']:.1%}")
                                last_report = time.monotonic()
            if partial.stat().st_size < entry["size"]:
                raise OSError("Model download ended before all bytes arrived")
            if not verify(partial, entry):
                partial.replace(partial.with_suffix(".part.invalid"))
                raise ValueError("Model file failed its official hash check")
            partial.replace(destination)
            report(f"Verified {entry['path']}")
            return destination
        except (OSError, ValueError, http.client.IncompleteRead) as error:
            if attempt == 7:
                raise
            report(f"Retry {entry['path']}: {type(error).__name__}; retaining downloaded bytes")
            time.sleep(min(2 ** attempt, 20))


def download_segmented(url, destination, entry, report=print, workers=4, chunk_size=64*1024*1024):
    """Resume stable ranges so a single large shard can use several connections."""
    if verify(destination, entry):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix >= entry['size']:
        return download_file(url, destination, entry, report)
    chunks = destination.with_name(destination.name + '.chunks')
    chunks.mkdir(exist_ok=True)
    starts = range((prefix // chunk_size) * chunk_size, entry['size'], chunk_size)
    lock = threading.Lock()
    progress = [prefix]
    def fetch(start):
        end = min(entry['size'], start + chunk_size) - 1
        path = chunks / str(start)
        length = end - start + 1
        for attempt in range(8):
            existing = path.stat().st_size if path.exists() else 0
            if existing == length:
                return path
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={start+existing}-{end}'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start+existing}-'):
                        raise ValueError('Server does not support precise ranged downloads')
                    with path.open('ab') as output:
                        remaining = length-existing
                        while remaining:
                            data = response.read(min(4*1024*1024, remaining))
                            if not data:
                                raise OSError('Incomplete range; retaining bytes')
                            output.write(data)
                            remaining -= len(data)
                with lock:
                    progress[0] += length-existing
                    if progress[0] // (512*1024*1024) != (progress[0]-length+existing) // (512*1024*1024):
                        report(f"{entry['path']}: {min(1,progress[0]/entry['size']):.1%}")
                return path
            except (OSError, ValueError, http.client.IncompleteRead):
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt,20))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(fetch, starts))
    with partial.open('ab') as output:
        for path in paths:
            with path.open('rb') as source:
                source.seek(max(0, prefix-int(path.name)))
                while data := source.read(8*1024*1024):
                    output.write(data)
    if not verify(partial, entry):
        raise ValueError('Assembled model failed its official hash check')
    partial.replace(destination)
    # Only delete this file's verified, now-redundant range files.
    for path in paths:
        path.unlink()
    report(f"Verified {entry['path']}")
    return destination


def ensure_model(directory: Path, *, revision=REVISION, endpoint=None, workers=4, report=None):
    from filelock import FileLock
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".download.lock")):
        return _ensure_model_locked(directory, revision=revision, endpoint=endpoint, workers=workers, report=report)


def _ensure_model_locked(directory: Path, *, revision, endpoint, workers, report):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    report = report or (lambda message: print(message, flush=True))
    marker = directory / ".verified-model.json"
    if marker.exists():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved["revision"] == revision and all(
                (directory / item["path"]).is_file()
                and (directory / item["path"]).stat().st_size == item["size"]
                for item in saved["files"]
            ):
                return directory
        except (OSError, ValueError, KeyError):
            pass
    entries = manifest(endpoint, revision)
    def fetch(entry):
        url = f"{endpoint.rstrip('/')}/{MODEL}/resolve/{revision}/{urllib.parse.quote(entry['path'])}"
        if entry['size'] > 512*1024*1024:
            return download_segmented(url, directory / entry['path'], entry, report, workers=max(1,min(workers,8)))
        return download_file(url, directory / entry["path"], entry, report)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, entries))
    marker.write_text(json.dumps({"model": MODEL, "revision": revision, "files": entries}, indent=2), encoding="utf-8")
    return directory
