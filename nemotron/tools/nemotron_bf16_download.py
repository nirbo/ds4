#!/usr/bin/env python3
"""Download and verify the BF16 shards named by one layer contract."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


STATE_FORMAT = "nemotron-bf16-download-state-v3"
DEFAULT_XET_FIXED_CONCURRENCY = 4
DEFAULT_XET_MIN_FETCH_MIB = 64
DEFAULT_XET_MAX_FETCH_MIB = 256
DEFAULT_XET_RANGE_MIB = 128
DEFAULT_XET_STALL_TIMEOUT_SECONDS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_profile(
    fixed_concurrency: int,
    min_fetch_mib: int,
    max_fetch_mib: int,
    range_mib: int,
    stall_timeout_seconds: int,
) -> dict[str, int | bool | str]:
    require(fixed_concurrency > 0, "Xet fixed concurrency must be positive")
    require(min_fetch_mib > 0, "Xet minimum fetch size must be positive")
    require(max_fetch_mib >= min_fetch_mib, "Xet maximum fetch size is below minimum")
    require(range_mib > 0, "Xet committed range size must be positive")
    require(stall_timeout_seconds > 0, "Xet stall timeout must be positive")
    return {
        "mode": "xet-ordered-range-v1",
        "high_performance": False,
        "adaptive_concurrency": False,
        "fixed_download_concurrency": fixed_concurrency,
        "min_reconstruction_fetch_mib": min_fetch_mib,
        "max_reconstruction_fetch_mib": max_fetch_mib,
        "committed_range_mib": range_mib,
        "stall_timeout_seconds": stall_timeout_seconds,
        "chunk_cache_bytes": 0,
        "max_concurrent_file_downloads": 1,
    }


def download_environment(
    job_dir: Path,
    profile: dict[str, int | bool | str],
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Keep every Hugging Face cache visible and local to this bounded job."""

    environment = os.environ.copy() if base_environment is None else dict(base_environment)
    original_home = Path(
        environment.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    token_path = Path(
        environment.get("HF_TOKEN_PATH", str(original_home / "token"))
    ).expanduser()
    for key in (
        "HF_XET_HIGH_PERFORMANCE",
        "HF_XET_NUM_CONCURRENT_RANGE_GETS",
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY",
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY",
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY",
    ):
        environment.pop(key, None)
    fixed_concurrency = int(profile["fixed_download_concurrency"])
    environment.update(
        {
            "HF_HOME": str((job_dir / "hf-home").resolve()),
            "HF_XET_CACHE": str((job_dir / "hf-xet").resolve()),
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            "HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY": "false",
            "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": str(fixed_concurrency),
            "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": str(fixed_concurrency),
            "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": str(fixed_concurrency),
            "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": str(fixed_concurrency),
            "HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE": (
                f"{profile['min_reconstruction_fetch_mib']}mb"
            ),
            "HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE": (
                f"{profile['max_reconstruction_fetch_mib']}mb"
            ),
            "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "1",
        }
    )
    if not environment.get("HF_TOKEN"):
        if token_path.is_file():
            environment["HF_TOKEN_PATH"] = str(token_path.resolve())
        else:
            environment.pop("HF_TOKEN_PATH", None)
    return environment


def state_identity(
    contract_path: Path,
    contract: dict,
    raw_dir: Path,
    profile: dict[str, int | bool | str],
    hub_runtime: dict[str, str],
    remote_files: dict[str, dict[str, str | int]],
) -> dict:
    required = contract.get("required_shards")
    require(isinstance(required, list) and required, "BF16 contract has no required shards")
    return {
        "format": STATE_FORMAT,
        "repository": contract["repository"],
        "revision": contract["source_revision"],
        "layer": contract["layer"],
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "raw_dir": str(raw_dir.resolve()),
        "tool_sha256": sha256_file(Path(__file__)),
        "download_profile": profile,
        "hub_runtime": hub_runtime,
        "remote_files": remote_files,
        "files": {
            entry["name"]: {
                "expected_bytes": entry["bytes"],
                "expected_sha256": entry["sha256"],
                "status": "pending",
                "attempts": 0,
                "completed_bytes": 0,
                "ranges": [],
            }
            for entry in required
        },
    }


def load_or_create_state(path: Path, identity: dict) -> dict:
    if path.exists():
        state = load_json(path)
        for key in (
            "format",
            "repository",
            "revision",
            "layer",
            "contract",
            "contract_sha256",
            "raw_dir",
            "tool_sha256",
            "download_profile",
            "hub_runtime",
            "remote_files",
        ):
            require(state.get(key) == identity[key], f"BF16 download state mismatch: {key}")
        require(set(state.get("files", {})) == set(identity["files"]), "BF16 download file set changed")
        for name, expected in identity["files"].items():
            entry = state["files"][name]
            require(
                entry.get("expected_bytes") == expected["expected_bytes"]
                and entry.get("expected_sha256") == expected["expected_sha256"],
                f"BF16 download identity changed: {name}",
            )
        return state
    state = {
        **identity,
        "status": "pending",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_json(path, state)
    return state


def validate_file(path: Path, entry: dict, hash_payload: bool = True) -> None:
    require(path.is_file(), f"missing BF16 source shard: {path}")
    require(not path.is_symlink(), f"BF16 source shard must be a direct file: {path}")
    require(path.stat().st_size == entry["expected_bytes"], f"BF16 shard size mismatch: {path}")
    if hash_payload:
        require(sha256_file(path) == entry["expected_sha256"], f"BF16 shard hash mismatch: {path}")


def verified_bytes(state: dict, raw_dir: Path, operation_log: OperationLog) -> int:
    total = 0
    for name, entry in state["files"].items():
        path = raw_dir / name
        if entry.get("status") == "verified":
            if not path.exists():
                entry["status"] = "pending"
                entry["missing_after_verification_at"] = utc_now()
                operation_log.write(f"bf16-download-reset file={name} reason=verified-file-missing")
                continue
            validate_file(path, entry, hash_payload=True)
            total += entry["expected_bytes"]
            operation_log.write(f"bf16-download-resume-verified file={name} bytes={entry['expected_bytes']}")
    return total


def disk_preflight(raw_dir: Path, state: dict, margin_gib: float) -> dict[str, int]:
    usage = shutil.disk_usage(raw_dir)
    required = sum(
        0
        if entry.get("status") == "verified" and (raw_dir / name).is_file()
        else entry["expected_bytes"] - int(entry.get("completed_bytes", 0))
        for name, entry in state["files"].items()
    )
    margin = int(margin_gib * 2**30)
    require(
        usage.free >= required + margin,
        "insufficient disk for BF16 download: "
        f"free={usage.free / 2**30:.2f}GiB required={required / 2**30:.2f}GiB "
        f"margin={margin_gib:.2f}GiB",
    )
    return {"free_bytes": usage.free, "remaining_download_bytes": required, "margin_bytes": margin}


def hub_runtime_identity() -> dict[str, str]:
    return {
        "python": str(Path(sys.executable).resolve()),
        "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        "hf_xet": importlib.metadata.version("hf_xet"),
    }


def configure_download_environment(environment: dict[str, str]) -> None:
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    for key, value in environment.items():
        if key.startswith("HF_"):
            os.environ[key] = value


def fetch_remote_files(contract: dict) -> tuple[dict[str, dict], dict[str, dict[str, str | int]]]:
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    token = get_token()
    headers = build_hf_headers(token=token)
    custom_headers = {
        key: value for key, value in headers.items() if key.lower() != "authorization"
    }
    objects = {}
    identity = {}
    for entry in contract["required_shards"]:
        name = entry["name"]
        metadata = get_hf_file_metadata(
            hf_hub_url(
                contract["repository"],
                name,
                revision=contract["source_revision"],
            ),
            token=token,
            timeout=30,
            retry_on_errors=True,
        )
        require(metadata.commit_hash == contract["source_revision"], f"remote revision mismatch: {name}")
        require(metadata.size == entry["bytes"], f"remote size mismatch: {name}")
        require(metadata.etag == entry["sha256"], f"remote ETag mismatch: {name}")
        require(metadata.xet_file_data is not None, f"remote file is not Xet-backed: {name}")
        xet_data = metadata.xet_file_data
        objects[name] = {
            "file_hash": xet_data.file_hash,
            "file_size": metadata.size,
            "refresh_route": xet_data.refresh_route,
            "token_headers": headers,
            "custom_headers": custom_headers,
        }
        identity[name] = {
            "commit": metadata.commit_hash,
            "etag": metadata.etag,
            "bytes": metadata.size,
            "xet_file_hash": xet_data.file_hash,
            "refresh_route_sha256": hashlib.sha256(
                xet_data.refresh_route.encode("utf-8")
            ).hexdigest(),
        }
    return objects, identity


def partial_path(raw_dir: Path, name: str) -> Path:
    return raw_dir / ".nemotron-range-parts" / f"{name}.part"


def sha256_range(path: Path, start: int, end: int) -> str:
    require(0 <= start <= end <= path.stat().st_size, f"invalid hash range: {path}")
    digest = hashlib.sha256()
    remaining = end - start
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            block = handle.read(min(8 * 2**20, remaining))
            require(bool(block), f"unexpected EOF while hashing partial: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def validate_partial(
    path: Path,
    entry: dict,
    operation_log: OperationLog,
) -> None:
    completed = int(entry.get("completed_bytes", 0))
    ranges = entry.get("ranges", [])
    require(0 <= completed <= entry["expected_bytes"], "partial completion is out of range")
    require(isinstance(ranges, list), "partial range records are invalid")
    if completed == 0:
        if path.exists() and path.stat().st_size:
            operation_log.write(
                f"bf16-range-recover-truncate path={path} from={path.stat().st_size} to=0"
            )
            with path.open("r+b") as handle:
                handle.truncate(0)
                handle.flush()
                os.fsync(handle.fileno())
        require(not ranges, "zero-length partial has committed ranges")
        return
    require(path.is_file() and not path.is_symlink(), f"missing direct partial file: {path}")
    actual = path.stat().st_size
    require(actual >= completed, f"partial file is shorter than committed state: {path}")
    if actual > completed:
        operation_log.write(f"bf16-range-recover-truncate path={path} from={actual} to={completed}")
        with path.open("r+b") as handle:
            handle.truncate(completed)
            handle.flush()
            os.fsync(handle.fileno())
    cursor = 0
    for record in ranges:
        start = int(record.get("start", -1))
        end = int(record.get("end", -1))
        require(start == cursor and start < end <= completed, "partial range coverage is invalid")
        require(record.get("bytes") == end - start, "partial range byte count is invalid")
        require(
            sha256_range(path, start, end) == record.get("sha256"),
            f"partial range hash mismatch: {path} [{start},{end})",
        )
        cursor = end
    require(cursor == completed, "partial ranges do not cover committed bytes")
    operation_log.write(
        f"bf16-range-resume-verified path={path} bytes={completed} ranges={len(ranges)}"
    )


def write_xet_range(
    path: Path,
    remote: dict,
    start: int,
    end: int,
    stall_timeout_seconds: int,
    operation_log: OperationLog,
) -> dict[str, int | float | str]:
    from hf_xet import XetFileInfo, XetSession

    require(start < end <= remote["file_size"], "invalid Xet download range")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    require(path.is_file() and not path.is_symlink(), f"partial path is not a direct file: {path}")
    session = XetSession()
    stop_watchdog = threading.Event()
    timed_out = threading.Event()
    last_progress = [time.monotonic()]

    def watchdog() -> None:
        while not stop_watchdog.wait(15.0):
            idle = time.monotonic() - last_progress[0]
            if idle < stall_timeout_seconds:
                continue
            timed_out.set()
            operation_log.write(
                f"bf16-range-stalled start={start} end={end} idle={idle:.1f}s action=abort"
            )
            try:
                session.sigint_abort()
            except Exception:
                pass
            return

    watchdog_thread = threading.Thread(target=watchdog, name="nemotron-xet-watchdog", daemon=True)
    watchdog_thread.start()
    started = time.perf_counter()
    digest = hashlib.sha256()
    received = 0
    report_step = 32 * 2**20
    next_report = report_step
    try:
        group = session.new_download_stream_group(
            token_refresh_url=remote["refresh_route"],
            token_refresh_headers=remote["token_headers"],
            custom_headers=remote["custom_headers"],
        )
        stream = group.download_stream(
            XetFileInfo(remote["file_hash"], remote["file_size"]),
            start=start,
            end=end,
        )
        with path.open("r+b") as handle:
            handle.truncate(start)
            handle.seek(start)
            try:
                for chunk in stream:
                    require(isinstance(chunk, bytes) and chunk, "Xet returned an empty range chunk")
                    require(received + len(chunk) <= end - start, "Xet range exceeded requested end")
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    last_progress[0] = time.monotonic()
                    if received >= next_report:
                        operation_log.write(
                            f"bf16-range-progress start={start} end={end} "
                            f"received={received}/{end - start}"
                        )
                        next_report += report_step
                require(not timed_out.is_set(), "Xet range stalled")
                require(received == end - start, "Xet range returned the wrong byte count")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                handle.truncate(start)
                handle.flush()
                os.fsync(handle.fileno())
                raise
    except BaseException:
        try:
            session.sigint_abort()
        except Exception:
            pass
        raise
    finally:
        stop_watchdog.set()
        watchdog_thread.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    result = {
        "start": start,
        "end": end,
        "bytes": received,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
    }
    del stream, group, session
    gc.collect()
    return result


def run_xet_range_download(
    name: str,
    entry: dict,
    remote: dict,
    raw_dir: Path,
    state: dict,
    state_path: Path,
    profile: dict[str, int | bool | str],
    operation_log: OperationLog,
    max_ranges: int | None = None,
) -> str | None:
    part = partial_path(raw_dir, name)
    validate_partial(part, entry, operation_log)
    range_bytes = int(profile["committed_range_mib"]) * 2**20
    processed_ranges = 0
    while int(entry.get("completed_bytes", 0)) < entry["expected_bytes"]:
        if max_ranges is not None and processed_ranges >= max_ranges:
            operation_log.write(
                f"bf16-range-stop file={name} max_ranges={max_ranges} "
                f"completed={entry.get('completed_bytes', 0)}/{entry['expected_bytes']}"
            )
            return None
        start = int(entry.get("completed_bytes", 0))
        end = min(start + range_bytes, entry["expected_bytes"])
        operation_log.write(
            f"bf16-range-start file={name} start={start} end={end} bytes={end - start}"
        )
        record = write_xet_range(
            part,
            remote,
            start,
            end,
            int(profile["stall_timeout_seconds"]),
            operation_log,
        )
        entry["ranges"].append({**record, "committed_at": utc_now()})
        entry["completed_bytes"] = end
        entry["status"] = "downloading"
        state["status"] = "running"
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        operation_log.write(
            f"bf16-range-committed file={name} end={end}/{entry['expected_bytes']} "
            f"sha256={record['sha256']} elapsed={record['elapsed_seconds']:.2f}s"
        )
        processed_ranges += 1
    require(part.stat().st_size == entry["expected_bytes"], f"completed partial size mismatch: {part}")
    operation_log.write(f"bf16-download-hash-start file={name} path={part}")
    digest = sha256_file(part)
    require(digest == entry["expected_sha256"], f"BF16 shard hash mismatch: {part}")
    destination = raw_dir / name
    require(not destination.exists(), f"BF16 destination already exists: {destination}")
    part.replace(destination)
    operation_log.write(f"bf16-download-hash-complete file={name} sha256={digest}")
    return digest


def cache_usage(job_dir: Path) -> dict[str, int]:
    result = {}
    for name in ("hf-home", "hf-xet"):
        root = job_dir / name
        total = 0
        if root.exists():
            total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        result[name] = total
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--margin-gib", type=float, default=5.0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-ranges", type=int)
    parser.add_argument(
        "--xet-fixed-concurrency",
        type=int,
        default=DEFAULT_XET_FIXED_CONCURRENCY,
    )
    parser.add_argument("--xet-min-fetch-mib", type=int, default=DEFAULT_XET_MIN_FETCH_MIB)
    parser.add_argument("--xet-max-fetch-mib", type=int, default=DEFAULT_XET_MAX_FETCH_MIB)
    parser.add_argument("--xet-range-mib", type=int, default=DEFAULT_XET_RANGE_MIB)
    parser.add_argument(
        "--xet-stall-timeout-seconds",
        type=int,
        default=DEFAULT_XET_STALL_TIMEOUT_SECONDS,
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    state_path = args.job_dir / "download-state.json"
    try:
        require(args.margin_gib >= 0.0, "disk margin must be nonnegative")
        require(args.max_shards is None or args.max_shards > 0, "max shards must be positive")
        require(args.max_ranges is None or args.max_ranges > 0, "max ranges must be positive")
        profile = download_profile(
            args.xet_fixed_concurrency,
            args.xet_min_fetch_mib,
            args.xet_max_fetch_mib,
            args.xet_range_mib,
            args.xet_stall_timeout_seconds,
        )
        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        args.job_dir.mkdir(parents=True, exist_ok=True)
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.job_dir / "download.log")
        environment = download_environment(args.job_dir, profile)
        configure_download_environment(environment)
        runtime = hub_runtime_identity()
        remote_objects, remote_identity = fetch_remote_files(contract)
        identity = state_identity(
            args.contract,
            contract,
            args.raw_dir,
            profile,
            runtime,
            remote_identity,
        )
        state = load_or_create_state(state_path, identity)
        operation_log.write(
            f"bf16-download-start layer={contract['layer']} revision={contract['source_revision']} "
            f"raw_dir={args.raw_dir.resolve()}"
        )
        verified_bytes(state, args.raw_dir, operation_log)
        for name, entry in state["files"].items():
            if entry.get("status") != "verified":
                validate_partial(partial_path(args.raw_dir, name), entry, operation_log)
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        preflight = disk_preflight(args.raw_dir, state, args.margin_gib)
        operation_log.write(
            f"bf16-download-preflight free={preflight['free_bytes'] / 2**30:.2f}GiB "
            f"remaining={preflight['remaining_download_bytes'] / 2**30:.2f}GiB "
            f"margin={preflight['margin_bytes'] / 2**30:.2f}GiB"
        )

        if args.validate_only:
            for name, entry in state["files"].items():
                validate_file(args.raw_dir / name, entry, hash_payload=True)
            state["status"] = "complete"
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            operation_log.write("bf16-download-validate-complete")
            return 0

        authentication = (
            "configured"
            if environment.get("HF_TOKEN") or environment.get("HF_TOKEN_PATH")
            else "anonymous"
        )
        operation_log.write(
            "bf16-download-cache-policy "
            f"HF_HOME={environment['HF_HOME']} HF_XET_CACHE={environment['HF_XET_CACHE']} "
            "HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_XET_HIGH_PERFORMANCE=disabled "
            f"fixed_concurrency={profile['fixed_download_concurrency']} "
            f"fetch_mib={profile['min_reconstruction_fetch_mib']}.."
            f"{profile['max_reconstruction_fetch_mib']} "
            f"range_mib={profile['committed_range_mib']} "
            f"stall_timeout={profile['stall_timeout_seconds']}s "
            f"hub={runtime['huggingface_hub']} xet={runtime['hf_xet']} "
            f"authentication={authentication}"
        )
        processed = 0
        range_stop = False
        for name, entry in state["files"].items():
            path = args.raw_dir / name
            if entry.get("status") == "verified":
                continue
            if args.max_shards is not None and processed >= args.max_shards:
                break
            if path.exists():
                operation_log.write(f"bf16-download-existing-validate file={name}")
                validate_file(path, entry, hash_payload=True)
            else:
                entry["status"] = "downloading"
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["started_at"] = utc_now()
                state["status"] = "running"
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
                operation_log.write(
                    f"bf16-download-file-start file={name} bytes={entry['expected_bytes']} "
                    f"attempt={entry['attempts']}"
                )
                digest = run_xet_range_download(
                    name,
                    entry,
                    remote_objects[name],
                    args.raw_dir,
                    state,
                    state_path,
                    profile,
                    operation_log,
                    args.max_ranges,
                )
                if digest is None:
                    range_stop = True
                    break
                require(digest == entry["expected_sha256"], f"BF16 shard hash mismatch: {name}")
                validate_file(path, entry, hash_payload=False)
            entry["status"] = "verified"
            entry["completed_bytes"] = entry["expected_bytes"]
            entry["verified_at"] = utc_now()
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            operation_log.write(
                f"bf16-download-file-verified file={name} bytes={entry['expected_bytes']} "
                f"sha256={entry['expected_sha256']}"
            )
            processed += 1

        if range_stop:
            operation_log.write("bf16-download-stop reason=max-ranges")

        complete = all(entry.get("status") == "verified" for entry in state["files"].values())
        state["status"] = "complete" if complete else "running"
        state["cache_bytes"] = cache_usage(args.job_dir)
        state["updated_at"] = utc_now()
        atomic_json(state_path, state)
        operation_log.write(
            f"bf16-download-stop status={state['status']} processed={processed} "
            f"verified={sum(entry.get('status') == 'verified' for entry in state['files'].values())}/"
            f"{len(state['files'])} hf_home_bytes={state['cache_bytes']['hf-home']} "
            f"hf_xet_bytes={state['cache_bytes']['hf-xet']}"
        )
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "state": str(state_path.resolve()),
                    "raw_dir": str(args.raw_dir.resolve()),
                    "cache_bytes": state["cache_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except KeyboardInterrupt:
        error = "interrupted"
        if state_path.exists():
            try:
                state = load_json(state_path)
                state["status"] = "failed"
                state["failure"] = {"at": utc_now(), "error": error}
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
            except (MetadataError, OSError, ValueError, KeyError):
                pass
        if operation_log is not None:
            operation_log.write(f"bf16-download-failed error={error}")
        print(f"nemotron BF16 download error: {error}", file=sys.stderr)
        return 130
    except Exception as exc:
        error = str(exc) or type(exc).__name__
        if state_path.exists():
            try:
                state = load_json(state_path)
                state["status"] = "failed"
                state["failure"] = {"at": utc_now(), "error": error}
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
            except (MetadataError, OSError, ValueError, KeyError):
                pass
        if operation_log is not None:
            operation_log.write(f"bf16-download-failed error={error}")
        print(f"nemotron BF16 download error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
