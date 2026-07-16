#!/usr/bin/env python3
"""Download and verify the BF16 shards named by one layer contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


STATE_FORMAT = "nemotron-bf16-download-state-v2"
DEFAULT_XET_FIXED_CONCURRENCY = 4
DEFAULT_XET_MIN_FETCH_MIB = 64
DEFAULT_XET_MAX_FETCH_MIB = 256


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_profile(
    fixed_concurrency: int,
    min_fetch_mib: int,
    max_fetch_mib: int,
) -> dict[str, int | bool]:
    require(fixed_concurrency > 0, "Xet fixed concurrency must be positive")
    require(min_fetch_mib > 0, "Xet minimum fetch size must be positive")
    require(max_fetch_mib >= min_fetch_mib, "Xet maximum fetch size is below minimum")
    return {
        "high_performance": False,
        "adaptive_concurrency": False,
        "fixed_download_concurrency": fixed_concurrency,
        "min_reconstruction_fetch_mib": min_fetch_mib,
        "max_reconstruction_fetch_mib": max_fetch_mib,
        "chunk_cache_bytes": 0,
        "max_concurrent_file_downloads": 1,
    }


def download_environment(
    job_dir: Path,
    profile: dict[str, int | bool],
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
    profile: dict[str, int | bool],
    hf_cli: dict[str, str],
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
        "hf_cli": hf_cli,
        "files": {
            entry["name"]: {
                "expected_bytes": entry["bytes"],
                "expected_sha256": entry["sha256"],
                "status": "pending",
                "attempts": 0,
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
            "hf_cli",
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
    existing = sum(
        entry["expected_bytes"]
        for name, entry in state["files"].items()
        if entry.get("status") == "verified" and (raw_dir / name).is_file()
    )
    required = sum(entry["expected_bytes"] for entry in state["files"].values()) - existing
    margin = int(margin_gib * 2**30)
    require(
        usage.free >= required + margin,
        "insufficient disk for BF16 download: "
        f"free={usage.free / 2**30:.2f}GiB required={required / 2**30:.2f}GiB "
        f"margin={margin_gib:.2f}GiB",
    )
    return {"free_bytes": usage.free, "remaining_download_bytes": required, "margin_bytes": margin}


def run_hf_download(
    hf_binary: str,
    contract: dict,
    name: str,
    raw_dir: Path,
    environment: dict[str, str],
    operation_log: OperationLog,
) -> None:
    command = [
        hf_binary,
        "download",
        contract["repository"],
        name,
        "--revision",
        contract["source_revision"],
        "--local-dir",
        str(raw_dir),
        "--max-workers",
        "1",
    ]
    operation_log.write("bf16-hf-command " + " ".join(command))
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    require(process.stdout is not None, "failed to capture hf output")
    for line in process.stdout:
        line = line.rstrip("\r\n")
        if line:
            operation_log.write(f"hf {line}")
    return_code = process.wait()
    require(return_code == 0, f"hf download failed for {name}: exit={return_code}")


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
    parser.add_argument("--hf-binary", default="hf")
    parser.add_argument("--margin-gib", type=float, default=5.0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument(
        "--xet-fixed-concurrency",
        type=int,
        default=DEFAULT_XET_FIXED_CONCURRENCY,
    )
    parser.add_argument("--xet-min-fetch-mib", type=int, default=DEFAULT_XET_MIN_FETCH_MIB)
    parser.add_argument("--xet-max-fetch-mib", type=int, default=DEFAULT_XET_MAX_FETCH_MIB)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    state_path = args.job_dir / "download-state.json"
    try:
        require(args.margin_gib >= 0.0, "disk margin must be nonnegative")
        require(args.max_shards is None or args.max_shards > 0, "max shards must be positive")
        profile = download_profile(
            args.xet_fixed_concurrency,
            args.xet_min_fetch_mib,
            args.xet_max_fetch_mib,
        )
        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        args.job_dir.mkdir(parents=True, exist_ok=True)
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.job_dir / "download.log")
        hf_binary = shutil.which(args.hf_binary)
        require(hf_binary is not None, f"Hugging Face CLI is unavailable: {args.hf_binary}")
        version_process = subprocess.run(
            [hf_binary, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        hf_version = (version_process.stdout or version_process.stderr).strip()
        require(bool(hf_version), "Hugging Face CLI returned no version")
        hf_cli = {"path": str(Path(hf_binary).resolve()), "version": hf_version}
        identity = state_identity(args.contract, contract, args.raw_dir, profile, hf_cli)
        state = load_or_create_state(state_path, identity)
        operation_log.write(
            f"bf16-download-start layer={contract['layer']} revision={contract['source_revision']} "
            f"raw_dir={args.raw_dir.resolve()}"
        )
        verified_bytes(state, args.raw_dir, operation_log)
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

        environment = download_environment(args.job_dir, profile)
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
            f"{profile['max_reconstruction_fetch_mib']} hf_cli={hf_version!r} "
            f"authentication={authentication}"
        )
        processed = 0
        for name, entry in state["files"].items():
            path = args.raw_dir / name
            if entry.get("status") == "verified":
                continue
            if args.max_shards is not None and processed >= args.max_shards:
                break
            if path.exists():
                operation_log.write(f"bf16-download-existing-validate file={name}")
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
                run_hf_download(
                    hf_binary,
                    contract,
                    name,
                    args.raw_dir,
                    environment,
                    operation_log,
                )
            operation_log.write(f"bf16-download-hash-start file={name}")
            validate_file(path, entry, hash_payload=True)
            entry["status"] = "verified"
            entry["verified_at"] = utc_now()
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            operation_log.write(
                f"bf16-download-file-verified file={name} bytes={entry['expected_bytes']} "
                f"sha256={entry['expected_sha256']}"
            )
            processed += 1

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
    except (
        MetadataError,
        OSError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as exc:
        error = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)
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
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
