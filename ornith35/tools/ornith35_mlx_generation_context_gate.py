#!/usr/bin/env python3
"""Measure the isolated production decode path across Ornith-35 contexts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_cache as persistent_cache
import ornith35_mlx_gdn as gdn
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_runtime_coordination as runtime_coordination
from ornith35_moe_reference import MoEError, require
import ornith35_nvfp4 as nvfp4
from ornith35_tokenizer import DEFAULT_ROOT, TokenizerError, load_text_tokenizer


FORMAT = "ornith35-generation-context-gate-v1"
JSON_PREFIX = "generation-context-json "
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = DEFAULT_ROOT / "experiments" / "generation-context-gate-v1" / "report.json"
_CASE_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class ContextCase:
    name: str
    context_profile: str
    prefix_tokens: int

    def canonical(self) -> dict[str, object]:
        return {
            "context_profile": self.context_profile,
            "name": self.name,
            "prefix_tokens": self.prefix_tokens,
        }


DEFAULT_CASES = (
    ContextCase("native-2k", context.NATIVE_PROFILE_ID, 2_048),
    ContextCase("native-128k", context.NATIVE_PROFILE_ID, 131_072),
    ContextCase("native-262k", context.NATIVE_PROFILE_ID, 262_000),
    ContextCase("yarn2-524k", context.YARN2_PROFILE_ID, 524_160),
)


def parse_case(value: str) -> ContextCase:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be NAME:CONTEXT_PROFILE:PREFIX_TOKENS")
    name, profile_id, prefix_text = parts
    if _CASE_NAME.fullmatch(name) is None:
        raise argparse.ArgumentTypeError("case name must be lowercase hyphenated text")
    if profile_id not in context.SUPPORTED_CONTEXT_PROFILES:
        raise argparse.ArgumentTypeError(f"unsupported context profile: {profile_id}")
    try:
        prefix_tokens = int(prefix_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case prefix must be an integer") from exc
    if prefix_tokens <= 0:
        raise argparse.ArgumentTypeError("case prefix must be positive")
    return ContextCase(name, profile_id, prefix_tokens)


def format_case(case: ContextCase) -> str:
    return f"{case.name}:{case.context_profile}:{case.prefix_tokens}"


def validate_cases(
    cases: Sequence[ContextCase],
    *,
    warmup: int,
    rounds: int,
) -> tuple[ContextCase, ...]:
    selected = tuple(cases)
    require(selected, "at least one context case is required")
    require(warmup >= 2, "context gate warmup must be at least two")
    require(rounds >= 10, "context gate rounds must be at least ten")
    require(
        len({case.name for case in selected}) == len(selected),
        "context case names must be unique",
    )
    require(
        len({(case.context_profile, case.prefix_tokens) for case in selected})
        == len(selected),
        "context profile/prefix pairs must be unique",
    )
    reserve = 1 + warmup + rounds
    for case in selected:
        require(_CASE_NAME.fullmatch(case.name) is not None, "invalid context case name")
        require(case.prefix_tokens > 0, "context case prefix must be positive")
        context.validate_range(case.context_profile, case.prefix_tokens, reserve)
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON value is not an object: {path}")
    return value


def repository_state(repo_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect repository state: {exc}") from exc
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "repository revision is invalid",
    )
    return revision, dirty


def sysctl_integer(name: str) -> int | None:
    try:
        output = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return int(output)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.is_symlink(), "context gate output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.is_dir() and not path.parent.is_symlink(), "unsafe output directory")
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def build_identity(
    root: Path,
    repo_root: Path,
    cases: Sequence[ContextCase],
    *,
    warmup: int,
    rounds: int,
    token_id: int,
    revision: str,
    dirty: bool,
) -> dict[str, object]:
    tokenizer = load_text_tokenizer(root)
    runtime_identities = {}
    for profile_id in sorted({case.context_profile for case in cases}):
        identity = persistent_cache.production_identity(
            root,
            repo_root,
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=True,
            quantized_lm_head=True,
            rope_profile=profile_id,
        )
        runtime_identities[profile_id] = asdict(identity)
    source_state_path = root / "source-nvfp4-state.json"
    source_state = load_json_object(source_state_path)
    weight = source_state.get("weight")
    require(isinstance(weight, dict), "source state has no weight identity")
    return {
        "cases": [case.canonical() for case in cases],
        "iogpu_wired_limit_mb": sysctl_integer("iogpu.wired_limit_mb"),
        "mlx_version": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "repository_dirty": dirty,
        "repository_revision": revision,
        "rounds": rounds,
        "runtime_identities": runtime_identities,
        "source_state_sha256": sha256_file(source_state_path),
        "source_weight_sha256": weight.get("sha256"),
        "synthetic_history": "shared-read-only-zero-bf16-kv-and-zero-gdn-state",
        "token_id": token_id,
        "tool_sha256": sha256_file(Path(__file__)),
        "warmup": warmup,
    }


def synthetic_state(
    weights: model.TextModelWeights,
    case: ContextCase,
) -> model.TextModelState:
    """Create exact shapes while sharing only the immutable all-zero seed K/V."""
    base = model.initial_state(
        weights,
        model.PRODUCTION_CONFIG,
        case.context_profile,
    )
    shape = (
        model.PRODUCTION_CONFIG.attention.num_kv_heads,
        case.prefix_tokens,
        model.PRODUCTION_CONFIG.attention.head_dim,
    )
    shared_attention = attention.MLXAttentionState(
        keys=mx.zeros(shape, dtype=mx.bfloat16),
        values=mx.zeros(shape, dtype=mx.bfloat16),
        context_profile=case.context_profile,
    )
    layers: list[model.LayerState] = []
    for kind, layer_state in zip(model.PRODUCTION_CONFIG.layer_types, base.layers):
        if kind == model.LAYER_GDN:
            require(isinstance(layer_state, gdn.MLXGDNState), "invalid synthetic GDN state")
            layers.append(layer_state)
        else:
            layers.append(shared_attention)
    mx.eval(shared_attention.keys, shared_attention.values)
    mx.synchronize()
    state = model.TextModelState(
        position=case.prefix_tokens,
        layers=tuple(layers),
        context_profile=case.context_profile,
    )
    model.validate_state(state, model.PRODUCTION_CONFIG)
    return state


def linear_cache_bytes(session: model.TextLinearDecodeSession) -> int:
    total = 0
    attention_layers = 0
    array_owners = set()
    for kind, layer_state in zip(session.config.layer_types, session.state.layers):
        if kind != model.LAYER_ATTENTION:
            continue
        require(
            isinstance(layer_state, attention.MLXLinearAttentionState),
            "context gate linear state mismatch",
        )
        attention_layers += 1
        array_owners.update((id(layer_state.keys), id(layer_state.values)))
        total += layer_state.keys.size * layer_state.keys.itemsize
        total += layer_state.values.size * layer_state.values.itemsize
    require(attention_layers == 10, "context gate attention layer count mismatch")
    require(len(array_owners) == 20, "context gate linear K/V buffers alias")
    return total


def percentile(values: Sequence[float], fraction: float) -> float:
    require(values, "percentile input is empty")
    require(0.0 <= fraction <= 1.0, "percentile fraction is invalid")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def measure_case(
    root: Path,
    case: ContextCase,
    *,
    warmup: int,
    rounds: int,
    token_id: int,
) -> dict[str, object]:
    mx.set_cache_limit(128 * 2**20)
    nvfp4.require_verified_source(root)
    mx.reset_peak_memory()
    load_started = time.perf_counter()
    weights = model.load_text_model(
        root,
        map_embedding=True,
        quantize_lm_head=True,
    )
    model_load_s = time.perf_counter() - load_started
    model_active_bytes = mx.get_active_memory()
    model_peak_bytes = mx.get_peak_memory()
    print(
        "generation-context-model-ready "
        f"case={case.name} load_s={model_load_s:.3f} "
        f"active_gib={model_active_bytes / 2**30:.3f}",
        flush=True,
    )

    capacity = case.prefix_tokens + 1 + warmup + rounds
    mx.reset_peak_memory()
    setup_started = time.perf_counter()
    source = synthetic_state(weights, case)
    session = model.start_linear_decode_session(
        weights,
        source,
        capacity,
        model.PRODUCTION_CONFIG,
        compile_gdn_layers=True,
        compile_attention_tails=True,
    )
    setup_s = time.perf_counter() - setup_started
    setup_peak_bytes = mx.get_peak_memory()
    cache_bytes = linear_cache_bytes(session)
    del source
    gc.collect()
    mx.clear_cache()
    decode_active_bytes = mx.get_active_memory()
    model.validate_linear_decode_session(session)
    require(session.state.position == case.prefix_tokens, "context gate setup position changed")
    print(
        "generation-context-cache-ready "
        f"case={case.name} profile={case.context_profile} prefix={case.prefix_tokens} "
        f"capacity={capacity} setup_s={setup_s:.3f} cache_gib={cache_bytes / 2**30:.3f} "
        f"active_gib={decode_active_bytes / 2**30:.3f} "
        f"setup_peak_gib={setup_peak_bytes / 2**30:.3f}",
        flush=True,
    )

    result = model.forward_linear_session_token(token_id, session)
    rng = random.Random(17)
    for _ in range(warmup):
        selected = generate.choose_next_token(
            result.logits,
            temperature=0.0,
            top_k=20,
            top_p=0.95,
            rng=rng,
            hidden=result.hidden,
            lm_head=weights.lm_head,
        )
        result = model.forward_linear_session_token(selected, session)

    mx.reset_peak_memory()
    durations: list[float] = []
    selected_tokens: list[int] = []
    for step in range(rounds):
        started = time.perf_counter()
        selected = generate.choose_next_token(
            result.logits,
            temperature=0.0,
            top_k=20,
            top_p=0.95,
            rng=rng,
            hidden=result.hidden,
            lm_head=weights.lm_head,
        )
        result = model.forward_linear_session_token(selected, session)
        durations.append(time.perf_counter() - started)
        selected_tokens.append(selected)
        if (step + 1) % 8 == 0 or step + 1 == rounds:
            elapsed = math.fsum(durations)
            print(
                "generation-context-progress "
                f"case={case.name} steps={step + 1}/{rounds} "
                f"tokens_s={(step + 1) / elapsed:.3f}",
                flush=True,
            )

    finite_logits = mx.all(mx.isfinite(result.logits))
    finite_hidden = mx.all(mx.isfinite(result.hidden))
    mx.eval(finite_logits, finite_hidden)
    require(bool(finite_logits.item()), "context gate produced non-finite logits")
    require(bool(finite_hidden.item()), "context gate produced non-finite hidden state")
    model.validate_linear_decode_session(session)
    expected_position = case.prefix_tokens + 1 + warmup + rounds
    require(session.state.position == expected_position, "context gate final position mismatch")
    total_s = math.fsum(durations)
    median_s = statistics.median(durations)
    record: dict[str, object] = {
        "active_gib": mx.get_active_memory() / 2**30,
        "cache_bytes": cache_bytes,
        "capacity": capacity,
        "case": case.canonical(),
        "decode_peak_gib": mx.get_peak_memory() / 2**30,
        "final_position": session.state.position,
        "finite_hidden": True,
        "finite_logits": True,
        "history_authority": "synthetic-zero-kv-performance-only",
        "median_ms": median_s * 1000,
        "median_tokens_s": 1.0 / median_s,
        "model_active_gib": model_active_bytes / 2**30,
        "model_load_s": model_load_s,
        "model_peak_gib": model_peak_bytes / 2**30,
        "p10_ms": percentile(durations, 0.10) * 1000,
        "p90_ms": percentile(durations, 0.90) * 1000,
        "rounds": rounds,
        "selected_token_sha256": hashlib.sha256(
            b"".join(value.to_bytes(4, "little") for value in selected_tokens)
        ).hexdigest(),
        "setup_peak_gib": setup_peak_bytes / 2**30,
        "setup_s": setup_s,
        "step_ms": [value * 1000 for value in durations],
        "tokens_s": rounds / total_s,
        "warmup": warmup,
    }
    print(
        "generation-context-result "
        f"case={case.name} profile={case.context_profile} prefix={case.prefix_tokens} "
        f"tokens_s={record['tokens_s']:.3f} median_tokens_s={record['median_tokens_s']:.3f} "
        f"p10_ms={record['p10_ms']:.3f} p90_ms={record['p90_ms']:.3f} "
        f"active_gib={record['active_gib']:.3f} peak_gib={record['decode_peak_gib']:.3f}",
        flush=True,
    )
    return record


def aggregate_results(
    cases: Sequence[ContextCase],
    results: dict[str, dict[str, object]],
) -> dict[str, object]:
    ordered = [results[case.name] for case in cases]
    return {
        "case_count": len(ordered),
        "maximum_active_gib": max(float(record["active_gib"]) for record in ordered),
        "maximum_decode_peak_gib": max(
            float(record["decode_peak_gib"]) for record in ordered
        ),
        "minimum_tokens_s": min(float(record["tokens_s"]) for record in ordered),
    }


def validate_report(
    state: dict[str, Any],
    identity: dict[str, object],
    cases: Sequence[ContextCase],
) -> None:
    require(state.get("format") == FORMAT, "context gate report format mismatch")
    require(state.get("identity") == identity, "context gate report identity mismatch")
    require(state.get("status") in ("running", "complete"), "invalid report status")
    results = state.get("results")
    require(isinstance(results, dict), "context gate results are invalid")
    expected = {case.name: case for case in cases}
    require(set(results) <= set(expected), "context gate report has an unknown case")
    for name, record in results.items():
        require(isinstance(record, dict), f"context gate result is invalid: {name}")
        require(record.get("case") == expected[name].canonical(), f"context case drift: {name}")
        require(record.get("finite_logits") is True, f"non-finite result retained: {name}")
        require(record.get("finite_hidden") is True, f"non-finite result retained: {name}")
    if state["status"] == "complete":
        require(set(results) == set(expected), "complete context gate is missing cases")
        require(
            state.get("aggregate") == aggregate_results(cases, results),
            "complete context gate aggregate mismatch",
        )


def worker_command(args: argparse.Namespace, case: ContextCase) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--root",
        str(args.root),
        "--repo-root",
        str(args.repo_root),
        "--worker-case",
        format_case(case),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--token",
        str(args.token),
        "--foreground-wait-seconds",
        str(args.foreground_wait_seconds),
    ]


def run_worker_process(args: argparse.Namespace, case: ContextCase) -> dict[str, object]:
    command = worker_command(args, case)
    print(
        "generation-context-worker-launch "
        f"case={case.name} command={' '.join(command)}",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    require(process.stdout is not None, "context gate worker has no stdout")
    records = []
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith(JSON_PREFIX):
            try:
                record = json.loads(line[len(JSON_PREFIX) :])
            except json.JSONDecodeError as exc:
                process.kill()
                raise RuntimeError(f"invalid context worker JSON: {exc}") from exc
            require(isinstance(record, dict), "context worker record is not an object")
            records.append(record)
    return_code = process.wait()
    require(return_code == 0, f"context worker failed: case={case.name} exit={return_code}")
    require(len(records) == 1, f"context worker emitted {len(records)} result records")
    require(records[0].get("case") == case.canonical(), "context worker returned wrong case")
    return records[0]


def run_worker(args: argparse.Namespace, case: ContextCase) -> int:
    cases = validate_cases((case,), warmup=args.warmup, rounds=args.rounds)
    require(len(cases) == 1, "context worker case validation failed")
    require(0 <= args.token < model.PRODUCTION_CONFIG.vocab_size, "token is out of range")
    with runtime_coordination.foreground_lease(
        args.root,
        timeout_s=args.foreground_wait_seconds,
    ) as lease:
        print(
            "generation-context-worker-start "
            f"case={case.name} foreground_wait_s={lease.waited_s:.3f} "
            f"wired_limit_mb={sysctl_integer('iogpu.wired_limit_mb')}",
            flush=True,
        )
        record = measure_case(
            args.root,
            case,
            warmup=args.warmup,
            rounds=args.rounds,
            token_id=args.token,
        )
        record["foreground_wait_s"] = lease.waited_s
        record["iogpu_wired_limit_mb"] = sysctl_integer("iogpu.wired_limit_mb")
        print(JSON_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


def run_coordinator(args: argparse.Namespace, cases: Sequence[ContextCase]) -> int:
    selected = validate_cases(cases, warmup=args.warmup, rounds=args.rounds)
    require(0 <= args.token < model.PRODUCTION_CONFIG.vocab_size, "token is out of range")
    require(args.max_cases is None or args.max_cases > 0, "max cases must be positive")
    require(not args.output.is_symlink(), "context gate output must not be a symlink")
    revision, dirty = repository_state(args.repo_root)
    require(not dirty or args.allow_dirty, "repository is dirty; commit before measurement")
    identity = build_identity(
        args.root,
        args.repo_root,
        selected,
        warmup=args.warmup,
        rounds=args.rounds,
        token_id=args.token,
        revision=revision,
        dirty=dirty,
    )
    if args.output.exists():
        state = load_json_object(args.output)
        validate_report(state, identity, selected)
        if state["status"] == "complete":
            print(
                "generation-context-complete-resume "
                f"cases={len(state['results'])} output={args.output}",
                flush=True,
            )
            return 0
    else:
        state = {
            "aggregate": None,
            "format": FORMAT,
            "identity": identity,
            "results": {},
            "status": "running",
        }
        atomic_json(args.output, state)
    pending = [case for case in selected if case.name not in state["results"]]
    if args.max_cases is not None:
        pending = pending[: args.max_cases]
    free_gib = (
        os.statvfs(args.output.parent).f_bavail
        * os.statvfs(args.output.parent).f_frsize
        / 2**30
    )
    print(
        "generation-context-start "
        f"cases={len(selected)} completed={len(state['results'])} pending={len(pending)} "
        f"rounds={args.rounds} warmup={args.warmup} dirty={str(dirty).lower()} "
        f"free_gib={free_gib:.3f} output={args.output}",
        flush=True,
    )
    for case in pending:
        record = run_worker_process(args, case)
        state["results"][case.name] = record
        atomic_json(args.output, state)
        print(
            "generation-context-checkpoint "
            f"case={case.name} completed={len(state['results'])}/{len(selected)} "
            f"output={args.output}",
            flush=True,
        )
    remaining = [case for case in selected if case.name not in state["results"]]
    if remaining:
        print(
            "generation-context-paused "
            f"remaining={','.join(case.name for case in remaining)} output={args.output}",
            flush=True,
        )
        return 0
    state["aggregate"] = aggregate_results(selected, state["results"])
    state["status"] = "complete"
    atomic_json(args.output, state)
    print(
        "generation-context-complete "
        f"minimum_tokens_s={state['aggregate']['minimum_tokens_s']:.3f} "
        f"maximum_active_gib={state['aggregate']['maximum_active_gib']:.3f} "
        f"output={args.output}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--token", type=int, default=9707)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--foreground-wait-seconds", type=float, default=60.0)
    parser.add_argument("--worker-case", type=parse_case, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.root.is_dir() and not args.root.is_symlink(), "model root is unsafe")
        require(args.repo_root.is_dir(), "repository root is missing")
        require(args.foreground_wait_seconds > 0.0, "foreground wait must be positive")
        if args.worker_case is not None:
            require(args.cases is None, "worker cannot select coordinator cases")
            return run_worker(args, args.worker_case)
        cases = DEFAULT_CASES if args.cases is None else tuple(args.cases)
        return run_coordinator(args, cases)
    except (
        MoEError,
        TokenizerError,
        context.ContextError,
        runtime_coordination.RuntimeCoordinationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"generation context gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
