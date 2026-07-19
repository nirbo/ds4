#!/usr/bin/env python3
"""Real-checkpoint persistence, restored-prefix TTFT, and suffix benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import mlx.core as mx

import ornith35_context as context
import ornith35_mlx_attention as attention
import ornith35_mlx_cache as cache
import ornith35_mlx_gdn as gdn
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import TokenizerError, load_text_tokenizer


JSON_PREFIX = "cache-bench-json "


def compare_state(
    expected: model.TextModelState,
    actual: model.TextModelState,
) -> int:
    require(expected.position == actual.position, "restored position mismatch")
    checks = []
    for left, right in zip(expected.layers, actual.layers):
        if isinstance(left, gdn.MLXGDNState):
            require(isinstance(right, gdn.MLXGDNState), "restored GDN type mismatch")
            checks.extend(
                (
                    mx.array_equal(left.conv, right.conv),
                    mx.array_equal(left.recurrent, right.recurrent),
                )
            )
            continue
        require(
            isinstance(left, attention.MLXLinearAttentionState)
            and isinstance(right, attention.MLXAttentionState),
            "restored attention type mismatch",
        )
        checks.extend(
            (
                mx.array_equal(left.keys[:, : left.position], right.keys),
                mx.array_equal(left.values[:, : left.position], right.values),
            )
        )
    mx.eval(*checks)
    mismatches = [index for index, check in enumerate(checks) if not bool(check.item())]
    require(not mismatches, f"restored state mismatch at tensors {mismatches}")
    return len(checks)


def directory_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.iterdir() if entry.is_file())


def deterministic_tokens(start: int, count: int) -> tuple[int, ...]:
    require(start >= 0 and count > 0, "invalid deterministic token range")
    return tuple(9707 + (start + index) % 17 for index in range(count))


def parse_suffixes(value: str) -> tuple[int, ...]:
    try:
        suffixes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise MoEError("suffixes must be comma-separated integers") from exc
    require(suffixes, "at least one suffix length is required")
    require(all(length > 0 for length in suffixes), "suffix lengths must be positive")
    require(len(set(suffixes)) == len(suffixes), "suffix lengths must be unique")
    return suffixes


def production_identity(
    args: argparse.Namespace,
) -> cache.CacheIdentity:
    tokenizer = load_text_tokenizer(args.root)
    return cache.production_identity(
        args.root,
        args.repo_root,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        chat_template_sha256=tokenizer.template_sha256,
        mapped_embedding=True,
        quantized_lm_head=args.quantized_lm_head,
        rope_profile=args.context_profile,
    )


def emit_record(record: dict[str, object]) -> None:
    print(JSON_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


def prepare_cache(args: argparse.Namespace) -> int:
    require(args.cache_root is not None, "prepare worker cache root is missing")
    selected_context = context.resolve_profile(args.context_profile)
    require(
        args.prefix_tokens + 1 <= selected_context.max_position_embeddings,
        "prefix exceeds the selected context profile",
    )
    tokens = deterministic_tokens(0, args.prefix_tokens)
    mx.reset_peak_memory()
    started = time.perf_counter()
    identity = production_identity(args)
    identity_elapsed = time.perf_counter() - started

    load_started = time.perf_counter()
    weights = model.load_text_model(
        args.root,
        map_embedding=True,
        quantize_lm_head=args.quantized_lm_head,
    )
    model_load_elapsed = time.perf_counter() - load_started
    initial = model.initial_state(
        weights,
        model.PRODUCTION_CONFIG,
        selected_context.profile_id,
    )
    session = model.start_linear_decode_session(
        weights,
        initial,
        args.prefix_tokens + 1,
        model.PRODUCTION_CONFIG,
        compile_gdn_layers=False,
        compile_attention_tails=False,
    )
    prefill_started = time.perf_counter()
    state, schedule = generate.prefill_state_prompt(
        tokens,
        session.state,
        weights,
        max_chunk=args.prefill_chunk,
        linear_session=session,
    )
    prefill_elapsed = time.perf_counter() - prefill_started
    require(state is session.state, "cache preparation lost linear-session ownership")

    save_started = time.perf_counter()
    path = cache.save_cache(
        args.cache_root,
        tokens,
        state,
        identity,
        model.PRODUCTION_CONFIG,
    )
    save_elapsed = time.perf_counter() - save_started
    restored = cache.load_cache(
        path,
        identity,
        model.PRODUCTION_CONFIG,
        expected_tokens=tokens,
    )
    exact_states = compare_state(session.state, restored.state)

    continuation_id = deterministic_tokens(args.prefix_tokens, 1)[0]
    expected = model.forward_linear_session_token(continuation_id, session)
    actual = model.forward_token(
        continuation_id,
        restored.state,
        weights,
        model.PRODUCTION_CONFIG,
    )
    model.evaluate_result(actual)
    logit_equal = mx.array_equal(expected.logits, actual.logits)
    mx.eval(logit_equal)
    require(bool(logit_equal.item()), "restored continuation logits mismatch")
    exact_continuation = compare_state(expected.state, actual.state)
    record: dict[str, object] = {
        "phase": "prepare",
        "prefix_tokens": args.prefix_tokens,
        "cache_path": str(path),
        "cache_bytes": directory_bytes(path),
        "identity_s": identity_elapsed,
        "model_load_s": model_load_elapsed,
        "prefill_s": prefill_elapsed,
        "prefill_tokens_s": args.prefix_tokens / prefill_elapsed,
        "prefill_schedule": generate.format_prefill_schedule(schedule),
        "save_s": save_elapsed,
        "verify_restore_s": restored.load_timing.total_s,
        "state_exact": exact_states,
        "continuation_exact": exact_continuation + 1,
        "active_gib": mx.get_active_memory() / 2**30,
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    print(
        "cache-bench-prepare "
        f"prefix_tokens={args.prefix_tokens} bytes={record['cache_bytes']} "
        f"prefill_s={prefill_elapsed:.3f} prefill_tokens_s={record['prefill_tokens_s']:.3f} "
        f"save_s={save_elapsed:.3f} verify_restore_s={restored.load_timing.total_s:.3f} "
        f"state_exact={exact_states} continuation_exact={exact_continuation + 1} "
        f"peak_gib={record['peak_gib']:.3f}",
        flush=True,
    )
    emit_record(record)
    return 0


def measure_ttft(args: argparse.Namespace) -> int:
    require(args.cache_path is not None, "measure worker cache path is missing")
    require(args.suffix_tokens is not None, "measure worker suffix length is missing")
    selected_context = context.resolve_profile(args.context_profile)
    capacity = args.prefix_tokens + args.suffix_tokens + args.decode_reserve
    require(
        capacity <= selected_context.max_position_embeddings,
        "prefix, suffix, and decode reserve exceed the selected context profile",
    )
    prefix = deterministic_tokens(0, args.prefix_tokens)
    suffix = list(deterministic_tokens(args.prefix_tokens, args.suffix_tokens))
    mx.reset_peak_memory()
    startup_started = time.perf_counter()

    tokenizer_started = time.perf_counter()
    tokenizer = load_text_tokenizer(args.root)
    tokenizer_elapsed = time.perf_counter() - tokenizer_started
    identity_started = time.perf_counter()
    identity = cache.production_identity(
        args.root,
        args.repo_root,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        chat_template_sha256=tokenizer.template_sha256,
        mapped_embedding=True,
        quantized_lm_head=args.quantized_lm_head,
        rope_profile=selected_context.profile_id,
    )
    identity_elapsed = time.perf_counter() - identity_started
    lookup = cache.find_longest_prefix(
        args.cache_path.parent,
        prefix + tuple(suffix),
        identity,
        model.PRODUCTION_CONFIG,
    )
    expected_path = lookup.path
    require(expected_path == args.cache_path, "cache lookup did not resolve the prepared entry")
    require(lookup.token_count == len(prefix), "cache lookup returned the wrong prefix length")
    require(expected_path.is_dir(), "prepared cache entry is absent")
    lookup_elapsed = lookup.elapsed_s

    restore_started = time.perf_counter()
    restored = cache.load_cache(
        expected_path,
        identity,
        model.PRODUCTION_CONFIG,
        expected_tokens=prefix,
    )
    restore_elapsed = time.perf_counter() - restore_started
    restore_timing = restored.load_timing
    load_started = time.perf_counter()
    weights = model.load_text_model(
        args.root,
        map_embedding=True,
        quantize_lm_head=args.quantized_lm_head,
    )
    model_load_elapsed = time.perf_counter() - load_started
    attach_started = time.perf_counter()
    session = model.start_linear_decode_session(
        weights,
        restored.state,
        capacity,
        model.PRODUCTION_CONFIG,
        compile_gdn_layers=args.compiled_gdn_layers,
        compile_attention_tails=args.compiled_attention_tails,
    )
    attach_elapsed = time.perf_counter() - attach_started
    restored = None
    gc.collect()

    suffix_started = time.perf_counter()
    result, schedule = generate.prefill_prompt(
        suffix,
        session.state,
        weights,
        max_chunk=args.prefill_chunk,
        linear_session=session,
    )
    suffix_elapsed = time.perf_counter() - suffix_started
    select_started = time.perf_counter()
    hidden = result.hidden[-1] if result.hidden.ndim == 2 else result.hidden
    selected_id = generate.choose_next_token(
        result.logits,
        temperature=0.0,
        top_k=20,
        top_p=0.95,
        rng=random.Random(17),
        hidden=hidden,
        lm_head=weights.lm_head,
    )
    select_elapsed = time.perf_counter() - select_started
    ttft_elapsed = time.perf_counter() - startup_started
    require(session.state.position == args.prefix_tokens + args.suffix_tokens, "suffix position mismatch")

    record: dict[str, object] = {
        "phase": "measure",
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "decode_reserve": args.decode_reserve,
        "tokenizer_s": tokenizer_elapsed,
        "identity_s": identity_elapsed,
        "lookup_s": lookup_elapsed,
        "lookup_scanned_entries": lookup.scanned_entries,
        "lookup_compatible_entries": lookup.compatible_entries,
        "lookup_matching_entries": lookup.matching_entries,
        "restore_s": restore_elapsed,
        "restore_manifest_s": restore_timing.manifest_s,
        "restore_tokens_s": restore_timing.tokens_s,
        "restore_verify_s": restore_timing.payload_verify_s,
        "restore_materialize_s": restore_timing.payload_materialize_s,
        "restore_finalize_s": restore_timing.finalize_s,
        "restore_payload_bytes": restore_timing.payload_bytes,
        "model_load_s": model_load_elapsed,
        "attach_s": attach_elapsed,
        "suffix_prefill_s": suffix_elapsed,
        "suffix_tokens_s": args.suffix_tokens / suffix_elapsed,
        "select_s": select_elapsed,
        "ttft_s": ttft_elapsed,
        "selected_id": selected_id,
        "prefill_schedule": generate.format_prefill_schedule(schedule),
        "os_page_cache_controlled": False,
        "active_gib": mx.get_active_memory() / 2**30,
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    print(
        "cache-bench-ttft "
        f"prefix_tokens={args.prefix_tokens} suffix_tokens={args.suffix_tokens} "
        f"lookup_s={lookup_elapsed:.3f} restore_s={restore_elapsed:.3f} "
        f"verify_s={restore_timing.payload_verify_s:.3f} "
        f"materialize_s={restore_timing.payload_materialize_s:.3f} "
        f"model_load_s={model_load_elapsed:.3f} attach_s={attach_elapsed:.3f} "
        f"suffix_s={suffix_elapsed:.3f} suffix_tokens_s={record['suffix_tokens_s']:.3f} "
        f"select_s={select_elapsed:.3f} ttft_s={ttft_elapsed:.3f} "
        f"selected_id={selected_id} peak_gib={record['peak_gib']:.3f}",
        flush=True,
    )
    emit_record(record)
    return 0


def worker_command(
    args: argparse.Namespace,
    worker: str,
    run_root: Path,
    *,
    cache_path: Path | None = None,
    suffix_tokens: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--root",
        str(args.root),
        "--repo-root",
        str(args.repo_root),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--decode-reserve",
        str(args.decode_reserve),
        "--context-profile",
        args.context_profile,
        "--cache-root",
        str(run_root),
    ]
    command.append("--quantized-lm-head" if args.quantized_lm_head else "--no-quantized-lm-head")
    command.append("--compiled-gdn-layers" if args.compiled_gdn_layers else "--no-compiled-gdn-layers")
    command.append(
        "--compiled-attention-tails"
        if args.compiled_attention_tails
        else "--no-compiled-attention-tails"
    )
    if cache_path is not None:
        command.extend(("--cache-path", str(cache_path)))
    if suffix_tokens is not None:
        command.extend(("--suffix-tokens", str(suffix_tokens)))
    return command


def run_worker(command: list[str]) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    record = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith(JSON_PREFIX):
            record = json.loads(line[len(JSON_PREFIX) :])
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    require(return_code == 0, f"cache benchmark worker failed with exit {return_code}")
    require(isinstance(record, dict), "cache benchmark worker emitted no record")
    return record, elapsed


def orchestrate(args: argparse.Namespace) -> int:
    suffixes = parse_suffixes(args.suffixes)
    selected_context = context.resolve_profile(args.context_profile)
    require(args.prefix_tokens > 0, "prefix token count must be positive")
    generate.prefill_schedule(1, args.prefill_chunk)
    require(args.decode_reserve > 0, "decode reserve must be positive")
    require(
        args.prefix_tokens + max(suffixes) + args.decode_reserve
        <= selected_context.max_position_embeddings,
        "benchmark exceeds the selected context profile",
    )
    base = args.cache_root if args.cache_root is not None else args.root / "experiments" / "prefix-ttft-v1"
    base.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="cache-bench-", dir=base))
    print(
        "cache-bench-start "
        f"prefix_tokens={args.prefix_tokens} suffixes={','.join(map(str, suffixes))} "
        f"decode_reserve={args.decode_reserve} os_page_cache=uncontrolled "
        f"cache_root={run_root}",
        flush=True,
    )
    try:
        prepared, prepare_wall = run_worker(worker_command(args, "prepare", run_root))
        cache_path = Path(str(prepared["cache_path"]))
        for sequence, suffix_tokens in enumerate(suffixes, start=1):
            measured, process_wall = run_worker(
                worker_command(
                    args,
                    "measure",
                    run_root,
                    cache_path=cache_path,
                    suffix_tokens=suffix_tokens,
                )
            )
            print(
                "cache-bench-summary "
                f"sequence={sequence} prefix_tokens={args.prefix_tokens} "
                f"suffix_tokens={suffix_tokens} "
                f"worker_ttft_s={float(measured['ttft_s']):.3f} "
                f"process_wall_s={process_wall:.3f} "
                f"prepare_wall_s={prepare_wall:.3f}",
                flush=True,
            )
    finally:
        if args.keep_cache:
            print(f"cache-bench-retained path={run_root}", flush=True)
        elif run_root.exists():
            shutil.rmtree(run_root)
            print(f"cache-bench-cleaned path={run_root}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tokens", "--prefix-tokens", dest="prefix_tokens", type=int, default=128)
    parser.add_argument("--suffixes", default="1,16,128")
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--decode-reserve", type=int, default=1024)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--context-profile",
        choices=tuple(context.SUPPORTED_CONTEXT_PROFILES),
        default=context.NATIVE_PROFILE_ID,
    )
    parser.add_argument(
        "--quantized-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compiled-gdn-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compiled-attention-tails",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--worker", choices=("prepare", "measure"), help=argparse.SUPPRESS)
    parser.add_argument("--cache-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--suffix-tokens", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.worker == "prepare":
            return prepare_cache(args)
        if args.worker == "measure":
            return measure_ttft(args)
        return orchestrate(args)
    except (
        MoEError,
        OSError,
        ValueError,
        TokenizerError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"cache benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
