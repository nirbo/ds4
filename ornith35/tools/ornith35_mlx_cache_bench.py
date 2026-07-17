#!/usr/bin/env python3
"""Real-checkpoint persistence, restore, and continuation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mlx.core as mx

import ornith35_mlx_attention as attention
import ornith35_mlx_cache as cache
import ornith35_mlx_gdn as gdn
import ornith35_mlx_model as model
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import TokenizerError, load_text_tokenizer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tokens", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = args.root / "cache-benchmark"
    try:
        require(args.tokens in (8, 16, 32, 64, 128), "invalid benchmark token count")
        tokenizer = load_text_tokenizer(args.root)
        identity = cache.production_identity(
            args.root,
            args.repo_root,
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=True,
            quantized_lm_head=False,
        )
        weights = model.load_text_model(args.root, map_embedding=True)
        initial = model.initial_state(weights, model.PRODUCTION_CONFIG)
        tokens = tuple(9707 + index % 17 for index in range(args.tokens))
        session = model.start_linear_decode_session(
            weights,
            initial,
            args.tokens + 1,
            model.PRODUCTION_CONFIG,
        )
        model.prefill_linear_session_chunk(
            tokens,
            session,
            project_logits=False,
            use_steel=False,
        )
        if cache_root.exists():
            shutil.rmtree(cache_root)
        mx.reset_peak_memory()
        started = time.perf_counter()
        path = cache.save_cache(
            cache_root,
            tokens,
            session.state,
            identity,
            model.PRODUCTION_CONFIG,
        )
        save_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        restored = cache.load_cache(
            path,
            identity,
            model.PRODUCTION_CONFIG,
            expected_tokens=tokens,
        )
        load_elapsed = time.perf_counter() - started
        exact_states = compare_state(session.state, restored.state)

        expected = model.forward_linear_session_token(11, session)
        actual = model.forward_token(
            11,
            restored.state,
            weights,
            model.PRODUCTION_CONFIG,
        )
        model.evaluate_result(actual)
        logit_equal = mx.array_equal(expected.logits, actual.logits)
        mx.eval(logit_equal)
        require(bool(logit_equal.item()), "restored continuation logits mismatch")
        exact_continuation = compare_state(expected.state, actual.state)
        print(
            "cache-bench "
            f"tokens={args.tokens} bytes={directory_bytes(path)} "
            f"save_s={save_elapsed:.3f} load_s={load_elapsed:.3f} "
            f"state_exact={exact_states} continuation_exact={exact_continuation + 1} "
            f"active_gib={mx.get_active_memory() / 2**30:.3f} "
            f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
            flush=True,
        )
    except (MoEError, OSError, ValueError, TokenizerError, subprocess.SubprocessError) as exc:
        print(f"cache benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if cache_root.exists():
            shutil.rmtree(cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
