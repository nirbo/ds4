#!/usr/bin/env python3
"""Resident-weight Nemotron MLX generator for a verified packed candidate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_attention import load_attention_layer
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mamba import load_mamba_layer
from nemotron_mlx_moe_layer import load_moe_layer


DEFAULT_MARGIN_GIB = 1.5


def iogpu_wired_limit_bytes() -> int:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) * 2**20
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0


def resident_requirement(payload_bytes: int, margin_gib: float = DEFAULT_MARGIN_GIB) -> int:
    require(payload_bytes > 0 and margin_gib >= 0.0, "invalid resident memory requirement")
    return payload_bytes + math.ceil(margin_gib * 2**30)


def preflight(model_dir: Path, margin_gib: float = DEFAULT_MARGIN_GIB) -> dict:
    report = load_json(model_dir / "nemotron_mlx_pack_report.json")
    require(report.get("format") == "nemotron-mlx-runtime-v1", "model is not a packed Nemotron runtime")
    require(report.get("status") == "complete", "packed Nemotron runtime is incomplete")
    payload = report.get("payload_bytes")
    require(isinstance(payload, int) and payload > 0, "runtime report has no payload size")
    device = mx.device_info()
    kernel_cap = iogpu_wired_limit_bytes()
    apple_cap = int(device.get("max_recommended_working_set_size", 0))
    effective_cap = kernel_cap or apple_cap
    required = resident_requirement(payload, margin_gib)
    return {
        "payload_bytes": payload,
        "payload_gib": payload / 2**30,
        "margin_gib": margin_gib,
        "required_bytes": required,
        "required_gib": required / 2**30,
        "required_mib_ceil": math.ceil(required / (256 * 2**20)) * 256,
        "kernel_cap_bytes": kernel_cap,
        "kernel_cap_gib": kernel_cap / 2**30,
        "apple_cap_bytes": apple_cap,
        "apple_cap_gib": apple_cap / 2**30,
        "effective_cap_bytes": effective_cap,
        "effective_cap_gib": effective_cap / 2**30,
        "memory_size_gib": int(device.get("memory_size", 0)) / 2**30,
        "safe_to_attempt": effective_cap >= required,
    }


class ResidentModel:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        self.pattern = self.config["hybrid_override_pattern"]
        self.index = load_json(model_dir / "model.safetensors.index.json")
        self.hidden_size = self.config["hidden_size"]
        self.embeddings = self._global("backbone.embeddings.weight")
        self.final_norm = self._global("backbone.norm_f.weight")
        self.lm_head = ModelOptBF16Linear(self._global("lm_head.weight"))
        self.blocks = []
        self.caches = {}
        for layer, kind in enumerate(self.pattern):
            if kind == "M":
                self.blocks.append(load_mamba_layer(model_dir, layer))
                self.caches[layer] = ArraysCache(size=2)
            elif kind == "E":
                self.blocks.append(load_moe_layer(model_dir, layer))
            elif kind == "*":
                self.blocks.append(load_attention_layer(model_dir, layer))
                self.caches[layer] = KVCache()
            else:
                raise MetadataError(f"unsupported layer type {kind!r} at {layer}")

    def _global(self, name: str) -> mx.array:
        shard_name = self.index["weight_map"].get(name)
        require(isinstance(shard_name, str), f"missing global tensor: {name}")
        tensors = mx.load(str(self.model_dir / shard_name))
        require(name in tensors, f"global tensor absent from shard: {name}")
        return tensors[name]

    def _cache_arrays(self) -> list[mx.array]:
        arrays = []
        for cache in self.caches.values():
            if isinstance(cache, ArraysCache):
                arrays.extend(value for value in cache.state if value is not None)
            elif cache.keys is not None:
                arrays.extend([cache.keys, cache.values])
        return arrays

    def logits(self, token_id: int) -> mx.array:
        require(0 <= token_id < self.embeddings.shape[0], "token ID out of range")
        x = self.embeddings[token_id].astype(mx.float32).reshape(1, 1, self.hidden_size)
        for layer, (kind, block) in enumerate(zip(self.pattern, self.blocks)):
            if kind == "M" or kind == "*":
                x = block(x, mask=None, cache=self.caches[layer])
            else:
                x = block(x)
        x = mx.fast.rms_norm(x, self.final_norm, self.config["layer_norm_epsilon"])
        logits = self.lm_head(x).reshape(-1)
        mx.eval(logits, *self._cache_arrays())
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--margin-gib", type=float, default=DEFAULT_MARGIN_GIB)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--token-timings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_new_tokens > 0, "max-new-tokens must be positive")
        result = preflight(args.model_dir, args.margin_gib)
        print("resident-preflight " + json.dumps(result, separators=(",", ":")), flush=True)
        if args.preflight_only:
            return 0 if result["safe_to_attempt"] else 2
        require(
            result["safe_to_attempt"],
            "Metal wired cap is too low; raise it deliberately before resident loading: "
            f"sudo sysctl -w iogpu.wired_limit_mb={result['required_mib_ceil']}",
        )
        previous_limit = mx.set_wired_limit(result["required_bytes"])
        mx.set_cache_limit(256 * 2**20)
        try:
            started = time.perf_counter()
            model = ResidentModel(args.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            require(token_ids, "prompt encoded to no tokens")
            logits = None
            for token_id in token_ids:
                logits = model.logits(token_id)
            load_prefill_seconds = time.perf_counter() - started
            require(logits is not None, "resident prefill produced no logits")
            generated = [int(mx.argmax(logits))]
            transition_seconds = []
            for _ in range(1, args.max_new_tokens):
                started = time.perf_counter()
                logits = model.logits(generated[-1])
                generated.append(int(mx.argmax(logits)))
                transition_seconds.append(time.perf_counter() - started)
            decode_seconds = sum(transition_seconds)
            measured_tokens = len(transition_seconds)
            decode_rate = measured_tokens / decode_seconds if decode_seconds else 0.0
            median_ms = (
                statistics.median(transition_seconds) * 1000 if transition_seconds else 0.0
            )
            p95_ms = (
                sorted(transition_seconds)[math.ceil(0.95 * measured_tokens) - 1] * 1000
                if transition_seconds
                else 0.0
            )
            print(
                f"resident-result prompt_tokens={len(token_ids)} generated_tokens={len(generated)} "
                f"load_prefill_seconds={load_prefill_seconds:.3f} decode_seconds={decode_seconds:.3f} "
                f"measured_decode_tokens={measured_tokens} tok_per_second={decode_rate:.3f} "
                f"decode_median_ms={median_ms:.3f} decode_p95_ms={p95_ms:.3f} "
                f"active_gib={mx.get_active_memory() / 2**30:.3f} peak_gib={mx.get_peak_memory() / 2**30:.3f} "
                f"token_ids={','.join(str(token) for token in generated)}"
            )
            if args.token_timings:
                print(
                    "resident-token-ms "
                    + ",".join(f"{elapsed * 1000:.3f}" for elapsed in transition_seconds)
                )
            print(tokenizer.decode(generated))
        finally:
            mx.set_wired_limit(previous_limit)
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron resident runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
