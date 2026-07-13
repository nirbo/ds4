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
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_attention import load_attention_layer
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mamba import load_mamba_layer, mamba_sequence_exact
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_mlx_mtp import NemotronMTPSidecar
from nemotron_paged_embeddings import PagedBF16Embedding, embedding_layout
from nemotron_prune_materialize import sha256_file


DEFAULT_MARGIN_GIB = 1.5
DEFAULT_EXTENDED_RUN_MEMORY_FRACTION = 0.80
MLX_ALLOCATOR_GC_FRACTION = 0.95
EXTENDED_RUN_CACHE_MIB = 512


def snapshot_caches(caches: dict[int, ArraysCache | KVCache]) -> dict[int, tuple]:
    snapshots = {}
    for layer, cache in caches.items():
        if isinstance(cache, ArraysCache):
            snapshots[layer] = (
                "arrays",
                tuple(cache.state),
                mx.array(cache.left_padding) if cache.left_padding is not None else None,
                mx.array(cache.lengths) if cache.lengths is not None else None,
            )
        elif isinstance(cache, KVCache):
            snapshots[layer] = ("kv", cache.keys, cache.values, cache.offset)
        else:
            raise MetadataError(f"unsupported resident cache type at layer {layer}: {type(cache).__name__}")
    return snapshots


def restore_caches(caches: dict[int, ArraysCache | KVCache], snapshots: dict[int, tuple]) -> None:
    require(set(caches) == set(snapshots), "resident cache snapshot layer mismatch")
    for layer, cache in caches.items():
        snapshot = snapshots[layer]
        if isinstance(cache, ArraysCache):
            require(snapshot[0] == "arrays", f"resident cache snapshot type mismatch at layer {layer}")
            cache.state = list(snapshot[1])
            cache.left_padding = snapshot[2]
            cache.lengths = snapshot[3]
        elif isinstance(cache, KVCache):
            require(snapshot[0] == "kv", f"resident cache snapshot type mismatch at layer {layer}")
            cache.keys, cache.values, cache.offset = snapshot[1:]
        else:
            raise MetadataError(f"unsupported resident cache type at layer {layer}: {type(cache).__name__}")


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


def alternate_mtp_head_report(head_dir: Path) -> dict:
    candidates = (
        (head_dir / "nemotron_mtp_head_report.json", "nemotron-mlx-mtp-head-v1"),
        (
            head_dir / "nemotron_mtp_vocab_head_report.json",
            "nemotron-mlx-mtp-vocab-head-v1",
        ),
    )
    existing = [(path, format_) for path, format_ in candidates if path.exists()]
    require(len(existing) == 1, "alternate MTP head must contain exactly one supported report")
    path, format_ = existing[0]
    report = load_json(path)
    require(
        report.get("format") == format_ and report.get("status") == "complete",
        "alternate MTP head is incomplete",
    )
    return report


def preflight(
    model_dir: Path,
    margin_gib: float = DEFAULT_MARGIN_GIB,
    mtp_sidecar: Path | None = None,
    mtp_lm_head: Path | None = None,
    paged_embeddings: bool = False,
) -> dict:
    target_report_path = model_dir / "nemotron_mlx_pack_report.json"
    report = load_json(target_report_path)
    require(report.get("format") == "nemotron-mlx-runtime-v1", "model is not a packed Nemotron runtime")
    require(report.get("status") == "complete", "packed Nemotron runtime is incomplete")
    target_payload = report.get("payload_bytes")
    require(isinstance(target_payload, int) and target_payload > 0, "runtime report has no payload size")
    mtp_payload = 0
    mtp_head_payload = 0
    require(mtp_lm_head is None or mtp_sidecar is not None, "MTP head requires an MTP sidecar")
    if mtp_sidecar is not None:
        mtp_report = load_json(mtp_sidecar / "nemotron_mtp_pack_report.json")
        require(
            mtp_report.get("format") == "nemotron-mlx-mtp-sidecar-v1"
            and mtp_report.get("status") == "complete",
            "MTP sidecar is incomplete",
        )
        require(
            mtp_report.get("source_revision") == report.get("source_revision"),
            "target/MTP source revision mismatch",
        )
        mtp_payload = mtp_report.get("payload_bytes")
        require(isinstance(mtp_payload, int) and mtp_payload > 0, "MTP sidecar has no payload size")
    if mtp_lm_head is not None:
        head_report = alternate_mtp_head_report(mtp_lm_head)
        require(
            head_report.get("source_revision") == report.get("source_revision"),
            "target/MTP head source revision mismatch",
        )
        require(
            head_report.get("source_report_sha256") == sha256_file(target_report_path),
            "target/MTP head packed-report mismatch",
        )
        mtp_head_payload = head_report.get("payload_bytes")
        require(
            isinstance(mtp_head_payload, int) and mtp_head_payload > 0,
            "alternate MTP head has no payload size",
        )
    paged_embedding_bytes = embedding_layout(model_dir)[3] if paged_embeddings else 0
    resident_target_payload = target_payload - paged_embedding_bytes
    require(resident_target_payload > 0, "paged embedding size exceeds target payload")
    payload = resident_target_payload + mtp_payload + mtp_head_payload
    device = mx.device_info()
    kernel_cap = iogpu_wired_limit_bytes()
    apple_cap = int(device.get("max_recommended_working_set_size", 0))
    effective_cap = kernel_cap or apple_cap
    required = resident_requirement(payload, margin_gib)
    memory_size = int(device.get("memory_size", 0))
    required_memory_fraction = required / memory_size if memory_size else math.inf
    allocator_gc_threshold = int(MLX_ALLOCATOR_GC_FRACTION * apple_cap)
    extended_required = math.ceil(required / MLX_ALLOCATOR_GC_FRACTION)
    return {
        "payload_bytes": payload,
        "payload_gib": payload / 2**30,
        "target_payload_gib": target_payload / 2**30,
        "resident_target_payload_gib": resident_target_payload / 2**30,
        "paged_embedding_gib": paged_embedding_bytes / 2**30,
        "mtp_payload_gib": mtp_payload / 2**30,
        "mtp_head_payload_gib": mtp_head_payload / 2**30,
        "margin_gib": margin_gib,
        "required_bytes": required,
        "required_gib": required / 2**30,
        "required_mib_ceil": math.ceil(required / (256 * 2**20)) * 256,
        "kernel_cap_bytes": kernel_cap,
        "kernel_cap_gib": kernel_cap / 2**30,
        "apple_cap_bytes": apple_cap,
        "apple_cap_gib": apple_cap / 2**30,
        "allocator_gc_threshold_bytes": allocator_gc_threshold,
        "allocator_gc_threshold_gib": allocator_gc_threshold / 2**30,
        "effective_cap_bytes": effective_cap,
        "effective_cap_gib": effective_cap / 2**30,
        "memory_size_gib": memory_size / 2**30,
        "required_memory_fraction": required_memory_fraction,
        "extended_run_memory_fraction": DEFAULT_EXTENDED_RUN_MEMORY_FRACTION,
        "extended_required_mib_ceil": (
            math.ceil(extended_required / (256 * 2**20)) * 256
        ),
        "safe_to_attempt": effective_cap >= required,
        "safe_for_extended_run": (
            effective_cap >= required
            and required_memory_fraction <= DEFAULT_EXTENDED_RUN_MEMORY_FRACTION
            and allocator_gc_threshold >= required
        ),
    }


def require_extended_run(memory: dict, allow_high_memory_risk: bool = False) -> None:
    require(memory["safe_to_attempt"], "resident preflight failed")
    require(
        memory["safe_for_extended_run"] or allow_high_memory_risk,
        "extended-run memory guard failed: "
        f"required={memory['required_memory_fraction']:.1%} "
        f"physical_limit={memory['extended_run_memory_fraction']:.1%} "
        f"allocator_gc={memory['allocator_gc_threshold_gib']:.3f}GiB; "
        "use a smaller candidate, raise the wired cap to at least "
        f"{memory['extended_required_mib_ceil']} MiB, or explicitly pass "
        "--allow-high-memory-risk",
    )


class ResidentModel:
    def __init__(
        self,
        model_dir: Path,
        mtp_sidecar: Path | None = None,
        mtp_lm_head: Path | None = None,
        paged_embeddings: bool = False,
        embedding_cache_rows: int = 256,
    ):
        require(mtp_lm_head is None or mtp_sidecar is not None, "MTP head requires an MTP sidecar")
        self.model_dir = model_dir
        self.config = load_json(model_dir / "config.json")
        self.pattern = self.config["hybrid_override_pattern"]
        self.index = load_json(model_dir / "model.safetensors.index.json")
        self.hidden_size = self.config["hidden_size"]
        self.embeddings = (
            PagedBF16Embedding(model_dir, embedding_cache_rows)
            if paged_embeddings
            else self._global("backbone.embeddings.weight")
        )
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
        self.mtp = (
            NemotronMTPSidecar(
                mtp_sidecar,
                self.embeddings,
                self.lm_head,
                alternate_lm_head=mtp_lm_head,
            )
            if mtp_sidecar is not None
            else None
        )

    def reset(self) -> None:
        """Reset sequence state without releasing reusable Metal cache buffers."""

        # Replacing evaluated cache arrays while prior Metal work is still retiring
        # can race IOGPU residency removal on near-cap workloads. Keep the allocated
        # KV storage and zero recurrent state in place across independent samples.
        mx.synchronize()
        next_caches = {}
        recurrent_arrays = []
        for layer, kind in enumerate(self.pattern):
            cache = self.caches.get(layer)
            if kind == "M":
                if not isinstance(cache, ArraysCache):
                    cache = ArraysCache(size=2)
                else:
                    for value in cache.state:
                        if value is not None:
                            value[:] = 0
                            recurrent_arrays.append(value)
                    cache.left_padding = None
                    cache.lengths = None
                next_caches[layer] = cache
            elif kind == "*":
                if not isinstance(cache, KVCache):
                    cache = KVCache()
                else:
                    cache.offset = 0
                next_caches[layer] = cache
        if recurrent_arrays:
            mx.eval(*recurrent_arrays)
            mx.synchronize()
        self.caches = next_caches

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

    def _forward_sequence(
        self,
        token_ids: list[int],
        capture_cache_at: int | tuple[int, ...] | None = None,
    ) -> tuple[mx.array, mx.array, dict | None]:
        require(token_ids, "resident sequence must contain at least one token")
        require(
            all(
                isinstance(token_id, int) and 0 <= token_id < self.embeddings.shape[0]
                for token_id in token_ids
            ),
            "token ID out of range",
        )
        tokens = mx.array(token_ids, dtype=mx.int32)
        x = self.embeddings[tokens].astype(mx.float32).reshape(1, len(token_ids), self.hidden_size)
        single_capture = isinstance(capture_cache_at, int)
        capture_indices = (
            (capture_cache_at,)
            if single_capture
            else (() if capture_cache_at is None else capture_cache_at)
        )
        captured_caches = (
            {index: {} for index in capture_indices} if capture_indices else None
        )
        if capture_cache_at is not None:
            require(
                len(token_ids) > 1
                and capture_indices
                and tuple(sorted(set(capture_indices))) == capture_indices
                and all(0 <= index < len(token_ids) for index in capture_indices),
                "resident cache capture index is out of range",
            )
        for layer, (kind, block) in enumerate(zip(self.pattern, self.blocks)):
            if kind == "M":
                cache = self.caches[layer]
                mask = create_ssm_mask(x, cache)
                if len(token_ids) > 1:
                    captured_states = {} if captured_caches is not None else None
                    x = mamba_sequence_exact(
                        block,
                        x,
                        cache,
                        mask,
                        capture_tokens=capture_indices if captured_caches is not None else None,
                        captured_states=captured_states,
                    )
                    if captured_caches is not None:
                        require(
                            cache.left_padding is None and cache.lengths is None,
                            "resident Mamba capture requires unpadded decode caches",
                        )
                        for index in capture_indices:
                            captured_caches[index][layer] = (
                                "arrays",
                                tuple(captured_states[index]),
                                None,
                                None,
                            )
                else:
                    x = block(x, mask=mask, cache=cache)
            elif kind == "*":
                cache = self.caches[layer]
                initial_offset = cache.offset
                x = block(x, mask=create_attention_mask(x, cache), cache=cache)
                if captured_caches is not None:
                    for index in capture_indices:
                        captured_caches[index][layer] = (
                            "kv",
                            cache.keys,
                            cache.values,
                            initial_offset + index + 1,
                        )
            else:
                x = block(x)
        x = mx.fast.rms_norm(x, self.final_norm, self.config["layer_norm_epsilon"])
        logits = self.lm_head(x).reshape(len(token_ids), -1)
        hidden = x.reshape(len(token_ids), self.hidden_size)
        captured_arrays = []
        if captured_caches is not None:
            require(
                all(set(snapshot) == set(self.caches) for snapshot in captured_caches.values()),
                "resident cache capture is incomplete",
            )
            for cache_snapshot in captured_caches.values():
                for snapshot in cache_snapshot.values():
                    if snapshot[0] == "arrays":
                        captured_arrays.extend(snapshot[1])
        mx.eval(logits, hidden, *self._cache_arrays(), *captured_arrays)
        if single_capture and captured_caches is not None:
            return logits, hidden, captured_caches[capture_indices[0]]
        return logits, hidden, captured_caches

    def forward_sequence(self, token_ids: list[int]) -> tuple[mx.array, mx.array]:
        logits, hidden, _ = self._forward_sequence(token_ids)
        return logits, hidden

    def verify_sequence(
        self,
        token_ids: list[int],
        accepted_index: int,
    ) -> tuple[mx.array, mx.array, dict[int, tuple]]:
        logits, hidden, snapshot = self._forward_sequence(token_ids, accepted_index)
        require(snapshot is not None, "resident verifier produced no accepted-state snapshot")
        return logits, hidden, snapshot

    def verify_sequence_prefixes(
        self,
        token_ids: list[int],
        accepted_indices: tuple[int, ...],
    ) -> tuple[mx.array, mx.array, dict[int, dict[int, tuple]]]:
        logits, hidden, snapshots = self._forward_sequence(token_ids, accepted_indices)
        require(
            isinstance(snapshots, dict) and set(snapshots) == set(accepted_indices),
            "resident verifier prefix snapshots are incomplete",
        )
        return logits, hidden, snapshots

    def logits_sequence(self, token_ids: list[int]) -> mx.array:
        return self.forward_sequence(token_ids)[0]

    def forward(self, token_id: int) -> tuple[mx.array, mx.array]:
        logits, hidden = self.forward_sequence([token_id])
        return logits[0], hidden[0]

    def logits(self, token_id: int) -> mx.array:
        return self.forward(token_id)[0]

    def snapshot(self) -> dict[int, tuple]:
        return snapshot_caches(self.caches)

    def restore(self, snapshot: dict[int, tuple]) -> None:
        restore_caches(self.caches, snapshot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--margin-gib", type=float, default=DEFAULT_MARGIN_GIB)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--token-timings", action="store_true")
    parser.add_argument("--paged-embeddings", action="store_true")
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument("--logits-out", type=Path)
    return parser.parse_args()


def save_logits(path: Path, logits: mx.array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part.npy")
    mx.save(str(temporary), logits.reshape(-1).astype(mx.float32))
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        require(args.max_new_tokens > 0, "max-new-tokens must be positive")
        require(args.embedding_cache_rows >= 0, "embedding cache rows cannot be negative")
        result = preflight(
            args.model_dir,
            args.margin_gib,
            paged_embeddings=args.paged_embeddings,
        )
        print("resident-preflight " + json.dumps(result, separators=(",", ":")), flush=True)
        if args.preflight_only:
            return 0 if result["safe_to_attempt"] else 2
        require(
            result["safe_to_attempt"],
            "Metal wired cap is too low; raise it deliberately before resident loading: "
            f"sudo sysctl -w iogpu.wired_limit_mb={result['required_mib_ceil']}",
        )
        mx.set_wired_limit(result["effective_cap_bytes"])
        mx.set_cache_limit(256 * 2**20)
        try:
            started = time.perf_counter()
            model = ResidentModel(
                args.model_dir,
                paged_embeddings=args.paged_embeddings,
                embedding_cache_rows=args.embedding_cache_rows,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            require(token_ids, "prompt encoded to no tokens")
            logits = None
            for token_id in token_ids:
                logits = model.logits(token_id)
            load_prefill_seconds = time.perf_counter() - started
            require(logits is not None, "resident prefill produced no logits")
            if args.logits_out is not None:
                save_logits(args.logits_out, logits)
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
                f"embedding_lookups={getattr(model.embeddings, 'lookups', 0)} "
                f"embedding_cache_hits={getattr(model.embeddings, 'cache_hits', 0)} "
                f"embedding_staging_ms={getattr(model.embeddings, 'staging_seconds', 0.0) * 1000:.3f} "
                f"token_ids={','.join(str(token) for token in generated)}"
            )
            if args.token_timings:
                print(
                    "resident-token-ms "
                    + ",".join(f"{elapsed * 1000:.3f}" for elapsed in transition_seconds)
                )
            print(tokenizer.decode(generated))
        finally:
            mx.set_wired_limit(result["effective_cap_bytes"])
        return 0
    except (MetadataError, OSError, ValueError, IndexError, RuntimeError) as exc:
        print(f"nemotron resident runtime error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
