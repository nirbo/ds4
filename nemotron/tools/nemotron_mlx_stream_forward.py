#!/usr/bin/env python3
"""Layer-streamed official Nemotron NVFP4 forward for baseline logits and routing."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_attention import load_attention_layer
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mamba import load_mamba_layer
from nemotron_mlx_moe import slice_expert_blocks
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_prune_materialize import load_source_state


def validate_virtual_plan(plan: dict, config: dict, source_revision: str) -> dict[str, list[int]]:
    require(plan.get("source_revision") == source_revision, "virtual plan/source revision mismatch")
    require(plan.get("old_num_experts") == config["n_routed_experts"], "virtual plan expert mismatch")
    kept_by_layer = plan.get("kept_by_layer")
    require(isinstance(kept_by_layer, dict), "virtual plan has no retained-expert map")
    expected = [layer for layer, kind in enumerate(config["hybrid_override_pattern"]) if kind == "E"]
    require(plan.get("model_moe_layers") == expected, "virtual plan layer catalog mismatch")
    for layer in expected:
        kept = kept_by_layer.get(str(layer))
        require(isinstance(kept, list), f"virtual plan has no layer {layer}")
        require(
            config["num_experts_per_tok"] <= len(kept) <= config["n_routed_experts"],
            f"virtual plan layer {layer} retained count is invalid",
        )
        require(kept == sorted(set(kept)), f"virtual plan layer {layer} is not sorted and unique")
        require(kept[0] >= 0 and kept[-1] < config["n_routed_experts"], f"virtual plan layer {layer} ID is invalid")
    return kept_by_layer


def validate_virtual_hybrid_plan(
    plan: dict, config: dict, source_revision: str
) -> tuple[dict[str, list[int]], dict[str, np.ndarray]]:
    require(plan.get("format") == "nemotron-hybrid-width-plan-v1", "invalid hybrid plan format")
    require(plan.get("source_revision") == source_revision, "hybrid plan/source revision mismatch")
    experts = config["n_routed_experts"]
    require(plan.get("old_num_experts") == experts, "hybrid plan expert mismatch")
    expected = [layer for layer, kind in enumerate(config["hybrid_override_pattern"]) if kind == "E"]
    require(plan.get("model_moe_layers") == expected, "hybrid plan layer catalog mismatch")
    modes = plan.get("layers")
    require(isinstance(modes, dict), "hybrid plan has no layer modes")
    retained = {}
    width = {}
    for layer in expected:
        item = modes.get(str(layer))
        require(isinstance(item, dict), f"hybrid plan has no layer {layer}")
        if item.get("mode") == "experts":
            kept = item.get("kept_experts")
            require(isinstance(kept, list), f"hybrid expert layer {layer} has no retained set")
            require(kept == sorted(set(kept)), f"hybrid expert layer {layer} retained set is invalid")
            require(config["num_experts_per_tok"] <= len(kept) <= experts, f"hybrid layer {layer} count is invalid")
            require(kept[0] >= 0 and kept[-1] < experts, f"hybrid layer {layer} expert ID is invalid")
            retained[str(layer)] = kept
        elif item.get("mode") == "width":
            blocks = np.asarray(item.get("kept_blocks"), dtype=np.int32)
            require(blocks.ndim == 2 and blocks.shape[0] == experts, f"hybrid width layer {layer} shape is invalid")
            require(blocks.shape[1] > 0, f"hybrid width layer {layer} removes every block")
            require(np.all((0 <= blocks) & (blocks < 168)), f"hybrid width layer {layer} block ID is invalid")
            require(np.all(np.diff(blocks, axis=1) > 0), f"hybrid width layer {layer} blocks are not ordered")
            width[str(layer)] = blocks
        else:
            raise MetadataError(f"hybrid layer {layer} has invalid mode")
    return retained, width


class StreamingForward:
    def __init__(
        self,
        source_dir: Path,
        retained_by_layer: dict[str, list[int]] | None = None,
        width_blocks_by_layer: dict[str, np.ndarray] | None = None,
    ):
        self.source_dir = source_dir
        self.config = load_json(source_dir / "config.json")
        self.pattern = self.config["hybrid_override_pattern"]
        self.hidden_size = self.config["hidden_size"]
        self.index = load_json(source_dir / "model.safetensors.index.json")
        self.embeddings = self._global_tensor("backbone.embeddings.weight")
        self.final_norm = self._global_tensor("backbone.norm_f.weight")
        self.lm_head = ModelOptBF16Linear(self._global_tensor("lm_head.weight"))
        self.caches: dict[int, ArraysCache | KVCache] = {}
        self.routing: dict[int, dict[str, list[float] | list[int]]] = {}
        self.layer_inputs: dict[int, np.ndarray] = {}
        self.retained_by_layer = retained_by_layer
        self.width_blocks_by_layer = width_blocks_by_layer

    def _global_tensor(self, name: str) -> mx.array:
        shard_name = self.index["weight_map"].get(name)
        require(isinstance(shard_name, str), f"missing global tensor: {name}")
        tensors = mx.load(str(self.source_dir / shard_name))
        require(name in tensors, f"global tensor absent from shard: {name}")
        return tensors[name]

    def _cache(self, layer: int, kind: str):
        cache = self.caches.get(layer)
        if cache is None:
            cache = ArraysCache(size=2) if kind == "M" else KVCache()
            self.caches[layer] = cache
        return cache

    @staticmethod
    def _eval_cache(cache) -> None:
        if isinstance(cache, ArraysCache):
            mx.eval(cache.state)
        elif isinstance(cache, KVCache) and cache.keys is not None:
            mx.eval(cache.keys, cache.values)

    def forward_token(self, token_id: int, max_layers: int | None = None, trace: bool = False) -> mx.array:
        require(0 <= token_id < self.embeddings.shape[0], "token ID out of range")
        x = self.embeddings[token_id].astype(mx.float32).reshape(1, 1, self.hidden_size)
        layer_limit = len(self.pattern) if max_layers is None else min(max_layers, len(self.pattern))
        for layer, kind in enumerate(self.pattern[:layer_limit]):
            started = time.perf_counter()
            if kind == "M":
                block = load_mamba_layer(self.source_dir, layer)
                cache = self._cache(layer, kind)
                x = block(x, mask=None, cache=cache)
            elif kind == "E":
                block = load_moe_layer(self.source_dir, layer)
                width_blocks = (
                    None if self.width_blocks_by_layer is None else self.width_blocks_by_layer.get(str(layer))
                )
                if width_blocks is not None:
                    block.experts = slice_expert_blocks(block.experts, width_blocks)
                retained = (
                    None if self.retained_by_layer is None else self.retained_by_layer.get(str(layer))
                )
                require(
                    self.retained_by_layer is None or isinstance(retained, list) or width_blocks is not None,
                    f"virtual prune plan has no layer {layer}",
                )
                if retained is None:
                    x, indices, scores, output_norms = block.forward_with_observation(x)
                else:
                    x, indices, scores, output_norms = block.forward_with_retained(x, retained)
                mx.eval(x, indices, scores, output_norms)
                self.routing[layer] = {
                    "indices": [int(value) for value in indices.reshape(-1).tolist()],
                    "scores": [float(value) for value in scores.reshape(-1).tolist()],
                    "output_norms": [float(value) for value in output_norms.reshape(-1).tolist()],
                }
                cache = None
            elif kind == "*":
                block = load_attention_layer(self.source_dir, layer)
                cache = self._cache(layer, kind)
                x = block(x, mask=None, cache=cache)
            else:
                raise MetadataError(f"unsupported Nemotron layer type {kind!r} at {layer}")
            mx.eval(x)
            if cache is not None:
                self._eval_cache(cache)
            elapsed = time.perf_counter() - started
            if trace:
                print(
                    f"layer={layer:02d} kind={kind} elapsed={elapsed:.3f}s "
                    f"active={mx.get_active_memory() / 2**30:.2f}GiB peak={mx.get_peak_memory() / 2**30:.2f}GiB",
                    flush=True,
                )
            del block
            gc.collect()
            mx.clear_cache()

        if layer_limit != len(self.pattern):
            return x
        normalized = mx.fast.rms_norm(x, self.final_norm, self.config["layer_norm_epsilon"])
        logits = self.lm_head(normalized)
        mx.eval(logits)
        return logits

    def forward_sequence(
        self,
        token_ids: list[int],
        max_layers: int | None = None,
        trace: bool = False,
        score_head: bool = True,
        capture_pair_cosines: bool = False,
        capture_layer_inputs: set[int] | None = None,
        capture_positions: list[int] | None = None,
    ) -> mx.array:
        require(token_ids, "token sequence is empty")
        require(not self.caches, "layer-major prefill requires fresh cache state")
        require(all(0 <= token_id < self.embeddings.shape[0] for token_id in token_ids), "token ID out of range")
        require(
            capture_positions is None or capture_layer_inputs is not None,
            "capture positions require layer-input capture",
        )
        if capture_positions is not None:
            require(
                capture_positions == sorted(set(capture_positions))
                and capture_positions
                and capture_positions[0] >= 0
                and capture_positions[-1] < len(token_ids),
                "capture positions are invalid",
            )
        x = self.embeddings[mx.array(token_ids, dtype=mx.uint32)].astype(mx.float32).reshape(
            1, len(token_ids), self.hidden_size
        )
        layer_limit = len(self.pattern) if max_layers is None else min(max_layers, len(self.pattern))
        for layer, kind in enumerate(self.pattern[:layer_limit]):
            started = time.perf_counter()
            if capture_layer_inputs is not None and layer in capture_layer_inputs:
                captured = x if capture_positions is None else x[:, capture_positions, :]
                self.layer_inputs[layer] = np.asarray(captured.astype(mx.float32))
            outputs = []
            route_indices = []
            route_scores = []
            route_norms = []
            route_outputs = []
            if kind == "M":
                block = load_mamba_layer(self.source_dir, layer)
                cache = self._cache(layer, kind)
                for position in range(len(token_ids)):
                    output = block(x[:, position : position + 1, :], mask=None, cache=cache)
                    mx.eval(output)
                    self._eval_cache(cache)
                    outputs.append(output)
            elif kind == "E":
                block = load_moe_layer(self.source_dir, layer)
                width_blocks = (
                    None if self.width_blocks_by_layer is None else self.width_blocks_by_layer.get(str(layer))
                )
                if width_blocks is not None:
                    block.experts = slice_expert_blocks(block.experts, width_blocks)
                cache = None
                retained = (
                    None if self.retained_by_layer is None else self.retained_by_layer.get(str(layer))
                )
                require(
                    self.retained_by_layer is None or isinstance(retained, list) or width_blocks is not None,
                    f"virtual prune plan has no layer {layer}",
                )
                require(
                    retained is None or not capture_pair_cosines,
                    "pair-cosine capture is unavailable with virtual pruning",
                )
                for position in range(len(token_ids)):
                    if capture_pair_cosines:
                        output, indices, scores, output_norms, selected_outputs = (
                            block.forward_with_expert_outputs(x[:, position : position + 1, :])
                        )
                        route_outputs.append(selected_outputs)
                    elif retained is None:
                        output, indices, scores, output_norms = block.forward_with_observation(
                            x[:, position : position + 1, :]
                        )
                    else:
                        output, indices, scores, output_norms = block.forward_with_retained(
                            x[:, position : position + 1, :], retained
                        )
                    outputs.append(output)
                    route_indices.append(indices)
                    route_scores.append(scores)
                    route_norms.append(output_norms)
            elif kind == "*":
                block = load_attention_layer(self.source_dir, layer)
                cache = self._cache(layer, kind)
                for position in range(len(token_ids)):
                    output = block(x[:, position : position + 1, :], mask=None, cache=cache)
                    mx.eval(output)
                    self._eval_cache(cache)
                    outputs.append(output)
            else:
                raise MetadataError(f"unsupported Nemotron layer type {kind!r} at {layer}")
            x = mx.concatenate(outputs, axis=1)
            evaluation = [x]
            if route_indices:
                indices = mx.concatenate(route_indices, axis=1)
                scores = mx.concatenate(route_scores, axis=1)
                output_norms = mx.concatenate(route_norms, axis=1)
                evaluation.extend([indices, scores, output_norms])
                if route_outputs:
                    selected_outputs = mx.concatenate(route_outputs, axis=1).astype(mx.float32)
                    unit_outputs = selected_outputs / mx.maximum(
                        output_norms[..., None].astype(mx.float32),
                        mx.array(1e-12, dtype=mx.float32),
                    )
                    pair_cosines = mx.matmul(unit_outputs, unit_outputs.swapaxes(-1, -2))
                    evaluation.append(pair_cosines)
            mx.eval(*evaluation)
            if cache is not None:
                self._eval_cache(cache)
            if route_indices:
                self.routing[layer] = {
                    "indices": [int(value) for value in indices.reshape(-1).tolist()],
                    "scores": [float(value) for value in scores.reshape(-1).tolist()],
                    "output_norms": [float(value) for value in output_norms.reshape(-1).tolist()],
                }
                if route_outputs:
                    self.routing[layer]["pair_cosines"] = [
                        float(value) for value in pair_cosines.reshape(-1).tolist()
                    ]
            if trace:
                print(
                    f"layer={layer:02d} kind={kind} positions={len(token_ids)} "
                    f"elapsed={time.perf_counter() - started:.3f}s "
                    f"active={mx.get_active_memory() / 2**30:.2f}GiB peak={mx.get_peak_memory() / 2**30:.2f}GiB",
                    flush=True,
                )
            del block, outputs
            gc.collect()
            mx.clear_cache()
        if layer_limit != len(self.pattern) or not score_head:
            return x
        normalized = mx.fast.rms_norm(x[:, -1:, :], self.final_norm, self.config["layer_norm_epsilon"])
        logits = self.lm_head(normalized)
        mx.eval(logits)
        return logits


def parse_token_ids(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise MetadataError(f"invalid token ID list: {value}") from exc
    require(result, "token ID list is empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--token-ids")
    inputs.add_argument("--prompt")
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--routing-out", type=Path)
    parser.add_argument("--logits-out", type=Path)
    parser.add_argument("--virtual-prune-plan", type=Path)
    parser.add_argument("--source-state", type=Path)
    parser.add_argument("--token-major", action="store_true")
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.max_layers is None or args.max_layers > 0, "max-layers must be positive")
        require(args.top_k > 0, "top-k must be positive")
        tokenizer = None
        if args.prompt is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.source_dir, local_files_only=True)
            token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
            require(token_ids, "prompt encoded to no tokens")
            print(f"prompt-token-ids={','.join(str(token_id) for token_id in token_ids)}")
        else:
            token_ids = parse_token_ids(args.token_ids or "0")
        retained_by_layer = None
        if args.virtual_prune_plan is not None:
            require(args.source_state is not None, "virtual pruning requires --source-state")
            source_state = load_source_state(args.source_state, args.source_dir)
            plan = load_json(args.virtual_prune_plan)
            retained_by_layer = validate_virtual_plan(plan, load_json(args.source_dir / "config.json"), source_state["revision"])
        runner = StreamingForward(args.source_dir, retained_by_layer)
        output = None
        started = time.perf_counter()
        if len(token_ids) > 1 and not args.token_major:
            output = runner.forward_sequence(token_ids, args.max_layers, args.trace)
            print(
                f"prefill-done positions={len(token_ids)} active={mx.get_active_memory() / 2**30:.2f}GiB "
                f"peak={mx.get_peak_memory() / 2**30:.2f}GiB",
                flush=True,
            )
        else:
            for position, token_id in enumerate(token_ids):
                output = runner.forward_token(token_id, args.max_layers, args.trace)
                print(
                    f"token-done position={position} token_id={token_id} "
                    f"active={mx.get_active_memory() / 2**30:.2f}GiB peak={mx.get_peak_memory() / 2**30:.2f}GiB",
                    flush=True,
                )
        require(output is not None, "forward produced no output")
        if args.max_layers is None:
            if args.logits_out:
                args.logits_out.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.logits_out.with_name(args.logits_out.name + ".part.npy")
                mx.save(str(temporary), output.reshape(-1).astype(mx.float32))
                temporary.replace(args.logits_out)
            k = min(args.top_k, output.shape[-1])
            indices = mx.argpartition(-output.reshape(-1), kth=k - 1)[:k]
            scores = output.reshape(-1)[indices]
            order = mx.argsort(-scores)
            mx.eval(indices, scores, order)
            ranked = []
            for index in order.tolist():
                token_id = int(indices[index])
                item = {"token_id": token_id, "score": float(scores[index])}
                if tokenizer is not None:
                    item["text"] = tokenizer.decode([token_id])
                ranked.append(item)
            print("top-k " + json.dumps(ranked, separators=(",", ":")))
        else:
            print(f"hidden-checksum={float(output[:, -1:, :].sum()):.9g}")
            if output.shape[1] > 1:
                print(f"hidden-sequence-checksum={float(output.sum()):.9g}")
        if args.routing_out:
            args.routing_out.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.routing_out.with_name(args.routing_out.name + ".part")
            temporary.write_text(json.dumps({"token_ids": token_ids, "layers": runner.routing}, indent=2) + "\n")
            temporary.replace(args.routing_out)
        print(f"forward-done tokens={len(token_ids)} elapsed={time.perf_counter() - started:.3f}s")
        return 0
    except (MetadataError, OSError, ValueError, IndexError) as exc:
        print(f"nemotron streaming forward error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
