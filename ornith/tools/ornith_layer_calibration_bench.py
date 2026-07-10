#!/usr/bin/env python3
"""Benchmark exact BF16 Ornith calibration one decoder layer at a time."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import time
import types
from pathlib import Path


MODEL_ID = "deepreinforce-ai/Ornith-1.0-397B"
FORMAT = "ornith-layer-calibration-benchmark-v1"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    part.replace(path)


def atomic_tensor_file(path: Path, tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    tensor.contiguous().numpy().astype("<f4", copy=False).tofile(part)
    with part.open("rb") as fp:
        os.fsync(fp.fileno())
    part.replace(path)


def required_layer_shards(index_path: Path, layer: int) -> list[str]:
    value = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"missing weight_map: {index_path}")
    prefix = f"model.language_model.layers.{layer}."
    names = sorted({shard for name, shard in weight_map.items() if name.startswith(prefix)})
    if not names:
        raise ValueError(f"layer {layer} has no tensors in {index_path}")
    return names


def find_shard(name: str, raw_dir: Path, preserved_dir: Path | None) -> Path:
    candidates = [raw_dir / name]
    if preserved_dir is not None:
        candidates.append(preserved_dir / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing raw shard {name}; searched: {', '.join(map(str, candidates))}")


def read_records(path: Path, limit: int) -> list[str | dict]:
    values: list[str | dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            value = json.loads(line)
            if "messages" in value:
                values.append(value)
            elif "text" in value:
                values.append(str(value["text"]))
            else:
                raise ValueError("JSONL calibration record needs messages or text")
        else:
            values.append(line)
        if len(values) == limit:
            break
    if not values:
        raise ValueError(f"no prompts in {path}")
    return values


def synchronize(torch, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def mps_memory(torch) -> dict[str, int]:
    if not torch.backends.mps.is_available():
        return {}
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
        "recommended_max": int(torch.mps.recommended_max_memory()),
    }


class LayerStats:
    def __init__(self, torch, experts: int, hidden: int, intermediate: int, device):
        self.torch = torch
        self.total_tokens = 0
        self.frequency = torch.zeros(experts, dtype=torch.int32, device=device)
        self.weight_sum = torch.zeros(experts, dtype=torch.float32, device=device)
        self.ean_sum = torch.zeros(experts, dtype=torch.float32, device=device)
        self.reap_sum = torch.zeros(experts, dtype=torch.float32, device=device)
        self.max_activation = torch.zeros(experts, dtype=torch.float32, device=device)
        self.gate_up_imatrix = torch.zeros((experts, hidden), dtype=torch.float32, device=device)
        self.down_imatrix = torch.zeros((experts, intermediate), dtype=torch.float32, device=device)

    def summary(self) -> dict:
        frequency = self.frequency.cpu()
        return {
            "tokens": self.total_tokens,
            "experts_selected": int((frequency > 0).sum().item()),
            "experts_selected_twice": int((frequency >= 2).sum().item()),
            "expert_assignments": int(frequency.sum().item()),
            "frequency_min": int(frequency.min().item()),
            "frequency_max": int(frequency.max().item()),
            "gate_up_imatrix_finite": bool(self.torch.isfinite(self.gate_up_imatrix).all().item()),
            "down_imatrix_finite": bool(self.torch.isfinite(self.down_imatrix).all().item()),
            "reap_finite": bool(self.torch.isfinite(self.reap_sum).all().item()),
            "gate_up_imatrix_sum": float(self.gate_up_imatrix.sum().item()),
            "down_imatrix_sum": float(self.down_imatrix.sum().item()),
            "reap_sum": float(self.reap_sum.sum().item()),
        }

    def write_artifacts(self, out: Path, layer: int, revision: str, prompts_path: Path) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        calibration_sha = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        frequency = self.frequency.cpu()
        weight_sum = self.weight_sum.cpu()
        ean_sum = self.ean_sum.cpu()
        reap_sum = self.reap_sum.cpu()
        safe_frequency = frequency.clamp_min(1).float()
        gate_name = f"layer-{layer:02d}-gate-up.f32"
        down_name = f"layer-{layer:02d}-down.f32"
        atomic_tensor_file(out / gate_name, self.gate_up_imatrix.cpu())
        atomic_tensor_file(out / down_name, self.down_imatrix.cpu())
        observer = {
            "format": "ornith-reap-observer-v2",
            "source_model": MODEL_ID,
            "source_precision": "bf16",
            "source_revision": revision,
            "quality_scope": "diagnostic-only",
            "calibration_sha256": calibration_sha,
            "calibration_tokens": self.total_tokens,
            "layers": {
                str(layer): {
                    "total_tokens": self.total_tokens,
                    "expert_frequency": frequency.tolist(),
                    "weighted_expert_frequency_sum": weight_sum.tolist(),
                    "ean_mean": (ean_sum / safe_frequency).tolist(),
                    "reap": (reap_sum / safe_frequency).tolist(),
                    "max_activations": self.max_activation.cpu().tolist(),
                }
            },
        }
        tensor_prefix = f"model.language_model.layers.{layer}.mlp.experts"
        manifest = {
            "format": "ornith-imatrix-v1",
            "source_model": MODEL_ID,
            "source_precision": "bf16",
            "source_revision": revision,
            "statistic": "sum_squared_input_activation_per_expert",
            "quality_scope": "diagnostic-only",
            "calibration_sha256": calibration_sha,
            "calibration_tokens": self.total_tokens,
            "tensors": {
                f"{tensor_prefix}.gate_up_proj": {
                    "file": gate_name,
                    "dtype": "float32-le",
                    "shape": list(self.gate_up_imatrix.shape),
                },
                f"{tensor_prefix}.down_proj": {
                    "file": down_name,
                    "dtype": "float32-le",
                    "shape": list(self.down_imatrix.shape),
                },
            },
        }
        digest = hashlib.sha256()
        for name in (down_name, gate_name):
            with (out / name).open("rb") as fp:
                while chunk := fp.read(1024 * 1024):
                    digest.update(chunk)
        manifest["payload_sha256"] = digest.hexdigest()
        observer_path = out / f"observations-layer-{layer:02d}.json"
        imatrix_path = out / f"imatrix-layer-{layer:02d}.json"
        atomic_json(observer_path, observer)
        atomic_json(imatrix_path, manifest)
        return {"observer": str(observer_path), "imatrix": str(imatrix_path)}


def install_observer(torch, layer, trace: dict | None = None) -> LayerStats:
    import torch.nn.functional as F

    experts = layer.mlp.experts
    stats = LayerStats(
        torch,
        experts.num_experts,
        experts.hidden_dim,
        experts.intermediate_dim,
        experts.gate_up_proj.device,
    )

    def observed_forward(module, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        stats.total_tokens += int(hidden_states.shape[0])
        if trace is not None:
            trace["moe_inputs"].append(hidden_states.detach().cpu())
            trace["router_indices"].append(top_k_index.detach().cpu())
            trace["router_weights"].append(top_k_weights.detach().cpu())
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current = hidden_states[token_idx]
                route = top_k_weights[token_idx, top_k_pos].float()
                gate, up = F.linear(current, module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                intermediate = module.act_fn(gate) * up
                expert_output = F.linear(intermediate, module.down_proj[expert_idx])
                norms = torch.linalg.vector_norm(expert_output.float(), dim=-1)

                stats.frequency[expert_idx] += token_idx.numel()
                stats.weight_sum[expert_idx] += route.sum()
                stats.ean_sum[expert_idx] += norms.sum()
                stats.reap_sum[expert_idx] += (norms * route).sum()
                stats.max_activation[expert_idx] = torch.maximum(stats.max_activation[expert_idx], norms.max())
                stats.gate_up_imatrix[expert_idx] += current.float().square().sum(dim=0)
                stats.down_imatrix[expert_idx] += (intermediate.float() * route[:, None]).square().sum(dim=0)

                weighted = expert_output * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, weighted.to(final.dtype))
        return final

    experts.forward = types.MethodType(observed_forward, experts)
    return stats


def timed_forward(torch, module, name: str, timings: dict[str, float], device: str) -> None:
    original = module.forward

    def wrapper(*args, **kwargs):
        synchronize(torch, device)
        started = time.perf_counter()
        value = original(*args, **kwargs)
        synchronize(torch, device)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        return value

    module.forward = wrapper


def load_layer(torch, model_dir: Path, shard_paths: list[Path], layer_id: int, device: str):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).text_config
    if layer_id < 0 or layer_id >= config.num_hidden_layers:
        raise ValueError(f"layer {layer_id} outside 0..{config.num_hidden_layers - 1}")
    with torch.device("meta"):
        layer = Qwen3_5MoeDecoderLayer(config, layer_id)
    expected = layer.state_dict()
    prefix = f"model.language_model.layers.{layer_id}."
    state = {}
    with contextlib.ExitStack() as stack:
        for path in shard_paths:
            handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name in handle.keys():
                if name.startswith(prefix):
                    short = name[len(prefix) :]
                    if short in state:
                        raise ValueError(f"duplicate tensor {name}")
                    state[short] = handle.get_tensor(name)
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        if missing or extra:
            raise ValueError(f"layer tensor mismatch missing={missing} extra={extra}")
        for name, tensor in state.items():
            if tuple(tensor.shape) != tuple(expected[name].shape):
                raise ValueError(f"shape mismatch {name}: {tuple(tensor.shape)} != {tuple(expected[name].shape)}")
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"expected BF16 tensor {name}, got {tensor.dtype}")
        layer.load_state_dict(state, strict=True, assign=True)
        del expected
        params = sum(parameter.numel() for parameter in layer.parameters())
        log(f"layer-copy-start layer={layer_id} device={device} params={params}")
        started = time.perf_counter()
        layer = layer.to(device=device, dtype=torch.bfloat16).eval()
        synchronize(torch, device)
        log(f"layer-copy-done layer={layer_id} elapsed={time.perf_counter() - started:.3f}s")
    return layer, config


def embedding_rows(torch, shard_paths: list[Path], token_ids, device: str):
    from safetensors import safe_open

    name = "model.language_model.embed_tokens.weight"
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                continue
            tensor_slice = handle.get_slice(name)
            unique, inverse = torch.unique(token_ids.cpu(), sorted=True, return_inverse=True)
            rows = torch.cat([tensor_slice[int(token) : int(token) + 1] for token in unique], dim=0)
            return rows[inverse].reshape(*token_ids.shape, rows.shape[-1]).to(device=device)
    raise ValueError(f"missing {name} in layer shards")


def rendered_prompt(tokenizer, value: str | dict) -> str:
    if isinstance(value, dict):
        return tokenizer.apply_chat_template(
            value["messages"],
            tools=value.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )
    return value


def rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def run(args: argparse.Namespace) -> dict:
    import torch
    import transformers
    from transformers import AutoTokenizer

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if args.layer != 0:
        raise ValueError("the initial benchmark intentionally supports layer 0 only")

    shard_names = required_layer_shards(args.index, args.layer)
    if len(shard_names) != 3:
        raise ValueError(f"Ornith layer {args.layer} expected 3 source shards, found {shard_names}")
    shard_paths = [find_shard(name, args.raw_dir, args.preserved_dir) for name in shard_names]
    for path in shard_paths:
        log(f"shard-ready file={path} bytes={path.stat().st_size}")

    records = read_records(args.prompts, args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    log(f"layer-load-start layer={args.layer}")
    load_started = time.perf_counter()
    layer, config = load_layer(torch, args.model_dir, shard_paths, args.layer, args.device)
    load_elapsed = time.perf_counter() - load_started
    trace = None
    if args.dump_output is not None:
        trace = {"attention_outputs": [], "moe_inputs": [], "router_indices": [], "router_weights": []}
    stats = install_observer(torch, layer, trace)
    timings: dict[str, float] = {}
    timed_forward(torch, layer.linear_attn, "attention", timings, args.device)
    timed_forward(torch, layer.mlp, "moe", timings, args.device)
    if trace is not None:
        layer.linear_attn.register_forward_hook(
            lambda _module, _inputs, output: trace["attention_outputs"].append(output.detach().cpu())
        )

    token_count = 0
    fingerprints = []
    dumped_outputs = []
    benchmark_started = time.perf_counter()
    with torch.inference_mode():
        for prompt_id, value in enumerate(records):
            rendered = rendered_prompt(tokenizer, value)
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=False,
            )
            token_ids = encoded["input_ids"]
            hidden = embedding_rows(torch, shard_paths, token_ids, args.device).to(torch.bfloat16)
            position_ids = torch.arange(token_ids.shape[1], device=args.device).unsqueeze(0)
            synchronize(torch, args.device)
            started = time.perf_counter()
            output = layer(
                hidden,
                position_embeddings=(None, None),
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
            )
            synchronize(torch, args.device)
            elapsed = time.perf_counter() - started
            output_float = output.float()
            if not bool(torch.isfinite(output_float).all().item()):
                raise ValueError(f"non-finite layer output for prompt {prompt_id}")
            fingerprints.append(
                {
                    "prompt": prompt_id,
                    "tokens": int(token_ids.numel()),
                    "elapsed_seconds": elapsed,
                    "mean": float(output_float.mean().item()),
                    "rms": float(output_float.square().mean().sqrt().item()),
                    "max_abs": float(output_float.abs().max().item()),
                }
            )
            if args.dump_output is not None:
                dumped_outputs.append(output.detach().cpu())
            token_count += int(token_ids.numel())
            log(
                f"prompt-done prompt={prompt_id + 1}/{len(records)} tokens={token_ids.numel()} "
                f"elapsed={elapsed:.3f}s rate={token_ids.numel() / elapsed:.3f}tok/s"
            )
            del output, output_float, hidden
    synchronize(torch, args.device)
    benchmark_elapsed = time.perf_counter() - benchmark_started
    stats_summary = stats.summary()
    finite_fields = ("gate_up_imatrix_finite", "down_imatrix_finite", "reap_finite")
    if not all(stats_summary[name] for name in finite_fields):
        raise ValueError("non-finite calibration statistics")
    expected_assignments = token_count * config.num_experts_per_tok
    if stats_summary["expert_assignments"] != expected_assignments:
        raise ValueError(f"expert assignment mismatch {stats_summary['expert_assignments']} != {expected_assignments}")
    artifacts = None
    if args.stats_out is not None:
        artifacts = stats.write_artifacts(args.stats_out, args.layer, args.revision, args.prompts)
    if args.dump_output is not None:
        args.dump_output.parent.mkdir(parents=True, exist_ok=True)
        part = args.dump_output.with_name(args.dump_output.name + ".part")
        torch.save(
            {
                "format": "ornith-layer-output-v1",
                "outputs": dumped_outputs,
                "attention_outputs": trace["attention_outputs"],
                "moe_inputs": trace["moe_inputs"],
                "router_indices": trace["router_indices"],
                "router_weights": trace["router_weights"],
            },
            part,
        )
        part.replace(args.dump_output)

    return {
        "format": FORMAT,
        "source_model": MODEL_ID,
        "source_revision": args.revision,
        "source_precision": "bf16",
        "quality_scope": "diagnostic-only",
        "layer": args.layer,
        "layer_type": config.layer_types[args.layer],
        "device": args.device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "shards": [
            {"name": name, "path": str(path), "bytes": path.stat().st_size}
            for name, path in zip(shard_names, shard_paths)
        ],
        "prompts": len(records),
        "tokens": token_count,
        "max_length": args.max_length,
        "load_seconds": load_elapsed,
        "benchmark_seconds": benchmark_elapsed,
        "tokens_per_second": token_count / benchmark_elapsed,
        "phase_seconds": timings,
        "stats": stats_summary,
        "calibration_artifacts": artifacts,
        "outputs": fingerprints,
        "output_tensor": str(args.dump_output) if args.dump_output is not None else None,
        "memory": {
            "rss": rss_bytes(),
            "peak_rss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "mps": mps_memory(torch),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--preserved-dir", type=Path)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--revision", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--max-prompts", type=int, default=1)
    p.add_argument("--max-length", type=int, default=16)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p.add_argument("--dump-output", type=Path)
    p.add_argument("--stats-out", type=Path)
    p.add_argument("--print-plan", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_prompts <= 0 or args.max_length <= 0:
        raise SystemExit("--max-prompts and --max-length must be positive")
    if args.print_plan:
        print(json.dumps({"layer": args.layer, "shards": required_layer_shards(args.index, args.layer)}, indent=2))
        return 0
    log(f"benchmark-start layer={args.layer} device={args.device} revision={args.revision}")
    try:
        result = run(args)
        atomic_json(args.out, result)
    except Exception as exc:
        log(f"benchmark-failed type={type(exc).__name__} error={exc}")
        raise
    log(
        f"benchmark-done tokens={result['tokens']} elapsed={result['benchmark_seconds']:.3f}s "
        f"rate={result['tokens_per_second']:.3f}tok/s out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
