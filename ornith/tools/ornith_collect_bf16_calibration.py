#!/usr/bin/env python3
"""Collect exact Ornith REAP saliency and routed-expert imatrices on BF16 weights.

This script is intended for a multi-GPU/cloud host that can load the original
model. It writes only calibration statistics; those artifacts can then be used
by the disk-streaming quantizer on the local machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from pathlib import Path


MODEL_ID = "deepreinforce-ai/Ornith-1.0-397B"


def prompts(path: Path) -> list[str | dict]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            item = json.loads(line)
            if "messages" in item:
                values.append(item)
            else:
                values.append(str(item["text"]))
        else:
            values.append(line)
    return values


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class LayerStats:
    def __init__(self, torch, experts: int, hidden: int, intermediate: int, device):
        self.torch = torch
        self.total_tokens = 0
        self.frequency = torch.zeros(experts, dtype=torch.int64, device=device)
        self.weight_sum = torch.zeros(experts, dtype=torch.float64, device=device)
        self.ean_sum = torch.zeros(experts, dtype=torch.float64, device=device)
        self.reap_sum = torch.zeros(experts, dtype=torch.float64, device=device)
        self.max_activation = torch.zeros(experts, dtype=torch.float32, device=device)
        self.gate_up_imatrix = torch.zeros((experts, hidden), dtype=torch.float32, device=device)
        self.down_imatrix = torch.zeros((experts, intermediate), dtype=torch.float32, device=device)

    def cpu_state(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "frequency": self.frequency.cpu(),
            "weight_sum": self.weight_sum.cpu(),
            "ean_sum": self.ean_sum.cpu(),
            "reap_sum": self.reap_sum.cpu(),
            "max_activation": self.max_activation.cpu(),
            "gate_up_imatrix": self.gate_up_imatrix.cpu(),
            "down_imatrix": self.down_imatrix.cpu(),
        }

    def load(self, value: dict) -> None:
        self.total_tokens = int(value["total_tokens"])
        for name in ("frequency", "weight_sum", "ean_sum", "reap_sum", "max_activation", "gate_up_imatrix", "down_imatrix"):
            getattr(self, name).copy_(value[name].to(getattr(self, name).device))


class Collector:
    def __init__(self, torch):
        self.torch = torch
        self.layers: dict[int, LayerStats] = {}

    def install(self, model) -> None:
        import torch.nn.functional as F

        base = model.model
        text_model = base.language_model if hasattr(base, "language_model") else base
        layers = text_model.layers
        for layer_id, layer in enumerate(layers):
            experts = layer.mlp.experts
            stats = LayerStats(
                self.torch,
                experts.num_experts,
                experts.hidden_dim,
                experts.intermediate_dim,
                experts.gate_up_proj.device,
            )
            self.layers[layer_id] = stats

            def observed_forward(module, hidden_states, top_k_index, top_k_weights, *, _stats=stats):
                final = self.torch.zeros_like(hidden_states)
                _stats.total_tokens += int(hidden_states.shape[0])
                with self.torch.no_grad():
                    expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts).permute(2, 1, 0)
                    expert_hit = self.torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
                    for expert_idx_tensor in expert_hit:
                        expert_idx = int(expert_idx_tensor.item())
                        top_k_pos, token_idx = self.torch.where(expert_mask[expert_idx])
                        current = hidden_states[token_idx]
                        route = top_k_weights[token_idx, top_k_pos].float()
                        gate, up = F.linear(current, module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                        intermediate = module.act_fn(gate) * up
                        expert_output = F.linear(intermediate, module.down_proj[expert_idx])
                        norms = self.torch.linalg.vector_norm(expert_output.float(), dim=-1)

                        _stats.frequency[expert_idx] += token_idx.numel()
                        _stats.weight_sum[expert_idx] += route.double().sum()
                        _stats.ean_sum[expert_idx] += norms.double().sum()
                        _stats.reap_sum[expert_idx] += (norms * route).double().sum()
                        _stats.max_activation[expert_idx] = self.torch.maximum(
                            _stats.max_activation[expert_idx], norms.max()
                        )
                        _stats.gate_up_imatrix[expert_idx] += current.float().square().sum(dim=0)
                        routed_intermediate = intermediate.float() * route[:, None]
                        _stats.down_imatrix[expert_idx] += routed_intermediate.square().sum(dim=0)

                        weighted = expert_output * top_k_weights[token_idx, top_k_pos, None]
                        final.index_add_(0, token_idx, weighted.to(final.dtype))
                return final

            experts.forward = types.MethodType(observed_forward, experts)

    def state_dict(self) -> dict:
        return {layer: stats.cpu_state() for layer, stats in self.layers.items()}

    def load_state_dict(self, value: dict) -> None:
        for layer, state in value.items():
            self.layers[int(layer)].load(state)


def save_checkpoint(torch, path: Path, next_prompt: int, collector: Collector) -> None:
    tmp = path.with_name(path.name + ".part")
    torch.save({"format": "ornith-bf16-calibration-checkpoint-v1", "next_prompt": next_prompt, "layers": collector.state_dict()}, tmp)
    tmp.replace(path)


def write_outputs(torch, out: Path, collector: Collector, prompt_path: Path, token_count: int, revision: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    calibration_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    observer = {
        "format": "ornith-reap-observer-v2",
        "source_model": MODEL_ID,
        "source_precision": "bf16",
        "source_revision": revision,
        "quality_scope": "candidate-final",
        "calibration_sha256": calibration_sha,
        "calibration_tokens": token_count,
        "layers": {},
    }
    manifest = {
        "format": "ornith-imatrix-v1",
        "source_model": MODEL_ID,
        "source_precision": "bf16",
        "source_revision": revision,
        "statistic": "sum_squared_input_activation_per_expert",
        "calibration_sha256": calibration_sha,
        "calibration_tokens": token_count,
        "tensors": {},
    }
    for layer_id, stats in sorted(collector.layers.items()):
        state = stats.cpu_state()
        freq = state["frequency"]
        safe_freq = freq.clamp_min(1).double()
        observer["layers"][str(layer_id)] = {
            "total_tokens": stats.total_tokens,
            "expert_frequency": freq.tolist(),
            "weighted_expert_frequency_sum": state["weight_sum"].tolist(),
            "ean_mean": (state["ean_sum"] / safe_freq).tolist(),
            "reap": (state["reap_sum"] / safe_freq).tolist(),
            "max_activations": state["max_activation"].tolist(),
        }
        prefix = f"layer-{layer_id:02d}"
        gate_file = f"{prefix}-gate-up.f32"
        down_file = f"{prefix}-down.f32"
        state["gate_up_imatrix"].contiguous().numpy().astype("<f4", copy=False).tofile(out / gate_file)
        state["down_imatrix"].contiguous().numpy().astype("<f4", copy=False).tofile(out / down_file)
        tensor_prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        manifest["tensors"][f"{tensor_prefix}.gate_up_proj"] = {
            "file": gate_file, "dtype": "float32-le", "shape": list(state["gate_up_imatrix"].shape)
        }
        manifest["tensors"][f"{tensor_prefix}.down_proj"] = {
            "file": down_file, "dtype": "float32-le", "shape": list(state["down_imatrix"].shape)
        }
    payload_digest = hashlib.sha256()
    for _name, entry in sorted(manifest["tensors"].items()):
        with (out / entry["file"]).open("rb") as fp:
            while raw := fp.read(1024 * 1024):
                payload_digest.update(raw)
    manifest["payload_sha256"] = payload_digest.hexdigest()
    atomic_json(out / "observations.json", observer)
    atomic_json(out / "imatrix.json", manifest)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, type=Path, help="rendered text, one line or JSON {text} per prompt")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision", required=True, help="immutable Hugging Face commit hash for the calibrated weights")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--checkpoint-every", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.model != MODEL_ID:
        raise SystemExit(f"this collector is model-specific and only accepts {MODEL_ID}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    values = prompts(args.prompts)
    if not values:
        raise SystemExit("no calibration prompts")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    collector = Collector(torch)
    collector.install(model)
    checkpoint = args.out / "checkpoint.pt"
    start = 0
    token_count = 0
    if args.resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") != "ornith-bf16-calibration-checkpoint-v1":
            raise SystemExit("unsupported calibration checkpoint")
        collector.load_state_dict(saved["layers"])
        start = int(saved["next_prompt"])
        token_count = sum(stats.total_tokens for stats in collector.layers.values()) // max(1, len(collector.layers))

    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for index in range(start, len(values)):
            prompt = values[index]
            if isinstance(prompt, dict):
                rendered = tokenizer.apply_chat_template(
                    prompt["messages"],
                    tools=prompt.get("tools"),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered = prompt
            encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=args.max_length)
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            model(**encoded, use_cache=False)
            token_count += int(encoded["input_ids"].numel())
            print(f"calibration prompt={index + 1}/{len(values)} tokens={token_count}", flush=True)
            if args.checkpoint_every and (index + 1) % args.checkpoint_every == 0:
                args.out.mkdir(parents=True, exist_ok=True)
                save_checkpoint(torch, checkpoint, index + 1, collector)
    write_outputs(torch, args.out, collector, args.prompts, token_count, args.revision)
    save_checkpoint(torch, checkpoint, len(values), collector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
