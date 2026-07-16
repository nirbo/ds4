#!/usr/bin/env python3
"""Exact BF16 reference for Nemotron 3 Super's official repeated MTP head."""

from __future__ import annotations

import math
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock, group_expert_select

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear, bf16_gather_matvec, bf16_switch_matmul
from nemotron_mlx_mtp_mixed import mixed_affine_switch
from nemotron_prune_materialize import sha256_file


MTP_ATTENTION_PREFIX = "mtp.layers.0"
MTP_MOE_PREFIX = "mtp.layers.1"


def binary_affine_supported() -> bool:
    """Return whether this MLX build can execute one-bit affine kernels."""

    try:
        restored = mx.dequantize(
            mx.zeros((1, 4), dtype=mx.uint32),
            mx.ones((1, 1), dtype=mx.bfloat16),
            -mx.ones((1, 1), dtype=mx.bfloat16),
            group_size=128,
            bits=1,
            mode="affine",
            dtype=mx.float32,
        )
        mx.eval(restored)
        return restored.shape == (1, 128)
    except (RuntimeError, ValueError):
        return False


class NemotronMTPCache:
    """Caller-owned attention state for one Nemotron MTP sequence."""

    def __init__(self):
        self.attention = KVCache()

    @property
    def offset(self) -> int:
        return self.attention.offset

    @property
    def nbytes(self) -> int:
        return self.attention.nbytes

    def checkpoint(self) -> int:
        return self.offset

    def restore(self, checkpoint: int) -> None:
        require(
            isinstance(checkpoint, int) and not isinstance(checkpoint, bool),
            "MTP cache checkpoint must be an integer",
        )
        require(0 <= checkpoint <= self.offset, "MTP cache checkpoint is out of range")
        trimmed = self.attention.trim(self.offset - checkpoint)
        require(trimmed >= 0 and self.offset == checkpoint, "MTP cache restore failed")

    def reset(self) -> None:
        self.attention.offset = 0


class _NemotronMTPAttentionState:
    def _attention_step(
        self,
        target_hidden: mx.array,
        accepted_token_id: int,
        cache: NemotronMTPCache | None,
    ) -> mx.array:
        require(
            0 <= accepted_token_id < self.embeddings.shape[0],
            "MTP token ID out of range",
        )
        hidden = target_hidden.astype(mx.float32).reshape(1, 1, -1)
        require(hidden.shape[-1] == self.embeddings.shape[1], "MTP hidden size mismatch")
        embedding = (
            self.embeddings[accepted_token_id]
            .astype(mx.float32)
            .reshape(1, 1, -1)
        )
        embedding = mx.fast.rms_norm(embedding, self.enorm_weight, self.epsilon)
        hidden = mx.fast.rms_norm(hidden, self.hnorm_weight, self.epsilon)
        fused = self.eh_proj(mx.concatenate([embedding, hidden], axis=-1))
        return self.attention(
            fused,
            mask=None,
            cache=cache.attention if cache is not None else None,
        )

    def advance_cache(
        self,
        target_hidden: mx.array,
        accepted_token_id: int,
        cache: NemotronMTPCache,
    ) -> None:
        """Append exact attention K/V without evaluating unused MoE or logits."""

        self._attention_step(target_hidden, accepted_token_id, cache)
        require(
            cache.attention.keys is not None and cache.attention.values is not None,
            "MTP attention step did not populate its cache",
        )
        mx.eval(cache.attention.keys, cache.attention.values)


def load_indexed_tensors(source_dir: Path, names: set[str]) -> dict[str, mx.array]:
    """Load an exact named subset while opening each source shard only once."""

    index = load_json(source_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    missing = sorted(names - set(weight_map))
    require(not missing, f"MTP tensors absent from source index: {missing[:4]}")
    by_shard: dict[str, set[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], set()).add(name)

    result: dict[str, mx.array] = {}
    for shard_name, shard_names in sorted(by_shard.items()):
        loaded = mx.load(str(source_dir / shard_name))
        absent = sorted(shard_names - set(loaded))
        require(not absent, f"MTP tensors absent from {shard_name}: {absent[:4]}")
        result.update((name, loaded[name]) for name in shard_names)
    return result


def mtp_tensor_names(config: dict, retained_experts: list[int] | None = None) -> set[str]:
    experts = config.get("n_routed_experts")
    require(isinstance(experts, int) and experts > 0, "invalid MTP expert count")
    retained = list(range(experts)) if retained_experts is None else retained_experts
    require(
        retained
        and len(set(retained)) == len(retained)
        and all(isinstance(expert, int) and 0 <= expert < experts for expert in retained),
        "invalid retained MTP expert list",
    )
    names = {
        "backbone.embeddings.weight",
        "lm_head.weight",
        f"{MTP_ATTENTION_PREFIX}.eh_proj.weight",
        f"{MTP_ATTENTION_PREFIX}.enorm.weight",
        f"{MTP_ATTENTION_PREFIX}.hnorm.weight",
        f"{MTP_ATTENTION_PREFIX}.norm.weight",
        f"{MTP_ATTENTION_PREFIX}.mixer.q_proj.weight",
        f"{MTP_ATTENTION_PREFIX}.mixer.k_proj.weight",
        f"{MTP_ATTENTION_PREFIX}.mixer.v_proj.weight",
        f"{MTP_ATTENTION_PREFIX}.mixer.o_proj.weight",
        f"{MTP_MOE_PREFIX}.norm.weight",
        f"{MTP_MOE_PREFIX}.final_layernorm.weight",
        f"{MTP_MOE_PREFIX}.mixer.gate.weight",
        f"{MTP_MOE_PREFIX}.mixer.gate.e_score_correction_bias",
        f"{MTP_MOE_PREFIX}.mixer.fc1_latent_proj.weight",
        f"{MTP_MOE_PREFIX}.mixer.fc2_latent_proj.weight",
        f"{MTP_MOE_PREFIX}.mixer.shared_experts.up_proj.weight",
        f"{MTP_MOE_PREFIX}.mixer.shared_experts.down_proj.weight",
    }
    for expert in retained:
        names.add(f"{MTP_MOE_PREFIX}.mixer.experts.{expert}.up_proj.weight")
        names.add(f"{MTP_MOE_PREFIX}.mixer.experts.{expert}.down_proj.weight")
    return names


def require_bf16(tensors: dict[str, mx.array], names: set[str]) -> None:
    for name in sorted(names):
        require(name in tensors, f"missing MTP tensor: {name}")
        require(tensors[name].dtype == mx.bfloat16, f"MTP tensor is not BF16: {name}")


class BF16LatentMoE:
    """Reference LatentMoE that pages only GPU-selected BF16 experts into work."""

    def __init__(
        self,
        args: ModelArgs,
        tensors: dict[str, mx.array],
        retained_experts: list[int] | None = None,
    ):
        prefix = f"{MTP_MOE_PREFIX}.mixer"
        fixed = {
            f"{MTP_MOE_PREFIX}.norm.weight",
            f"{prefix}.gate.weight",
            f"{prefix}.fc1_latent_proj.weight",
            f"{prefix}.fc2_latent_proj.weight",
            f"{prefix}.shared_experts.up_proj.weight",
            f"{prefix}.shared_experts.down_proj.weight",
        }
        require_bf16(tensors, fixed)
        correction_name = f"{prefix}.gate.e_score_correction_bias"
        require(
            correction_name in tensors and tensors[correction_name].dtype == mx.float32,
            "MTP router correction bias must be F32",
        )
        self.norm_weight = tensors[f"{MTP_MOE_PREFIX}.norm.weight"]
        self.epsilon = args.layer_norm_epsilon
        experts = args.n_routed_experts
        self.expert_ids = list(range(experts)) if retained_experts is None else retained_experts
        require(len(self.expert_ids) >= args.num_experts_per_tok, "MTP expert budget is below top-k")
        require(args.n_group == 1, "pruned MTP reference requires a single routing group")
        self.expert_id_array = mx.array(self.expert_ids, dtype=mx.int32)
        self.gate_weight = tensors[f"{prefix}.gate.weight"][self.expert_id_array]
        self.correction_bias = tensors[correction_name][self.expert_id_array]
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.fc1_latent = ModelOptBF16Linear(tensors[f"{prefix}.fc1_latent_proj.weight"])
        self.fc2_latent = ModelOptBF16Linear(tensors[f"{prefix}.fc2_latent_proj.weight"])
        self.shared_up = ModelOptBF16Linear(tensors[f"{prefix}.shared_experts.up_proj.weight"])
        self.shared_down = ModelOptBF16Linear(tensors[f"{prefix}.shared_experts.down_proj.weight"])
        self.expert_up = []
        self.expert_down = []
        for expert in self.expert_ids:
            up = f"{prefix}.experts.{expert}.up_proj.weight"
            down = f"{prefix}.experts.{expert}.down_proj.weight"
            require_bf16(tensors, {up, down})
            self.expert_up.append(ModelOptBF16Linear(tensors[up]))
            self.expert_down.append(ModelOptBF16Linear(tensors[down]))

    def route(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        return group_expert_select(
            hidden @ self.gate_weight.T,
            self.correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        hidden = mx.fast.rms_norm(x, self.norm_weight, self.epsilon)
        indices, scores = self.route(hidden)
        latent = self.fc1_latent(hidden)
        mx.eval(indices, scores, latent)
        selected = [int(index) for index in indices.reshape(-1).tolist()]
        require(len(selected) == self.top_k, "BF16 MTP reference currently requires one token")

        expert_outputs = []
        for expert in selected:
            activated = mx.square(mx.maximum(self.expert_up[expert](latent), 0.0))
            expert_outputs.append(self.expert_down[expert](activated))
        routed_latent = (
            mx.stack(expert_outputs, axis=-2) * scores[..., None]
        ).sum(axis=-2)
        routed = self.fc2_latent(routed_latent)
        shared = self.shared_down(mx.square(mx.maximum(self.shared_up(hidden), 0.0)))
        return x + routed + shared, self.expert_id_array[indices], scores


class NemotronMTPReference(_NemotronMTPAttentionState):
    """Official one-depth MTP head with optional caller-owned attention state."""

    def __init__(self, source_dir: Path, retained_experts: list[int] | None = None):
        self.source_dir = source_dir
        self.config = load_json(source_dir / "config.json")
        require(self.config.get("num_nextn_predict_layers") == 1, "expected one MTP depth")
        require(self.config.get("mtp_hybrid_override_pattern") == "*E", "expected MTP *E pattern")
        args = ModelArgs.from_dict(self.config)
        tensors = load_indexed_tensors(
            source_dir,
            mtp_tensor_names(self.config, retained_experts),
        )

        top = {
            "backbone.embeddings.weight",
            "lm_head.weight",
            f"{MTP_ATTENTION_PREFIX}.eh_proj.weight",
            f"{MTP_ATTENTION_PREFIX}.enorm.weight",
            f"{MTP_ATTENTION_PREFIX}.hnorm.weight",
            f"{MTP_ATTENTION_PREFIX}.norm.weight",
            f"{MTP_ATTENTION_PREFIX}.mixer.q_proj.weight",
            f"{MTP_ATTENTION_PREFIX}.mixer.k_proj.weight",
            f"{MTP_ATTENTION_PREFIX}.mixer.v_proj.weight",
            f"{MTP_ATTENTION_PREFIX}.mixer.o_proj.weight",
            f"{MTP_MOE_PREFIX}.final_layernorm.weight",
        }
        require_bf16(tensors, top)
        self.embeddings = tensors["backbone.embeddings.weight"]
        self.enorm_weight = tensors[f"{MTP_ATTENTION_PREFIX}.enorm.weight"]
        self.hnorm_weight = tensors[f"{MTP_ATTENTION_PREFIX}.hnorm.weight"]
        self.epsilon = args.layer_norm_epsilon
        self.eh_proj = ModelOptBF16Linear(tensors[f"{MTP_ATTENTION_PREFIX}.eh_proj.weight"])

        self.attention = NemotronHBlock(args, "*")
        self.attention.norm.weight = tensors[f"{MTP_ATTENTION_PREFIX}.norm.weight"]
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                self.attention.mixer,
                projection,
                ModelOptBF16Linear(tensors[f"{MTP_ATTENTION_PREFIX}.mixer.{projection}.weight"]),
            )
        self.attention.eval()
        self.moe = BF16LatentMoE(args, tensors, retained_experts)
        self.retained_experts = self.moe.expert_ids
        self.final_norm_weight = tensors[f"{MTP_MOE_PREFIX}.final_layernorm.weight"]
        self.lm_head = ModelOptBF16Linear(tensors["lm_head.weight"])

    def reset(self) -> None:
        """Compatibility no-op; sequence state is owned by ``NemotronMTPCache``."""

    def __call__(
        self,
        target_hidden: mx.array,
        accepted_token_id: int,
        *,
        project_logits: bool = True,
    ) -> tuple[mx.array | None, mx.array, mx.array]:
        logits, _, indices, scores = self.draft_step(
            target_hidden,
            accepted_token_id,
            project_logits=project_logits,
        )
        return logits, indices, scores

    def draft_step(
        self,
        target_hidden: mx.array,
        accepted_token_id: int,
        *,
        project_logits: bool = True,
        cache: NemotronMTPCache | None = None,
    ) -> tuple[mx.array | None, mx.array, mx.array, mx.array]:
        """Return the official draft hidden state and full-expert route."""

        fused = self._attention_step(target_hidden, accepted_token_id, cache)
        fused, indices, scores = self.moe(fused)
        fused = mx.fast.rms_norm(fused, self.final_norm_weight, self.epsilon)
        logits = self.lm_head(fused).reshape(-1) if project_logits else None
        values = [fused, indices, scores]
        if logits is not None:
            values.append(logits)
        mx.eval(*values)
        return logits, fused.reshape(-1), indices.reshape(-1), scores.reshape(-1)


class BF16LatentMoEGPU:
    """Stacked BF16 LatentMoE with GPU-owned expert selection."""

    def __init__(self, args: ModelArgs, tensors: dict[str, mx.array]):
        prefix = f"{MTP_MOE_PREFIX}.mixer"
        self.norm_weight = tensors[f"{MTP_MOE_PREFIX}.norm.weight"]
        self.epsilon = args.layer_norm_epsilon
        self.gate_weight = tensors[f"{prefix}.gate.weight"]
        self.correction_bias = tensors[f"{prefix}.gate.e_score_correction_bias"]
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.fc1_latent = ModelOptBF16Linear(tensors[f"{prefix}.fc1_latent_proj.weight"])
        self.fc2_latent = ModelOptBF16Linear(tensors[f"{prefix}.fc2_latent_proj.weight"])
        self.shared_up = ModelOptBF16Linear(tensors[f"{prefix}.shared_experts.up_proj.weight"])
        self.shared_down = ModelOptBF16Linear(tensors[f"{prefix}.shared_experts.down_proj.weight"])
        self.up_weight = tensors[f"{prefix}.switch_mlp.up_proj.weight"]
        self.down_weight = tensors[f"{prefix}.switch_mlp.down_proj.weight"]
        retained = self.gate_weight.shape[0]
        require(
            self.gate_weight.dtype == mx.bfloat16
            and self.correction_bias.dtype == mx.float32
            and self.up_weight.dtype == mx.bfloat16
            and self.down_weight.dtype == mx.bfloat16,
            "invalid MTP sidecar MoE dtypes",
        )
        require(
            self.correction_bias.shape == (retained,)
            and self.up_weight.shape[0] == retained
            and self.down_weight.shape[0] == retained
            and retained >= self.top_k,
            "invalid MTP sidecar expert shapes",
        )

    def route(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        return group_expert_select(
            hidden @ self.gate_weight.T,
            self.correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        hidden = mx.fast.rms_norm(x, self.norm_weight, self.epsilon)
        indices, scores = self.route(hidden)
        latent = self.fc1_latent(hidden)
        expert_hidden = bf16_switch_matmul(self.up_weight, latent, indices)
        expert_hidden = mx.square(mx.maximum(expert_hidden, 0.0))
        expert_output = bf16_switch_matmul(
            self.down_weight,
            expert_hidden,
            indices,
        )
        routed = self.fc2_latent((expert_output * scores[..., None]).sum(axis=-2))
        shared = self.shared_down(mx.square(mx.maximum(self.shared_up(hidden), 0.0)))
        return x + routed + shared, indices, scores


class MLXQuantizedLinear:
    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array | None,
        group_size: int,
        bits: int,
        mode: str,
    ):
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


class QuantizedMTPHead(MLXQuantizedLinear):
    def __init__(self, head_dir: Path, expected_revision: str):
        report_path = head_dir / "nemotron_mtp_head_report.json"
        report = load_json(report_path)
        require(
            report.get("format") == "nemotron-mlx-mtp-head-v1"
            and report.get("status") == "complete",
            "quantized MTP head is incomplete",
        )
        require(report.get("source_revision") == expected_revision, "MTP head revision mismatch")
        artifact = report.get("artifact")
        require(
            isinstance(artifact, str) and artifact and Path(artifact).name == artifact,
            "quantized MTP head has an invalid artifact",
        )
        artifact_path = head_dir / artifact
        artifact_hash = report.get("artifact_sha256")
        require(
            isinstance(artifact_hash, str) and sha256_file(artifact_path) == artifact_hash,
            "quantized MTP head artifact hash mismatch",
        )
        tensors, metadata = mx.load(str(artifact_path), return_metadata=True)
        require(
            metadata.get("format") == "nemotron-mlx-mtp-head-v1",
            "invalid MTP head metadata",
        )
        require(
            set(tensors) in ({"weight", "scales"}, {"weight", "scales", "biases"}),
            "invalid MTP head tensors",
        )
        settings = report.get("quantization")
        require(
            isinstance(settings, dict)
            and settings.get("group_size") in (16, 32)
            and settings.get("bits") in (4, 8)
            and settings.get("mode") in ("nvfp4", "mxfp8"),
            "quantized MTP head has invalid settings",
        )
        require(
            (settings["mode"], settings["group_size"], settings["bits"])
            in (("nvfp4", 16, 4), ("mxfp8", 32, 8)),
            "quantized MTP head settings are inconsistent",
        )
        payload_bytes = report.get("payload_bytes")
        require(
            isinstance(payload_bytes, int)
            and payload_bytes > 0
            and sum(value.nbytes for value in tensors.values()) == payload_bytes,
            "quantized MTP head payload size mismatch",
        )
        super().__init__(
            tensors["weight"],
            tensors["scales"],
            tensors.get("biases"),
            settings["group_size"],
            settings["bits"],
            settings["mode"],
        )
        source_shape = report.get("source_shape")
        require(
            isinstance(source_shape, list)
            and len(source_shape) == 2
            and all(isinstance(value, int) and value > 0 for value in source_shape),
            "invalid MTP head source shape",
        )
        require(report.get("source_dtype") == "mlx.core.bfloat16", "invalid MTP head source dtype")
        self.shape = tuple(source_shape)


class ReducedVocabMTPHead:
    def __init__(
        self,
        head_dir: Path,
        expected_revision: str,
        shared_lm_head: ModelOptBF16Linear | None,
    ):
        report = load_json(head_dir / "nemotron_mtp_vocab_head_report.json")
        require(
            report.get("format") == "nemotron-mlx-mtp-vocab-head-v1"
            and report.get("status") == "complete",
            "reduced-vocabulary MTP head is incomplete",
        )
        require(report.get("source_revision") == expected_revision, "MTP head revision mismatch")
        artifact = report.get("artifact")
        require(
            isinstance(artifact, str) and artifact and Path(artifact).name == artifact,
            "reduced-vocabulary MTP head has an invalid artifact",
        )
        artifact_path = head_dir / artifact
        artifact_hash = report.get("artifact_sha256")
        require(
            isinstance(artifact_hash, str) and sha256_file(artifact_path) == artifact_hash,
            "reduced-vocabulary MTP head artifact hash mismatch",
        )
        tensors, metadata = mx.load(str(artifact_path), return_metadata=True)
        require(
            metadata.get("format") == "nemotron-mlx-mtp-vocab-head-v1",
            "invalid reduced-vocabulary MTP head metadata",
        )
        storage = report.get("storage", "copied-bf16")
        require(storage in ("copied-bf16", "shared-target-bf16"), "invalid reduced MTP storage")
        require(metadata.get("storage", "copied-bf16") == storage, "reduced MTP storage mismatch")
        expected_tensors = (
            {"weight", "target_token_ids"}
            if storage == "copied-bf16"
            else {"target_token_ids"}
        )
        require(set(tensors) == expected_tensors, "invalid reduced MTP head tensors")
        source_shape = report.get("source_shape")
        require(
            isinstance(source_shape, list)
            and len(source_shape) == 2
            and all(isinstance(value, int) and value > 0 for value in source_shape),
            "invalid reduced MTP head source shape",
        )
        require(report.get("source_dtype") == "mlx.core.bfloat16", "invalid MTP head source dtype")
        budget = report.get("budget")
        require(
            isinstance(budget, int)
            and 0 < budget < source_shape[0]
            and (
                storage != "copied-bf16"
                or (
                    tensors["weight"].dtype == mx.bfloat16
                    and tensors["weight"].shape == (budget, source_shape[1])
                )
            ),
            "reduced-vocabulary MTP head budget or weight mismatch",
        )
        require(
            tensors["target_token_ids"].dtype == mx.int32
            and tensors["target_token_ids"].shape == (budget,),
            "reduced-vocabulary MTP token map mismatch",
        )
        token_ids = tensors["target_token_ids"].tolist()
        require(
            token_ids == sorted(set(token_ids))
            and token_ids[0] >= 0
            and token_ids[-1] < source_shape[0],
            "reduced-vocabulary MTP token IDs are invalid",
        )
        payload_bytes = report.get("payload_bytes")
        require(
            isinstance(payload_bytes, int)
            and payload_bytes > 0
            and sum(value.nbytes for value in tensors.values()) == payload_bytes,
            "reduced-vocabulary MTP head payload size mismatch",
        )
        if storage == "shared-target-bf16":
            require(
                isinstance(shared_lm_head, ModelOptBF16Linear)
                and shared_lm_head.weight.dtype == mx.bfloat16
                and shared_lm_head.weight.shape == tuple(source_shape),
                "shared reduced-vocabulary MTP head requires the exact target head shape",
            )
        self.storage = storage
        self.linear = (
            ModelOptBF16Linear(tensors["weight"])
            if storage == "copied-bf16"
            else shared_lm_head
        )
        self.target_token_ids = tensors["target_token_ids"]
        self.shape = tuple(source_shape)

    def __call__(self, x: mx.array) -> mx.array:
        if self.storage == "copied-bf16":
            return self.linear(x)
        require(math.prod(x.shape[:-1]) == 1, "shared reduced MTP head requires one token")
        output = bf16_gather_matvec(
            self.linear.weight,
            self.target_token_ids,
            x.reshape(-1).astype(mx.float32),
        )
        return output.reshape(*x.shape[:-1], self.target_token_ids.size)


def alternate_mtp_head_uses_shared_target(head_dir: Path) -> bool:
    report_path = head_dir / "nemotron_mtp_vocab_head_report.json"
    return report_path.exists() and load_json(report_path).get("storage") == "shared-target-bf16"


def load_alternate_mtp_head(
    head_dir: Path,
    expected_revision: str,
    shared_lm_head: ModelOptBF16Linear | None,
):
    quantized_report = head_dir / "nemotron_mtp_head_report.json"
    vocabulary_report = head_dir / "nemotron_mtp_vocab_head_report.json"
    require(
        quantized_report.exists() != vocabulary_report.exists(),
        "alternate MTP head must contain exactly one supported report",
    )
    if vocabulary_report.exists():
        return ReducedVocabMTPHead(head_dir, expected_revision, shared_lm_head)
    return QuantizedMTPHead(head_dir, expected_revision)


def load_sidecar_linear(
    tensors: dict[str, mx.array],
    prefix: str,
    quantization: dict | None,
):
    weight = tensors[f"{prefix}.weight"]
    if weight.dtype == mx.bfloat16:
        require(f"{prefix}.scales" not in tensors, f"BF16 linear has quantization scales: {prefix}")
        require(f"{prefix}.biases" not in tensors, f"BF16 linear has quantization biases: {prefix}")
        return ModelOptBF16Linear(weight)
    require(quantization is not None, f"quantized linear has no settings: {prefix}")
    settings = sidecar_quantization_settings(quantization, f"{prefix}.weight")
    biases = tensors.get(f"{prefix}.biases")
    require(
        (settings["mode"] == "affine") == (biases is not None),
        f"quantized linear bias payload does not match mode: {prefix}",
    )
    return MLXQuantizedLinear(
        weight,
        tensors[f"{prefix}.scales"],
        biases,
        settings["group_size"],
        settings["bits"],
        settings["mode"],
    )


def validate_sidecar_quantization_settings(settings: dict, context: str) -> None:
    require(isinstance(settings, dict), f"MTP quantization settings are invalid: {context}")
    group_size = settings.get("group_size")
    bits = settings.get("bits")
    mode = settings.get("mode")
    valid = (
        mode == "affine"
        and group_size in (32, 64, 128)
        and bits in (2, 3, 4, 5, 6, 8)
    ) or (
        mode == "affine"
        and group_size == 128
        and bits == 1
    ) or (mode, group_size, bits) in {
        ("mxfp4", 32, 4),
        ("nvfp4", 16, 4),
    }
    require(valid, f"unsupported MTP quantization settings: {context}")
    require(
        bits != 1 or binary_affine_supported(),
        f"one-bit MTP payload requires an MLX build with affine bits=1 support: {context}",
    )
    recipe = settings.get("recipe")
    require(recipe is None or isinstance(recipe, str), f"invalid MTP recipe: {context}")


def sidecar_quantization_settings(quantization: dict, weight_name: str) -> dict:
    validate_sidecar_quantization_settings(quantization, "default")
    overrides = quantization.get("tensor_modes", {})
    require(isinstance(overrides, dict), "MTP tensor-mode map is invalid")
    settings = overrides.get(weight_name, quantization)
    validate_sidecar_quantization_settings(settings, weight_name)
    return settings


def validate_expert_bank_metadata(metadata: dict) -> list[dict]:
    require(
        isinstance(metadata, dict)
        and metadata.get("format") == "nemotron-mtp-lowbit-banks-v1"
        and isinstance(metadata.get("banks"), list)
        and len(metadata["banks"]) >= 2,
        "MTP expert-bank metadata is invalid",
    )
    return metadata["banks"]


class QuantizedLatentMoEGPU:
    def __init__(
        self,
        args: ModelArgs,
        tensors: dict[str, mx.array],
        quantization: dict,
        use_mixed_metal: bool | None = None,
    ):
        prefix = f"{MTP_MOE_PREFIX}.mixer"
        self.norm_weight = tensors[f"{MTP_MOE_PREFIX}.norm.weight"]
        self.epsilon = args.layer_norm_epsilon
        self.gate_weight = tensors[f"{prefix}.gate.weight"]
        self.correction_bias = tensors[f"{prefix}.gate.e_score_correction_bias"]
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.fc1_latent = load_sidecar_linear(tensors, f"{prefix}.fc1_latent_proj", quantization)
        self.fc2_latent = load_sidecar_linear(tensors, f"{prefix}.fc2_latent_proj", quantization)
        self.shared_up = load_sidecar_linear(tensors, f"{prefix}.shared_experts.up_proj", quantization)
        self.shared_down = load_sidecar_linear(tensors, f"{prefix}.shared_experts.down_proj", quantization)
        self.up_weight = tensors[f"{prefix}.switch_mlp.up_proj.weight"]
        self.up_scales = tensors[f"{prefix}.switch_mlp.up_proj.scales"]
        self.up_biases = tensors.get(f"{prefix}.switch_mlp.up_proj.biases")
        self.down_weight = tensors[f"{prefix}.switch_mlp.down_proj.weight"]
        self.down_scales = tensors[f"{prefix}.switch_mlp.down_proj.scales"]
        self.down_biases = tensors.get(f"{prefix}.switch_mlp.down_proj.biases")
        self.projection_settings = {
            projection: sidecar_quantization_settings(
                quantization,
                f"{prefix}.switch_mlp.{projection}_proj.weight",
            )
            for projection in ("up", "down")
        }
        for projection in ("up", "down"):
            settings = self.projection_settings[projection]
            biases = getattr(self, f"{projection}_biases")
            require(
                (settings["mode"] == "affine") == (biases is not None),
                f"MTP expert {projection} bias payload does not match mode",
            )
        self.banks = None
        self.mixed_metal_banks = None
        self.use_mixed_metal = (
            os.environ.get("NEMOTRON_MTP_MIXED_METAL", "1") != "0"
            if use_mixed_metal is None
            else use_mixed_metal
        )
        bank_metadata = quantization.get("expert_banks")
        if bank_metadata is not None:
            self.banks = []
            covered = set()
            for metadata in validate_expert_bank_metadata(bank_metadata):
                require(isinstance(metadata, dict), "MTP expert bank is invalid")
                expert_ids = metadata.get("sidecar_expert_indices")
                tensor_prefix = metadata.get("tensor_prefix")
                require(
                    isinstance(expert_ids, list)
                    and expert_ids
                    and all(isinstance(expert, int) for expert in expert_ids)
                    and expert_ids == sorted(set(expert_ids))
                    and all(0 <= expert < args.n_routed_experts for expert in expert_ids),
                    "MTP expert-bank sidecar indices are invalid",
                )
                require(not (covered & set(expert_ids)), "MTP expert banks overlap")
                covered.update(expert_ids)
                require(
                    isinstance(tensor_prefix, str) and tensor_prefix,
                    "MTP expert-bank tensor prefix is invalid",
                )
                original_to_local = [-1] * args.n_routed_experts
                for local, original in enumerate(expert_ids):
                    original_to_local[original] = local
                bank = {
                    "original_to_local": mx.array(original_to_local, dtype=mx.int32),
                    "expert_ids": expert_ids,
                }
                for projection in ("up", "down"):
                    projection_prefix = f"{tensor_prefix}.{projection}_proj"
                    weight_name = f"{projection_prefix}.weight"
                    settings = sidecar_quantization_settings(quantization, weight_name)
                    weight = tensors[weight_name]
                    scales = tensors[f"{projection_prefix}.scales"]
                    biases = tensors.get(f"{projection_prefix}.biases")
                    require(
                        weight.shape[0] == len(expert_ids)
                        and scales.shape[0] == len(expert_ids),
                        f"MTP expert bank {projection} count mismatch",
                    )
                    require(
                        (settings["mode"] == "affine") == (biases is not None),
                        f"MTP expert bank {projection} bias payload does not match mode",
                    )
                    bank[projection] = {
                        "weight": weight,
                        "scales": scales,
                        "biases": biases,
                        "settings": settings,
                    }
                self.banks.append(bank)
            require(
                covered == set(range(args.n_routed_experts)),
                "MTP expert banks do not partition every sidecar expert",
            )
            if len(self.banks) == 2:
                by_bits = {
                    bank["up"]["settings"]["bits"]: bank for bank in self.banks
                }
                if set(by_bits) == {1, 3}:
                    self.mixed_metal_banks = (by_bits[1], by_bits[3])
        self.overlay = None
        overlay = quantization.get("expert_overlay")
        if overlay is not None:
            require(isinstance(overlay, dict), "MTP expert overlay metadata is invalid")
            expert_ids = overlay.get("sidecar_expert_indices")
            overlay_prefix = overlay.get("tensor_prefix")
            require(
                isinstance(expert_ids, list)
                and expert_ids
                and all(isinstance(expert, int) for expert in expert_ids)
                and expert_ids == sorted(set(expert_ids))
                and all(0 <= expert < args.n_routed_experts for expert in expert_ids),
                "MTP expert overlay sidecar indices are invalid",
            )
            require(
                isinstance(overlay_prefix, str) and overlay_prefix,
                "MTP expert overlay tensor prefix is invalid",
            )
            original_to_overlay = [-1] * args.n_routed_experts
            for local, original in enumerate(expert_ids):
                original_to_overlay[original] = local
            self.overlay = {
                "original_to_local": mx.array(original_to_overlay, dtype=mx.int32),
                "expert_ids": expert_ids,
            }
            for projection in ("up", "down"):
                tensor_prefix = f"{overlay_prefix}.{projection}_proj"
                weight_name = f"{tensor_prefix}.weight"
                settings = sidecar_quantization_settings(quantization, weight_name)
                weight = tensors[weight_name]
                scales = tensors[f"{tensor_prefix}.scales"]
                biases = tensors.get(f"{tensor_prefix}.biases")
                require(
                    weight.shape[0] == len(expert_ids)
                    and scales.shape[0] == len(expert_ids),
                    f"MTP expert overlay {projection} count mismatch",
                )
                require(
                    (settings["mode"] == "affine") == (biases is not None),
                    f"MTP expert overlay {projection} bias payload does not match mode",
                )
                self.overlay[projection] = {
                    "weight": weight,
                    "scales": scales,
                    "biases": biases,
                    "settings": settings,
                }

    def route(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        return group_expert_select(
            hidden @ self.gate_weight.T,
            self.correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    def switch(self, x: mx.array, indices: mx.array, projection: str) -> mx.array:
        if self.banks is not None:
            if (
                self.mixed_metal_banks is not None
                and self.use_mixed_metal
            ):
                return mixed_affine_switch(
                    x,
                    indices,
                    self.mixed_metal_banks[0],
                    self.mixed_metal_banks[1],
                    projection,
                )
            output = None
            for bank in self.banks:
                mapped = bank["original_to_local"][indices]
                selected = mapped >= 0
                safe_indices = mx.maximum(mapped, 0)
                payload = bank[projection]
                bank_output = mx.gather_qmm(
                    x,
                    payload["weight"],
                    payload["scales"],
                    payload["biases"],
                    rhs_indices=safe_indices,
                    transpose=True,
                    group_size=payload["settings"]["group_size"],
                    bits=payload["settings"]["bits"],
                    mode=payload["settings"]["mode"],
                )
                mask = selected.reshape(
                    *selected.shape,
                    *([1] * (bank_output.ndim - selected.ndim)),
                )
                output = (
                    mx.where(mask, bank_output, mx.zeros_like(bank_output))
                    if output is None
                    else mx.where(mask, bank_output, output)
                )
            require(output is not None, "MTP expert-bank dispatch produced no output")
            return output
        settings = self.projection_settings[projection]
        output = mx.gather_qmm(
            x,
            getattr(self, f"{projection}_weight"),
            getattr(self, f"{projection}_scales"),
            getattr(self, f"{projection}_biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=settings["group_size"],
            bits=settings["bits"],
            mode=settings["mode"],
        )
        if self.overlay is None:
            return output
        mapped = self.overlay["original_to_local"][indices]
        selected = mapped >= 0
        safe_indices = mx.maximum(mapped, 0)
        bank = self.overlay[projection]
        overlay_output = mx.gather_qmm(
            x,
            bank["weight"],
            bank["scales"],
            bank["biases"],
            rhs_indices=safe_indices,
            transpose=True,
            group_size=bank["settings"]["group_size"],
            bits=bank["settings"]["bits"],
            mode=bank["settings"]["mode"],
        )
        mask = selected.reshape(*selected.shape, *([1] * (output.ndim - selected.ndim)))
        return mx.where(mask, overlay_output, output)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        hidden = mx.fast.rms_norm(x, self.norm_weight, self.epsilon)
        indices, scores = self.route(hidden)
        latent = self.fc1_latent(hidden)
        expert_input = mx.expand_dims(latent, (-2, -3))
        expert_hidden = mx.square(mx.maximum(self.switch(expert_input, indices, "up"), 0.0))
        expert_output = self.switch(expert_hidden, indices, "down").squeeze(-2)
        routed = self.fc2_latent((expert_output * scores[..., None]).sum(axis=-2))
        shared = self.shared_down(mx.square(mx.maximum(self.shared_up(hidden), 0.0)))
        return x + routed + shared, indices, scores


class NemotronMTPSidecar(_NemotronMTPAttentionState):
    """Resident MTP head backed by an exact packed sidecar."""

    def __init__(
        self,
        sidecar_dir: Path,
        embeddings: mx.array,
        lm_head: ModelOptBF16Linear | None,
        alternate_lm_head: Path | None = None,
        use_mixed_metal: bool | None = None,
    ):
        self.sidecar_dir = sidecar_dir
        self.config = load_json(sidecar_dir / "config.json")
        runtime = self.config.get("nemotron_mtp_runtime", {})
        require(runtime.get("format") == "nemotron-mlx-mtp-sidecar-v1", "invalid MTP sidecar")
        require(isinstance(runtime.get("source_revision"), str), "MTP sidecar has no source revision")
        args = ModelArgs.from_dict(self.config)
        index = load_json(sidecar_dir / "model.safetensors.index.json")
        shard_names = set(index.get("weight_map", {}).values())
        require(len(shard_names) == 1, "MTP sidecar must occupy one shard")
        tensors = mx.load(str(sidecar_dir / next(iter(shard_names))))
        require(set(index["weight_map"]) == set(tensors), "MTP sidecar index/tensor mismatch")

        require(
            embeddings.dtype == mx.bfloat16
            and embeddings.shape == (args.vocab_size, args.hidden_size),
            "MTP shared embedding mismatch",
        )
        self.embeddings = embeddings
        require(lm_head is not None or alternate_lm_head is not None, "MTP sidecar has no vocabulary head")
        self.lm_head = (
            load_alternate_mtp_head(
                alternate_lm_head,
                runtime["source_revision"],
                lm_head,
            )
            if alternate_lm_head is not None
            else lm_head
        )
        if isinstance(self.lm_head, (QuantizedMTPHead, ReducedVocabMTPHead)):
            require(
                self.lm_head.shape == (args.vocab_size, args.hidden_size),
                "alternate MTP head shape mismatch",
            )
        self.draft_token_ids = (
            self.lm_head.target_token_ids
            if isinstance(self.lm_head, ReducedVocabMTPHead)
            else None
        )
        self.enorm_weight = tensors[f"{MTP_ATTENTION_PREFIX}.enorm.weight"]
        self.hnorm_weight = tensors[f"{MTP_ATTENTION_PREFIX}.hnorm.weight"]
        self.epsilon = args.layer_norm_epsilon
        quantization = runtime.get("quantization")
        if quantization is not None:
            require(
                quantization.get("format") == "nemotron-mtp-sidecar-quant-v1",
                "unsupported MTP sidecar quantization format",
            )
            validate_sidecar_quantization_settings(quantization, "default")
            declared_bf16 = quantization.get("bf16_tensors", [])
            require(
                isinstance(declared_bf16, list)
                and all(isinstance(name, str) for name in declared_bf16)
                and declared_bf16 == sorted(set(declared_bf16)),
                "MTP mixed-precision tensor list is invalid",
            )
            actual_bf16 = sorted(
                name
                for name, value in tensors.items()
                if name.endswith(".weight")
                and value.ndim >= 2
                and value.dtype == mx.bfloat16
                and not name.endswith(".gate.weight")
            )
            require(
                actual_bf16 == declared_bf16,
                "MTP mixed-precision tensor payload does not match its config",
            )
            tensor_modes = quantization.get("tensor_modes", {})
            require(
                isinstance(tensor_modes, dict)
                and all(isinstance(name, str) for name in tensor_modes),
                "MTP tensor-mode map is invalid",
            )
            quantized_weights = {
                name
                for name, value in tensors.items()
                if name.endswith(".weight")
                and value.ndim >= 2
                and value.dtype != mx.bfloat16
            }
            for name, settings in tensor_modes.items():
                require(name in quantized_weights, f"MTP tensor-mode payload is missing: {name}")
                validate_sidecar_quantization_settings(settings, name)
        self.eh_proj = load_sidecar_linear(
            tensors,
            f"{MTP_ATTENTION_PREFIX}.eh_proj",
            quantization,
        )
        self.attention = NemotronHBlock(args, "*")
        self.attention.norm.weight = tensors[f"{MTP_ATTENTION_PREFIX}.norm.weight"]
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                self.attention.mixer,
                projection,
                load_sidecar_linear(
                    tensors,
                    f"{MTP_ATTENTION_PREFIX}.mixer.{projection}",
                    quantization,
                ),
            )
        self.attention.eval()
        self.moe = (
            QuantizedLatentMoEGPU(
                args,
                tensors,
                quantization,
                use_mixed_metal=use_mixed_metal,
            )
            if quantization is not None
            else BF16LatentMoEGPU(args, tensors)
        )
        self.final_norm_weight = tensors[f"{MTP_MOE_PREFIX}.final_layernorm.weight"]
        self.retained_experts = runtime["original_expert_ids"]
        self.original_expert_ids = mx.array(self.retained_experts, dtype=mx.int32)

    def token_ids(self, draft_indices: mx.array) -> mx.array:
        return (
            draft_indices
            if self.draft_token_ids is None
            else self.draft_token_ids[draft_indices]
        )

    def argmax_token(self, logits: mx.array) -> int:
        index = mx.argmax(logits)
        return int(self.token_ids(index))

    def top_token_ids(self, logits: mx.array, count: int) -> list[int]:
        require(0 < count <= logits.size, "invalid MTP top-token count")
        indices = mx.argpartition(-logits, kth=count - 1)[:count]
        return self.token_ids(indices).tolist()

    def __call__(
        self, target_hidden: mx.array, accepted_token_id: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        logits, _, indices, scores = self.draft_step(target_hidden, accepted_token_id)
        return logits, indices, scores

    def draft_step(
        self,
        target_hidden: mx.array,
        accepted_token_id: int,
        *,
        project_logits: bool = True,
        cache: NemotronMTPCache | None = None,
    ) -> tuple[mx.array | None, mx.array, mx.array, mx.array]:
        fused = self._attention_step(target_hidden, accepted_token_id, cache)
        fused, indices, scores = self.moe(fused)
        fused = mx.fast.rms_norm(fused, self.final_norm_weight, self.epsilon)
        logits = self.lm_head(fused).reshape(-1) if project_logits else None
        return (
            logits,
            fused.reshape(-1),
            self.original_expert_ids[indices].reshape(-1),
            scores.reshape(-1),
        )


def mtp_payload_estimate(config: dict, retained_experts: int) -> int:
    """Estimate exact BF16 MTP payload from architecture dimensions."""

    experts = config["n_routed_experts"]
    require(0 < retained_experts <= experts, "invalid retained MTP expert count")
    hidden = config["hidden_size"]
    latent = config["moe_latent_size"]
    intermediate = config["moe_intermediate_size"]
    expert_bytes = retained_experts * 2 * latent * intermediate * 2
    full_expert_bytes = experts * 2 * latent * intermediate * 2
    router_row_bytes = hidden * 2 + 4
    full_router_bytes = experts * router_row_bytes
    retained_router_bytes = retained_experts * router_row_bytes
    # The pinned checkpoint's total is authoritative; subtracting its expert
    # matrices retains every fixed tensor and the full router exactly.
    total_bytes = 5_884_651_520
    require(full_expert_bytes < total_bytes, "invalid MTP payload constants")
    return (
        total_bytes
        - full_expert_bytes
        - full_router_bytes
        + expert_bytes
        + retained_router_bytes
    )
