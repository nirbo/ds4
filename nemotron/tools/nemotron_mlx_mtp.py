#!/usr/bin/env python3
"""Exact BF16 reference for Nemotron 3 Super's official repeated MTP head."""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock, group_expert_select

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear, bf16_gather_matvec, bf16_switch_matmul
from nemotron_prune_materialize import sha256_file


MTP_ATTENTION_PREFIX = "mtp.layers.0"
MTP_MOE_PREFIX = "mtp.layers.1"


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


class NemotronMTPReference:
    """Official one-depth MTP head with persistent attention cache."""

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
        """MTP is stateless in official speculative inference."""

    def __call__(
        self,
        target_hidden: mx.array,
        accepted_token_id: int,
        *,
        project_logits: bool = True,
    ) -> tuple[mx.array | None, mx.array, mx.array]:
        require(0 <= accepted_token_id < self.embeddings.shape[0], "MTP token ID out of range")
        hidden = target_hidden.astype(mx.float32).reshape(1, 1, -1)
        require(hidden.shape[-1] == self.embeddings.shape[1], "MTP hidden size mismatch")
        embedding = self.embeddings[accepted_token_id].astype(mx.float32).reshape(1, 1, -1)
        embedding = mx.fast.rms_norm(embedding, self.enorm_weight, self.epsilon)
        hidden = mx.fast.rms_norm(hidden, self.hnorm_weight, self.epsilon)
        fused = self.eh_proj(mx.concatenate([embedding, hidden], axis=-1))
        fused = self.attention(fused, mask=None, cache=None)
        fused, indices, scores = self.moe(fused)
        fused = mx.fast.rms_norm(fused, self.final_norm_weight, self.epsilon)
        logits = self.lm_head(fused).reshape(-1) if project_logits else None
        values = [fused, indices, scores]
        if logits is not None:
            values.append(logits)
        mx.eval(*values)
        return logits, indices.reshape(-1), scores.reshape(-1)


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
    if quantization is None:
        return ModelOptBF16Linear(tensors[f"{prefix}.weight"])
    biases = tensors.get(f"{prefix}.biases")
    return MLXQuantizedLinear(
        tensors[f"{prefix}.weight"],
        tensors[f"{prefix}.scales"],
        biases,
        quantization["group_size"],
        quantization["bits"],
        quantization["mode"],
    )


class QuantizedLatentMoEGPU:
    def __init__(
        self,
        args: ModelArgs,
        tensors: dict[str, mx.array],
        quantization: dict,
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
        self.group_size = quantization["group_size"]
        self.bits = quantization["bits"]
        self.mode = quantization["mode"]

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
        return mx.gather_qmm(
            x,
            getattr(self, f"{projection}_weight"),
            getattr(self, f"{projection}_scales"),
            getattr(self, f"{projection}_biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

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


class NemotronMTPSidecar:
    """Resident MTP head backed by an exact packed sidecar."""

    def __init__(
        self,
        sidecar_dir: Path,
        embeddings: mx.array,
        lm_head: ModelOptBF16Linear | None,
        alternate_lm_head: Path | None = None,
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
            QuantizedLatentMoEGPU(args, tensors, quantization)
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
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        hidden = target_hidden.astype(mx.float32).reshape(1, 1, -1)
        embedding = self.embeddings[accepted_token_id].astype(mx.float32).reshape(1, 1, -1)
        embedding = mx.fast.rms_norm(embedding, self.enorm_weight, self.epsilon)
        hidden = mx.fast.rms_norm(hidden, self.hnorm_weight, self.epsilon)
        fused = self.eh_proj(mx.concatenate([embedding, hidden], axis=-1))
        fused = self.attention(fused, mask=None, cache=None)
        fused, indices, scores = self.moe(fused)
        fused = mx.fast.rms_norm(fused, self.final_norm_weight, self.epsilon)
        logits = self.lm_head(fused).reshape(-1)
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
