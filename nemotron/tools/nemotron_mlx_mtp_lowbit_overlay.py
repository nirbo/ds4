#!/usr/bin/env python3
"""Overlay selected fitted one-bit MTP experts with a higher-precision bank."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mtp import sidecar_quantization_settings
from nemotron_mlx_mtp_lowbit_plan import FORMAT as PLAN_FORMAT
from nemotron_mlx_mtp_quantize import (
    BINARY_MODES,
    MODES,
    QUANT_FORMAT,
    quantize_tensor,
    tensor_error,
)
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


SIDECAR_FORMAT = "nemotron-mlx-mtp-sidecar-v1"
OVERLAY_FORMAT = "nemotron-mtp-lowbit-overlay-v1"
BANK_FORMAT = "nemotron-mtp-lowbit-banks-v1"
EXPERT_PREFIX = "mtp.layers.1.mixer.switch_mlp"
OVERLAY_PREFIX = f"{EXPERT_PREFIX}.overlay"


def load_sidecar(path: Path) -> tuple[dict, dict, dict[str, mx.array], Path]:
    config = load_json(path / "config.json")
    runtime = config.get("nemotron_mtp_runtime", {})
    require(runtime.get("format") == SIDECAR_FORMAT, f"invalid MTP sidecar: {path}")
    report = load_json(path / "nemotron_mtp_pack_report.json")
    require(
        report.get("format") == SIDECAR_FORMAT and report.get("status") == "complete",
        f"incomplete MTP sidecar: {path}",
    )
    index = load_json(path / "model.safetensors.index.json")
    shard_names = set(index.get("weight_map", {}).values())
    require(len(shard_names) == 1, f"MTP sidecar must occupy one shard: {path}")
    payload_path = path / next(iter(shard_names))
    tensors = mx.load(str(payload_path))
    require(set(tensors) == set(index["weight_map"]), f"MTP sidecar index mismatch: {path}")
    return config, report, tensors, payload_path


def atomic_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(str(temporary), tensors, metadata={"format": QUANT_FORMAT})
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-sidecar", required=True, type=Path)
    parser.add_argument("--bf16-sidecar", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--high-mode", default="affine3-g128", choices=sorted(MODES))
    parser.add_argument(
        "--split-base",
        action="store_true",
        help="remove promoted experts from the one-bit bank instead of retaining an overlay copy",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.high_mode not in BINARY_MODES, "overlay bank must exceed one-bit precision")
        require(
            args.output_dir.resolve() not in {
                args.binary_sidecar.resolve(),
                args.bf16_sidecar.resolve(),
            },
            "overlay output cannot replace a source sidecar",
        )
        binary_config, binary_report, binary_tensors, binary_payload = load_sidecar(
            args.binary_sidecar
        )
        bf16_config, bf16_report, bf16_tensors, bf16_payload = load_sidecar(
            args.bf16_sidecar
        )
        revision = binary_report["source_revision"]
        require(bf16_report["source_revision"] == revision, "overlay source revisions differ")
        require(
            binary_config["nemotron_mtp_runtime"]["original_expert_ids"]
            == bf16_config["nemotron_mtp_runtime"]["original_expert_ids"],
            "overlay source expert mappings differ",
        )
        expert_count = len(binary_config["nemotron_mtp_runtime"]["original_expert_ids"])
        require(binary_report["budget"] == bf16_report["budget"] == expert_count, "overlay budgets differ")
        binary_quantization = binary_config["nemotron_mtp_runtime"].get("quantization")
        require(binary_quantization is not None, "overlay base is not quantized")
        for projection in ("up", "down"):
            settings = sidecar_quantization_settings(
                binary_quantization,
                f"{EXPERT_PREFIX}.{projection}_proj.weight",
            )
            require(settings["bits"] == 1, f"overlay base {projection} tensor is not one-bit")

        plan = load_json(args.plan)
        require(plan.get("format") == PLAN_FORMAT, "invalid MTP low-bit plan")
        require(plan.get("source_revision") == revision, "overlay plan revision differs")
        require(plan.get("expert_count") == expert_count, "overlay plan expert count differs")
        require(
            plan.get("fit_artifact_sha256") == sha256_file(binary_payload),
            "overlay plan was not ranked from this binary artifact",
        )
        expert_ids = plan.get("budgets", {}).get(str(args.budget))
        require(
            isinstance(expert_ids, list)
            and len(expert_ids) == args.budget
            and expert_ids == sorted(set(expert_ids))
            and all(isinstance(expert, int) and 0 <= expert < expert_count for expert in expert_ids),
            "overlay plan budget is invalid",
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "overlay.log")
        operation_log.write(
            f"mtp-lowbit-overlay-start budget={args.budget} mode={args.high_mode} "
            f"binary_sha256={sha256_file(binary_payload)}"
        )
        output = dict(binary_tensors)
        errors = {}
        settings = MODES[args.high_mode]
        tensor_modes = copy.deepcopy(binary_quantization.get("tensor_modes", {}))
        selected = mx.array(expert_ids, dtype=mx.int32)
        low_expert_ids = sorted(set(range(expert_count)) - set(expert_ids))
        low_selected = mx.array(low_expert_ids, dtype=mx.int32)
        if args.split_base:
            require(low_expert_ids, "split low-bit bank cannot be empty")
            for projection in ("up", "down"):
                prefix = f"{EXPERT_PREFIX}.{projection}_proj"
                for suffix in ("weight", "scales", "biases"):
                    name = f"{prefix}.{suffix}"
                    if name in binary_tensors:
                        output[name] = binary_tensors[name][low_selected]
            operation_log.write(
                f"mtp-lowbit-overlay-split low_experts={len(low_expert_ids)} "
                f"high_experts={len(expert_ids)}"
            )
        for projection in ("up", "down"):
            source_name = f"{EXPERT_PREFIX}.{projection}_proj.weight"
            require(
                bf16_tensors[source_name].dtype == mx.bfloat16,
                f"overlay BF16 source is not BF16: {source_name}",
            )
            original = bf16_tensors[source_name][selected]
            weight, scales, biases = quantize_tensor(original, args.high_mode)
            mx.eval(weight, scales, *([biases] if biases is not None else []))
            prefix = f"{OVERLAY_PREFIX}.{projection}_proj"
            output[f"{prefix}.weight"] = weight
            output[f"{prefix}.scales"] = scales
            if biases is not None:
                output[f"{prefix}.biases"] = biases
            tensor_modes[f"{prefix}.weight"] = {
                "recipe": args.high_mode,
                **settings,
            }
            errors[f"{prefix}.weight"] = {
                **tensor_error(original, weight, scales, biases, settings),
                "recipe": args.high_mode,
            }
            operation_log.write(
                f"mtp-lowbit-overlay-tensor projection={projection} experts={len(expert_ids)} "
                f"relative_l2={errors[f'{prefix}.weight']['relative_l2']:.9g}"
            )
            mx.clear_cache()

        output_path = args.output_dir / "mtp.safetensors"
        atomic_safetensors(output_path, output)
        loaded = mx.load(str(output_path))
        require(set(loaded) == set(output), "overlay tensor set mismatch")
        sliced_names = set()
        if args.split_base:
            for projection in ("up", "down"):
                prefix = f"{EXPERT_PREFIX}.{projection}_proj"
                sliced_names.update(
                    f"{prefix}.{suffix}"
                    for suffix in ("weight", "scales", "biases")
                    if f"{prefix}.{suffix}" in binary_tensors
                )
        for name, value in binary_tensors.items():
            expected = value[low_selected] if name in sliced_names else value
            require(bool(mx.array_equal(loaded[name], expected)), f"overlay changed base tensor: {name}")
        for name, value in output.items():
            require(
                loaded[name].shape == value.shape and loaded[name].dtype == value.dtype,
                f"overlay tensor metadata changed: {name}",
            )

        config = copy.deepcopy(binary_config)
        quantization = config["nemotron_mtp_runtime"]["quantization"]
        quantization["tensor_modes"] = tensor_modes
        high_bank = {
            "sidecar_expert_indices": expert_ids,
            "tensor_prefix": OVERLAY_PREFIX,
            "mode": args.high_mode,
        }
        provenance = {
            "plan_sha256": sha256_file(args.plan),
            "bf16_source_sha256": sha256_file(bf16_payload),
        }
        if args.split_base:
            quantization.pop("expert_overlay", None)
            quantization["expert_banks"] = {
                "format": BANK_FORMAT,
                "banks": [
                    {
                        "sidecar_expert_indices": low_expert_ids,
                        "tensor_prefix": EXPERT_PREFIX,
                        "mode": "fitted-binary1",
                    },
                    high_bank,
                ],
                **provenance,
            }
        else:
            quantization.pop("expert_banks", None)
            quantization["expert_overlay"] = {
                "format": OVERLAY_FORMAT,
                **high_bank,
                **provenance,
            }
        atomic_json(args.output_dir / "config.json", config)
        payload_bytes = sum(value.nbytes for value in loaded.values())
        atomic_json(
            args.output_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": payload_bytes},
                "weight_map": {name: output_path.name for name in loaded},
            },
        )
        report = copy.deepcopy(binary_report)
        report.update(
            {
                "status": "complete",
                "payload_bytes": payload_bytes,
                "payload_gib": payload_bytes / 2**30,
                "source_sidecar_sha256": sha256_file(binary_payload),
                "artifact_sha256": sha256_file(output_path),
                (
                    "lowbit_banks" if args.split_base else "lowbit_overlay"
                ): quantization[
                    "expert_banks" if args.split_base else "expert_overlay"
                ],
                "overlay_full_tensor_error": errors,
            }
        )
        atomic_json(args.output_dir / "nemotron_mtp_pack_report.json", report)
        operation_log.write(
            f"mtp-lowbit-overlay-complete payload_gib={payload_bytes / 2**30:.6f} "
            f"sha256={report['artifact_sha256']}"
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, KeyError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-lowbit-overlay-failed error={exc}")
        print(f"nemotron MTP low-bit overlay error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
