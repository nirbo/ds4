#!/usr/bin/env python3
"""Quantize only a packed Nemotron MTP sidecar for speculative inference."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-mlx-mtp-sidecar-v1"
QUANT_FORMAT = "nemotron-mtp-sidecar-quant-v1"
MODES = {
    "affine": {"group_size": 64, "bits": 4, "mode": "affine"},
    "binary1-g128": {"group_size": 128, "bits": 1, "mode": "affine"},
    "binary1-kmeans-g128": {"group_size": 128, "bits": 1, "mode": "affine"},
    "affine2-g32": {"group_size": 32, "bits": 2, "mode": "affine"},
    "affine2-g64": {"group_size": 64, "bits": 2, "mode": "affine"},
    "affine2-g128": {"group_size": 128, "bits": 2, "mode": "affine"},
    "affine3-g32": {"group_size": 32, "bits": 3, "mode": "affine"},
    "affine3-g64": {"group_size": 64, "bits": 3, "mode": "affine"},
    "affine3-g128": {"group_size": 128, "bits": 3, "mode": "affine"},
    "ternary2-g64": {"group_size": 64, "bits": 2, "mode": "affine"},
    "ternary2-g128": {"group_size": 128, "bits": 2, "mode": "affine"},
    "mxfp4": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    "nvfp4": {"group_size": 16, "bits": 4, "mode": "nvfp4"},
}
BINARY_MODES = {"binary1-g128", "binary1-kmeans-g128"}
TERNARY_MODES = {"ternary2-g64", "ternary2-g128"}


def quantizable(name: str, value: mx.array) -> bool:
    return (
        name.endswith(".weight")
        and value.dtype == mx.bfloat16
        and value.ndim >= 2
        and not name.endswith(".gate.weight")
    )


def binary_affine_supported() -> bool:
    """Return whether this MLX build can execute its one-bit affine decoder."""

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


def binary_quantize(
    value: mx.array,
    group_size: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pack least-squares symmetric {-scale, +scale} groups as one-bit affine."""

    require(value.ndim >= 2, "binary quantization requires a matrix")
    require(value.shape[-1] % group_size == 0, "binary group size does not divide tensor")
    require(group_size % 32 == 0, "binary groups must align to packed uint32 words")
    grouped = value.astype(mx.float32).reshape(
        *value.shape[:-1], value.shape[-1] // group_size, group_size
    )
    magnitude = mx.mean(mx.abs(grouped), axis=-1)
    codes = (grouped >= 0).astype(mx.uint32)
    codes = codes.reshape(*value.shape[:-1], value.shape[-1] // 32, 32)
    shifts = mx.arange(32, dtype=mx.uint32)
    packed = mx.sum(codes << shifts, axis=-1).astype(mx.uint32)
    scales = (2.0 * magnitude).astype(value.dtype)
    biases = (-magnitude).astype(value.dtype)
    return packed, scales, biases


def binary_kmeans_quantize(
    value: mx.array,
    group_size: int,
    iterations: int = 6,
    chunk_values: int = 8 * 1024 * 1024,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fit two affine reconstruction levels per group in bounded row chunks."""

    require(value.ndim >= 2, "binary quantization requires a matrix")
    require(value.shape[-1] % group_size == 0, "binary group size does not divide tensor")
    require(group_size % 32 == 0, "binary groups must align to packed uint32 words")
    rows = math.prod(value.shape[:-1])
    width = value.shape[-1]
    rows_per_chunk = max(1, chunk_values // width)
    flat = value.reshape(rows, width)
    packed_chunks = []
    scale_chunks = []
    bias_chunks = []
    shifts = mx.arange(32, dtype=mx.uint32)
    for start in range(0, rows, rows_per_chunk):
        grouped = flat[start : start + rows_per_chunk].astype(mx.float32).reshape(
            -1, width // group_size, group_size
        )
        threshold = mx.mean(grouped, axis=-1)
        low = threshold
        high = threshold
        for _ in range(iterations):
            low_mask = grouped <= threshold[..., None]
            low_count = mx.sum(low_mask, axis=-1)
            high_count = group_size - low_count
            low = mx.sum(mx.where(low_mask, grouped, 0.0), axis=-1) / mx.maximum(
                low_count, 1
            )
            high = mx.sum(mx.where(low_mask, 0.0, grouped), axis=-1) / mx.maximum(
                high_count, 1
            )
            low = mx.where(low_count > 0, low, high)
            high = mx.where(high_count > 0, high, low)
            threshold = (low + high) * 0.5
        codes = (grouped > threshold[..., None]).astype(mx.uint32)
        codes = codes.reshape(-1, width // 32, 32)
        packed = mx.sum(codes << shifts, axis=-1).astype(mx.uint32)
        scales = (high - low).astype(value.dtype)
        biases = low.astype(value.dtype)
        mx.eval(packed, scales, biases)
        packed_chunks.append(packed)
        scale_chunks.append(scales)
        bias_chunks.append(biases)
        mx.clear_cache()
    packed = mx.concatenate(packed_chunks, axis=0).reshape(*value.shape[:-1], width // 32)
    group_shape = (*value.shape[:-1], width // group_size)
    scales = mx.concatenate(scale_chunks, axis=0).reshape(group_shape)
    biases = mx.concatenate(bias_chunks, axis=0).reshape(group_shape)
    return packed, scales, biases


def ternary_quantize(
    value: mx.array,
    group_size: int,
    iterations: int = 4,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pack symmetric {-scale, 0, +scale} groups into MLX affine 2-bit storage."""

    require(value.ndim >= 2, "ternary quantization requires a matrix")
    require(value.shape[-1] % group_size == 0, "ternary group size does not divide tensor")
    require(group_size % 16 == 0, "ternary groups must align to packed uint32 words")
    grouped = value.astype(mx.float32).reshape(
        *value.shape[:-1], value.shape[-1] // group_size, group_size
    )
    absolute = mx.abs(grouped)
    scale = mx.mean(absolute, axis=-1)
    for _ in range(iterations):
        selected = absolute > scale[..., None] * 0.5
        count = mx.sum(selected, axis=-1)
        selected_sum = mx.sum(mx.where(selected, absolute, 0.0), axis=-1)
        scale = mx.where(count > 0, selected_sum / mx.maximum(count, 1), 0.0)
    threshold = scale[..., None] * 0.5
    codes = mx.where(grouped > threshold, 2, mx.where(grouped < -threshold, 0, 1)).astype(
        mx.uint32
    )
    codes = codes.reshape(*value.shape[:-1], value.shape[-1] // 16, 16)
    shifts = mx.arange(16, dtype=mx.uint32) * 2
    packed = mx.sum(codes << shifts, axis=-1).astype(mx.uint32)
    scales = scale.astype(value.dtype)
    biases = (-scale).astype(value.dtype)
    return packed, scales, biases


def quantize_tensor(
    value: mx.array,
    recipe: str,
) -> tuple[mx.array, mx.array, mx.array | None]:
    settings = MODES[recipe]
    if recipe == "binary1-kmeans-g128":
        return binary_kmeans_quantize(value, settings["group_size"])
    if recipe in BINARY_MODES:
        return binary_quantize(value, settings["group_size"])
    if recipe in TERNARY_MODES:
        return ternary_quantize(value, settings["group_size"])
    result = mx.quantize(value, **settings)
    require(len(result) in (2, 3), "unexpected MLX quantization result")
    return result[0], result[1], result[2] if len(result) == 3 else None


def tensor_error(
    original: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    settings: dict,
) -> dict[str, float]:
    restored = mx.dequantize(
        weight,
        scales,
        biases,
        **settings,
        dtype=mx.float32,
    )
    reference = original.astype(mx.float32)
    difference = restored - reference
    error2 = mx.sum(mx.square(difference))
    reference2 = mx.sum(mx.square(reference))
    mse = mx.mean(mx.square(difference))
    maximum = mx.max(mx.abs(difference))
    mx.eval(error2, reference2, mse, maximum)
    return {
        "relative_l2": math.sqrt(float(error2) / max(float(reference2), 1e-30)),
        "mse": float(mse),
        "max_abs": float(maximum),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sidecar", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument(
        "--tensor-mode",
        action="append",
        default=[],
        metavar="TENSOR=MODE",
        help="override the quantization recipe for one exact weight tensor; repeat as needed",
    )
    parser.add_argument(
        "--keep-bf16",
        action="append",
        default=[],
        metavar="TENSOR",
        help="retain an exact BF16 matrix instead of quantizing it; repeat as needed",
    )
    return parser.parse_args()


def parse_tensor_modes(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, recipe = value.partition("=")
        require(separator and name and recipe, f"invalid --tensor-mode value: {value}")
        require(recipe in MODES, f"unsupported tensor quantization recipe: {recipe}")
        require(name not in result, f"duplicate --tensor-mode tensor: {name}")
        result[name] = recipe
    return result


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.source_sidecar.resolve() != args.output_dir.resolve(), "source and output are identical")
        source_report_path = args.source_sidecar / "nemotron_mtp_pack_report.json"
        source_report = load_json(source_report_path)
        require(
            source_report.get("format") == FORMAT and source_report.get("status") == "complete",
            "source MTP sidecar is incomplete",
        )
        config = load_json(args.source_sidecar / "config.json")
        index = load_json(args.source_sidecar / "model.safetensors.index.json")
        shard_names = set(index.get("weight_map", {}).values())
        require(len(shard_names) == 1, "source MTP sidecar must occupy one shard")
        source_path = args.source_sidecar / next(iter(shard_names))
        tensors = mx.load(str(source_path))
        require(set(tensors) == set(index["weight_map"]), "source MTP sidecar index mismatch")
        kept_bf16 = set(args.keep_bf16)
        tensor_modes = parse_tensor_modes(args.tensor_mode)
        requested_recipes = {args.mode, *tensor_modes.values()}
        require(
            not (requested_recipes & BINARY_MODES) or binary_affine_supported(),
            "one-bit MTP quantization requires an MLX build with affine bits=1 support",
        )
        require(len(kept_bf16) == len(args.keep_bf16), "duplicate --keep-bf16 tensor")
        for name in sorted(kept_bf16):
            require(name in tensors, f"kept BF16 tensor is missing: {name}")
            require(quantizable(name, tensors[name]), f"tensor cannot be selectively retained: {name}")
        for name in sorted(tensor_modes):
            require(name in tensors, f"tensor-mode tensor is missing: {name}")
            require(quantizable(name, tensors[name]), f"tensor-mode tensor cannot be quantized: {name}")
            require(name not in kept_bf16, f"tensor has both BF16 and quantized policies: {name}")

        output: dict[str, mx.array] = {}
        errors = {}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "quantize.log")
        operation_log.write(
            f"mtp-quant-start mode={args.mode} overrides={len(tensor_modes)} "
            f"source_sha256={sha256_file(source_path)}"
        )
        for name in sorted(tensors):
            value = tensors[name]
            if not quantizable(name, value) or name in kept_bf16:
                output[name] = value
                if name in kept_bf16:
                    operation_log.write(
                        f"mtp-quant-retain-bf16 name={name} bytes={value.nbytes}"
                    )
                continue
            prefix = name[: -len(".weight")]
            recipe = tensor_modes.get(name, args.mode)
            settings = MODES[recipe]
            weight, scales, biases = quantize_tensor(value, recipe)
            mx.eval(weight, scales, *([biases] if biases is not None else []))
            errors[name] = tensor_error(value, weight, scales, biases, settings)
            errors[name]["recipe"] = recipe
            output[name] = weight
            output[f"{prefix}.scales"] = scales
            if biases is not None:
                output[f"{prefix}.biases"] = biases
            operation_log.write(
                f"mtp-quant-tensor name={name} recipe={recipe} "
                f"relative_l2={errors[name]['relative_l2']:.9g} "
                f"max_abs={errors[name]['max_abs']:.9g}"
            )
            mx.clear_cache()

        output_path = args.output_dir / "mtp.safetensors"
        temporary = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
        temporary.unlink(missing_ok=True)
        mx.save_safetensors(str(temporary), output, metadata={"format": QUANT_FORMAT})
        temporary.replace(output_path)
        loaded = mx.load(str(output_path))
        require(set(loaded) == set(output), "quantized MTP output tensor mismatch")
        require(
            all(loaded[name].shape == value.shape and loaded[name].dtype == value.dtype for name, value in output.items()),
            "quantized MTP output metadata mismatch",
        )

        runtime_config = copy.deepcopy(config)
        runtime = runtime_config["nemotron_mtp_runtime"]
        quantization = {
            "format": QUANT_FORMAT,
            "recipe": args.mode,
            "producer_mlx_version": getattr(mx, "__version__", "unknown"),
            **MODES[args.mode],
            "bf16_tensors": sorted(kept_bf16),
            "tensor_modes": {
                name: {"recipe": recipe, **MODES[recipe]}
                for name, recipe in sorted(tensor_modes.items())
            },
        }
        runtime["quantization"] = quantization
        atomic_json(args.output_dir / "config.json", runtime_config)
        total_size = sum(value.nbytes for value in loaded.values())
        atomic_json(
            args.output_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_size},
                "weight_map": {name: output_path.name for name in loaded},
            },
        )
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_report["source_revision"],
            "source_sidecar_sha256": sha256_file(source_path),
            "source_report_sha256": sha256_file(source_report_path),
            "quantization": quantization,
            "budget": source_report["budget"],
            "payload_bytes": total_size,
            "payload_gib": total_size / 2**30,
            "tensors": len(loaded),
            "quantized_tensors": len(errors),
            "retained_bf16_tensors": sorted(kept_bf16),
            "retained_bf16_bytes": sum(tensors[name].nbytes for name in kept_bf16),
            "full_tensor_error": errors,
            "max_relative_l2": max(
                (value["relative_l2"] for value in errors.values()), default=0.0
            ),
            "max_abs": max((value["max_abs"] for value in errors.values()), default=0.0),
        }
        atomic_json(args.output_dir / "nemotron_mtp_pack_report.json", report)
        operation_log.write(
            f"mtp-quant-complete mode={args.mode} payload_gib={report['payload_gib']:.6f} "
            f"sha256={sha256_file(output_path)}"
        )
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-quant-failed error={exc}")
        print(f"nemotron MTP quantization error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
