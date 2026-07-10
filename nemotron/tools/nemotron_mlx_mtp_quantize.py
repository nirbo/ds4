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
    "mxfp4": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    "nvfp4": {"group_size": 16, "bits": 4, "mode": "nvfp4"},
}


def quantizable(name: str, value: mx.array) -> bool:
    return (
        name.endswith(".weight")
        and value.dtype == mx.bfloat16
        and value.ndim >= 2
        and not name.endswith(".gate.weight")
    )


def quantize_tensor(value: mx.array, settings: dict) -> tuple[mx.array, mx.array, mx.array | None]:
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
    return parser.parse_args()


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

        settings = MODES[args.mode]
        output: dict[str, mx.array] = {}
        errors = {}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "quantize.log")
        operation_log.write(
            f"mtp-quant-start mode={args.mode} source_sha256={sha256_file(source_path)}"
        )
        for name in sorted(tensors):
            value = tensors[name]
            if not quantizable(name, value):
                output[name] = value
                continue
            prefix = name[: -len(".weight")]
            weight, scales, biases = quantize_tensor(value, settings)
            mx.eval(weight, scales, *([biases] if biases is not None else []))
            errors[name] = tensor_error(value, weight, scales, biases, settings)
            output[name] = weight
            output[f"{prefix}.scales"] = scales
            if biases is not None:
                output[f"{prefix}.biases"] = biases
            operation_log.write(
                f"mtp-quant-tensor name={name} relative_l2={errors[name]['relative_l2']:.9g} "
                f"max_abs={errors[name]['max_abs']:.9g}"
            )

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
        runtime["quantization"] = {"format": QUANT_FORMAT, **settings}
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
            "quantization": {"format": QUANT_FORMAT, **settings},
            "budget": source_report["budget"],
            "payload_bytes": total_size,
            "payload_gib": total_size / 2**30,
            "tensors": len(loaded),
            "full_tensor_error": errors,
            "max_relative_l2": max(value["relative_l2"] for value in errors.values()),
            "max_abs": max(value["max_abs"] for value in errors.values()),
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
