#!/usr/bin/env python3
"""Build a quantized vocabulary head used only by verified MTP drafts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-mlx-mtp-head-v1"
MODES = {
    "mxfp8": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    "nvfp4": {"group_size": 16, "bits": 4, "mode": "nvfp4"},
}


def locate_lm_head(model_dir: Path) -> tuple[Path, mx.array]:
    index = load_json(model_dir / "model.safetensors.index.json")
    shard_name = index.get("weight_map", {}).get("lm_head.weight")
    require(isinstance(shard_name, str), "target runtime has no lm_head.weight")
    shard_path = model_dir / shard_name
    tensors = mx.load(str(shard_path))
    require("lm_head.weight" in tensors, "lm_head.weight is absent from its shard")
    weight = tensors["lm_head.weight"]
    require(weight.dtype == mx.bfloat16 and weight.ndim == 2, "MTP source head must be BF16")
    return shard_path, weight


def quantize_weight(weight: mx.array, settings: dict) -> tuple[mx.array, mx.array, mx.array | None]:
    result = mx.quantize(weight, **settings)
    require(len(result) in (2, 3), "unexpected MTP head quantization result")
    return result[0], result[1], result[2] if len(result) == 3 else None


def chunked_error(
    original: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    settings: dict,
    rows_per_chunk: int = 1024,
) -> dict[str, float]:
    error2 = 0.0
    reference2 = 0.0
    maximum = 0.0
    elements = 0
    for start in range(0, original.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, original.shape[0])
        restored = mx.dequantize(
            weight[start:end],
            scales[start:end],
            biases[start:end] if biases is not None else None,
            **settings,
            dtype=mx.float32,
        )
        reference = original[start:end].astype(mx.float32)
        difference = restored - reference
        chunk_error2 = mx.sum(mx.square(difference))
        chunk_reference2 = mx.sum(mx.square(reference))
        chunk_maximum = mx.max(mx.abs(difference))
        mx.eval(chunk_error2, chunk_reference2, chunk_maximum)
        error2 += float(chunk_error2)
        reference2 += float(chunk_reference2)
        maximum = max(maximum, float(chunk_maximum))
        elements += reference.size
    return {
        "relative_l2": math.sqrt(error2 / max(reference2, 1e-30)),
        "mse": error2 / elements,
        "max_abs": maximum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        report_path = args.model_dir / "nemotron_mlx_pack_report.json"
        source_report = load_json(report_path)
        require(
            source_report.get("format") == "nemotron-mlx-runtime-v1"
            and source_report.get("status") == "complete",
            "source target runtime is incomplete",
        )
        require(args.model_dir.resolve() != args.output_dir.resolve(), "source and output are identical")
        settings = MODES[args.mode]
        source_path, original = locate_lm_head(args.model_dir)
        source_hash = sha256_file(source_path)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "quantize.log")
        operation_log.write(
            f"mtp-head-start mode={args.mode} source={source_path} source_sha256={source_hash}"
        )
        weight, scales, biases = quantize_weight(original, settings)
        mx.eval(weight, scales, *([biases] if biases is not None else []))
        error = chunked_error(original, weight, scales, biases, settings)
        operation_log.write(
            f"mtp-head-error relative_l2={error['relative_l2']:.9g} "
            f"mse={error['mse']:.9g} max_abs={error['max_abs']:.9g}"
        )

        output = {"weight": weight, "scales": scales}
        if biases is not None:
            output["biases"] = biases
        output_path = args.output_dir / "lm_head.safetensors"
        temporary = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
        temporary.unlink(missing_ok=True)
        mx.save_safetensors(
            str(temporary),
            output,
            metadata={"format": FORMAT, "mode": args.mode},
        )
        temporary.replace(output_path)
        loaded, metadata = mx.load(str(output_path), return_metadata=True)
        require(set(loaded) == set(output), "quantized MTP head tensor mismatch")
        require(metadata == {"format": FORMAT, "mode": args.mode}, "quantized MTP head metadata mismatch")
        require(
            all(loaded[name].shape == value.shape and loaded[name].dtype == value.dtype for name, value in output.items()),
            "quantized MTP head shape or dtype mismatch",
        )
        payload_bytes = sum(value.nbytes for value in loaded.values())
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_report["source_revision"],
            "source_model_dir": str(args.model_dir.resolve()),
            "source_report_sha256": sha256_file(report_path),
            "source_shard": source_path.name,
            "source_shard_sha256": source_hash,
            "source_shape": list(original.shape),
            "source_dtype": str(original.dtype),
            "quantization": settings,
            "payload_bytes": payload_bytes,
            "payload_gib": payload_bytes / 2**30,
            "full_tensor_error": error,
            "artifact": output_path.name,
            "artifact_sha256": sha256_file(output_path),
        }
        atomic_json(args.output_dir / "nemotron_mtp_head_report.json", report)
        operation_log.write(
            f"mtp-head-complete payload_gib={report['payload_gib']:.6f} "
            f"artifact_sha256={report['artifact_sha256']}"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-head-failed error={exc}")
        print(f"nemotron MTP head quantization error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
