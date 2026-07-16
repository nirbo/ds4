#!/usr/bin/env python3
"""Materialize one disjoint fitted-binary/exact-NVFP4 backbone layer."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import mlx.core as mx

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_backbone_fit import STATE_FORMAT as FIT_STATE_FORMAT
from nemotron_mlx_backbone_fit import validate_expert_artifact
from nemotron_mlx_backbone_lowbit import AffineWeight, BinaryExpert
from nemotron_mlx_backbone_mixed import (
    FILE_FORMAT,
    BinaryExpertMLP,
    MixedExpertMLP,
    binary_switch_from_affine,
    build_maps,
    compact_native,
    load_mixed_file,
    mixed_file_tensors,
)
from nemotron_mlx_backbone_plan import FORMAT as PLAN_FORMAT
from nemotron_mlx_moe import load_expert_layer
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
)


REPORT_FORMAT = "nemotron-backbone-lowbit-pack-report-v1"


def binary_expert_from_arrays(arrays: dict[str, mx.array], latent: int, hidden: int) -> BinaryExpert:
    result = BinaryExpert(
        up=AffineWeight(
            arrays["up.weight"],
            arrays["up.scales"],
            arrays["up.biases"],
            bits=1,
            group_size=128,
            rows=hidden,
            columns=latent,
        ),
        down=AffineWeight(
            arrays["down.weight"],
            arrays["down.scales"],
            arrays["down.biases"],
            bits=1,
            group_size=128,
            rows=latent,
            columns=hidden,
        ),
    )
    result.validate()
    return result


def require_array_identity(left: mx.array, right: mx.array, label: str) -> None:
    require(left.dtype == right.dtype and left.shape == right.shape, f"packed identity shape mismatch: {label}")
    require(bool(mx.array_equal(left, right)), f"packed identity payload mismatch: {label}")


def validate_existing(
    output: Path,
    report_path: Path,
    *,
    contract_sha256: str,
    fit_state_sha256: str,
    plan_sha256: str,
    budget: int,
) -> dict | None:
    if not output.exists() and not report_path.exists():
        return None
    if output.is_file() and not report_path.exists():
        return None
    require(output.is_file() and report_path.is_file(), "committed mixed layer artifact is missing its payload")
    report = load_json(report_path)
    require(report.get("format") == REPORT_FORMAT and report.get("status") == "complete", "invalid pack report")
    require(report.get("contract_sha256") == contract_sha256, "existing pack contract mismatch")
    require(report.get("fit_state_sha256") == fit_state_sha256, "existing pack fit mismatch")
    require(report.get("plan_sha256") == plan_sha256, "existing pack plan mismatch")
    require(report.get("native_budget") == budget, "existing pack budget mismatch")
    require(output.stat().st_size == report.get("file_bytes"), "existing mixed file size mismatch")
    require(sha256_file(output) == report.get("file_sha256"), "existing mixed file hash mismatch")
    weights, metadata = load_mixed_file(output)
    require(int(metadata.get("native_budget", -1)) == budget, "existing mixed metadata budget mismatch")
    require(weights.native.up.experts == budget, "existing mixed native bank size mismatch")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--native-budget", required=True, type=int)
    parser.add_argument("--proxy-source-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        fit_state_path = args.fit_dir / "state.json"
        fit_state = load_json(fit_state_path)
        require(
            fit_state.get("format") == FIT_STATE_FORMAT and fit_state.get("status") == "complete",
            "backbone fit is incomplete",
        )
        plan = load_json(args.plan)
        require(
            plan.get("format") == PLAN_FORMAT and plan.get("status") == "complete",
            "backbone precision plan is incomplete",
        )
        contract_sha256 = sha256_file(args.contract)
        fit_state_sha256 = sha256_file(fit_state_path)
        plan_sha256 = sha256_file(args.plan)
        require(fit_state.get("contract_sha256") == contract_sha256, "fit/contract mismatch")
        require(plan.get("contract_sha256") == contract_sha256, "plan/contract mismatch")
        require(plan.get("fit_state_sha256") == fit_state_sha256, "plan/fit mismatch")
        require(plan.get("layer") == contract["layer"] == fit_state["layer"], "mixed layer index mismatch")
        proxy_state = load_source_state(args.proxy_source_state, args.proxy_source_dir)
        require(plan.get("proxy_source_revision") == proxy_state["revision"], "plan/native source mismatch")
        require(
            plan.get("proxy_source_state_sha256") == sha256_file(args.proxy_source_state),
            "plan/native source state mismatch",
        )
        assignment = plan.get("budgets", {}).get(str(args.native_budget))
        require(isinstance(assignment, dict), "native budget is absent from precision plan")
        binary_ids = assignment.get("binary_experts")
        native_ids = assignment.get("native_nvfp4_experts")
        experts = contract["architecture"]["experts"]
        require(
            isinstance(binary_ids, list)
            and isinstance(native_ids, list)
            and binary_ids
            and native_ids
            and len(native_ids) == args.native_budget,
            "mixed precision assignment has invalid bank sizes",
        )
        binary_map, native_map = build_maps(experts, binary_ids, native_ids)
        report_path = args.output.with_suffix(".report.json")
        existing = validate_existing(
            args.output,
            report_path,
            contract_sha256=contract_sha256,
            fit_state_sha256=fit_state_sha256,
            plan_sha256=plan_sha256,
            budget=args.native_budget,
        )
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

        args.output.parent.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        operation_log.write(
            f"backbone-pack-start layer={contract['layer']} binary={len(binary_ids)} "
            f"native={len(native_ids)} output={args.output}"
        )
        entries = {entry["expert"]: entry for entry in fit_state["completed"]}
        identity = {
            "source_revision": fit_state["source_revision"],
            "layer": fit_state["layer"],
            "architecture": fit_state["architecture"],
            "validation_context_rows": fit_state["validation_context_rows"],
            "group_size": fit_state["group_size"],
            "fit_strategy": fit_state.get("fit_strategy", "bf16-endpoint"),
            "contract_sha256": fit_state["contract_sha256"],
            "context_state_sha256": fit_state["context_state_sha256"],
        }
        latent = contract["architecture"]["latent_width"]
        hidden = contract["architecture"]["hidden_width"]
        binary_experts = []
        for offset, expert in enumerate(binary_ids):
            require(expert in entries, f"binary expert has no fitted artifact: {expert}")
            entry = entries[expert]
            arrays = validate_expert_artifact(args.fit_dir / "experts" / entry["file"], entry, identity)
            binary_experts.append(binary_expert_from_arrays(arrays, latent, hidden))
            if (offset + 1) % 64 == 0 or offset + 1 == len(binary_ids):
                operation_log.write(f"backbone-pack-binary-progress experts={offset + 1}/{len(binary_ids)}")
        binary = BinaryExpertMLP(
            up=binary_switch_from_affine([expert.up for expert in binary_experts]),
            down=binary_switch_from_affine([expert.down for expert in binary_experts]),
        )
        del binary_experts
        gc.collect()
        mx.clear_cache()

        operation_log.write("backbone-pack-native-load-start")
        source_native = load_expert_layer(args.proxy_source_dir, contract["layer"])
        native = compact_native(source_native, native_ids)
        mixed = MixedExpertMLP(binary, native, binary_map, native_map)
        mixed.validate()
        payload_bytes = sum(value.nbytes for value in mixed_file_tensors(mixed).values())
        require(
            payload_bytes == assignment["layer_payload_bytes"] + 2 * experts * 4,
            "mixed payload does not match precision-plan accounting",
        )
        temporary = args.output.with_name(args.output.stem + ".part" + args.output.suffix)
        temporary.unlink(missing_ok=True)
        mx.save_safetensors(
            str(temporary),
            mixed_file_tensors(mixed),
            metadata={
                "format": FILE_FORMAT,
                "layer": str(contract["layer"]),
                "bf16_source_revision": fit_state["source_revision"],
                "native_source_revision": proxy_state["revision"],
                "fit_strategy": fit_state.get("fit_strategy", "bf16-endpoint"),
                "contract_sha256": contract_sha256,
                "fit_state_sha256": fit_state_sha256,
                "plan_sha256": plan_sha256,
                "native_budget": str(args.native_budget),
            },
        )
        temporary.replace(args.output)
        loaded, metadata = load_mixed_file(args.output)
        require(metadata.get("contract_sha256") == contract_sha256, "mixed file contract mismatch")
        require(metadata.get("fit_state_sha256") == fit_state_sha256, "mixed file fit mismatch")
        require(metadata.get("plan_sha256") == plan_sha256, "mixed file plan mismatch")
        require(
            metadata.get("fit_strategy", "bf16-endpoint")
            == fit_state.get("fit_strategy", "bf16-endpoint"),
            "mixed file fit strategy mismatch",
        )
        for projection in ("up", "down"):
            expected_binary = getattr(binary, projection)
            actual_binary = getattr(loaded.binary, projection)
            expected_native = getattr(source_native, projection)
            actual_native = getattr(loaded.native, projection)
            selected = mx.array(native_ids, dtype=mx.uint32)
            require_array_identity(actual_binary.weight, expected_binary.weight, f"binary.{projection}.weight")
            require_array_identity(actual_binary.scales, expected_binary.scales, f"binary.{projection}.scales")
            require_array_identity(actual_binary.biases, expected_binary.biases, f"binary.{projection}.biases")
            require_array_identity(actual_native.weight, expected_native.weight[selected], f"native.{projection}.weight")
            require_array_identity(actual_native.scales, expected_native.scales[selected], f"native.{projection}.scales")
            require_array_identity(
                actual_native.global_scales,
                expected_native.global_scales[selected],
                f"native.{projection}.global_scales",
            )
        require_array_identity(loaded.binary_map, binary_map, "maps.binary")
        require_array_identity(loaded.native_map, native_map, "maps.native")
        report = {
            "format": REPORT_FORMAT,
            "status": "complete",
            "layer": contract["layer"],
            "bf16_source_revision": fit_state["source_revision"],
            "native_source_revision": proxy_state["revision"],
            "fit_strategy": fit_state.get("fit_strategy", "bf16-endpoint"),
            "contract_sha256": contract_sha256,
            "fit_state_sha256": fit_state_sha256,
            "plan_sha256": plan_sha256,
            "native_source_state_sha256": sha256_file(args.proxy_source_state),
            "native_budget": args.native_budget,
            "binary_experts": binary_ids,
            "native_nvfp4_experts": native_ids,
            "payload_bytes": payload_bytes,
            "payload_gib": payload_bytes / 2**30,
            "file": str(args.output.resolve()),
            "file_bytes": args.output.stat().st_size,
            "file_sha256": sha256_file(args.output),
            "retained_native_payload_identity": "passed",
            "fitted_binary_payload_identity": "passed",
            "tool_sha256": sha256_file(Path(__file__)),
        }
        atomic_json(report_path, report)
        operation_log.write(
            f"backbone-pack-complete bytes={report['file_bytes']} sha256={report['file_sha256']} "
            "native_identity=passed binary_identity=passed"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-pack-failed error={exc}")
        print(f"nemotron backbone pack error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
