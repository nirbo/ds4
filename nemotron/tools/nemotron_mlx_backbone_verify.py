#!/usr/bin/env python3
"""Verify a physical mixed backbone layer against its causal virtual plan."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_backbone_context import load_context_rows
from nemotron_mlx_backbone_fit import STATE_FORMAT as FIT_STATE_FORMAT
from nemotron_mlx_backbone_mixed import load_mixed_file, mixed_layer_forward
from nemotron_mlx_backbone_pack import REPORT_FORMAT as PACK_REPORT_FORMAT
from nemotron_mlx_backbone_plan import (
    FORMAT as PLAN_FORMAT,
    aggregate_binary_residual,
    canonical_routes,
    load_evidence,
)
from nemotron_mlx_moe_layer import load_moe_layer
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
)


FORMAT = "nemotron-backbone-lowbit-physical-verify-v1"


def parity_metrics(
    *,
    full_reference2: float,
    virtual_error2: float,
    physical_error2: float,
    physical_virtual_error2: float,
    physical_max_abs: float,
    physical_virtual_max_abs: float,
) -> dict[str, float]:
    require(full_reference2 > 0.0, "physical verification reference norm is zero")
    return {
        "full_layer_reference2": full_reference2,
        "virtual_error2": virtual_error2,
        "physical_error2": physical_error2,
        "physical_virtual_error2": physical_virtual_error2,
        "virtual_full_relative_l2": math.sqrt(virtual_error2 / full_reference2),
        "physical_full_relative_l2": math.sqrt(physical_error2 / full_reference2),
        "physical_virtual_full_relative_l2": math.sqrt(
            physical_virtual_error2 / full_reference2
        ),
        "physical_virtual_error_relative_l2": math.sqrt(
            physical_virtual_error2 / max(virtual_error2, 1e-30)
        ),
        "physical_max_abs": physical_max_abs,
        "physical_virtual_max_abs": physical_virtual_max_abs,
    }


def gate_metrics(
    metrics: dict[str, float],
    expected: dict[str, float],
    *,
    plan_relative_tolerance: float,
    physical_plan_tolerance: float,
    physical_virtual_tolerance: float,
    physical_virtual_max_abs_tolerance: float,
) -> dict:
    plan_reference_relative_delta = abs(
        metrics["full_layer_reference2"] - expected["full_layer_reference2"]
    ) / max(expected["full_layer_reference2"], 1e-30)
    plan_error_relative_delta = abs(metrics["virtual_error2"] - expected["error2"]) / max(
        expected["error2"], 1e-30
    )
    physical_plan_delta = abs(
        metrics["physical_full_relative_l2"] - expected["full_layer_relative_l2"]
    )
    checks = {
        "plan_reference_reproduced": plan_reference_relative_delta <= plan_relative_tolerance,
        "plan_error_reproduced": plan_error_relative_delta <= plan_relative_tolerance,
        "physical_plan_error_preserved": physical_plan_delta <= physical_plan_tolerance,
        "physical_virtual_full_parity": (
            metrics["physical_virtual_full_relative_l2"] <= physical_virtual_tolerance
        ),
        "physical_virtual_max_abs": (
            metrics["physical_virtual_max_abs"] <= physical_virtual_max_abs_tolerance
        ),
    }
    return {
        "result": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "observed": {
            "plan_reference_relative_delta": plan_reference_relative_delta,
            "plan_error_relative_delta": plan_error_relative_delta,
            "physical_plan_relative_l2_delta": physical_plan_delta,
        },
        "tolerances": {
            "plan_relative": plan_relative_tolerance,
            "physical_plan_relative_l2": physical_plan_tolerance,
            "physical_virtual_full_relative_l2": physical_virtual_tolerance,
            "physical_virtual_max_abs": physical_virtual_max_abs_tolerance,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--fit-dir", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--native-budget", required=True, type=int)
    parser.add_argument("--mixed-layer", required=True, type=Path)
    parser.add_argument("--proxy-source-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-state", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--plan-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--physical-plan-tolerance", type=float, default=5e-4)
    parser.add_argument("--physical-virtual-tolerance", type=float, default=5e-4)
    parser.add_argument("--physical-virtual-max-abs-tolerance", type=float, default=1e-2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.chunk_rows > 0, "chunk rows must be positive")
        for name in (
            "plan_relative_tolerance",
            "physical_plan_tolerance",
            "physical_virtual_tolerance",
            "physical_virtual_max_abs_tolerance",
        ):
            require(getattr(args, name) >= 0.0, f"{name} must be nonnegative")

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
        assignment = plan.get("budgets", {}).get(str(args.native_budget))
        require(isinstance(assignment, dict), "native budget is absent from precision plan")
        context_state_path = args.contexts / "state.json"
        contract_sha256 = sha256_file(args.contract)
        fit_state_sha256 = sha256_file(fit_state_path)
        plan_sha256 = sha256_file(args.plan)
        context_state_sha256 = sha256_file(context_state_path)
        require(fit_state.get("contract_sha256") == contract_sha256, "fit/contract mismatch")
        require(plan.get("contract_sha256") == contract_sha256, "plan/contract mismatch")
        require(plan.get("fit_state_sha256") == fit_state_sha256, "plan/fit mismatch")
        require(plan.get("context_state_sha256") == context_state_sha256, "plan/context mismatch")
        require(
            contract["layer"] == fit_state["layer"] == plan["layer"],
            "physical verification layer mismatch",
        )

        source_state = load_source_state(args.proxy_source_state, args.proxy_source_dir)
        require(plan.get("proxy_source_revision") == source_state["revision"], "plan/source mismatch")
        require(
            plan.get("proxy_source_state_sha256") == sha256_file(args.proxy_source_state),
            "plan/source state mismatch",
        )
        pack_report_path = args.mixed_layer.with_suffix(".report.json")
        pack_report = load_json(pack_report_path)
        require(
            pack_report.get("format") == PACK_REPORT_FORMAT
            and pack_report.get("status") == "complete",
            "mixed layer pack report is incomplete",
        )
        require(pack_report.get("contract_sha256") == contract_sha256, "pack/contract mismatch")
        require(pack_report.get("fit_state_sha256") == fit_state_sha256, "pack/fit mismatch")
        require(pack_report.get("plan_sha256") == plan_sha256, "pack/plan mismatch")
        require(pack_report.get("native_budget") == args.native_budget, "pack budget mismatch")
        require(args.mixed_layer.stat().st_size == pack_report.get("file_bytes"), "mixed file size mismatch")
        require(sha256_file(args.mixed_layer) == pack_report.get("file_sha256"), "mixed file hash mismatch")

        mixed, metadata = load_mixed_file(args.mixed_layer)
        require(metadata.get("contract_sha256") == contract_sha256, "mixed/contract mismatch")
        require(metadata.get("fit_state_sha256") == fit_state_sha256, "mixed/fit mismatch")
        require(metadata.get("plan_sha256") == plan_sha256, "mixed/plan mismatch")
        require(int(metadata.get("native_budget", -1)) == args.native_budget, "mixed budget mismatch")
        binary_ids = [
            expert for expert, local in enumerate(mixed.binary_map.tolist()) if local >= 0
        ]
        native_ids = [
            expert for expert, local in enumerate(mixed.native_map.tolist()) if local >= 0
        ]
        require(binary_ids == assignment["binary_experts"], "mixed binary assignment mismatch")
        require(native_ids == assignment["native_nvfp4_experts"], "mixed native assignment mismatch")

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
        validation = load_context_rows(args.contexts, "validation")
        evidence = load_evidence(args.fit_dir, fit_state, identity)
        residual = aggregate_binary_residual(
            evidence,
            assignment["binary_experts"],
            validation["layer_input"].shape[0],
            contract["architecture"]["latent_width"],
        )
        block = load_moe_layer(args.proxy_source_dir, contract["layer"])
        rows = validation["layer_input"].shape[0]
        operation_log = OperationLog(args.output.with_suffix(".log"))
        operation_log.write(
            f"backbone-physical-verify-start layer={contract['layer']} budget={args.native_budget} "
            f"rows={rows} chunk_rows={args.chunk_rows}"
        )

        full_reference2 = 0.0
        virtual_error2 = 0.0
        physical_error2 = 0.0
        physical_virtual_error2 = 0.0
        physical_max_abs = 0.0
        physical_virtual_max_abs = 0.0
        for start in range(0, rows, args.chunk_rows):
            stop = min(start + args.chunk_rows, rows)
            x = validation["layer_input"][start:stop].reshape(1, stop - start, -1)
            native, native_indices, native_scores, _, _ = block.forward_with_expert_outputs(x)
            physical, physical_indices, physical_scores = mixed_layer_forward(block, mixed, x)
            virtual_delta = block.fc2_latent(
                mx.array(residual[start:stop], dtype=mx.float32).reshape(1, stop - start, -1)
            )
            virtual = native + virtual_delta
            native_indices, native_scores = canonical_routes(native_indices, native_scores)
            physical_indices, physical_scores = canonical_routes(physical_indices, physical_scores)
            mx.eval(
                native,
                physical,
                virtual,
                native_indices,
                native_scores,
                physical_indices,
                physical_scores,
            )
            require(
                bool(mx.array_equal(native_indices.astype(mx.int32), physical_indices.astype(mx.int32))),
                "physical/native route expert set drifted",
            )
            require(
                bool(mx.allclose(native_scores, physical_scores, rtol=1e-5, atol=2e-6)),
                "physical/native route scores drifted",
            )
            physical_delta = physical.astype(mx.float32) - native.astype(mx.float32)
            virtual_delta = virtual.astype(mx.float32) - native.astype(mx.float32)
            parity_delta = physical.astype(mx.float32) - virtual.astype(mx.float32)
            full_reference2 += float(mx.sum(mx.square(native.astype(mx.float32))))
            virtual_error2 += float(mx.sum(mx.square(virtual_delta)))
            physical_error2 += float(mx.sum(mx.square(physical_delta)))
            physical_virtual_error2 += float(mx.sum(mx.square(parity_delta)))
            physical_max_abs = max(physical_max_abs, float(mx.max(mx.abs(physical_delta))))
            physical_virtual_max_abs = max(
                physical_virtual_max_abs,
                float(mx.max(mx.abs(parity_delta))),
            )
            if stop == rows or stop % 512 == 0:
                operation_log.write(f"backbone-physical-verify-progress rows={stop}/{rows}")

        metrics = parity_metrics(
            full_reference2=full_reference2,
            virtual_error2=virtual_error2,
            physical_error2=physical_error2,
            physical_virtual_error2=physical_virtual_error2,
            physical_max_abs=physical_max_abs,
            physical_virtual_max_abs=physical_virtual_max_abs,
        )
        expected = assignment["causal_layer_output"]
        gate = gate_metrics(
            metrics,
            expected,
            plan_relative_tolerance=args.plan_relative_tolerance,
            physical_plan_tolerance=args.physical_plan_tolerance,
            physical_virtual_tolerance=args.physical_virtual_tolerance,
            physical_virtual_max_abs_tolerance=args.physical_virtual_max_abs_tolerance,
        )
        report = {
            "format": FORMAT,
            "status": "complete",
            "result": gate["result"],
            "layer": contract["layer"],
            "native_budget": args.native_budget,
            "fit_strategy": fit_state.get("fit_strategy", "bf16-endpoint"),
            "contract_sha256": contract_sha256,
            "fit_state_sha256": fit_state_sha256,
            "context_state_sha256": context_state_sha256,
            "plan_sha256": plan_sha256,
            "mixed_layer_sha256": sha256_file(args.mixed_layer),
            "pack_report_sha256": sha256_file(pack_report_path),
            "proxy_source_revision": source_state["revision"],
            "proxy_source_state_sha256": sha256_file(args.proxy_source_state),
            "validation_rows": rows,
            "metrics": metrics,
            "expected_plan_metrics": expected,
            "gate": gate,
            "tool_sha256": sha256_file(Path(__file__)),
            "mixed_runtime_tool_sha256": sha256_file(
                Path(__file__).with_name("nemotron_mlx_backbone_mixed.py")
            ),
        }
        atomic_json(args.output, report)
        operation_log.write(
            f"backbone-physical-verify-complete result={gate['result']} "
            f"physical_relative_l2={metrics['physical_full_relative_l2']:.9g} "
            f"virtual_relative_l2={metrics['virtual_full_relative_l2']:.9g} "
            f"physical_virtual_relative_l2={metrics['physical_virtual_full_relative_l2']:.9g} "
            f"max_abs={metrics['physical_virtual_max_abs']:.9g}"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        del block, mixed, evidence, validation
        gc.collect()
        mx.clear_cache()
        return 0 if gate["result"] == "passed" else 1
    except (MetadataError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-physical-verify-failed error={exc}")
        print(f"nemotron backbone physical verify error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
