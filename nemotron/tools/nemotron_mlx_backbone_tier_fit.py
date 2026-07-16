#!/usr/bin/env python3
"""Pilot activation-fitted affine tiers against the native Nemotron expert target."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

from nemotron_bf16_source import FORMAT as CONTRACT_FORMAT
from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_backbone_context import STATE_FORMAT as CONTEXT_STATE_FORMAT
from nemotron_mlx_backbone_context import load_context_rows
from nemotron_mlx_backbone_fit import NVFP4TargetWeights, expert_context_rows, parse_experts
from nemotron_mlx_backbone_lowbit import (
    affine_weight,
    dequantize_affine,
    expert_output,
    fit_affine_expert,
)
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
)


FORMAT = "nemotron-backbone-lowbit-tier-fit-pilot-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--proxy-source-dir", required=True, type=Path)
    parser.add_argument("--proxy-source-state", required=True, type=Path)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--bits", action="append", type=int)
    parser.add_argument("--route-weight-power", type=float, default=2.0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--endpoint-margin", type=float, default=0.25)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        bits = sorted(set(args.bits or (1, 2, 3)))
        require(bits and all(value in (1, 2, 3, 4) for value in bits), "invalid affine fit tiers")
        require(args.route_weight_power > 0.0, "route-weight power must be positive")
        require(args.group_size > 0, "group size must be positive")
        require(args.ridge >= 0.0 and args.endpoint_margin >= 0.0, "fit regularizers must be nonnegative")

        contract = load_json(args.contract)
        require(contract.get("format") == CONTRACT_FORMAT, "unsupported BF16 layer contract")
        context_state_path = args.contexts / "state.json"
        context_state = load_json(context_state_path)
        require(context_state.get("format") == CONTEXT_STATE_FORMAT, "unsupported context state")
        require(context_state.get("status") == "complete", "context capture is incomplete")
        proxy_state = load_source_state(args.proxy_source_state, args.proxy_source_dir)
        require(context_state.get("source_revision") == proxy_state["revision"], "context/source revision mismatch")
        require(context_state.get("layer") == contract["layer"], "context/contract layer mismatch")

        expert_count = contract["architecture"]["experts"]
        experts = parse_experts(args.experts, expert_count)
        identity = {
            "format": FORMAT,
            "source_repository": contract["repository"],
            "source_revision": contract["source_revision"],
            "proxy_source_revision": proxy_state["revision"],
            "layer": contract["layer"],
            "architecture": contract["architecture"],
            "contract": str(args.contract.resolve()),
            "contract_sha256": sha256_file(args.contract),
            "context_state": str(context_state_path.resolve()),
            "context_state_sha256": sha256_file(context_state_path),
            "proxy_source_state_sha256": sha256_file(args.proxy_source_state),
            "tool_sha256": sha256_file(Path(__file__)),
            "experts": experts,
            "bits": bits,
            "route_weight_power": args.route_weight_power,
            "group_size": args.group_size,
            "ridge": args.ridge,
            "endpoint_margin": args.endpoint_margin,
        }
        operation_log = OperationLog(args.output.with_suffix(".log"))
        if args.output.exists():
            existing = load_json(args.output)
            require(existing.get("status") == "complete", "existing tier fit pilot is incomplete")
            for key, value in identity.items():
                require(existing.get(key) == value, f"existing tier fit identity mismatch: {key}")
            operation_log.write(f"backbone-tier-fit-validated output={args.output}")
            return 0

        operation_log.write(
            f"backbone-tier-fit-start layer={contract['layer']} experts={len(experts)} "
            f"bits={','.join(map(str, bits))}"
        )
        train = load_context_rows(args.contexts, "train")
        validation = load_context_rows(args.contexts, "validation")
        target = NVFP4TargetWeights(
            args.proxy_source_dir,
            contract["layer"],
            expert_count,
            contract["architecture"]["latent_width"],
            contract["architecture"]["hidden_width"],
        )
        results = []
        for expert in experts:
            train_latent, train_weights, _, _ = expert_context_rows(
                train,
                expert,
                args.route_weight_power,
            )
            validation_latent, validation_weights, _, validation_scores = expert_context_rows(
                validation,
                expert,
                args.route_weight_power,
            )
            target_up, target_down = target.expert(expert)
            quant_source_up = target_up.astype(mx.bfloat16)
            quant_source_down = target_down.astype(mx.bfloat16)
            target_validation = expert_output(validation_latent, target_up, target_down)
            mx.eval(target_validation)
            tiers = {}
            for bit_width in bits:
                started = time.perf_counter()
                fitted, metrics = fit_affine_expert(
                    quant_source_up,
                    quant_source_down,
                    train_latent,
                    validation_latent,
                    train_weights,
                    validation_weights,
                    bits=bit_width,
                    group_size=args.group_size,
                    ridge=args.ridge,
                    endpoint_margin=args.endpoint_margin,
                    target_up=target_up,
                    target_down=target_down,
                )
                stock_up = dequantize_affine(affine_weight(quant_source_up, bit_width, args.group_size))
                stock_down = dequantize_affine(affine_weight(quant_source_down, bit_width, args.group_size))
                stock_output = expert_output(validation_latent, stock_up, stock_down)
                fitted_output = expert_output(
                    validation_latent,
                    dequantize_affine(fitted.up),
                    dequantize_affine(fitted.down),
                )
                stock_residual = (stock_output - target_validation) * validation_scores[:, None]
                fitted_residual = (fitted_output - target_validation) * validation_scores[:, None]
                stock_error2 = mx.sum(mx.square(stock_residual.astype(mx.float32)))
                fitted_error2 = mx.sum(mx.square(fitted_residual.astype(mx.float32)))
                mx.eval(stock_error2, fitted_error2)
                stock_value = float(stock_error2)
                fitted_value = float(fitted_error2)
                tiers[str(bit_width)] = {
                    "payload_bytes": fitted.payload_bytes,
                    "metrics": metrics,
                    "validation_weighted_residual_error2": {
                        "stock": stock_value,
                        "fitted": fitted_value,
                        "relative_change": (fitted_value - stock_value) / max(stock_value, 1e-30),
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
                operation_log.write(
                    f"backbone-tier-fit-expert-done expert={expert} bits={bit_width} "
                    f"routes={validation_latent.shape[0]} stock={metrics['validation']['stock_relative_l2']:.7g} "
                    f"codebook={metrics['validation']['codebook_relative_l2']:.7g} "
                    f"fitted={metrics['validation']['fitted_relative_l2']:.7g} "
                    f"weighted_change={(fitted_value - stock_value) / max(stock_value, 1e-30):.3%} "
                    f"elapsed={tiers[str(bit_width)]['elapsed_seconds']:.2f}s"
                )
                del (
                    fitted,
                    stock_up,
                    stock_down,
                    stock_output,
                    fitted_output,
                    stock_residual,
                    fitted_residual,
                )
                gc.collect()
                mx.clear_cache()
            results.append(
                {
                    "expert": expert,
                    "train_routes": train_latent.shape[0],
                    "validation_routes": validation_latent.shape[0],
                    "tiers": tiers,
                }
            )
            del target_up, target_down, quant_source_up, quant_source_down, target_validation
            gc.collect()
            mx.clear_cache()

        summary = {}
        for bit_width in bits:
            rows = [row["tiers"][str(bit_width)] for row in results]
            changes = [row["validation_weighted_residual_error2"]["relative_change"] for row in rows]
            stock_total = sum(row["validation_weighted_residual_error2"]["stock"] for row in rows)
            fitted_total = sum(row["validation_weighted_residual_error2"]["fitted"] for row in rows)
            guarded_total = sum(
                min(
                    row["validation_weighted_residual_error2"]["stock"],
                    row["validation_weighted_residual_error2"]["fitted"],
                )
                for row in rows
            )
            validation_changes = [
                (
                    row["metrics"]["validation"]["fitted_relative_l2"]
                    - row["metrics"]["validation"]["stock_relative_l2"]
                )
                / max(row["metrics"]["validation"]["stock_relative_l2"], 1e-30)
                for row in rows
            ]
            summary[str(bit_width)] = {
                "experts_improved": sum(value < 0.0 for value in changes),
                "experts_regressed": sum(value > 0.0 for value in changes),
                "median_weighted_residual_change": statistics.median(changes),
                "aggregate_weighted_residual_change": (fitted_total - stock_total) / max(stock_total, 1e-30),
                "guarded_best_of_stock_or_fitted_change": (guarded_total - stock_total)
                / max(stock_total, 1e-30),
                "median_validation_relative_l2_change": statistics.median(validation_changes),
            }
        report = {
            **identity,
            "status": "complete",
            "method": (
                "Target-derived fixed affine codes with weighted least-squares group endpoints; "
                "the down projection is fit on hidden activations produced by the fitted up projection."
            ),
            "summary": summary,
            "results": results,
        }
        atomic_json(args.output, report)
        operation_log.write(
            f"backbone-tier-fit-complete output={args.output} bytes={args.output.stat().st_size} "
            f"sha256={sha256_file(args.output)}"
        )
        print(json.dumps(summary, indent=2))
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"backbone-tier-fit-failed error={exc}")
        print(f"nemotron backbone tier fit error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
