#!/usr/bin/env python3
"""Export an exact Router KD/source-layer composition after bounded quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_plan_compare import parse_router_revert_layers
from nemotron_mlx_router_distill import load_router_artifact
from nemotron_mlx_stream_forward import validate_virtual_plan
from nemotron_mlx_streamed_router_kd import load_source_router
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-router-kd-composed-v1"


def compose_routers(
    trained: dict[str, mx.array],
    source: dict[str, mx.array],
    revert_layers: list[int],
) -> dict[str, mx.array]:
    require(set(trained) == set(source), "trained/source router catalog mismatch")
    require(set(revert_layers) <= {int(layer) for layer in trained}, "router reversion layer is absent")
    result = dict(trained)
    for layer in revert_layers:
        result[str(layer)] = source[str(layer)]
    return result


def save_artifact(path: Path, routers: dict[str, mx.array], metadata: dict[str, str]) -> None:
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    mx.save_safetensors(
        str(temporary),
        {f"layer_{int(layer):03d}.gate.weight": tensor for layer, tensor in routers.items()},
        metadata={"format": FORMAT, **metadata},
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--router-report", required=True, type=Path)
    parser.add_argument("--revert-layers", required=True)
    parser.add_argument("--quality-report", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operation_log = OperationLog(args.output_dir / "run.log")
    try:
        require(len(args.quality_report) >= 2, "at least two quality reports are required")
        require(
            len({path.resolve() for path in args.quality_report}) == len(args.quality_report),
            "quality reports must be distinct",
        )
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        retained = validate_virtual_plan(plan, config, source_state["revision"])
        trained, upstream = load_router_artifact(
            args.router_report,
            args.plan,
            source_state["revision"],
            retained,
            config["hidden_size"],
        )
        revert_layers = parse_router_revert_layers(args.revert_layers)
        require(revert_layers, "at least one reverted router layer is required")
        source = {
            layer: load_source_router(args.source_dir, int(layer), retained[layer])
            for layer in retained
        }
        routers = compose_routers(trained, source, revert_layers)
        quality_rows = []
        for path in args.quality_report:
            report = load_json(path)
            require(report.get("format") == "nemotron-plan-logit-comparison-v1", "invalid quality report")
            require(report.get("source_revision") == source_state["revision"], "quality/source mismatch")
            require(report.get("nonuniform_plan_sha256") == sha256_file(args.plan), "quality/plan mismatch")
            require(
                report.get("nonuniform_router_artifact_sha256") == upstream["artifact_sha256"],
                "quality/router mismatch",
            )
            require(report.get("nonuniform_router_revert_layers") == revert_layers, "quality reversion mismatch")
            require(report.get("cases"), "quality report has no cases")
            quality_rows.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "cases": len(report["cases"]),
                    "summary": report["summary"]["nonuniform"],
                }
            )

        artifact = args.output_dir / "router.safetensors"
        tool_hash = sha256_file(Path(__file__))
        save_artifact(
            artifact,
            routers,
            {
                "source_revision": source_state["revision"],
                "plan_sha256": sha256_file(args.plan),
                "upstream_router_sha256": upstream["artifact_sha256"],
                "revert_layers": ",".join(str(layer) for layer in revert_layers),
                "tool_sha256": tool_hash,
            },
        )
        loaded, metadata = mx.load(str(artifact), return_metadata=True)
        expected_names = {f"layer_{int(layer):03d}.gate.weight" for layer in routers}
        require(set(loaded) == expected_names, "exported router catalog mismatch")
        exact_layers = []
        for layer, expected in routers.items():
            actual = loaded[f"layer_{int(layer):03d}.gate.weight"]
            require(actual.dtype == mx.bfloat16 and actual.shape == expected.shape, "exported router shape mismatch")
            exact = np.array_equal(
                np.asarray(actual.astype(mx.float32)),
                np.asarray(expected.astype(mx.float32)),
            )
            require(exact, f"exported router layer {layer} changed")
            exact_layers.append(int(layer))
        require(metadata.get("format") == FORMAT, "exported router metadata mismatch")
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "plan_sha256": sha256_file(args.plan),
            "upstream_router_report_sha256": sha256_file(args.router_report),
            "upstream_router_artifact_sha256": upstream["artifact_sha256"],
            "revert_layers": revert_layers,
            "trained_layers": sorted(set(int(layer) for layer in retained) - set(revert_layers)),
            "quality_reports": quality_rows,
            "tool_sha256": tool_hash,
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "exact_layers": sorted(exact_layers),
        }
        report_path = args.output_dir / "report.json"
        atomic_json(report_path, report)
        operation_log.write(
            f"run-complete trained_layers={len(report['trained_layers'])} "
            f"reverted_layers={len(revert_layers)} artifact_sha256={report['artifact_sha256']}"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        operation_log.write(f"run-failed error={exc}")
        print(f"nemotron Router KD export error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
