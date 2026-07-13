#!/usr/bin/env python3
"""Evaluate a recursive MTP hidden adapter on an independent target trace."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_linear import ModelOptBF16Linear
from nemotron_mlx_mtp import NemotronMTPSidecar, load_indexed_tensors
from nemotron_mlx_mtp_hidden_adapter import FORMAT as ADAPTER_FORMAT
from nemotron_mlx_mtp_hidden_adapter import recursive_metrics
from nemotron_prune_materialize import atomic_json, sha256_file


FORMAT = "nemotron-mtp-hidden-adapter-independent-v1"


def aggregate(left: dict, right: dict) -> dict:
    return {
        "depths": {
            depth: {
                "attempts": left["depths"][depth]["attempts"]
                + right["depths"][depth]["attempts"],
                "matches": left["depths"][depth]["matches"]
                + right["depths"][depth]["matches"],
            }
            for depth in left["depths"]
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--mtp-lm-head", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        adapter_report_path = args.adapter_dir / "report.json"
        adapter_report = load_json(adapter_report_path)
        require(
            adapter_report.get("format") == ADAPTER_FORMAT
            and adapter_report.get("status") == "diagnostic",
            "hidden adapter report is invalid",
        )
        training_head_report = load_json(
            Path(adapter_report["mtp_lm_head"]) / "nemotron_mtp_vocab_head_report.json"
        )
        current_head_report = load_json(
            args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
        )
        require(
            training_head_report.get("artifact_sha256")
            == current_head_report.get("artifact_sha256"),
            "independent MTP token map differs from adapter training",
        )
        artifact = args.adapter_dir / adapter_report["artifact"]
        require(
            artifact.is_file()
            and sha256_file(artifact) == adapter_report.get("artifact_sha256"),
            "hidden adapter artifact hash mismatch",
        )
        adapter, metadata = mx.load(str(artifact), return_metadata=True)
        require(
            metadata.get("format") == ADAPTER_FORMAT and set(adapter) == {"up", "down"},
            "hidden adapter artifact is invalid",
        )
        arrays, trace_metadata = mx.load(str(args.trace), return_metadata=True)
        require(
            trace_metadata.get("format") == "nemotron-mtp-target-trace-v1"
            and trace_metadata.get("model_dir") == str(args.source_dir.resolve()),
            "independent trace does not belong to the target candidate",
        )
        globals_ = load_indexed_tensors(
            args.source_dir,
            {"backbone.embeddings.weight", "lm_head.weight"},
        )
        model = NemotronMTPSidecar(
            args.sidecar,
            globals_["backbone.embeddings.weight"],
            ModelOptBF16Linear(globals_["lm_head.weight"]),
            alternate_lm_head=args.mtp_lm_head,
        )
        prompts = arrays["prompt_indices"].tolist()
        scored = arrays["scored"].tolist()
        zero_up = mx.zeros_like(adapter["up"])
        baseline_parts = [
            recursive_metrics(model, arrays, prompts, scored, zero_up, adapter["down"], 0, parity)
            for parity in (0, 1)
        ]
        adapter_parts = [
            recursive_metrics(
                model,
                arrays,
                prompts,
                scored,
                adapter["up"],
                adapter["down"],
                adapter_report["best"]["epoch"],
                parity,
            )
            for parity in (0, 1)
        ]
        baseline = aggregate(*baseline_parts)
        candidate = aggregate(*adapter_parts)
        for result in (baseline, candidate):
            for depth in result["depths"].values():
                depth["conditional_acceptance"] = depth["matches"] / depth["attempts"]
        improvement = (
            candidate["depths"]["2"]["matches"]
            - baseline["depths"]["2"]["matches"]
        )
        status = "candidate" if improvement > 0 else "rejected"
        report = {
            "format": FORMAT,
            "status": status,
            "source_revision": model.config["nemotron_mtp_runtime"]["source_revision"],
            "source_dir": str(args.source_dir.resolve()),
            "source_report_sha256": sha256_file(args.source_dir / "nemotron_mlx_pack_report.json"),
            "trace": str(args.trace.resolve()),
            "trace_sha256": sha256_file(args.trace),
            "adapter_report_sha256": sha256_file(adapter_report_path),
            "artifact_sha256": sha256_file(artifact),
            "sidecar_report_sha256": sha256_file(args.sidecar / "nemotron_mtp_pack_report.json"),
            "mtp_lm_head_report_sha256": sha256_file(
                args.mtp_lm_head / "nemotron_mtp_vocab_head_report.json"
            ),
            "mtp_lm_head_artifact_sha256": current_head_report["artifact_sha256"],
            "tool_sha256": sha256_file(Path(__file__)),
            "mlx_version": version("mlx"),
            "baseline": baseline,
            "candidate": candidate,
            "depth_two_match_improvement": improvement,
            "decision": "candidate-pending-exact-resident-throughput" if status == "candidate" else "rejected",
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"mtp-hidden-adapter-independent path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        print(f"nemotron MTP hidden adapter evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
