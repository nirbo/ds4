#!/usr/bin/env python3
"""Materialize an exact, stacked BF16 MTP sidecar for resident inference."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mtp import MTP_MOE_PREFIX, mtp_payload_estimate
from nemotron_mlx_pack import (
    append_tensor,
    source_catalog,
    validate_done,
    write_group,
)
from nemotron_prune_materialize import (
    OperationLog,
    atomic_json,
    load_source_state,
    sha256_file,
)


FORMAT = "nemotron-mlx-mtp-sidecar-v1"
PLAN_FORMAT = "nemotron-mtp-expert-plan-v1"
STATE_FORMAT = "nemotron-mlx-mtp-pack-state-v1"


def build_mtp_group(
    catalog: dict[str, dict[str, Any]],
    config: dict[str, Any],
    retained_experts: list[int],
) -> dict[str, dict[str, Any]]:
    experts = config["n_routed_experts"]
    require(
        len(retained_experts) >= config["num_experts_per_tok"]
        and len(set(retained_experts)) == len(retained_experts)
        and all(0 <= expert < experts for expert in retained_experts),
        "invalid retained MTP expert list",
    )
    output: dict[str, dict[str, Any]] = {}
    mixer = f"{MTP_MOE_PREFIX}.mixer"
    expert_prefix = f"{mixer}.experts."
    router_names = {
        f"{mixer}.gate.weight",
        f"{mixer}.gate.e_score_correction_bias",
    }

    for name, source in catalog.items():
        if not name.startswith("mtp.") or name.startswith(expert_prefix):
            continue
        if name in router_names:
            require(source["shape"][0] == experts, f"MTP router shape mismatch: {name}")
            row_bytes = source["size"] // experts
            require(row_bytes * experts == source["size"], f"MTP router row mismatch: {name}")
            segments = [
                {
                    **source,
                    "offset": source["offset"] + expert * row_bytes,
                    "size": row_bytes,
                }
                for expert in retained_experts
            ]
            append_tensor(
                output,
                name,
                source["dtype"],
                [len(retained_experts), *source["shape"][1:]],
                segments,
            )
        else:
            append_tensor(output, name, source["dtype"], source["shape"], [source])

    for projection in ("up_proj", "down_proj"):
        parts = []
        for expert in retained_experts:
            name = f"{expert_prefix}{expert}.{projection}.weight"
            require(name in catalog, f"missing MTP expert tensor: {name}")
            parts.append(catalog[name])
        first = parts[0]
        require(
            all(part["dtype"] == "BF16" and part["shape"] == first["shape"] for part in parts),
            f"MTP {projection} expert metadata mismatch",
        )
        append_tensor(
            output,
            f"{mixer}.switch_mlp.{projection}.weight",
            "BF16",
            [len(parts), *first["shape"]],
            parts,
        )
    require(output, "MTP sidecar has no tensors")
    return output


def sidecar_config(
    config: dict[str, Any],
    retained_experts: list[int],
    source_revision: str,
    plan_sha256: str,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["nemotron_mtp_runtime"] = {
        "format": FORMAT,
        "source_revision": source_revision,
        "plan_sha256": plan_sha256,
        "original_experts": config["n_routed_experts"],
        "retained_experts": len(retained_experts),
        "original_expert_ids": retained_experts,
        "expert_layout": "stacked-bf16",
        "shared_globals": ["backbone.embeddings.weight", "lm_head.weight"],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.source_dir.resolve() != args.output_dir.resolve(), "source and output are identical")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        plan = load_json(args.plan)
        require(plan.get("format") == PLAN_FORMAT, "unsupported MTP expert plan")
        retained = plan.get("budgets", {}).get(str(args.budget))
        require(isinstance(retained, list) and len(retained) == args.budget, "plan budget mismatch")
        plan_digest = sha256_file(args.plan)
        index = load_json(args.source_dir / "model.safetensors.index.json")
        group = build_mtp_group(source_catalog(args.source_dir, index), config, retained)
        projected_payload = sum(tensor["size"] for tensor in group.values())
        require(
            projected_payload == mtp_payload_estimate(config, args.budget),
            "MTP sidecar payload estimate mismatch",
        )
        message = (
            f"mtp-sidecar-projected experts={config['n_routed_experts']}->{args.budget} "
            f"tensors={len(group)} payload_gib={projected_payload / 2**30:.6f}"
        )
        print(message, flush=True)
        if args.dry_run:
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "pack.log")
        identity = {
            "format": STATE_FORMAT,
            "source_dir": str(args.source_dir.resolve()),
            "source_revision": source_state["revision"],
            "plan_sha256": plan_digest,
            "budget": args.budget,
        }
        state_path = args.output_dir / "pack-state.json"
        output_path = args.output_dir / "mtp.safetensors"
        if state_path.exists():
            state = load_json(state_path)
            for key, value in identity.items():
                require(state.get(key) == value, f"MTP pack state identity mismatch: {key}")
        else:
            require(not list(args.output_dir.glob("*.safetensors*")), "MTP output exists without state")
            state = {**identity, "status": "running"}
            atomic_json(state_path, state)

        record = state.get("output")
        if record and record.get("status") == "done":
            validate_done(output_path, record)
            operation_log.write("mtp-sidecar-skip status=verified")
        else:
            require(
                shutil.disk_usage(args.output_dir).free >= projected_payload + 2 * 2**30,
                "insufficient disk with 2 GiB safety margin",
            )
            operation_log.write(
                f"mtp-sidecar-start source_revision={source_state['revision']} "
                f"plan_sha256={plan_digest} budget={args.budget}"
            )
            state["output"] = {"status": "processing"}
            atomic_json(state_path, state)
            record = write_group(output_path, group)
            state["output"] = record
            atomic_json(state_path, state)
            operation_log.write(
                f"mtp-sidecar-written bytes={record['bytes']} "
                f"payload_sha256={record['payload_sha256']}"
            )

        weight_map = {name: output_path.name for name in group}
        atomic_json(
            args.output_dir / "model.safetensors.index.json",
            {"metadata": {"total_size": projected_payload}, "weight_map": weight_map},
        )
        atomic_json(
            args.output_dir / "config.json",
            sidecar_config(config, retained, source_state["revision"], plan_digest),
        )
        report = {
            "format": FORMAT,
            "status": "complete",
            "source_revision": source_state["revision"],
            "source_state_sha256": sha256_file(args.source_state),
            "plan_sha256": plan_digest,
            "budget": args.budget,
            "payload_bytes": projected_payload,
            "payload_gib": projected_payload / 2**30,
            "tensors": len(group),
            "retained_payloads": "byte-identical",
            "router_rows": "exact-source-subset",
        }
        atomic_json(args.output_dir / "nemotron_mtp_pack_report.json", report)
        state["status"] = "complete"
        state["report"] = report
        atomic_json(state_path, state)
        operation_log.write(f"mtp-sidecar-complete payload_gib={report['payload_gib']:.6f}")
        return 0
    except (MetadataError, OSError, ValueError) as exc:
        if operation_log is not None:
            operation_log.write(f"mtp-sidecar-failed error={exc}")
        print(f"nemotron MTP pack error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
