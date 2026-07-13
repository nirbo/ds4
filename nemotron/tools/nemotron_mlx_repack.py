#!/usr/bin/env python3
"""Incrementally repack a Nemotron plan over a validated MLX runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_pack import (
    FORMAT as RUNTIME_FORMAT,
    build_groups,
    encode_group,
    finalize,
    read_header,
    source_catalog,
    validate_done,
    validate_pack_plan,
    write_group,
)
from nemotron_paged_embeddings import write_catalog
from nemotron_prune_materialize import OperationLog, atomic_json, load_source_state, sha256_file


FORMAT = "nemotron-mlx-repack-state-v1"


def changed_group_names(
    base: dict[int, dict[int, int]], target: dict[int, dict[int, int]]
) -> set[str]:
    require(set(base) == set(target), "base/target MoE layer mismatch")
    return {
        f"layer-{layer:03d}"
        for layer in sorted(base)
        if base[layer] != target[layer]
    }


def validate_base_runtime(
    base_runtime: Path,
    base_plan_sha256: str,
    source_revision: str,
    omit_mtp: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = base_runtime / "nemotron_mlx_pack_report.json"
    state_path = base_runtime / "pack-state.json"
    report = load_json(report_path)
    state = load_json(state_path)
    require(
        report.get("format") == RUNTIME_FORMAT and report.get("status") == "complete",
        "base runtime report is incomplete",
    )
    require(state.get("format") == "nemotron-mlx-pack-state-v1", "invalid base pack state")
    require(state.get("status") == "complete", "base pack state is incomplete")
    require(report.get("source_revision") == source_revision, "base runtime source revision mismatch")
    require(state.get("source_revision") == source_revision, "base state source revision mismatch")
    require(report.get("plan_sha256") == base_plan_sha256, "base runtime plan mismatch")
    require(state.get("plan_sha256") == base_plan_sha256, "base state plan mismatch")
    require(report.get("mtp_omitted") is omit_mtp, "base runtime MTP policy mismatch")
    require(state.get("omit_mtp") is omit_mtp, "base state MTP policy mismatch")
    require(isinstance(state.get("groups"), dict), "base pack state has no groups")
    return report, state


def group_projected_bytes(tensors: dict[str, dict[str, Any]]) -> int:
    return sum(tensor["size"] for tensor in tensors.values())


def validate_group_schema(path: Path, tensors: dict[str, dict[str, Any]]) -> None:
    actual, _ = read_header(path)
    encoded, _ = encode_group(tensors)
    expected = json.loads(encoded)
    require(actual == expected, f"runtime group schema mismatch: {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--target-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--omit-mtp", action="store_true")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.source_dir.resolve() != args.output_dir.resolve(), "source and output are identical")
        require(args.base_runtime.resolve() != args.output_dir.resolve(), "base and output are identical")
        require(args.max_groups is None or args.max_groups > 0, "max-groups must be positive")
        source_state = load_source_state(args.source_state, args.source_dir)
        config = load_json(args.source_dir / "config.json")
        index = load_json(args.source_dir / "model.safetensors.index.json")
        base_plan = load_json(args.base_plan)
        target_plan = load_json(args.target_plan)
        base_mappings = validate_pack_plan(base_plan, config, source_state["revision"])
        target_mappings = validate_pack_plan(target_plan, config, source_state["revision"])
        base_plan_digest = sha256_file(args.base_plan)
        target_plan_digest = sha256_file(args.target_plan)
        base_report, base_state = validate_base_runtime(
            args.base_runtime,
            base_plan_digest,
            source_state["revision"],
            args.omit_mtp,
        )
        catalog = source_catalog(args.source_dir, index)
        groups = build_groups(catalog, config, target_mappings, args.omit_mtp)
        group_names = ["global", *[f"layer-{layer:03d}" for layer in range(config["num_hidden_layers"])]]
        changed = changed_group_names(base_mappings, target_mappings)
        linked = set(group_names) - changed
        projected_payload = sum(group_projected_bytes(groups[group]) for group in group_names)
        changed_payload = sum(group_projected_bytes(groups[group]) for group in changed)
        base_payload = int(base_report["payload_bytes"])
        payload_delta = projected_payload - base_payload
        message = (
            f"repack-projected groups={len(group_names)} changed={len(changed)} linked={len(linked)} "
            f"payload={projected_payload / 2**30:.4f}GiB delta={payload_delta / 2**30:+.4f}GiB "
            f"additional={changed_payload / 2**30:.4f}GiB "
            f"omit_mtp={args.omit_mtp}"
        )
        print(message, flush=True)
        if args.dry_run:
            return 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        operation_log = OperationLog(args.output_dir / "repack.log")
        state_path = args.output_dir / "repack-state.json"
        identity = {
            "format": FORMAT,
            "tool_sha256": sha256_file(Path(__file__)),
            "source_dir": str(args.source_dir.resolve()),
            "source_revision": source_state["revision"],
            "base_runtime": str(args.base_runtime.resolve()),
            "base_report_sha256": sha256_file(args.base_runtime / "nemotron_mlx_pack_report.json"),
            "base_state_sha256": sha256_file(args.base_runtime / "pack-state.json"),
            "base_plan_sha256": base_plan_digest,
            "target_plan_sha256": target_plan_digest,
            "omit_mtp": args.omit_mtp,
            "changed_groups": sorted(changed),
            "linked_groups": sorted(linked),
        }
        if state_path.exists():
            state = load_json(state_path)
            for key, value in identity.items():
                require(state.get(key) == value, f"repack state identity mismatch: {key}")
        else:
            require(
                not list(args.output_dir.glob("*.safetensors*")),
                "repack output has shards without state",
            )
            state = {**identity, "status": "running", "groups": {}}
            atomic_json(state_path, state)
        operation_log.write(
            f"run-start source_revision={source_state['revision']} base_plan={base_plan_digest} "
            f"target_plan={target_plan_digest}"
        )
        operation_log.write(message)

        remaining = sum(
            group_projected_bytes(groups[group])
            for group in changed
            if state["groups"].get(group, {}).get("status") != "done"
        )
        require(
            shutil.disk_usage(args.output_dir).free >= remaining + 4 * 2**30,
            "insufficient disk with 4 GiB safety margin",
        )

        processed = 0
        for group_name in group_names:
            output_path = args.output_dir / f"{group_name}.safetensors"
            record = state["groups"].get(group_name)
            if record and record.get("status") == "done":
                validate_done(output_path, record)
                operation_log.write(
                    f"group-skip group={group_name} method={record['method']} status=verified"
                )
                continue
            if args.max_groups is not None and processed >= args.max_groups:
                break
            output_path.unlink(missing_ok=True)
            state["groups"][group_name] = {"status": "processing"}
            atomic_json(state_path, state)
            if group_name in linked:
                base_record = base_state["groups"].get(group_name)
                require(
                    isinstance(base_record, dict) and base_record.get("status") == "done",
                    f"base group is incomplete: {group_name}",
                )
                base_path = args.base_runtime / output_path.name
                validate_done(base_path, base_record)
                os.link(base_path, output_path)
                result = {**base_record, "method": "hardlink"}
            else:
                result = {**write_group(output_path, groups[group_name]), "method": "rewrite"}
            validate_done(output_path, result)
            validate_group_schema(output_path, groups[group_name])
            state["groups"][group_name] = result
            atomic_json(state_path, state)
            processed += 1
            operation_log.write(
                f"group-done group={group_name} method={result['method']} "
                f"bytes={result['bytes']} sha256={result['payload_sha256']}"
            )

        if all(state["groups"].get(group, {}).get("status") == "done" for group in group_names):
            experts_by_layer = {layer: len(mapping) for layer, mapping in target_mappings.items()}
            report = finalize(
                args.source_dir,
                args.output_dir,
                group_names,
                groups,
                config,
                experts_by_layer,
                args.omit_mtp,
                source_state["revision"],
                target_plan_digest,
                None,
                None,
                identity["tool_sha256"],
            )
            report.update(
                {
                    "materialization": "incremental-plan-repack",
                    "base_runtime": str(args.base_runtime.resolve()),
                    "base_report_sha256": sha256_file(
                        args.base_runtime / "nemotron_mlx_pack_report.json"
                    ),
                    "base_plan_sha256": base_plan_digest,
                    "changed_groups": sorted(changed),
                    "linked_groups": sorted(linked),
                    "additional_payload_bytes": changed_payload,
                    "payload_delta_bytes": payload_delta,
                }
            )
            atomic_json(args.output_dir / "nemotron_mlx_pack_report.json", report)
            pack_state = {
                "format": "nemotron-mlx-pack-state-v1",
                "tool_sha256": identity["tool_sha256"],
                "source_dir": str(args.source_dir.resolve()),
                "source_revision": source_state["revision"],
                "plan_sha256": target_plan_digest,
                "experts_by_layer": {
                    str(layer): count for layer, count in sorted(experts_by_layer.items())
                },
                "omit_mtp": args.omit_mtp,
                "router_report_sha256": None,
                "router_artifact_sha256": None,
                "status": "complete",
                "groups": state["groups"],
                "report": report,
            }
            atomic_json(args.output_dir / "pack-state.json", pack_state)
            write_catalog(args.output_dir)
            state["status"] = "complete"
            state["report"] = report
            atomic_json(state_path, state)
            operation_log.write(
                f"run-complete payload_gib={report['payload_gib']:.4f} "
                f"additional_gib={changed_payload / 2**30:.4f} validation=passed"
            )
            print(
                f"repack-complete output={args.output_dir} payload={report['payload_gib']:.4f}GiB "
                f"additional={changed_payload / 2**30:.4f}GiB",
                flush=True,
            )
        else:
            operation_log.write(f"run-paused processed={processed}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron MLX repack error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
