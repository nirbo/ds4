#!/usr/bin/env python3
"""Ablate packed K/V precision per attention layer after one shared prefill."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import mlx.core as mx

import ornith35_mlx_cache as persistent_cache
import ornith35_mlx_generate as generate
import ornith35_mlx_model as model
import ornith35_mlx_turboquant_coding_gate as coding_gate
from ornith35_mlx_turboquant_runtime_gate import compare_logits
from ornith35_moe_reference import MoEError, require
from ornith35_nvfp4 import DEFAULT_ROOT
from ornith35_tokenizer import TokenizerError, load_text_tokenizer


FORMAT = "ornith35-turboquant-norm-ablation-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ATTENTION_LAYERS = tuple(
    index
    for index, kind in enumerate(model.PRODUCTION_CONFIG.layer_types)
    if kind == model.LAYER_ATTENTION
)
DEFAULT_CASES = (
    "postgres-online-unique-migration:seed-17",
    "c-tlv-parser-security-review:seed-17",
)


@dataclass(frozen=True)
class AblationCase:
    prompt_name: str
    mode: str
    seed: int | None


@dataclass(frozen=True)
class NormPolicy:
    name: str
    bf16_layers: frozenset[int]
    exact_layers: frozenset[int] = frozenset()
    k8_layers: frozenset[int] = frozenset()


def parse_case(value: str) -> AblationCase:
    try:
        prompt_name, mode = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be PROMPT:MODE") from exc
    if not prompt_name:
        raise argparse.ArgumentTypeError("case prompt name is empty")
    if mode == "greedy":
        return AblationCase(prompt_name, mode, None)
    if not mode.startswith("seed-") or not mode[5:].isdigit():
        raise argparse.ArgumentTypeError("case mode must be greedy or seed-U32")
    seed = int(mode[5:])
    if seed >= 2**32:
        raise argparse.ArgumentTypeError("case seed is outside U32")
    return AblationCase(prompt_name, mode, seed)


def norm_policies() -> tuple[NormPolicy, ...]:
    all_layers = frozenset(ATTENTION_LAYERS)
    return (
        NormPolicy("fp32-all", frozenset()),
        NormPolicy("bf16-all", all_layers),
        *(
            NormPolicy(f"exact-layer-{layer_index}", frozenset(), frozenset((layer_index,)))
            for layer_index in ATTENTION_LAYERS
        ),
        *(
            NormPolicy(f"bf16-layer-{layer_index}", frozenset((layer_index,)))
            for layer_index in ATTENTION_LAYERS
        ),
    )


def parse_exact_layers(value: str) -> NormPolicy:
    try:
        layers = tuple(sorted(int(part) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("exact layers must be comma-separated integers") from exc
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("exact layers must be non-empty and unique")
    if any(layer not in ATTENTION_LAYERS for layer in layers):
        raise argparse.ArgumentTypeError(
            f"exact layers must be selected from {','.join(map(str, ATTENTION_LAYERS))}"
        )
    return NormPolicy(
        f"exact-layers-{'-'.join(map(str, layers))}",
        frozenset(),
        frozenset(layers),
    )


def _parse_layer_indices(value: str, label: str) -> frozenset[int]:
    try:
        layers = tuple(sorted(int(part) for part in value.split(","))) if value else ()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError(f"{label} must be unique")
    if any(layer not in ATTENTION_LAYERS for layer in layers):
        raise argparse.ArgumentTypeError(
            f"{label} must be selected from {','.join(map(str, ATTENTION_LAYERS))}"
        )
    return frozenset(layers)


def parse_layer_policy(value: str) -> NormPolicy:
    fields: dict[str, frozenset[int]] = {}
    for clause in value.split(";"):
        key, separator, raw = clause.partition("=")
        if not separator or key not in ("exact", "k8", "bf16norm"):
            raise argparse.ArgumentTypeError(
                "layer policy uses exact=..., k8=..., and optional bf16norm=..."
            )
        if key in fields:
            raise argparse.ArgumentTypeError(f"layer policy repeats {key}")
        fields[key] = _parse_layer_indices(raw, key)
    exact = fields.get("exact", frozenset())
    k8 = fields.get("k8", frozenset())
    bf16 = fields.get("bf16norm", frozenset())
    if exact & (k8 | bf16):
        raise argparse.ArgumentTypeError("exact layers cannot also select packed precision")
    require(fields, "layer policy must not be empty")
    name = "mixed-" + "-".join(
        (
            f"exact{'-'.join(map(str, sorted(exact))) or 'none'}",
            f"k8-{'-'.join(map(str, sorted(k8))) or 'none'}",
            f"bf16norm-{'-'.join(map(str, sorted(bf16))) or 'none'}",
        )
    )
    return NormPolicy(name, bf16, exact, k8)


def select_policies(
    names: list[str],
    exact_layer_policies: list[NormPolicy] | None = None,
    layer_policies: list[NormPolicy] | None = None,
) -> tuple[NormPolicy, ...]:
    policies = norm_policies()
    custom = tuple(exact_layer_policies or ()) + tuple(layer_policies or ())
    if not names and not custom:
        return policies
    require(len(set(names)) == len(names), "ablation policies must be unique")
    by_name = {policy.name: policy for policy in policies}
    require(all(name in by_name for name in names), "unknown ablation policy")
    selected = tuple(by_name[name] for name in names) + custom
    require(
        len({policy.name for policy in selected}) == len(selected),
        "ablation policies must be unique",
    )
    return selected


def summarize_steps(reports: list[dict[str, float | int | bool]]) -> dict[str, Any]:
    require(reports, "norm ablation produced no comparable steps")
    return {
        "steps": len(reports),
        "top1": sum(int(bool(report["top1"])) for report in reports),
        "top8_recall_mean": statistics.fmean(
            float(report["top8_recall"]) for report in reports
        ),
        "kl_mean": statistics.fmean(float(report["kl"]) for report in reports),
        "kl_max": max(float(report["kl"]) for report in reports),
        "max_abs": max(float(report["max_abs"]) for report in reports),
    }


def aggregate_policy_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    require(cases, "norm policy has no cases")
    steps = sum(int(case["summary"]["steps"]) for case in cases)
    return {
        "cases": len(cases),
        "steps": steps,
        "top1": sum(int(case["summary"]["top1"]) for case in cases),
        "top8_recall_mean": sum(
            float(case["summary"]["top8_recall_mean"])
            * int(case["summary"]["steps"])
            for case in cases
        )
        / steps,
        "kl_mean": sum(
            float(case["summary"]["kl_mean"]) * int(case["summary"]["steps"])
            for case in cases
        )
        / steps,
        "kl_max": max(float(case["summary"]["kl_max"]) for case in cases),
        "material_mismatches": sum(int(case["material_mismatches"]) for case in cases),
    }


def source_trajectory(
    initial: model.TextModelResult | model.TextModelChunkResult,
    session: model.TextLinearDecodeSession,
    weights: model.TextModelWeights,
    eos_token_ids: frozenset[int],
    case: AblationCase,
    steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[tuple[int, ...], tuple[mx.array, ...]]:
    result = initial
    rng = random.Random(case.seed) if case.seed is not None else None
    token_ids: list[int] = []
    logits: list[mx.array] = []
    for _ in range(steps):
        token_id = coding_gate.select_source_token(
            result,
            case.mode,
            rng,
            weights,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if token_id in eos_token_ids:
            break
        token_ids.append(token_id)
        result = model.forward_linear_session_token(token_id, session)
        retained = mx.contiguous(result.logits)
        mx.eval(retained)
        logits.append(retained)
    require(token_ids, f"source trajectory ended immediately for {case.prompt_name}")
    return tuple(token_ids), tuple(logits)


def evaluate_policy(
    policy: NormPolicy,
    token_ids: tuple[int, ...],
    exact_logits: tuple[mx.array, ...],
    source_state: model.TextModelState,
    capacity: int,
    weights: model.TextModelWeights,
    material_margin: float,
) -> dict[str, Any]:
    candidate = model.start_turboquant_decode_session(
        weights,
        source_state,
        capacity,
        bf16_norm_layers=policy.bf16_layers,
        exact_attention_layers=policy.exact_layers,
        k8_attention_layers=policy.k8_layers,
    )
    reports: list[dict[str, float | int | bool]] = []
    mismatches = []
    material_mismatches = 0
    for step, (token_id, source_logits) in enumerate(zip(token_ids, exact_logits, strict=True), 1):
        result = model.forward_turboquant_session_token(token_id, candidate)
        report = compare_logits(source_logits, result.logits)
        reports.append(report)
        if not bool(report["top1"]):
            mismatch = {
                "step": step,
                "source_top": int(report["source_top"]),
                "candidate_top": int(report["candidate_top"]),
                "source_margin": float(report["source_margin"]),
                "kl": float(report["kl"]),
            }
            mismatches.append(mismatch)
            if float(report["source_margin"]) >= material_margin:
                material_mismatches += 1
    summary = summarize_steps(reports)
    del result
    del candidate
    gc.collect()
    mx.clear_cache()
    return {
        "summary": summary,
        "material_mismatches": material_mismatches,
        "mismatches": mismatches,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--case", type=parse_case, action="append", default=[])
    result.add_argument(
        "--policy",
        action="append",
        default=[],
        choices=tuple(policy.name for policy in norm_policies()),
    )
    result.add_argument(
        "--exact-layers",
        type=parse_exact_layers,
        action="append",
        default=[],
        metavar="LAYER[,LAYER...]",
        help="retain an explicit set of full-attention layers as exact BF16 K/V",
    )
    result.add_argument(
        "--layer-policy",
        type=parse_layer_policy,
        action="append",
        default=[],
        metavar="exact=L,...;k8=L,...[;bf16norm=L,...]",
        help="evaluate an explicit mixed exact/K8/K9 layer policy",
    )
    result.add_argument("--prefix-tokens", type=int, default=65_536)
    result.add_argument("--steps", type=int, default=64)
    result.add_argument("--temperature", type=float, default=0.6)
    result.add_argument("--top-k", type=int, default=20)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--chunk", type=int, default=128)
    result.add_argument("--progress-tokens", type=int, default=4096)
    result.add_argument(
        "--prefix-cache-root",
        type=Path,
        help="strictly restore or atomically save the exact shared BF16 prefix",
    )
    result.add_argument("--material-margin", type=float, default=0.5)
    result.add_argument("--report", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        cases = tuple(args.case) if args.case else tuple(parse_case(case) for case in DEFAULT_CASES)
        policies = select_policies(args.policy, args.exact_layers, args.layer_policy)
        require(128 <= args.prefix_tokens < 262_144, "invalid ablation prefix length")
        require(4 <= args.steps <= 256, "invalid ablation trajectory length")
        require(args.chunk in (8, 16, 32, 64, 128), "invalid prefill chunk")
        require(
            args.progress_tokens >= args.chunk and args.progress_tokens % args.chunk == 0,
            "invalid progress interval",
        )
        require(args.temperature > 0.0, "sampling temperature must be positive")
        require(0 < args.top_k <= model.PRODUCTION_CONFIG.vocab_size, "invalid top-k")
        require(0.0 < args.top_p <= 1.0, "invalid top-p")
        require(args.material_margin >= 0.0, "invalid material margin")

        prompts = {
            prompt.name: prompt
            for prompt in coding_gate.load_coding_prompts(coding_gate.DEFAULT_PROMPTS)
        }
        require(
            len({case.prompt_name for case in cases}) == len(cases),
            "ablation prompt cases must be unique",
        )
        require(
            all(case.prompt_name in prompts for case in cases),
            "ablation case names an unknown prompt",
        )
        tokenizer = load_text_tokenizer(args.root)
        prefix = coding_gate.build_long_system_prefix(tokenizer, args.prefix_tokens)
        tails = {
            case.prompt_name: coding_gate.prompt_tail_ids(
                tokenizer,
                prefix,
                prompts[case.prompt_name],
            )
            for case in cases
        }
        capacity = len(prefix.token_ids) + max(map(len, tails.values())) + args.steps
        identity = persistent_cache.production_identity(
            args.root,
            REPOSITORY_ROOT,
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=False,
            quantized_lm_head=False,
            turboquant_kv=True,
        )
        prefix_identity = persistent_cache.production_identity(
            args.root,
            REPOSITORY_ROOT,
            tokenizer_sha256=tokenizer.tokenizer_sha256,
            chat_template_sha256=tokenizer.template_sha256,
            mapped_embedding=False,
            quantized_lm_head=False,
            turboquant_kv=False,
        )
        print(
            "turboquant-norm-ablation-plan "
            f"cases={len(cases)} policies={len(policies)} steps={args.steps} "
            f"prefix_tokens={len(prefix.token_ids)} capacity={capacity} "
            f"runtime_sha256={identity.runtime_sha256} "
            f"prefix_cache_root={args.prefix_cache_root if args.prefix_cache_root is not None else 'disabled'}",
            flush=True,
        )
        weights = model.load_text_model(args.root)
        prefix_setup = coding_gate.prepare_exact_prefix(
            prefix.token_ids,
            weights,
            capacity,
            prefix_identity,
            cache_root=args.prefix_cache_root,
            chunk=args.chunk,
            progress_tokens=args.progress_tokens,
        )
        exact = prefix_setup.session
        prefix_prefill_s = prefix_setup.prefill_s
        prefix_setup_s = prefix_setup.setup_s
        prefix_cache_record = prefix_setup.cache
        del prefix_setup
        base_checkpoint = model.checkpoint_linear_session_state(exact)
        policy_cases: dict[str, list[dict[str, Any]]] = {
            policy.name: [] for policy in policies
        }
        source_cases = []
        for case in cases:
            model.restore_linear_session_checkpoint(exact, base_checkpoint)
            initial, _ = generate.prefill_prompt(
                list(tails[case.prompt_name]),
                exact.state,
                weights,
                max_chunk=args.chunk,
                linear_session=exact,
            )
            prompt_checkpoint = model.checkpoint_linear_session_state(exact)
            token_ids, exact_logits = source_trajectory(
                initial,
                exact,
                weights,
                tokenizer.eos_token_ids,
                case,
                args.steps,
                args.temperature,
                args.top_k,
                args.top_p,
            )
            source_cases.append(
                {
                    **asdict(case),
                    "prompt_tail_tokens": len(tails[case.prompt_name]),
                    "prompt_tail_sha256": coding_gate.token_sha256(tails[case.prompt_name]),
                    "steps": len(token_ids),
                    "token_sha256": coding_gate.token_sha256(token_ids),
                }
            )
            for policy in policies:
                model.restore_linear_session_checkpoint(exact, prompt_checkpoint)
                report = evaluate_policy(
                    policy,
                    token_ids,
                    exact_logits,
                    exact.state,
                    capacity,
                    weights,
                    args.material_margin,
                )
                report.update(asdict(case))
                policy_cases[policy.name].append(report)
                summary = report["summary"]
                print(
                    "turboquant-norm-ablation-case "
                    f"prompt={case.prompt_name} mode={case.mode} policy={policy.name} "
                    f"top1={summary['top1']}/{summary['steps']} "
                    f"top8={summary['top8_recall_mean']:.6f} "
                    f"mean_kl={summary['kl_mean']:.9g} max_kl={summary['kl_max']:.9g} "
                    f"material_mismatches={report['material_mismatches']}",
                    flush=True,
                )
            del exact_logits
            mx.clear_cache()

        policy_reports = {}
        for policy in policies:
            cases_report = policy_cases[policy.name]
            policy_reports[policy.name] = {
                "bf16_layers": sorted(policy.bf16_layers),
                "exact_layers": sorted(policy.exact_layers),
                "k8_layers": sorted(policy.k8_layers),
                "summary": aggregate_policy_cases(cases_report),
                "cases": cases_report,
            }
        report = {
            "format": FORMAT,
            "identity": asdict(identity),
            "tool_sha256": coding_gate.sha256_file(Path(__file__)),
            "coding_gate_sha256": coding_gate.sha256_file(Path(coding_gate.__file__)),
            "configuration": {
                "prefix_tokens": args.prefix_tokens,
                "steps": args.steps,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "chunk": args.chunk,
                "material_margin": args.material_margin,
            },
            "prefix": {
                "tokens": len(prefix.token_ids),
                "token_sha256": coding_gate.token_sha256(prefix.token_ids),
            },
            "source_cases": source_cases,
            "policies": policy_reports,
            "prefix_prefill_s": prefix_prefill_s,
            "prefix_setup_s": prefix_setup_s,
            "prefix_cache": {
                **prefix_cache_record,
                "identity": asdict(prefix_identity),
            },
            "peak_gib": mx.get_peak_memory() / 2**30,
        }
        report_sha256 = coding_gate.atomic_json(args.report, report)
        print(
            f"turboquant-norm-ablation-report path={args.report} sha256={report_sha256}",
            flush=True,
        )
    except (MoEError, OSError, TokenizerError, ValueError) as exc:
        print(f"TurboQuant norm ablation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
