#!/usr/bin/env python3
"""Run resumable, sandboxed LiveCodeBench on a resident Nemotron candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mbpp import extract_code, generate
from nemotron_livecodebench_private_index import (
    FORMAT as PRIVATE_INDEX_FORMAT,
    read_indexed_row,
    validate_index_files,
)
from nemotron_mlx_resident import ResidentModel, preflight
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-livecodebench-v1"
DEFAULT_EXEC_TIMEOUT = 6
DATED_FORMAT = "nemotron-livecodebench-dated-v1"
PRIVATE_FORMAT = "nemotron-livecodebench-private-v1"
OFFICIAL_SPLITS = {
    ("release_v5", "2024-07-01", "2024-12-31"),
    ("release_v6", "2024-08-01", "2025-05-31"),
}
OFFICIAL_PRELUDE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from functools import *
from itertools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""


def split_reasoning(response: str, enabled: bool) -> tuple[str, str]:
    if not enabled:
        return "", response
    reasoning, marker, final = response.partition("</think>")
    return (reasoning.strip(), final.strip()) if marker else (response.strip(), "")


def nvidia_protocol_mismatches(
    args: argparse.Namespace,
    dataset_state: dict | None = None,
    private_tests: bool = False,
) -> list[str]:
    mismatches = []
    if not args.enable_thinking:
        mismatches.append("thinking_disabled")
    if args.protocol_profile == "standard" and args.low_effort:
        mismatches.append("low_effort_enabled")
    if args.protocol_profile == "low-budget" and not args.low_effort:
        mismatches.append("low_effort_disabled")
    if args.temperature != 1.0:
        mismatches.append("temperature_not_1.0")
    if args.top_p != 0.95:
        mismatches.append("top_p_not_0.95")
    if args.repeats != 8:
        mismatches.append("repeats_not_8")
    if args.max_new_tokens != 131072:
        mismatches.append("max_new_tokens_not_131072")
    if args.max_public_cases != 0:
        mismatches.append("public_case_limit_enabled")
    if dataset_state is None:
        mismatches.append("official_dated_split_unverified")
    else:
        split = (
            dataset_state.get("config"),
            dataset_state.get("start_date"),
            dataset_state.get("end_date"),
        )
        if split not in OFFICIAL_SPLITS:
            mismatches.append("nonstandard_dated_split")
        if not private_tests:
            mismatches.append("public_tests_only")
    return mismatches


def summarize_results(results: list[dict], total_tasks: int) -> dict:
    passed = sum(row["passed"] for row in results)
    tasks_with_pass = len({row["task_id"] for row in results if row["passed"]})
    samples = len(results)
    summary = {
        "completed_samples": samples,
        "passed_samples": passed,
        "sample_pass_at_1": passed / samples,
        "tasks_with_pass": tasks_with_pass,
        "task_pass_any": tasks_with_pass / total_tasks,
        "truncated_samples": sum(row["truncated"] for row in results),
        "generation_seconds": sum(row["generation_seconds"] for row in results),
        "generated_tokens": sum(row["generated_tokens"] for row in results),
    }
    for key, output_key in (
        ("difficulty", "by_difficulty"),
        ("test_type", "by_test_type"),
    ):
        values = sorted({str(row.get(key, "")) for row in results})
        summary[output_key] = {
            value: {
                "samples": sum(str(row.get(key, "")) == value for row in results),
                "passed": sum(
                    row["passed"] and str(row.get(key, "")) == value
                    for row in results
                ),
            }
            for value in values
        }
    return summary


def load_items(path: Path) -> list[dict]:
    items = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cases = item.get("public_test_cases", [])
        if isinstance(cases, str):
            cases = json.loads(cases)
        if not isinstance(cases, list) or not cases:
            continue
        if not all(case.get("testtype") in ("stdin", "functional") for case in cases):
            continue
        item["public_test_cases"] = cases
        items.append(item)
    require(items, "LiveCodeBench dataset contains no stdin problems")
    items.sort(key=lambda item: hashlib.sha256(str(item["question_id"]).encode()).digest())
    return items


def deterministic_items(path: Path, count: int, offset: int) -> list[dict]:
    items = load_items(path)
    require(0 <= offset < len(items), "sample offset exceeds LiveCodeBench dataset")
    require(count > 0 and offset + count <= len(items), "sample range exceeds LiveCodeBench dataset")
    return items[offset : offset + count]


def stratified_items(path: Path, count: int, offset: int) -> list[dict]:
    require(count > 0 and offset >= 0, "invalid stratified sample range")
    strata = {difficulty: [] for difficulty in ("easy", "medium", "hard")}
    for item in load_items(path):
        difficulty = str(item.get("difficulty", "")).lower()
        if difficulty in strata:
            strata[difficulty].append(item)
    for difficulty, rows in strata.items():
        require(
            offset + count <= len(rows),
            f"insufficient {difficulty} LiveCodeBench rows for requested stratum",
        )
    selected = []
    for index in range(offset, offset + count):
        selected.extend(strata[difficulty][index] for difficulty in ("easy", "medium", "hard"))
    return selected


def attach_private_tests(
    items: list[dict],
    private_dir: Path,
    private_state_path: Path,
    private_index_path: Path,
    dataset_state_path: Path,
) -> tuple[dict, dict]:
    private_state = load_json(private_state_path)
    require(private_state.get("format") == PRIVATE_FORMAT, "unsupported private state")
    require(private_state.get("status") == "complete", "private tests are incomplete")
    require(
        private_state.get("public_state_sha256") == sha256_file(dataset_state_path),
        "private tests are bound to a different public state",
    )
    private_index = load_json(private_index_path)
    require(private_index.get("format") == PRIVATE_INDEX_FORMAT, "unsupported private index")
    require(private_index.get("status") == "complete", "private index is incomplete")
    require(
        private_index.get("private_state_sha256") == sha256_file(private_state_path),
        "private index state mismatch",
    )
    require(
        Path(private_index["private_dir"]).resolve() == private_dir.resolve(),
        "private index directory mismatch",
    )
    validate_index_files(private_dir, private_index)
    for item in items:
        private_row = read_indexed_row(
            private_dir, private_index, str(item["question_id"])
        )
        item["private_test_cases"] = private_row["private_test_cases"]
    return private_state, private_index


def prompt_for(item: dict) -> str:
    starter = item.get("starter_code", "").strip()
    if starter:
        starter_message = (
            "\n\nSolve the problem starting with the provided function header.\n\n"
            f"Function header:\n```\n{starter}\n```"
        )
        formatting = (
            "Please place the solution code in the following format:\n"
            "```python\n# Your solution code here\n```"
        )
    else:
        starter_message = ""
        formatting = (
            "Write Python code to solve the problem. Please place the solution code in "
            "the following format:\n```python\n# Your solution code here\n```"
        )
    question = f"{item['question_content']}{starter_message}\n\n{formatting}"
    return (
        f"### Question:\n{question}\n\n"
        "### Answer: (use the provided format with backticks)"
    )


def normalize_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _limits(timeout: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 2, timeout + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2**20, 8 * 2**20))
    resource.setrlimit(resource.RLIMIT_NPROC, (4, 4))


def execute_program(
    code: str,
    stdin_input: str,
    work_root: Path,
    python: Path,
    timeout: int = DEFAULT_EXEC_TIMEOUT,
) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory(prefix="lcb-", dir=work_root) as temporary:
        task_dir = Path(temporary).resolve()
        script = task_dir / "candidate.py"
        script.write_text(code + "\n")
        profile = (
            "(version 1) (deny default) (allow process-exec) (allow file-read*) "
            f'(allow file-write* (subpath "{task_dir}"))'
        )
        try:
            result = subprocess.run(
                ["/usr/bin/sandbox-exec", "-p", profile, str(python), str(script)],
                input=stdin_input,
                cwd=task_dir,
                env={
                    "PATH": str(python.parent),
                    "HOME": str(task_dir),
                    "TMPDIR": str(task_dir),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "LANG": "en_US.UTF-8",
                },
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=lambda: _limits(timeout),
            )
            if result.returncode == 0:
                return True, result.stdout, ""
            return False, result.stdout, (result.stderr or result.stdout)[:1000]
        except subprocess.TimeoutExpired:
            return False, "", "execution timed out"


def execute_functional(
    code: str,
    case: dict,
    function_name: str,
    work_root: Path,
    python: Path,
    timeout: int = DEFAULT_EXEC_TIMEOUT,
) -> tuple[bool, str, str]:
    try:
        arguments = [json.loads(line) for line in str(case.get("input", "")).splitlines()]
    except json.JSONDecodeError as exc:
        return False, "", f"invalid functional input: {exc}"
    harness = (
        OFFICIAL_PRELUDE
        + "\n"
        + code
        + "\n\n"
        + "import json as _json\n"
        + f"_arguments = _json.loads({json.dumps(json.dumps(arguments))})\n"
        + f"_result = Solution().{function_name}(*_arguments)\n"
        + "print(_json.dumps(_result, ensure_ascii=True, separators=(',', ':')))\n"
    )
    return execute_program(harness, "", work_root, python, timeout)


def check_cases(
    code: str,
    item: dict,
    work_root: Path,
    python: Path,
    max_cases: int = 0,
    timeout: int = DEFAULT_EXEC_TIMEOUT,
) -> tuple[bool, str, int]:
    public_cases = (
        item["public_test_cases"][:max_cases]
        if max_cases
        else item["public_test_cases"]
    )
    cases = public_cases + item.get("private_test_cases", [])
    for index, case in enumerate(cases):
        if case.get("testtype") == "functional":
            metadata = item.get("metadata", {})
            function_name = metadata.get("func_name") if isinstance(metadata, dict) else None
            if not isinstance(function_name, str) or not function_name:
                return False, f"case {index}: missing functional method name", index + 1
            success, stdout, error = execute_functional(
                code, case, function_name, work_root, python, timeout
            )
        else:
            success, stdout, error = execute_program(
                OFFICIAL_PRELUDE + "\n" + code,
                str(case.get("input", "")),
                work_root,
                python,
                timeout,
            )
        if not success:
            return False, f"case {index}: {error}", index + 1
        if case.get("testtype") == "functional":
            try:
                actual = json.loads(stdout)
                expected = json.loads(str(case.get("output", "")))
            except json.JSONDecodeError as exc:
                return False, f"case {index}: invalid functional output: {exc}", index + 1
        else:
            actual = normalize_output(stdout)
            expected = normalize_output(str(case.get("output", "")))
        if actual != expected:
            return (
                False,
                f"case {index}: expected={repr(expected)[:300]} "
                f"actual={repr(actual)[:300]}",
                index + 1,
            )
    return True, "", len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-state", type=Path)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--private-state", type=Path)
    parser.add_argument("--private-index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--samples-per-difficulty",
        type=int,
        default=0,
        help="Select this many each of easy, medium, and hard, interleaved deterministically",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-public-cases",
        type=int,
        default=0,
        help="Maximum public cases per task; zero runs every available case",
    )
    parser.add_argument("--execution-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--low-effort", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--protocol-profile",
        choices=("standard", "low-budget"),
        default="standard",
    )
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(
            args.max_new_tokens > 0
            and args.max_public_cases >= 0
            and args.execution_timeout > 0
            and args.repeats > 0
            and 0 <= args.temperature
            and 0 <= args.top_p <= 1,
            "invalid evaluation limits",
        )
        require(not args.low_effort or args.enable_thinking, "low effort requires thinking")
        require(shutil.which("sandbox-exec") is not None, "sandbox-exec is required")
        require(args.python.is_file(), "sandbox Python interpreter is unavailable")
        dataset_state = None
        if args.dataset_state is not None:
            dataset_state = load_json(args.dataset_state)
            require(dataset_state.get("format") == DATED_FORMAT, "unsupported dated dataset state")
            require(dataset_state.get("status") == "complete", "dated dataset is incomplete")
            require(
                dataset_state.get("output_sha256") == sha256_file(args.dataset),
                "dated dataset hash mismatch",
            )
        if args.samples_per_difficulty:
            items = stratified_items(
                args.dataset, args.samples_per_difficulty, args.sample_offset
            )
            sampling = {
                "mode": "stratified",
                "samples_per_difficulty": args.samples_per_difficulty,
                "offset_per_difficulty": args.sample_offset,
            }
        else:
            items = deterministic_items(args.dataset, args.sample_size, args.sample_offset)
            sampling = {
                "mode": "global",
                "sample_size": args.sample_size,
                "sample_offset": args.sample_offset,
            }
        private_paths = (args.private_dir, args.private_state, args.private_index)
        require(
            all(path is None for path in private_paths)
            or all(path is not None for path in private_paths),
            "private-dir, private-state, and private-index must be supplied together",
        )
        private_state = None
        private_index = None
        if args.private_dir is not None:
            require(dataset_state is not None, "private tests require dated dataset state")
            private_state, private_index = attach_private_tests(
                items,
                args.private_dir,
                args.private_state,
                args.private_index,
                args.dataset_state,
            )
        report_path = args.model_dir / "nemotron_mlx_pack_report.json"
        runtime_files = [
            args.model_dir / name
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        ]
        helper_path = Path(__file__).with_name("nemotron_mlx_mbpp.py")
        require(all(path.is_file() for path in runtime_files), "candidate runtime metadata is incomplete")
        identity = {
            "format": FORMAT,
            "evaluator_sha256": sha256_file(Path(__file__)),
            "generation_helper_sha256": sha256_file(helper_path),
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(report_path),
            "runtime_file_sha256": {path.name: sha256_file(path) for path in runtime_files},
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_state_sha256": (
                sha256_file(args.dataset_state) if args.dataset_state is not None else None
            ),
            "private_state_sha256": (
                sha256_file(args.private_state) if args.private_state is not None else None
            ),
            "private_index_sha256": (
                sha256_file(args.private_index) if args.private_index is not None else None
            ),
            "sandbox_python": str(args.python.resolve()),
            "sandbox_python_version": subprocess.check_output(
                [str(args.python.resolve()), "--version"], text=True
            ).strip(),
            "sampling": sampling,
            "task_ids": [str(item["question_id"]) for item in items],
            "max_new_tokens": args.max_new_tokens,
            "max_public_cases": args.max_public_cases,
            "execution_timeout": args.execution_timeout,
            "generation": {
                "enable_thinking": args.enable_thinking,
                "low_effort": args.low_effort,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repeats": args.repeats,
                "seed": args.seed,
                "protocol_profile": args.protocol_profile,
            },
            "nvidia_reference_protocol": {
                "matches_reference_protocol": not nvidia_protocol_mismatches(
                    args, dataset_state, private_index is not None
                ),
                "mismatches": nvidia_protocol_mismatches(
                    args, dataset_state, private_index is not None
                ),
            },
        }
        memory = preflight(args.model_dir, args.margin_gib, paged_embeddings=True)
        require(memory["safe_to_attempt"], "resident preflight failed")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "sampling": sampling,
                        "tasks": len(items),
                        "difficulty_counts": {
                            difficulty: sum(
                                str(item.get("difficulty", "")).lower() == difficulty
                                for item in items
                            )
                            for difficulty in ("easy", "medium", "hard")
                        },
                        "task_ids": identity["task_ids"],
                        "required_gib": memory["required_gib"],
                        "effective_cap_gib": memory["effective_cap_gib"],
                        "nvidia_reference_protocol": identity[
                            "nvidia_reference_protocol"
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            report = load_json(args.output)
            for key, value in identity.items():
                require(report.get(key) == value, f"LiveCodeBench report identity mismatch: {key}")
        else:
            report = {**identity, "status": "running", "results": []}
            atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        completed = {(row["task_id"], row.get("repeat", 0)) for row in report["results"]}
        mx.set_wired_limit(memory["effective_cap_bytes"])
        mx.set_cache_limit(256 * 2**20)
        operation_log.write(
            f"run-start completed={len(completed)} total={len(items)} required_gib={memory['required_gib']:.3f}"
        )
        model = ResidentModel(args.model_dir, paged_embeddings=True, embedding_cache_rows=args.embedding_cache_rows)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        sandbox_root = args.output.parent / ".livecodebench-sandbox"
        sandbox_root.mkdir(exist_ok=True)
        for item in items:
            task_id = str(item["question_id"])
            for repeat in range(args.repeats):
                if (task_id, repeat) in completed:
                    continue
                sample_seed = int.from_bytes(
                    hashlib.sha256(f"{args.seed}:{task_id}:{repeat}".encode()).digest()[:4],
                    "little",
                )
                operation_log.write(
                    f"task-start task_id={task_id} repeat={repeat} difficulty={item.get('difficulty', '')} seed={sample_seed}"
                )
                response, generated_tokens, elapsed = generate(
                    model,
                    tokenizer,
                    prompt_for(item),
                    args.max_new_tokens,
                    assistant_prefix=None if args.enable_thinking else "```python\n",
                    enable_thinking=args.enable_thinking,
                    low_effort=args.low_effort,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=sample_seed,
                )
                reasoning, final_response = split_reasoning(response, args.enable_thinking)
                code = extract_code(final_response)
                passed, error, cases_run = check_cases(
                    code,
                    item,
                    sandbox_root,
                    args.python.resolve(),
                    args.max_public_cases,
                    args.execution_timeout,
                )
                truncated = generated_tokens == args.max_new_tokens
                report["results"].append(
                    {
                        "task_id": task_id,
                        "repeat": repeat,
                        "seed": sample_seed,
                        "difficulty": item.get("difficulty", ""),
                        "test_type": item["public_test_cases"][0]["testtype"],
                        "contest_date": item.get("contest_date"),
                        "passed": passed,
                        "truncated": truncated,
                        "cases_run": cases_run,
                        "generated_tokens": generated_tokens,
                        "generation_seconds": elapsed,
                        "response": final_response,
                        "reasoning": reasoning,
                        "code": code,
                        "error": error,
                    }
                )
                report["summary"] = summarize_results(report["results"], len(items))
                atomic_json(args.output, report)
                operation_log.write(
                    f"task-done task_id={task_id} repeat={repeat} passed={passed} cases={cases_run} "
                    f"tokens={generated_tokens} truncated={truncated} elapsed={elapsed:.2f}s "
                    f"error={error[:120]!r}"
                )
        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write(
            "run-complete "
            f"passed={report['summary']['passed_samples']}/"
            f"{report['summary']['completed_samples']}"
        )
        print(json.dumps(report["summary"], sort_keys=True))
        print(f"livecodebench-report path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron LiveCodeBench error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
