#!/usr/bin/env python3
"""Run resumable, sandboxed HumanEval on a resident Nemotron candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_mbpp import execute_tests, generate
from nemotron_mlx_resident import (
    EXTENDED_RUN_CACHE_MIB,
    ResidentModel,
    preflight,
    require_extended_run,
    require_runtime_headroom,
)
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-humaneval-v1"


def deterministic_items(path: Path, count: int, offset: int) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(
        all(item.get("task_id") and item.get("prompt") and item.get("test") and item.get("entry_point") for item in items),
        "HumanEval dataset contains incomplete rows",
    )
    items.sort(key=lambda item: hashlib.sha256(str(item["task_id"]).encode()).digest())
    require(0 <= offset < len(items), "sample offset exceeds HumanEval dataset")
    require(count > 0 and offset + count <= len(items), "sample range exceeds HumanEval dataset")
    return items[offset : offset + count]


def prompt_for(item: dict) -> str:
    return (
        "Complete the following Python function. Return only the complete function "
        "implementation in one Python code block, with no explanation.\n\n"
        + item["prompt"]
    )


def _prompt_preamble(prompt: str, entry_point: str | None) -> str:
    if entry_point is not None:
        target = re.search(rf"^def\s+{re.escape(entry_point)}\s*\(", prompt, re.MULTILINE)
        if target:
            return prompt[: target.start()].rstrip()
    return "\n".join(
        line for line in prompt.splitlines() if line.startswith(("import ", "from "))
    )


def extract_completion(response: str, prompt: str, entry_point: str | None = None) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    code = blocks[-1].strip("\n") if blocks else response.strip("\n")
    if "def " not in code:
        if code and not code[0].isspace():
            code = "\n".join("    " + line if line else line for line in code.splitlines())
        return prompt + code
    preamble = _prompt_preamble(prompt, entry_point)
    if preamble:
        return preamble + "\n\n" + code
    return code


def human_eval_tests(code: str, item: dict, work_root: Path, python: Path) -> tuple[bool, str]:
    harness = {
        "test_setup_code": "",
        "test_list": [item["test"], f"check({item['entry_point']})"],
    }
    return execute_tests(code, harness, work_root, python)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--allow-high-memory-risk", action="store_true")
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(
            args.max_new_tokens > 0
            and args.embedding_cache_rows >= 0
            and args.prefill_chunk_size > 0,
            "invalid generation limits",
        )
        require(shutil.which("sandbox-exec") is not None, "sandbox-exec is required for generated code")
        require(args.python.is_file(), "sandbox Python interpreter is unavailable")
        items = deterministic_items(args.dataset, args.sample_size, args.sample_offset)
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
            "resident_runtime_sha256": sha256_file(
                Path(__file__).with_name("nemotron_mlx_resident.py")
            ),
            "code_eval_helper_sha256": sha256_file(helper_path),
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(report_path),
            "runtime_file_sha256": {path.name: sha256_file(path) for path in runtime_files},
            "dataset_sha256": sha256_file(args.dataset),
            "sandbox_python": str(args.python.resolve()),
            "sandbox_python_version": subprocess.check_output(
                [str(args.python.resolve()), "--version"], text=True
            ).strip(),
            "sample_size": args.sample_size,
            "sample_offset": args.sample_offset,
            "task_ids": [str(item["task_id"]) for item in items],
            "max_new_tokens": args.max_new_tokens,
            "prefill_chunk_size": args.prefill_chunk_size,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            report = load_json(args.output)
            for key, value in identity.items():
                require(report.get(key) == value, f"HumanEval report identity mismatch: {key}")
        else:
            report = {**identity, "status": "running", "results": []}
            atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        completed = {row["task_id"] for row in report["results"]}
        memory = preflight(args.model_dir, args.margin_gib, paged_embeddings=True)
        require_extended_run(memory, args.allow_high_memory_risk)
        mx.set_wired_limit(memory["effective_cap_bytes"])
        mx.set_cache_limit(EXTENDED_RUN_CACHE_MIB * 2**20)
        operation_log.write(
            f"run-start completed={len(completed)} total={len(items)} required_gib={memory['required_gib']:.3f}"
        )
        model = ResidentModel(
            args.model_dir,
            paged_embeddings=True,
            embedding_cache_rows=args.embedding_cache_rows,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        sandbox_root = args.output.parent / ".humaneval-sandbox"
        sandbox_root.mkdir(exist_ok=True)
        for item in items:
            task_id = str(item["task_id"])
            if task_id in completed:
                continue
            operation_log.write(f"task-start task_id={task_id}")
            response, generated_tokens, elapsed = generate(
                model,
                tokenizer,
                prompt_for(item),
                args.max_new_tokens,
                prefill_chunk_size=args.prefill_chunk_size,
            )
            code = extract_completion(response, item["prompt"], item["entry_point"])
            passed, error = human_eval_tests(code, item, sandbox_root, args.python.resolve())
            report["results"].append(
                {
                    "task_id": task_id,
                    "passed": passed,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": elapsed,
                    "response": response,
                    "code": code,
                    "error": error,
                }
            )
            passed_count = sum(row["passed"] for row in report["results"])
            report["summary"] = {
                "completed": len(report["results"]),
                "passed": passed_count,
                "pass_at_1": passed_count / len(report["results"]),
                "generation_seconds": sum(row["generation_seconds"] for row in report["results"]),
                "generated_tokens": sum(row["generated_tokens"] for row in report["results"]),
            }
            atomic_json(args.output, report)
            operation_log.write(
                f"task-done task_id={task_id} passed={passed} tokens={generated_tokens} "
                f"elapsed={elapsed:.2f}s active_gib={mx.get_active_memory() / 2**30:.3f} "
                f"cache_gib={mx.get_cache_memory() / 2**30:.3f} "
                f"peak_gib={mx.get_peak_memory() / 2**30:.3f} error={error[:120]!r}"
            )
            require_runtime_headroom(memory)
        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write(
            f"run-complete passed={report['summary']['passed']}/{report['summary']['completed']}"
        )
        print(json.dumps(report["summary"], sort_keys=True))
        print(f"humaneval-report path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron HumanEval error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
