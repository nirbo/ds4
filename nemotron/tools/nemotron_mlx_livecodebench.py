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
from nemotron_mlx_resident import ResidentModel, preflight
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-livecodebench-v1"
EXEC_TIMEOUT = 30


def deterministic_items(path: Path, count: int, offset: int) -> list[dict]:
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
        if not all(case.get("testtype") == "stdin" for case in cases):
            continue
        item["public_test_cases"] = cases
        items.append(item)
    require(items, "LiveCodeBench dataset contains no stdin problems")
    items.sort(key=lambda item: hashlib.sha256(str(item["question_id"]).encode()).digest())
    require(0 <= offset < len(items), "sample offset exceeds LiveCodeBench dataset")
    require(count > 0 and offset + count <= len(items), "sample range exceeds LiveCodeBench dataset")
    return items[offset : offset + count]


def prompt_for(item: dict) -> str:
    starter = item.get("starter_code", "").strip()
    suffix = f"\n\nStarter code:\n{starter}" if starter else ""
    return (
        "Solve the following programming problem in Python. Read input from stdin and "
        "print output to stdout. Return only complete Python code in one code block, "
        "with no explanation.\n\n"
        f"Problem:\n{item['question_content']}{suffix}\n\nSolution:"
    )


def normalize_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT + 2, EXEC_TIMEOUT + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2**20, 8 * 2**20))
    resource.setrlimit(resource.RLIMIT_NPROC, (4, 4))


def execute_program(
    code: str,
    stdin_input: str,
    work_root: Path,
    python: Path,
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
                timeout=EXEC_TIMEOUT,
                preexec_fn=_limits,
            )
            if result.returncode == 0:
                return True, result.stdout, ""
            return False, result.stdout, (result.stderr or result.stdout)[:1000]
        except subprocess.TimeoutExpired:
            return False, "", "execution timed out"


def check_cases(
    code: str,
    item: dict,
    work_root: Path,
    python: Path,
    max_cases: int = 3,
) -> tuple[bool, str, int]:
    cases = item["public_test_cases"][:max_cases]
    for index, case in enumerate(cases):
        success, stdout, error = execute_program(code, str(case.get("input", "")), work_root, python)
        if not success:
            return False, f"case {index}: {error}", index + 1
        actual = normalize_output(stdout)
        expected = normalize_output(str(case.get("output", "")))
        if actual != expected:
            return False, f"case {index}: expected={expected[:300]!r} actual={actual[:300]!r}", index + 1
    return True, "", len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-public-cases", type=int, default=3)
    parser.add_argument("--margin-gib", type=float, default=0.5)
    parser.add_argument("--embedding-cache-rows", type=int, default=256)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operation_log = None
    try:
        require(args.max_new_tokens > 0 and args.max_public_cases > 0, "invalid evaluation limits")
        require(shutil.which("sandbox-exec") is not None, "sandbox-exec is required")
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
            "generation_helper_sha256": sha256_file(helper_path),
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
            "task_ids": [str(item["question_id"]) for item in items],
            "max_new_tokens": args.max_new_tokens,
            "max_public_cases": args.max_public_cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            report = load_json(args.output)
            for key, value in identity.items():
                require(report.get(key) == value, f"LiveCodeBench report identity mismatch: {key}")
        else:
            report = {**identity, "status": "running", "results": []}
            atomic_json(args.output, report)
        operation_log = OperationLog(args.output.with_suffix(".log"))
        completed = {row["task_id"] for row in report["results"]}
        memory = preflight(args.model_dir, args.margin_gib, paged_embeddings=True)
        require(memory["safe_to_attempt"], "resident preflight failed")
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
            if task_id in completed:
                continue
            operation_log.write(f"task-start task_id={task_id} difficulty={item.get('difficulty', '')}")
            response, generated_tokens, elapsed = generate(
                model,
                tokenizer,
                prompt_for(item),
                args.max_new_tokens,
                assistant_prefix="```python\n",
            )
            code = extract_code(response)
            passed, error, cases_run = check_cases(code, item, sandbox_root, args.python.resolve(), args.max_public_cases)
            truncated = generated_tokens == args.max_new_tokens
            report["results"].append({
                "task_id": task_id,
                "difficulty": item.get("difficulty", ""),
                "passed": passed,
                "truncated": truncated,
                "cases_run": cases_run,
                "generated_tokens": generated_tokens,
                "generation_seconds": elapsed,
                "response": response,
                "code": code,
                "error": error,
            })
            passed_count = sum(row["passed"] for row in report["results"])
            truncated_count = sum(row["truncated"] for row in report["results"])
            report["summary"] = {
                "completed": len(report["results"]),
                "passed": passed_count,
                "pass_at_1": passed_count / len(report["results"]),
                "truncated": truncated_count,
                "generation_seconds": sum(row["generation_seconds"] for row in report["results"]),
                "generated_tokens": sum(row["generated_tokens"] for row in report["results"]),
            }
            atomic_json(args.output, report)
            operation_log.write(
                f"task-done task_id={task_id} passed={passed} cases={cases_run} "
                f"tokens={generated_tokens} truncated={truncated} elapsed={elapsed:.2f}s "
                f"error={error[:120]!r}"
            )
        report["status"] = "complete"
        atomic_json(args.output, report)
        operation_log.write(f"run-complete passed={report['summary']['passed']}/{report['summary']['completed']}")
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
