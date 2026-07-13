#!/usr/bin/env python3
"""Run resumable, sandboxed MBPP evaluation on a resident Nemotron candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from nemotron_metadata import MetadataError, load_json, require
from nemotron_mlx_resident import (
    EXTENDED_RUN_CACHE_MIB,
    ResidentModel,
    preflight,
    require_extended_run,
    require_runtime_headroom,
)
from nemotron_prune_materialize import OperationLog, atomic_json, sha256_file


FORMAT = "nemotron-mbpp-eval-v1"
EXEC_TIMEOUT = 15


def extract_code(response: str) -> str:
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    lines = response.strip().splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("def ", "class ", "import ", "from ", "#")):
            return "\n".join(lines[index:]).strip()
    return response.strip()


def deterministic_items(path: Path, count: int, offset: int) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(all(item.get("test_list") for item in items), "MBPP dataset contains untestable rows")
    items.sort(key=lambda item: hashlib.sha256(str(item["task_id"]).encode()).digest())
    require(0 <= offset < len(items), "sample offset exceeds MBPP dataset")
    require(count > 0 and offset + count <= len(items), "sample range exceeds MBPP dataset")
    return items[offset : offset + count]


def prompt_for(item: dict) -> str:
    tests = "\n".join(item["test_list"][:3])
    return (
        "Write a Python function to solve the following problem. Return only the complete "
        "implementation in one Python code block, with no explanation.\n\n"
        f"Problem: {item['prompt']}\n\nTests:\n{tests}\n\nSolution:"
    )


def chat_token_ids(
    tokenizer,
    prompt: str,
    assistant_prefix: str | None = None,
    enable_thinking: bool = False,
    low_effort: bool = False,
) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if assistant_prefix is not None:
        messages.append({"role": "assistant", "content": assistant_prefix})
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=assistant_prefix is None,
        continue_final_message=assistant_prefix is not None,
        enable_thinking=enable_thinking,
        low_effort=low_effort,
    )
    token_ids = encoded if isinstance(encoded, list) else encoded["input_ids"]
    require(
        isinstance(token_ids, list)
        and token_ids
        and all(isinstance(token_id, int) for token_id in token_ids),
        "chat template produced invalid token IDs",
    )
    return token_ids


def single_token_delimiter(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    require(
        len(token_ids) == 1 and tokenizer.decode(token_ids) == text,
        f"expected a single exact token for delimiter {text!r}",
    )
    return token_ids[0]


def advance_completion(
    token: int,
    enable_thinking: bool,
    think_end_token: int,
    fence_token: int,
    thinking_complete: bool,
    fence_count: int,
) -> tuple[bool, int, bool]:
    if enable_thinking and token == think_end_token:
        thinking_complete = True
        fence_count = 0
    elif thinking_complete and token == fence_token:
        fence_count += 1
    return thinking_complete, fence_count, thinking_complete and fence_count >= 2


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT + 2, EXEC_TIMEOUT + 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 2**20, 8 * 2**20))
    resource.setrlimit(resource.RLIMIT_NPROC, (4, 4))


def execute_tests(code: str, item: dict, work_root: Path, python: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="mbpp-", dir=work_root) as temporary:
        task_dir = Path(temporary).resolve()
        script = task_dir / "candidate.py"
        script.write_text(
            item.get("test_setup_code", "") + "\n" + code + "\n" + "\n".join(item["test_list"]) + "\n"
        )
        profile = (
            "(version 1) (deny default) (allow process-exec) (allow file-read*) "
            f'(allow file-write* (subpath "{task_dir}"))'
        )
        try:
            result = subprocess.run(
                ["/usr/bin/sandbox-exec", "-p", profile, str(python), str(script)],
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
                return True, ""
            return False, (result.stderr or result.stdout)[:1000]
        except subprocess.TimeoutExpired:
            return False, "execution timed out"


def generate(
    model: ResidentModel,
    tokenizer,
    prompt: str,
    max_tokens: int,
    assistant_prefix: str | None = None,
    enable_thinking: bool = False,
    low_effort: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.0,
    seed: int = 0,
    prefill_chunk_size: int = 128,
) -> tuple[str, int, float]:
    require(prefill_chunk_size > 0, "prefill chunk size must be positive")
    model.reset()
    token_ids = chat_token_ids(
        tokenizer, prompt, assistant_prefix, enable_thinking, low_effort
    )
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    think_end_token = single_token_delimiter(tokenizer, "</think>")
    fence_token = single_token_delimiter(tokenizer, "```")
    thinking_complete = not enable_thinking
    fence_count = (assistant_prefix or "").count("```")
    started = time.perf_counter()
    next_logits, _ = model.prefill(token_ids, prefill_chunk_size)
    generated = []
    eos = set(tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id])
    for _ in range(max_tokens):
        logprobs = next_logits - mx.logsumexp(next_logits, keepdims=True)
        token = int(sampler(logprobs))
        if token in eos:
            break
        generated.append(token)
        thinking_complete, fence_count, complete = advance_completion(
            token,
            enable_thinking,
            think_end_token,
            fence_token,
            thinking_complete,
            fence_count,
        )
        if complete:
            break
        next_logits = model.logits(token)
    response = (assistant_prefix or "") + tokenizer.decode(generated, skip_special_tokens=True)
    return response, len(generated), time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
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
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        ]
        require(all(path.is_file() for path in runtime_files), "candidate runtime metadata is incomplete")
        identity = {
            "format": FORMAT,
            "evaluator_sha256": sha256_file(Path(__file__)),
            "resident_runtime_sha256": sha256_file(
                Path(__file__).with_name("nemotron_mlx_resident.py")
            ),
            "model_dir": str(args.model_dir.resolve()),
            "model_report_sha256": sha256_file(report_path),
            "runtime_file_sha256": {
                path.name: sha256_file(path) for path in runtime_files
            },
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
                require(report.get(key) == value, f"MBPP report identity mismatch: {key}")
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
        sandbox_root = args.output.parent / ".mbpp-sandbox"
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
            code = extract_code(response)
            passed, error = execute_tests(code, item, sandbox_root, args.python.resolve())
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
        print(f"mbpp-report path={args.output} sha256={sha256_file(args.output)}")
        return 0
    except (MetadataError, OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        if operation_log is not None:
            operation_log.write(f"run-failed error={exc}")
        print(f"nemotron MBPP error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
