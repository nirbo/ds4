#!/usr/bin/env python3
"""Bounded orchestration checks for the Ornith-35 context decode gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_context as context
import ornith35_mlx_generation_context_gate as gate
from ornith35_moe_reference import MoEError


class GenerationContextGateTest(unittest.TestCase):
    def test_case_parser_and_formatter_are_strict(self) -> None:
        case = gate.parse_case("native-test:native-262k:2048")
        self.assertEqual(
            case,
            gate.ContextCase("native-test", context.NATIVE_PROFILE_ID, 2048),
        )
        self.assertEqual(gate.format_case(case), "native-test:native-262k:2048")
        for invalid in (
            "missing-fields",
            "Upper:native-262k:2048",
            "name:unknown:2048",
            "name:native-262k:0",
            "name:native-262k:nope",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                gate.parse_case(invalid)

    def test_case_validation_binds_unique_names_ranges_and_reserve(self) -> None:
        selected = gate.validate_cases(gate.DEFAULT_CASES, warmup=4, rounds=32)
        self.assertEqual(selected, gate.DEFAULT_CASES)
        duplicate = gate.ContextCase("native-2k", context.NATIVE_PROFILE_ID, 4096)
        with self.assertRaisesRegex(MoEError, "names"):
            gate.validate_cases((gate.DEFAULT_CASES[0], duplicate), warmup=4, rounds=32)
        overflow = gate.ContextCase("overflow", context.NATIVE_PROFILE_ID, 262_140)
        with self.assertRaises(context.ContextError):
            gate.validate_cases((overflow,), warmup=4, rounds=32)

    def test_aggregate_and_complete_report_validation_are_deterministic(self) -> None:
        cases = gate.DEFAULT_CASES[:2]
        results = {
            case.name: {
                "active_gib": 20.0 + index,
                "case": case.canonical(),
                "decode_peak_gib": 21.0 + index,
                "finite_hidden": True,
                "finite_logits": True,
                "tokens_s": 50.0 - index * 10,
            }
            for index, case in enumerate(cases)
        }
        aggregate = gate.aggregate_results(cases, results)
        self.assertEqual(
            aggregate,
            {
                "case_count": 2,
                "maximum_active_gib": 21.0,
                "maximum_decode_peak_gib": 22.0,
                "minimum_tokens_s": 40.0,
            },
        )
        identity = {"fixture": True}
        state = {
            "aggregate": aggregate,
            "format": gate.FORMAT,
            "identity": identity,
            "results": results,
            "status": "complete",
        }
        gate.validate_report(state, identity, cases)
        state["results"][cases[0].name]["finite_logits"] = False
        with self.assertRaisesRegex(MoEError, "non-finite"):
            gate.validate_report(state, identity, cases)

    def test_worker_command_reproduces_measurement_settings(self) -> None:
        args = SimpleNamespace(
            foreground_wait_seconds=45.0,
            repo_root=Path("/repo"),
            root=Path("/model"),
            rounds=24,
            token=17,
            warmup=3,
        )
        case = gate.ContextCase("fixture", context.YARN2_PROFILE_ID, 4096)
        command = gate.worker_command(args, case)
        self.assertEqual(command[0], sys.executable)
        self.assertIn("fixture:yarn2-524k:4096", command)
        self.assertEqual(command[command.index("--rounds") + 1], "24")
        self.assertEqual(command[command.index("--warmup") + 1], "3")
        self.assertEqual(command[command.index("--token") + 1], "17")


if __name__ == "__main__":
    unittest.main()
