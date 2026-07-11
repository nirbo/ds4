#!/usr/bin/env python3
"""Tests for random-access LiveCodeBench private rows."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_livecodebench_private_index import (  # noqa: E402
    index_file,
    read_indexed_row,
)


class LiveCodeBenchPrivateIndexTest(unittest.TestCase):
    def test_indexes_and_reads_one_row_without_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "part.jsonl"
            rows = [
                {"question_id": "a", "private_test_cases": [{"input": "1"}]},
                {"question_id": "b", "private_test_cases": [{"input": "2"}, {"input": "3"}]},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            entries = {
                key: {"file": path.name, **value}
                for key, value in index_file(path).items()
            }
            index = {"entries": entries}
            self.assertEqual(read_indexed_row(root, index, "b"), rows[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
