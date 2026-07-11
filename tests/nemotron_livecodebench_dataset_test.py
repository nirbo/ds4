#!/usr/bin/env python3
"""Tests for official dated LiveCodeBench dataset validation."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_livecodebench_dataset import (  # noqa: E402
    dated_rows,
    join_official_rows,
    validate_manifest,
)
from nemotron_metadata import MetadataError  # noqa: E402


def rows() -> tuple[list[dict], list[dict]]:
    local = [
        {
            "question_id": "a",
            "question_title": "Title",
            "question_content": "Question",
            "starter_code": "",
            "difficulty": "hard",
            "public_test_cases": '[{"input":"1","output":"2","testtype":"stdin"}]',
        }
    ]
    official = [
        {
            **local[0],
            "public_test_cases": '[{"input":"1","output":"2","testtype":"stdin"}]',
            "contest_date": "2025-01-15T00:00:00",
            "platform": "atcoder",
            "contest_id": "x",
            "metadata": "{}",
            "_official_shard": "part.parquet",
            "_official_row": 3,
            "_official_row_group": 0,
        }
    ]
    return official, local


class LiveCodeBenchDatasetTest(unittest.TestCase):
    def test_manifest_requires_range_verification_hashes(self) -> None:
        manifest = {
            "format": "nemotron-livecodebench-parquet-manifest-v1",
            "repository": "owner/dataset",
            "revision": "a" * 40,
            "files": [
                {
                    "path": "release/test.parquet",
                    "size": 9,
                    "lfs_sha256": "b" * 64,
                    "xet_hash": "c" * 64,
                }
            ],
        }
        self.assertEqual(validate_manifest(manifest), manifest["files"])
        del manifest["files"][0]["xet_hash"]
        with self.assertRaisesRegex(MetadataError, "Xet hash"):
            validate_manifest(manifest)

    def test_joins_byte_equivalent_public_dataset(self) -> None:
        official, local = rows()
        joined = join_official_rows(official, local)
        self.assertEqual(joined[0]["contest_date"], "2025-01-15T00:00:00")
        self.assertEqual(joined[0]["official_row"], 3)

    def test_rejects_changed_problem_content(self) -> None:
        official, local = rows()
        local[0]["question_content"] = "changed"
        with self.assertRaisesRegex(MetadataError, "question_content"):
            join_official_rows(official, local)

    def test_allows_newer_local_release_rows(self) -> None:
        official, local = rows()
        local.append({**local[0], "question_id": "newer"})
        self.assertEqual(len(join_official_rows(official, local)), 1)

    def test_selects_inclusive_date_window(self) -> None:
        official, local = rows()
        joined = join_official_rows(official, local)
        selected = dated_rows(
            joined, datetime(2025, 1, 15), datetime(2025, 1, 15, 23, 59)
        )
        self.assertEqual([row["question_id"] for row in selected], ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
