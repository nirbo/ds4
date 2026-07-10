#!/usr/bin/env python3
"""Tests for bounded Nemotron prompt/generated-token lookup drafting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))
from nemotron_ngram_lookup import NGramLookup  # noqa: E402


class NGramLookupTest(unittest.TestCase):
    def test_longest_suffix_returns_prior_continuation(self) -> None:
        lookup = NGramLookup(
            [1, 2, 3, 4, 1, 2, 3],
            min_key_tokens=2,
            max_key_tokens=3,
            min_matching_continuations=1,
        )
        draft = lookup.propose(4, 3)
        self.assertIsNotNone(draft)
        self.assertEqual(draft.key_tokens, 3)
        self.assertEqual(draft.source_position, 1)
        self.assertEqual(draft.token_ids, (1, 2, 3))

    def test_generated_tokens_become_available(self) -> None:
        lookup = NGramLookup(
            [7, 8, 9, 10],
            min_key_tokens=2,
            max_key_tokens=2,
            min_matching_continuations=1,
        )
        self.assertIsNone(lookup.propose(7, 2))
        lookup.extend([7, 8, 9])
        self.assertEqual(lookup.propose(10, 2).token_ids, (7, 8))

    def test_tables_and_positions_are_bounded(self) -> None:
        lookup = NGramLookup([], min_key_tokens=2, max_key_tokens=3, max_entries=6)
        lookup.extend(list(range(20)))
        self.assertLessEqual(lookup.entries, 6)
        self.assertTrue(all(len(positions) <= 4 for table in lookup.tables.values() for positions in table.values()))

    def test_does_not_return_a_match_without_known_continuation(self) -> None:
        lookup = NGramLookup([1, 2, 3], min_key_tokens=3, max_key_tokens=3)
        self.assertIsNone(lookup.propose(3, 2))

    def test_requires_matching_continuations(self) -> None:
        lookup = NGramLookup(
            [1, 2, 8, 1, 2, 9, 1, 2, 8, 1, 2],
            min_key_tokens=2,
            max_key_tokens=2,
            min_matching_continuations=2,
        )
        draft = lookup.propose(8, 2)
        self.assertIsNotNone(draft)
        self.assertEqual(draft.token_ids, (1, 2))

    def test_rejects_disagreeing_continuations(self) -> None:
        lookup = NGramLookup(
            [1, 2, 8, 1, 2, 9, 1, 2],
            min_key_tokens=2,
            max_key_tokens=2,
            min_matching_continuations=2,
        )
        self.assertIsNone(lookup.propose(8, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
