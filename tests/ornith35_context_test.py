#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ornith35" / "tools"))

import ornith35_context as context


class ContextProfileTest(unittest.TestCase):
    def test_authoritative_profiles(self) -> None:
        native = context.resolve_profile(context.NATIVE_PROFILE_ID)
        yarn = context.resolve_profile(context.YARN2_PROFILE_ID)
        self.assertEqual(native.max_position_embeddings, 262_144)
        self.assertEqual(native.rope_type, "default")
        self.assertEqual(yarn.max_position_embeddings, 524_288)
        self.assertEqual(yarn.rope_type, "yarn")
        self.assertEqual(yarn.factor, 2.0)
        self.assertEqual(yarn.original_max_position_embeddings, 262_144)
        self.assertEqual(set(context.SUPPORTED_CONTEXT_PROFILES), {native.profile_id, yarn.profile_id})

    def test_native_parameters_match_checkpoint_equation(self) -> None:
        inverse, scaling = context.rope_parameters(
            context.NATIVE_PROFILE_ID,
            64,
            10_000_000.0,
        )
        expected = tuple(
            1.0 / (10_000_000.0 ** (index / 64))
            for index in range(0, 64, 2)
        )
        self.assertEqual(inverse, expected)
        self.assertEqual(scaling, 1.0)

    def test_yarn2_matches_pinned_transformers_equation(self) -> None:
        profile = context.YARN2_PROFILE
        low, high = context.yarn_correction_range(profile, 64, 10_000_000.0)
        self.assertEqual((low, high), (14, 22))
        inverse, scaling = context.rope_parameters(profile, 64, 10_000_000.0)
        native, _ = context.rope_parameters(
            context.NATIVE_PROFILE_ID,
            64,
            10_000_000.0,
        )
        self.assertEqual(inverse[14], native[14])
        self.assertTrue(math.isclose(inverse[18], native[18] * 0.75, rel_tol=1e-15))
        self.assertEqual(inverse[22], native[22] * 0.5)
        self.assertEqual(inverse[-1], native[-1] * 0.5)
        self.assertTrue(
            math.isclose(
                scaling,
                1.0 + 0.1 * math.log(2.0),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )

    def test_context_ranges_are_count_based(self) -> None:
        context.validate_range(context.NATIVE_PROFILE_ID, 262_143, 1)
        context.validate_range(context.YARN2_PROFILE_ID, 524_287, 1)
        with self.assertRaisesRegex(context.ContextError, "native-262k"):
            context.validate_range(context.NATIVE_PROFILE_ID, 262_144, 1)
        with self.assertRaisesRegex(context.ContextError, "yarn2-524k"):
            context.validate_range(context.YARN2_PROFILE_ID, 524_288, 1)

    def test_rejects_unbound_profile_objects(self) -> None:
        forged = context.ContextProfile(
            profile_id=context.YARN2_PROFILE_ID,
            max_position_embeddings=524_288,
            rope_type="yarn",
            factor=2.0,
            original_max_position_embeddings=262_144,
            attention_factor=1.0,
        )
        with self.assertRaisesRegex(context.ContextError, "authoritative"):
            context.resolve_profile(forged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
