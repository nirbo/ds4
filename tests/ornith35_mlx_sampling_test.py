#!/usr/bin/env python3
"""Sampling distribution tests for exact Ornith-35 generation."""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import unittest

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith35" / "tools"
sys.path.insert(0, str(TOOLS))

import ornith35_mlx_sampling as sampling


class MLXSamplingTest(unittest.TestCase):
    def test_candidate_distribution_is_stable_normalized_and_top_p_bounded(self) -> None:
        distribution = sampling.candidate_distribution(
            [30, 10, 20],
            [0.0, 4.0, 1.0],
            temperature=1.0,
            top_p=0.95,
        )
        self.assertEqual(distribution.token_ids, (10, 20))
        self.assertAlmostEqual(math.fsum(distribution.probabilities), 1.0)
        self.assertGreater(distribution.probability(10), distribution.probability(20))
        self.assertEqual(distribution.probability(30), 0.0)

    def test_delta_draft_acceptance_and_residual_reconstruct_target_distribution(self) -> None:
        distribution = sampling.TokenDistribution(
            token_ids=(3, 7, 11),
            probabilities=(0.6, 0.3, 0.1),
        )
        draft = 3
        reconstructed = {token_id: 0.0 for token_id in distribution.token_ids}
        reconstructed[draft] = distribution.probability(draft)
        rejection = 1.0 - distribution.probability(draft)
        for token_id in distribution.token_ids:
            if token_id != draft:
                reconstructed[token_id] = (
                    rejection
                    * distribution.probability(token_id)
                    / rejection
                )
        for token_id in distribution.token_ids:
            self.assertAlmostEqual(
                reconstructed[token_id],
                distribution.probability(token_id),
            )

        rng = random.Random(19)
        self.assertIn(distribution.sample_excluding(draft, rng), (7, 11))

    def test_general_draft_acceptance_and_residual_reconstruct_target(self) -> None:
        target = sampling.TokenDistribution((3, 7, 11), (0.6, 0.3, 0.1))
        draft = sampling.TokenDistribution((3, 7, 13), (0.5, 0.2, 0.3))
        residual = sampling.residual_distribution(target, draft)
        overlap = sum(
            min(target.probability(token_id), draft.probability(token_id))
            for token_id in set((*target.token_ids, *draft.token_ids))
        )
        rejection = 1.0 - overlap
        for token_id in target.token_ids:
            accepted = min(
                target.probability(token_id),
                draft.probability(token_id),
            )
            reconstructed = accepted + rejection * residual.probability(token_id)
            self.assertAlmostEqual(reconstructed, target.probability(token_id))
        self.assertAlmostEqual(
            sampling.speculative_acceptance_probability(target, draft, 3),
            1.0,
        )
        self.assertAlmostEqual(
            sampling.speculative_acceptance_probability(target, draft, 13),
            0.0,
        )

    def test_dense_target_distribution_is_seed_deterministic(self) -> None:
        logits = mx.array([0.0, 1.0, 2.0, 3.0], dtype=mx.bfloat16)
        distribution = sampling.target_distribution(
            logits,
            temperature=0.6,
            top_k=3,
            top_p=0.95,
        )
        first = random.Random(23)
        second = random.Random(23)
        self.assertEqual(
            [distribution.sample(first) for _ in range(20)],
            [distribution.sample(second) for _ in range(20)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
