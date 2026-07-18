#!/usr/bin/env python3
"""Exact bounded sampling distributions for the Ornith-35 target runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import mlx.core as mx

import ornith35_mlx_vocab as vocab
from ornith35_moe_reference import require


@dataclass(frozen=True)
class TokenDistribution:
    token_ids: tuple[int, ...]
    probabilities: tuple[float, ...]

    def probability(self, token_id: int) -> float:
        for candidate, probability in zip(self.token_ids, self.probabilities):
            if candidate == token_id:
                return probability
        return 0.0

    def sample(self, rng: random.Random) -> int:
        return _sample_weighted(self.token_ids, self.probabilities, rng)

    def sample_excluding(self, token_id: int, rng: random.Random) -> int:
        retained = [
            (candidate, probability)
            for candidate, probability in zip(self.token_ids, self.probabilities)
            if candidate != token_id
        ]
        require(retained, "sampled residual distribution is empty")
        total = math.fsum(probability for _, probability in retained)
        require(total > 0.0, "sampled residual distribution has no mass")
        return _sample_weighted(
            tuple(candidate for candidate, _ in retained),
            tuple(probability / total for _, probability in retained),
            rng,
        )


def validate_distribution(distribution: TokenDistribution) -> None:
    require(isinstance(distribution, TokenDistribution), "invalid token distribution")
    require(
        len(distribution.token_ids) == len(distribution.probabilities) > 0,
        "sampling distribution is empty",
    )
    require(
        len(set(distribution.token_ids)) == len(distribution.token_ids)
        and all(isinstance(token_id, int) and token_id >= 0 for token_id in distribution.token_ids),
        "sampling distribution token IDs are invalid",
    )
    require(
        all(math.isfinite(value) and value >= 0.0 for value in distribution.probabilities),
        "sampling distribution probabilities are invalid",
    )
    require(
        math.isclose(math.fsum(distribution.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9),
        "sampling distribution is not normalized",
    )


def speculative_acceptance_probability(
    target: TokenDistribution,
    draft: TokenDistribution,
    token_id: int,
) -> float:
    """Return the exact speculative acceptance probability for one draft token."""
    validate_distribution(target)
    validate_distribution(draft)
    draft_probability = draft.probability(token_id)
    require(draft_probability > 0.0, "draft token has zero draft probability")
    return min(1.0, target.probability(token_id) / draft_probability)


def residual_distribution(
    target: TokenDistribution,
    draft: TokenDistribution,
) -> TokenDistribution:
    """Normalize ``max(target - draft, 0)`` after speculative rejection."""
    validate_distribution(target)
    validate_distribution(draft)
    residual = [
        (token_id, max(0.0, probability - draft.probability(token_id)))
        for token_id, probability in zip(target.token_ids, target.probabilities)
    ]
    retained = [
        (token_id, probability)
        for token_id, probability in residual
        if probability > 0.0
    ]
    require(retained, "speculative residual distribution is empty")
    total = math.fsum(probability for _, probability in retained)
    require(total > 0.0 and math.isfinite(total), "speculative residual is invalid")
    return TokenDistribution(
        token_ids=tuple(token_id for token_id, _ in retained),
        probabilities=tuple(probability / total for _, probability in retained),
    )


def _sample_weighted(
    token_ids: Sequence[int],
    probabilities: Sequence[float],
    rng: random.Random,
) -> int:
    require(len(token_ids) == len(probabilities) > 0, "sampling distribution is empty")
    total = math.fsum(probabilities)
    require(total > 0.0 and math.isfinite(total), "sampling distribution is invalid")
    threshold = rng.random() * total
    cumulative = 0.0
    for token_id, probability in zip(token_ids, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return token_id
    return token_ids[-1]


def candidate_distribution(
    token_ids: Sequence[int],
    logits: Sequence[float],
    *,
    temperature: float,
    top_p: float,
) -> TokenDistribution:
    """Build the target's stable top-k then top-p categorical distribution."""
    require(len(token_ids) == len(logits) > 0, "candidate shape mismatch")
    require(temperature > 0.0, "sampling temperature must be positive")
    require(0.0 < top_p <= 1.0, "top-p must be in (0, 1]")
    require(len(set(token_ids)) == len(token_ids), "candidate token IDs are not unique")
    ranked = sorted(zip(token_ids, logits), key=lambda item: (-item[1], item[0]))
    maximum = ranked[0][1]
    weights = [math.exp((value - maximum) / temperature) for _, value in ranked]
    total = math.fsum(weights)
    require(total > 0.0 and math.isfinite(total), "candidate softmax is invalid")
    probabilities = [weight / total for weight in weights]

    retained = 1
    cumulative = probabilities[0]
    while retained < len(probabilities) and cumulative < top_p:
        cumulative += probabilities[retained]
        retained += 1
    retained_total = math.fsum(probabilities[:retained])
    return TokenDistribution(
        token_ids=tuple(token_id for token_id, _ in ranked[:retained]),
        probabilities=tuple(
            probability / retained_total for probability in probabilities[:retained]
        ),
    )


def target_distribution(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    hidden: mx.array | None = None,
    lm_head: mx.array | vocab.MLXAffineQuantizedMatrix | None = None,
) -> TokenDistribution:
    """Materialize exactly the distribution used by normal target generation."""
    require(logits.ndim == 1, "target logits must be a vector")
    require(temperature >= 0.0, "temperature must be nonnegative")
    if isinstance(lm_head, vocab.MLXAffineQuantizedMatrix) and lm_head.reference is not None:
        require(hidden is not None, "hybrid LM head requires final hidden state")
        require(
            temperature == 0.0 or top_k <= 256,
            "hybrid LM head supports sampled top-k at most 256",
        )
        candidate_count = 64 if temperature == 0.0 else max(64, top_k)
        token_ids, values = vocab.exact_candidate_scores(
            lm_head,
            logits,
            hidden,
            candidate_count=candidate_count,
        )
        ranked = sorted(zip(token_ids, values), key=lambda item: (-item[1], item[0]))
        if temperature == 0.0:
            return TokenDistribution((ranked[0][0],), (1.0,))
        selected = ranked[:top_k]
        return candidate_distribution(
            [token_id for token_id, _ in selected],
            [value for _, value in selected],
            temperature=temperature,
            top_p=top_p,
        )
    if temperature == 0.0:
        return TokenDistribution((int(mx.argmax(logits).item()),), (1.0,))
    require(0 < top_k <= logits.size, "top-k is outside the vocabulary")
    require(0.0 < top_p <= 1.0, "top-p must be in (0, 1]")
    indices = mx.argpartition(logits, logits.size - top_k)[-top_k:]
    values = mx.take(logits, indices).astype(mx.float32)
    mx.eval(indices, values)
    return candidate_distribution(
        [int(value) for value in indices.tolist()],
        [float(value) for value in values.tolist()],
        temperature=temperature,
        top_p=top_p,
    )
