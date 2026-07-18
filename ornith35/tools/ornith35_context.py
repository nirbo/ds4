#!/usr/bin/env python3
"""Authoritative text context and RoPE profiles for Ornith-35."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


NATIVE_CONTEXT_TOKENS = 262_144
YARN2_CONTEXT_TOKENS = 524_288
NATIVE_PROFILE_ID = "native-262k"
YARN2_PROFILE_ID = "yarn2-524k"


class ContextError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContextError(message)


@dataclass(frozen=True)
class ContextProfile:
    profile_id: str
    max_position_embeddings: int
    rope_type: str
    factor: float
    original_max_position_embeddings: int
    attention_factor: float | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    truncate: bool = True

    def __post_init__(self) -> None:
        require(bool(self.profile_id), "context profile ID must be nonempty")
        require(
            self.max_position_embeddings > 0,
            "maximum position embeddings must be positive",
        )
        require(self.rope_type in ("default", "yarn"), "unsupported RoPE type")
        require(self.factor >= 1.0, "RoPE factor must be at least one")
        require(
            0 < self.original_max_position_embeddings <= self.max_position_embeddings,
            "invalid original maximum position embeddings",
        )
        require(
            self.attention_factor is None or self.attention_factor > 0.0,
            "attention factor must be positive",
        )
        require(self.beta_fast >= self.beta_slow > 0.0, "invalid YaRN beta range")
        if self.rope_type == "default":
            require(self.factor == 1.0, "default RoPE factor must be one")
            require(
                self.original_max_position_embeddings == self.max_position_embeddings,
                "default RoPE original context mismatch",
            )
        else:
            require(self.factor > 1.0, "YaRN factor must exceed one")
            require(
                math.isclose(
                    self.factor,
                    self.max_position_embeddings / self.original_max_position_embeddings,
                ),
                "YaRN factor does not match the context extension",
            )

    def descriptor(self) -> dict[str, Any]:
        return asdict(self)


NATIVE_PROFILE = ContextProfile(
    profile_id=NATIVE_PROFILE_ID,
    max_position_embeddings=NATIVE_CONTEXT_TOKENS,
    rope_type="default",
    factor=1.0,
    original_max_position_embeddings=NATIVE_CONTEXT_TOKENS,
    attention_factor=1.0,
)

YARN2_PROFILE = ContextProfile(
    profile_id=YARN2_PROFILE_ID,
    max_position_embeddings=YARN2_CONTEXT_TOKENS,
    rope_type="yarn",
    factor=2.0,
    original_max_position_embeddings=NATIVE_CONTEXT_TOKENS,
)

SUPPORTED_CONTEXT_PROFILES = {
    NATIVE_PROFILE_ID: NATIVE_PROFILE,
    YARN2_PROFILE_ID: YARN2_PROFILE,
}


def resolve_profile(profile: str | ContextProfile) -> ContextProfile:
    if isinstance(profile, ContextProfile):
        require(
            SUPPORTED_CONTEXT_PROFILES.get(profile.profile_id) == profile,
            "context profile is not an authoritative supported profile",
        )
        return profile
    require(isinstance(profile, str), "context profile must be a string or ContextProfile")
    selected = SUPPORTED_CONTEXT_PROFILES.get(profile)
    require(selected is not None, f"unsupported context profile: {profile}")
    return selected


def validate_range(
    profile: str | ContextProfile,
    position: int,
    tokens: int = 0,
) -> ContextProfile:
    selected = resolve_profile(profile)
    require(isinstance(position, int) and position >= 0, "context position must be nonnegative")
    require(isinstance(tokens, int) and tokens >= 0, "context token count must be nonnegative")
    require(
        position + tokens <= selected.max_position_embeddings,
        f"context range exceeds {selected.profile_id}",
    )
    return selected


def yarn_correction_range(
    profile: ContextProfile,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[float, float]:
    def correction_dim(rotations: float) -> float:
        return (
            rotary_dim
            * math.log(
                profile.original_max_position_embeddings / (rotations * 2.0 * math.pi)
            )
            / (2.0 * math.log(rope_theta))
        )

    low = correction_dim(profile.beta_fast)
    high = correction_dim(profile.beta_slow)
    if profile.truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0.0), min(high, float(rotary_dim - 1))


def profile_attention_factor(profile: str | ContextProfile) -> float:
    selected = resolve_profile(profile)
    if selected.attention_factor is not None:
        return selected.attention_factor
    return 1.0 + 0.1 * math.log(selected.factor)


def rope_parameters(
    profile: str | ContextProfile,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[tuple[float, ...], float]:
    """Return inverse frequencies and scaling from Transformers 5.10.1."""
    selected = resolve_profile(profile)
    require(rotary_dim > 0 and rotary_dim % 2 == 0, "rotary dimension must be positive and even")
    require(rope_theta > 1.0, "RoPE theta must exceed one")
    base = tuple(
        1.0 / (rope_theta ** (index / rotary_dim))
        for index in range(0, rotary_dim, 2)
    )
    if selected.rope_type == "default":
        return base, profile_attention_factor(selected)

    low, high = yarn_correction_range(selected, rotary_dim, rope_theta)
    if low == high:
        high += 0.001
    inverse_frequencies = []
    for index, extrapolated in enumerate(base):
        ramp = min(max((index - low) / (high - low), 0.0), 1.0)
        interpolated = extrapolated / selected.factor
        inverse_frequencies.append(interpolated * ramp + extrapolated * (1.0 - ramp))
    return tuple(inverse_frequencies), profile_attention_factor(selected)
