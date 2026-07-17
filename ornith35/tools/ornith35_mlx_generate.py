#!/usr/bin/env python3
"""Bounded text generation from the verified Ornith-35 target."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import sys
import time

import mlx.core as mx

import ornith35_mlx_model as model
from ornith35_moe_reference import MoEError, require as require_model
from ornith35_tokenizer import (
    DEFAULT_ROOT,
    TokenizerError,
    load_text_tokenizer,
    render_text_prompt,
)


def sample_candidates(
    token_ids: list[int],
    logits: list[float],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> int:
    """Sample a preselected top-k set using a deterministic CPU RNG."""
    require_model(len(token_ids) == len(logits) > 0, "candidate shape mismatch")
    require_model(temperature > 0.0, "sampling temperature must be positive")
    require_model(0.0 < top_p <= 1.0, "top-p must be in (0, 1]")
    ranked = sorted(zip(token_ids, logits), key=lambda item: item[1], reverse=True)
    maximum = ranked[0][1]
    weights = [math.exp((value - maximum) / temperature) for _, value in ranked]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]

    retained = 1
    cumulative = probabilities[0]
    while retained < len(probabilities) and cumulative < top_p:
        cumulative += probabilities[retained]
        retained += 1
    threshold = rng.random() * math.fsum(probabilities[:retained])
    cumulative = 0.0
    for (token_id, _), probability in zip(ranked[:retained], probabilities[:retained]):
        cumulative += probability
        if threshold <= cumulative:
            return token_id
    return ranked[retained - 1][0]


def choose_next_token(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
) -> int:
    require_model(logits.ndim == 1, "target logits must be a vector")
    require_model(temperature >= 0.0, "temperature must be nonnegative")
    if temperature == 0.0:
        return int(mx.argmax(logits).item())
    require_model(0 < top_k <= logits.size, "top-k is outside the vocabulary")
    require_model(0.0 < top_p <= 1.0, "top-p must be in (0, 1]")
    indices = mx.argpartition(logits, logits.size - top_k)[-top_k:]
    values = mx.take(logits, indices).astype(mx.float32)
    mx.eval(indices, values)
    return sample_candidates(
        [int(value) for value in indices.tolist()],
        [float(value) for value in values.tolist()],
        temperature=temperature,
        top_p=top_p,
        rng=rng,
    )


def split_reasoning_response(response: str) -> tuple[str | None, str]:
    marker = "</think>"
    if marker not in response:
        return None, response
    reasoning, final = response.split(marker, 1)
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>") :]
    return reasoning.strip(), final.lstrip()


def generate(
    root: Path,
    prompt: str,
    *,
    system: str | None,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> str:
    require_model(0 < max_tokens <= 4096, "max tokens must be between 1 and 4096")
    require_model(temperature >= 0.0, "temperature must be nonnegative")
    require_model(0 < top_k <= model.PRODUCTION_CONFIG.vocab_size, "invalid top-k")
    require_model(0.0 < top_p <= 1.0, "invalid top-p")
    tokenizer = load_text_tokenizer(root)
    rendered = render_text_prompt(
        prompt,
        system=system,
        enable_thinking=enable_thinking,
    )
    prompt_ids = tokenizer.encode(rendered)
    require_model(prompt_ids, "rendered prompt produced no tokens")
    print(
        "generate-start "
        f"prompt_tokens={len(prompt_ids)} max_tokens={max_tokens} "
        f"thinking={str(enable_thinking).lower()} temperature={temperature:.6g} "
        f"top_k={top_k} top_p={top_p:.6g} seed={seed}",
        flush=True,
    )

    load_started = time.perf_counter()
    weights = model.load_text_model(root)
    load_elapsed = time.perf_counter() - load_started
    print(
        f"generate-model-ready load_s={load_elapsed:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f}",
        flush=True,
    )

    state = model.initial_state(weights, model.PRODUCTION_CONFIG)
    prefill_started = time.perf_counter()
    for token_id in prompt_ids[:-1]:
        transition = model.forward_hidden_token(
            token_id,
            state,
            weights,
            model.PRODUCTION_CONFIG,
        )
        model.evaluate_transition(transition)
        state = transition.state
    result = model.forward_token(
        prompt_ids[-1],
        state,
        weights,
        model.PRODUCTION_CONFIG,
    )
    model.evaluate_result(result)
    state = result.state
    prefill_elapsed = time.perf_counter() - prefill_started
    print(
        "generate-prefill-done "
        f"tokens={len(prompt_ids)} elapsed_s={prefill_elapsed:.3f} "
        f"tokens_s={len(prompt_ids) / prefill_elapsed:.3f} "
        "logit_projections=1",
        flush=True,
    )

    generated: list[int] = []
    transition_elapsed = 0.0
    stop = "length"
    rng = random.Random(seed)
    for step in range(max_tokens):
        started = time.perf_counter()
        next_id = choose_next_token(
            result.logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        generated.append(next_id)
        if next_id in tokenizer.eos_token_ids:
            stop = "eos"
            break
        if step + 1 == max_tokens:
            break
        result = model.forward_token(next_id, state, weights, model.PRODUCTION_CONFIG)
        model.evaluate_result(result)
        transition_elapsed += time.perf_counter() - started
        state = result.state

    measured = max(len(generated) - 1, 0)
    speed = measured / transition_elapsed if transition_elapsed > 0.0 else 0.0
    response_ids = (
        generated[:-1]
        if generated and generated[-1] in tokenizer.eos_token_ids
        else generated
    )
    response = tokenizer.decode(response_ids)
    print(
        "generate-done "
        f"tokens={len(generated)} stop={stop} decode_tokens_s={speed:.3f} "
        f"active_gib={mx.get_active_memory() / 2**30:.3f} "
        f"peak_gib={mx.get_peak_memory() / 2**30:.3f}",
        flush=True,
    )
    reasoning, final = split_reasoning_response(response)
    if reasoning is not None:
        print("reasoning-begin", flush=True)
        print(reasoning, flush=True)
        print("reasoning-end", flush=True)
        print("response-begin", flush=True)
        print(final, flush=True)
        print("response-end", flush=True)
    else:
        print("response-begin", flush=True)
        print(response, flush=True)
        print("response-end", flush=True)
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        generate(
            args.root,
            args.prompt,
            system=args.system,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
    except (MoEError, TokenizerError, OSError, ValueError) as exc:
        print(f"ornith35 generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
