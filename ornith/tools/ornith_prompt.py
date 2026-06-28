#!/usr/bin/env python3
"""Small Ornith prompt-rendering reference.

This covers text-only chat. Vision and tools are intentionally left out until
the text path is validated against the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def special_token_ids(tokenizer: dict) -> dict[str, int]:
    out = {}
    for token in tokenizer.get("added_tokens", []):
        content = token.get("content")
        token_id = token.get("id")
        if isinstance(content, str) and isinstance(token_id, int):
            out[content] = token_id
    return out


def render_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and (
                "image" in item or "image_url" in item or "video" in item or item.get("type") in ("image", "video")
            ):
                raise ValueError("text-only renderer does not support vision/tool content blocks")
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                raise ValueError("text-only renderer does not support vision/tool content blocks")
        return "".join(parts)
    raise ValueError("unexpected message content type")


def last_user_query_index(messages: list[dict]) -> int:
    last = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            text = render_content(msg.get("content")).strip()
            if not (text.startswith("<tool_response>") and text.endswith("</tool_response>")):
                last = i
    if last < 0:
        raise ValueError("no user query found in messages")
    return last


def split_reasoning(content: str, explicit_reasoning) -> tuple[str, str]:
    if isinstance(explicit_reasoning, str):
        return explicit_reasoning.strip(), content.strip()
    if "</think>" in content:
        before, after = content.split("</think>", 1)
        reasoning = before.rstrip("\n").split("<think>")[-1].lstrip("\n")
        return reasoning.strip(), after.lstrip("\n").strip()
    return "", content.strip()


def render_text_chat(
    messages: list[dict],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    if not messages:
        raise ValueError("no messages provided")

    out: list[str] = []
    start = 0
    if messages[0].get("role") == "system":
        content = render_content(messages[0].get("content")).strip()
        out.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        start = 1

    last_query = last_user_query_index(messages)
    for i, msg in enumerate(messages[start:], start=start):
        role = msg.get("role")
        content = render_content(msg.get("content")).strip()
        if role == "system":
            raise ValueError("system message must be first")
        if role == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            reasoning, visible = split_reasoning(content, msg.get("reasoning_content"))
            if i > last_query:
                out.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{visible}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>assistant\n{visible}<|im_end|>\n")
        else:
            raise ValueError(f"unsupported text-only role: {role}")

    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        if enable_thinking:
            out.append("<think>\n")
        else:
            out.append("<think>\n\n</think>\n\n")
    return "".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--messages", type=Path)
    p.add_argument("--nothink", action="store_true")
    p.add_argument("--no-generation-prompt", action="store_true")
    p.add_argument("--print-specials", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_specials:
        if not args.tokenizer:
            raise SystemExit("--print-specials requires --tokenizer")
        specials = special_token_ids(load_json(args.tokenizer))
        for token, token_id in sorted(specials.items(), key=lambda kv: kv[1]):
            print(f"{token_id}\t{token}")

    if args.messages:
        messages = load_json(args.messages)
        print(render_text_chat(
            messages,
            add_generation_prompt=not args.no_generation_prompt,
            enable_thinking=not args.nothink,
        ), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
