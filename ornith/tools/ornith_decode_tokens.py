#!/usr/bin/env python3
"""Decode Ornith/Qwen byte-level BPE token ids without external packages."""

from __future__ import annotations

import argparse
import json


def byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def byte_encoder() -> dict[int, str]:
    return {v: k for k, v in byte_decoder().items()}


def load_tokenizer(path: str) -> dict:
    return json.load(open(path, "r", encoding="utf-8"))


def load_id_to_token(path: str) -> dict[int, str]:
    data = load_tokenizer(path)
    out = {int(v): k for k, v in data["model"]["vocab"].items()}
    for tok in data.get("added_tokens", []):
        out[int(tok["id"])] = tok["content"]
    return out


def bpe(piece: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    parts = list(piece)
    while len(parts) > 1:
        best_i = -1
        best_rank = len(ranks) + 1
        for i in range(len(parts) - 1):
            rank = ranks.get((parts[i], parts[i + 1]), best_rank)
            if rank < best_rank:
                best_i = i
                best_rank = rank
        if best_i < 0:
            break
        parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return parts


def encode(text: str, tokenizer: dict) -> list[int]:
    vocab = tokenizer["model"]["vocab"]
    ranks = {tuple(pair): i for i, pair in enumerate(tokenizer["model"].get("merges", []))}
    specials = {
        tok["content"]: int(tok["id"])
        for tok in tokenizer.get("added_tokens", [])
        if isinstance(tok.get("content"), str)
    }
    benc = byte_encoder()
    out: list[int] = []
    i = 0
    special_keys = sorted(specials, key=len, reverse=True)
    while i < len(text):
        matched = next((s for s in special_keys if text.startswith(s, i)), None)
        if matched:
            out.append(specials[matched])
            i += len(matched)
            continue
        j = min((text.find(s, i) for s in special_keys if text.find(s, i) >= 0), default=len(text))
        piece = "".join(benc[b] for b in text[i:j].encode("utf-8"))
        out.extend(vocab[token] for token in bpe(piece, ranks))
        i = j
    return out


def decode(ids: list[int], id_to_token: dict[int, str]) -> str:
    bdec = byte_decoder()
    chunks: list[str] = []
    raw = bytearray()
    for token_id in ids:
        token = id_to_token.get(token_id, "")
        if token.startswith("<|") and token.endswith("|>"):
            if raw:
                chunks.append(raw.decode("utf-8", "replace"))
                raw.clear()
            chunks.append(token)
            continue
        for ch in token:
            raw.append(bdec.get(ch, ord(ch) if ord(ch) < 256 else ord("?")))
    if raw:
        chunks.append(raw.decode("utf-8", "replace"))
    return "".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--encode", help="Text to encode instead of token ids to decode")
    ap.add_argument("ids", nargs="*", help="Token ids or comma-separated token id lists")
    args = ap.parse_args()
    if args.encode is not None:
        print(",".join(str(i) for i in encode(args.encode, load_tokenizer(args.tokenizer))))
        return 0
    ids: list[int] = []
    for value in args.ids:
        ids.extend(int(part) for part in value.replace("\n", ",").split(",") if part)
    print(decode(ids, load_id_to_token(args.tokenizer)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
