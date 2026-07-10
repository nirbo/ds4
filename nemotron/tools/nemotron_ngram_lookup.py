#!/usr/bin/env python3
"""Bounded prompt and generated-token lookup drafts for Nemotron."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

from nemotron_metadata import require


@dataclass(frozen=True)
class LookupDraft:
    token_ids: tuple[int, ...]
    key_tokens: int
    source_position: int


class NGramLookup:
    """Index prior token suffixes without retaining unbounded position lists."""

    def __init__(
        self,
        token_ids: list[int],
        min_key_tokens: int = 3,
        max_key_tokens: int = 8,
        max_entries: int = 65_536,
        positions_per_key: int = 4,
        min_matching_continuations: int = 2,
    ):
        require(
            1 <= min_key_tokens <= max_key_tokens,
            "invalid lookup key-token range",
        )
        require(max_entries >= max_key_tokens - min_key_tokens + 1, "lookup table is too small")
        require(positions_per_key > 0, "lookup positions per key must be positive")
        require(
            1 <= min_matching_continuations <= positions_per_key,
            "invalid lookup continuation-match requirement",
        )
        self.min_key_tokens = min_key_tokens
        self.max_key_tokens = max_key_tokens
        self.positions_per_key = positions_per_key
        self.min_matching_continuations = min_matching_continuations
        table_count = max_key_tokens - min_key_tokens + 1
        self.max_entries_per_table = max(1, max_entries // table_count)
        self.tokens: list[int] = []
        self.tables: dict[int, OrderedDict[tuple[int, ...], deque[int]]] = {
            key_tokens: OrderedDict()
            for key_tokens in range(min_key_tokens, max_key_tokens + 1)
        }
        self.extend(token_ids)

    @property
    def entries(self) -> int:
        return sum(len(table) for table in self.tables.values())

    def append(self, token_id: int) -> None:
        require(isinstance(token_id, int) and token_id >= 0, "invalid lookup token ID")
        self.tokens.append(token_id)
        for key_tokens, table in self.tables.items():
            if len(self.tokens) < key_tokens:
                continue
            start = len(self.tokens) - key_tokens
            key = tuple(self.tokens[start:])
            positions = table.pop(key, None)
            if positions is None:
                positions = deque(maxlen=self.positions_per_key)
            positions.append(start)
            table[key] = positions
            while len(table) > self.max_entries_per_table:
                table.popitem(last=False)

    def extend(self, token_ids: list[int] | tuple[int, ...]) -> None:
        for token_id in token_ids:
            self.append(token_id)

    def propose(self, next_token_id: int, max_draft_tokens: int) -> LookupDraft | None:
        require(max_draft_tokens > 0, "lookup draft limit must be positive")
        require(isinstance(next_token_id, int) and next_token_id >= 0, "invalid lookup token ID")
        for key_tokens in range(self.max_key_tokens, self.min_key_tokens - 1, -1):
            if len(self.tokens) + 1 < key_tokens:
                continue
            prefix = self.tokens[-(key_tokens - 1) :] if key_tokens > 1 else []
            key = tuple([*prefix, next_token_id])
            positions = self.tables[key_tokens].get(key)
            if positions is None:
                continue
            candidates: dict[tuple[int, ...], list[int]] = {}
            for source_position in positions:
                continuation = source_position + key_tokens
                available = len(self.tokens) - continuation
                if available <= 0:
                    continue
                token_ids = tuple(
                    self.tokens[continuation : continuation + min(max_draft_tokens, available)]
                )
                if token_ids:
                    candidates.setdefault(token_ids, []).append(source_position)
            supported = [
                (token_ids, source_positions)
                for token_ids, source_positions in candidates.items()
                if len(source_positions) >= self.min_matching_continuations
            ]
            if supported:
                token_ids, source_positions = max(
                    supported,
                    key=lambda item: (len(item[1]), item[1][-1]),
                )
                return LookupDraft(token_ids, key_tokens, source_positions[-1])
        return None
