#!/usr/bin/env python3
"""Contract tests for resumable Nemotron backbone fitting contexts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nemotron" / "tools"))

from nemotron_mlx_backbone_context import (  # noqa: E402
    FORMAT,
    STATE_FORMAT,
    atomic_safetensors,
    context_corpus_samples,
    load_context_rows,
    split_for_sample,
    validate_context_arrays,
)
from nemotron_prune_materialize import sha256_file  # noqa: E402


class BackboneContextTest(unittest.TestCase):
    def arrays(self, offset: int = 0) -> dict[str, mx.array]:
        return {
            "layer_input": mx.arange(offset, offset + 3 * 16, dtype=mx.float32).reshape(3, 16),
            "latent": mx.arange(offset, offset + 3 * 8, dtype=mx.float32).reshape(3, 8),
            "indices": mx.array([[0, 1], [2, 3], [1, 3]], dtype=mx.int32),
            "scores": mx.array([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]], dtype=mx.float32),
        }

    def test_split_is_deterministic_and_prompt_disjoint(self) -> None:
        validation = f"{5:064x}"
        train = f"{6:064x}"
        self.assertEqual(split_for_sample(validation, 5), "validation")
        self.assertEqual(split_for_sample(validation, 5), "validation")
        self.assertEqual(split_for_sample(train, 5), "train")

    def test_array_contract_rejects_dtype_drift(self) -> None:
        arrays = self.arrays()
        self.assertEqual(validate_context_arrays(arrays, hidden_size=16, latent_size=8, top_k=2), 3)
        arrays["latent"] = arrays["latent"].astype(mx.bfloat16)
        with self.assertRaisesRegex(Exception, "latent values must be float32"):
            validate_context_arrays(arrays, hidden_size=16, latent_size=8, top_k=2)

    def test_balanced_prompt_jsonl_loader_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                '{"category":"code","prompt":"def f(): pass"}\n'
                '{"category":"reasoning","prompt":"Compute 2+2."}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                context_corpus_samples(path),
                [("code", "def f(): pass"), ("reasoning", "Compute 2+2.")],
            )

    def test_split_loader_rereads_hash_bound_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            completed = []
            for batch, split in ((0, "validation"), (1, "train"), (2, "train")):
                name = f"batch-{batch:05d}-{split}.safetensors"
                path = output / name
                atomic_safetensors(
                    path,
                    self.arrays(batch * 100),
                    {
                        "format": FORMAT,
                        "source_revision": "revision",
                        "layer": "1",
                        "batch": str(batch),
                        "split": split,
                    },
                )
                completed.append({
                    "batch": batch,
                    "split": split,
                    "rows": 3,
                    "file": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
            state = {
                "format": STATE_FORMAT,
                "status": "complete",
                "source_revision": "revision",
                "layer": 1,
                "architecture": {"hidden_size": 16, "latent_size": 8, "top_k": 2},
                "completed": completed,
            }
            (output / "state.json").write_text(json.dumps(state), encoding="utf-8")
            train = load_context_rows(output, "train")
            self.assertEqual(train["latent"].shape, (6, 8))
            self.assertEqual(float(train["latent"][0, 0]), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
