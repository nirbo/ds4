#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "ornith" / "tools" / "ornith_iq1.py"
spec = importlib.util.spec_from_file_location("ornith_iq1", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def demo():
    values = [-2.0, -1.0, 0.5, 4.0, -0.25]
    q = mod.quantize_iq1(values, block_size=4)
    assert q.n == 5
    assert q.block_size == 4
    assert q.scales == (1.875, 0.25)
    assert [mod.sign_at(q.signs, i) for i in range(q.n)] == [-1.0, -1.0, 1.0, 1.0, -1.0]

    restored = mod.dequantize_iq1(q)
    assert restored == [-1.875, -1.875, 1.875, 1.875, -0.25]
    x = [0.25, -0.5, 1.0, 2.0, -4.0]
    assert abs(mod.dot_iq1(q, x) - mod.dot(restored, x)) < 1e-12
    assert mod.mse(values, restored) > 0.0
    assert abs(mod.bits_per_weight(q, 16) - ((5 + 2 * 16) / 5)) < 1e-12

    rows = [
        [1.0, -2.0, 3.0, -4.0],
        [-0.5, 1.5, -2.5, 3.5],
    ]
    qm = mod.quantize_iq1_rows(rows, block_size=2)
    x2 = [0.25, -0.5, 1.0, 2.0]
    restored_rows = [mod.dequantize_iq1(row) for row in qm.row_vectors]
    assert qm.rows == 2
    assert qm.cols == 4
    assert mod.matvec_iq1(qm, x2) == mod.matvec(restored_rows, x2)

    weighted = mod.quantize_iq1([1.0, 3.0], block_size=2, importance=[100.0, 1.0])
    unweighted = mod.quantize_iq1([1.0, 3.0], block_size=2)
    assert abs(unweighted.scales[0] - 2.0) < 1e-12
    assert abs(weighted.scales[0] - (103.0 / 101.0)) < 1e-12
    source = [1.0, 3.0]
    imp = [100.0, 1.0]
    assert mod.weighted_mse(source, mod.dequantize_iq1(weighted), imp) < mod.weighted_mse(
        source,
        mod.dequantize_iq1(unweighted),
        imp,
    )

    stats = mod.demo(seed=7, n=64, block_size=16)
    assert stats["mse"] > 0.0
    assert stats["weighted_scale_mse"] <= stats["weighted_mse"]
    assert stats["packed_vs_restored_dot_abs"] < 1e-12
    assert stats["matvec_packed_vs_restored_max_abs"] < 1e-12
    assert stats["bits_per_weight_f16_scales"] == 2.0
    assert stats["bits_per_weight_f32_scales"] == 3.0
    assert stats["scale_count"] == 4.0


if __name__ == "__main__":
    demo()
    print("ornith_iq1_test: ok")
