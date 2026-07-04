#!/usr/bin/env python3

import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ornith" / "tools"
sys.path.insert(0, str(TOOLS))
MOD_PATH = TOOLS / "ornith_reap_calibrate.py"
spec = importlib.util.spec_from_file_location("ornith_reap_calibrate", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Args:
    pass


def demo():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fake = root / "observe"
        fake.write_text("""#!/bin/sh
test "$3" = "--prompts"
grep -qx '1,2' "$4"
grep -qx '3,4' "$4"
out="$9"
cat >"$out" <<'JSON'
{"format":"ornith-reap-observer-v1","layers":{"0":{"total_tokens":2,"expert_frequency":[2,0],"weighted_expert_frequency_sum":[0.5,0],"ean_mean":[4,0],"reap":[1,0],"max_activations":[3,0]}}}
JSON
""", encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        prompts = root / "prompts.txt"
        prompts.write_text("1,2,99\n3,4,99\n", encoding="utf-8")
        args = Args()
        args.binary = fake
        args.catalog = root / "catalog.tsv"
        args.shards = root
        args.prompts = prompts
        args.out = root / "out.json"
        args.tokenizer = None
        args.text_prompts = False
        args.max_prompts = 0
        args.max_prompt_tokens = 2
        args.max_new = 1
        args.layers = 1
        args.expert_top_k = 1
        args.vocab_limit = 32
        report = mod.run(args)
        layer = report["layers"]["0"]
        assert layer["total_tokens"] == 2
        assert layer["expert_frequency"] == [2, 0]
        assert layer["weighted_expert_frequency_sum"] == [0.5, 0.0]
        assert layer["ean_mean"] == [4.0, 0.0]
        assert layer["reap"] == [1.0, 0.0]
        assert layer["max_activations"] == [3.0, 0.0]


if __name__ == "__main__":
    demo()
    print("ornith_reap_calibrate_test: ok")
