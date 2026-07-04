#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fake = tmp / "fake_generate"
        fake.write_text(
            "#!/usr/bin/env sh\n"
            "max_new=\"$4\"\n"
            "printf 'backend=metal generated=%s layers=%s expert_top_k=%s vocab_limit=%s seconds=2.000000\\n' \"$max_new\" \"$5\" \"$6\" \"$7\"\n"
            "i=0\n"
            "while [ \"$i\" -lt \"$max_new\" ]; do printf '%s\\t%s\\t1.5\\n' \"$i\" \"$((100+i))\"; i=$((i+1)); done\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        out = tmp / "bench.log"
        ledger = tmp / "ledger.jsonl"
        env = os.environ.copy()
        env.update(
            {
                "BIN": str(fake),
                "CATALOG": "catalog.tsv",
                "SHARDS": "shards",
                "PROMPT": "0,1",
                "LAYERS": "4",
                "TOP_K": "1",
                "VOCAB_LIMIT": "32",
                "MAX_NEW_LIST": "2",
                "CACHE_MB_LIST": "7",
                "OUT": str(out),
                "LEDGER": str(ledger),
            }
        )
        subprocess.run([str(root / "ornith/tools/bench_metal_decode.sh")], check=True, env=env, cwd=root)
        subprocess.run([str(root / "ornith/tools/bench_metal_decode.sh")], check=True, env=env, cwd=root)
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["case"] == "cache_7"
        assert rows[0]["tok_s"] == 1.0
        assert rows[0]["tokens"] == [100, 101]
        assert out.read_text(encoding="utf-8").count("ornith metal decode bench") == 2
    print("ornith_bench_metal_decode_test: ok")


if __name__ == "__main__":
    main()
