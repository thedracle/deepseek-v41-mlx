"""Convert the raw DeepSeek-V4.1-Flash release to an MLX build.

    python scripts/convert_cli.py --src /path/DeepSeek-V4.1-Flash-src --dst out-mixed \
        --bits 8 --expert-bits 4 --engram-bits 4

Recommended for 512 GB machines: --bits 8 --expert-bits 4 --engram-bits 4
(~427 GB). Without --engram-bits the two native-fp8 engram tables alone are
203 GB and no expert width fits under 512 GB.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from deepseek_v41_mlx.convert import convert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--expert-bits", type=int, default=None)
    p.add_argument("--engram-bits", type=int, default=None)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--resume", action="store_true",
                   help="skip groups whose output shard + manifest already exist")
    a = p.parse_args()
    t0 = time.time()
    acct = convert(a.src, a.dst, bits=a.bits, expert_bits=a.expert_bits,
                   engram_bits=a.engram_bits, group_size=a.group_size,
                   resume=a.resume)
    print(f"done in {time.time() - t0:.0f}s: {acct}")


if __name__ == "__main__":
    main()
