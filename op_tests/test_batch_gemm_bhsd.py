# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
BHSD-fused batch GEMM for the DeepSeek-V4 MLA output projection.

The attention output is produced in BHSD layout ``A = [B, H, S, D]`` and the
grouped output-LoRA weight is ``W = [G, R, K]`` with ``H = G * heads_per_group``
and ``K = heads_per_group * D``. The op fuses the BHSD->BSHD transpose into the
A-matrix address calculation, so it consumes A in its native layout and writes
``out = [B, S, G, R]`` directly.

    out[b, s, g, r] = sum_d A[b, g*hpg+h, s, d] * W[g, r, h*D+d]
    == torch.einsum("bsgd,grd->bsgr", A.transpose(1, 2).reshape(B, S, G, -1), W)

Correctness note: the opus wrapper's default is ``kid=608`` (split-barrier
splitk). That family is only correct when each K-split maps within a single
head, i.e. ``splitK == heads_per_group``; ``splitK=0`` (split_k==1) spans every
head in one split. The ``err`` column surfaces this -- sweep ``--splitk 0 8`` to
compare.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.ops.opus.gemm_op_a16w16 import batch_gemm_a16w16_bhsd_opus
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# BHSD a_offset-remapping pipeline is a gfx950-only opus kernel.
SUPPORTED_GFX = ["gfx950"]


def run_torch(A, W, heads_per_group, dtype=dtypes.bf16):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    B, H, S, D = A.shape
    G = H // heads_per_group
    o = A.transpose(1, 2).reshape(B, S, G, heads_per_group * D)
    out = torch.einsum("bsgd,grd->bsgr", o.to(dtypes.fp32), W.to(dtypes.fp32))
    return out.to(dtype)


@benchmark()
def test_bhsd_gemm(B, H, S, D, G, R, dtype, splitk):
    heads_per_group = H // G
    K = heads_per_group * D

    # Faithful to the model call: A is native BHSD [B, H, S, D], W is [G, R, K].
    A = torch.randn(B, H, S, D, dtype=dtype)
    W = torch.randn(G, R, K, dtype=dtype)

    ref = run_torch(A, W, heads_per_group, dtype)

    candidates = {
        # opus BHSD-fused kernel -- the path the model really runs.
        "opus": lambda: batch_gemm_a16w16_bhsd_opus(
            A, W, heads_per_group, splitK=splitk
        ),
        # torch.einsum on the model's natural [B, S, G, d]/[G, R, d] operands.
        "torch_einsum": lambda: torch.einsum(
            "bsgd,grd->bsgr",
            A.transpose(1, 2).reshape(B, S, G, K),
            W,
        ),
    }

    # Batched GEMM of B*G x ([S, K] @ [R, K]^T -> [S, R]):
    #   FLOPs = 2 * (B*G) * S * R * K  (multiply-add)
    #   bytes = A + W + out elements * dtype size (weight counted once, logical)
    flops = 2 * (B * G) * S * R * K
    nbytes = (B * H * S * D + G * R * K + B * S * G * R) * A.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: bhsd batch gemm",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("bhsd batch gemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        # DeepSeek-V4 MLA output projection: (B, H, S, D, G, R).
        #   H=64 heads, D=512 head_dim, G=8 groups (hpg=8), R=1024 o_lora_rank,
        #   K=hpg*D=4096. S is the swept token dim (decode -> long prefill).
        default=[
            (1, 64, 1, 512, 8, 1024),
            (1, 64, 4, 512, 8, 1024),
            (1, 64, 128, 512, 8, 1024),
            (1, 64, 1024, 512, 8, 1024),
            (1, 64, 4096, 512, 8, 1024),
            (4, 64, 1, 512, 8, 1024),
            (4, 64, 128, 512, 8, 1024),
        ],
        help="""Shape (B, H, S, D, G, R).
        e.g.: -s 1,64,128,512,8,1024""",
    )
    parser.add_argument(
        "-k",
        "--splitk",
        type=int,
        nargs="*",
        # 0 == split_k 1 (spans all heads -> incorrect for hpg>1); hpg (=8)
        # keeps each split within one head.
        default=[0],
        help="""Split-K factor(s) to sweep.
        e.g.: -k 0 8""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for shape, splitk in itertools.product(args.shape, args.splitk):
            B, H, S, D, G, R = shape
            df.append(test_bhsd_gemm(B, H, S, D, G, R, dtype, splitk))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "bhsd batch gemm summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
