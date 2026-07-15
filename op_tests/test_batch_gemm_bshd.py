# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
BSHD-fused batch GEMM for the DeepSeek-V4 MLA output projection.

The attention output is produced in BSHD layout ``A = [B, S, H, D]`` (the DSv4
default out of sparse_attn) and the grouped output-LoRA weight is
``W = [G, R, K]`` with ``H = G * heads_per_group`` and ``K = heads_per_group*D``.
The op avoids the full HBM transpose by feeding the kernel strided views and
writes ``out = [B, S, G, R]`` directly.

    out[b, s, g, r] = sum_d A[b, s, g*hpg+h, d] * W[g, r, h*D+d]
    == torch.einsum("bsgd,grd->bsgr", A.reshape(B, S, G, -1), W)

Two pipeline variants (swept via ``--pipeline``):
  standard   -- standard a16w16 batch GEMM over a strided [G, S, K] view.
  bhsd_remap -- BHSD a_offset remapping over a strided [G, hpg, S, D] view.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.ops.opus.gemm_op_a16w16 import batch_gemm_a16w16_bshd_opus
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# Both BSHD pipelines route through gfx950-only opus a16w16 kernels.
SUPPORTED_GFX = ["gfx950"]


def run_torch(A, W, heads_per_group, dtype=dtypes.bf16):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    B, S, H, D = A.shape
    G = H // heads_per_group
    o = A.reshape(B, S, G, heads_per_group * D)
    out = torch.einsum("bsgd,grd->bsgr", o, W)
    return out.to(dtype)


@benchmark()
def test_bshd_gemm(B, S, H, D, G, R, dtype, pipeline, splitk, kernelId=None):
    heads_per_group = H // G
    K = heads_per_group * D
    use_standard = pipeline == "standard"

    # Faithful to the model call: A is native BSHD [B, S, H, D], W is [G, R, K].
    A = torch.randn(B, S, H, D, dtype=dtype)
    W = torch.randn(G, R, K, dtype=dtype)

    ref = run_torch(A, W, heads_per_group, dtype)

    candidates = {
        # opus BSHD-fused kernel -- the path the model really runs.
        # kernelId=None lets the op pick its default kid (208 standard / 608
        # bhsd_remap); pass an explicit kid to force a specific kernel.
        "opus": lambda: batch_gemm_a16w16_bshd_opus(
            A,
            W,
            heads_per_group,
            kernelId=kernelId,
            splitK=splitk,
            use_standard_pipeline=use_standard,
        ),
        # torch.einsum on the model's natural [B, S, G, d]/[G, R, d] operands.
        "torch_einsum": lambda: torch.einsum(
            "bsgd,grd->bsgr", A.reshape(B, S, G, K), W
        ),
    }

    # Batched GEMM of B*G x ([S, K] @ [R, K]^T -> [S, R]):
    #   FLOPs = 2 * (B*G) * S * R * K  (multiply-add)
    #   bytes = A + W + out elements * dtype size (weight counted once, logical)
    flops = 2 * (B * G) * S * R * K
    nbytes = (B * S * H * D + G * R * K + B * S * G * R) * A.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: bshd batch gemm [{pipeline}]",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("bshd batch gemm unsupported on %s; skipping", get_gfx())
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
        # DeepSeek-V4 MLA output projection: (B, S, H, D, G, R).
        #   H=64 heads, D=512 head_dim, G=8 groups (hpg=8), R=1024 o_lora_rank,
        #   K=hpg*D=4096. S is the swept token dim (decode -> long prefill).
        default=[
            (1, 1, 64, 512, 8, 1024),
            (1, 4, 64, 512, 8, 1024),
            (1, 128, 64, 512, 8, 1024),
            (1, 1024, 64, 512, 8, 1024),
            (1, 4096, 64, 512, 8, 1024),
            (4, 1, 64, 512, 8, 1024),
            (4, 128, 64, 512, 8, 1024),
        ],
        help="""Shape (B, S, H, D, G, R).
        e.g.: -s 1,128,64,512,8,1024""",
    )
    parser.add_argument(
        "-l",
        "--pipeline",
        type=str,
        choices=["standard", "bhsd_remap"],
        nargs="*",
        default=["standard", "bhsd_remap"],
        help="""BSHD pipeline variant(s) to sweep.
        standard   = strided [G, S, K] view + standard a16w16 kids.
        bhsd_remap = strided [G, hpg, S, D] view + BHSD a_offset remap kids.
        e.g.: -l standard""",
    )
    parser.add_argument(
        "-k",
        "--splitk",
        type=int,
        nargs="*",
        default=[0],
        help="""Split-K factor(s) to sweep.
        e.g.: -k 0 8""",
    )
    parser.add_argument(
        "--kernelId",
        type=int,
        default=None,
        help="""Force a specific opus kid instead of the op's default
        (208 standard / 608 bhsd_remap). e.g. --kernelId 9 for the
        512x256x256 split-barrier kernel (T>=4096 winner).""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for shape, pipeline, splitk in itertools.product(
            args.shape, args.pipeline, args.splitk
        ):
            B, S, H, D, G, R = shape
            df.append(
                test_bshd_gemm(B, S, H, D, G, R, dtype, pipeline, splitk, args.kernelId)
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "bshd batch gemm summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
