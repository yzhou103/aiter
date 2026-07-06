# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test BHSD-fused batch GEMM for MLA output projection.

Reference:
    out[b,s,g,r] = sum_d A[b, g*hpg+h, s, d] * W[g, r, h*D+d]
    Equivalent to: torch.einsum("bsgd,grd->bsgr", A.transpose(1,2).view(B,S,G,-1), W)
"""

import argparse

import torch
from aiter.test_common import checkAllclose, perftest, run_perftest

TEST_NUM_ITERS = 100


def reference_einsum(A, W, heads_per_group):
    """Torch reference: transpose + reshape + einsum."""
    B, H, S, D = A.shape
    G = H // heads_per_group
    o = A.transpose(1, 2).contiguous().reshape(B, S, G, heads_per_group * D)
    wo_a = W  # [G, R, K]
    return torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).to(A.dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_reference(A, W, heads_per_group):
    return reference_einsum(A, W, heads_per_group)


@perftest(num_iters=TEST_NUM_ITERS)
def run_bhsd_opus(A, W, heads_per_group, kernelId, splitK):
    from aiter.ops.opus.gemm_op_a16w16 import batch_gemm_a16w16_bhsd_opus

    return batch_gemm_a16w16_bhsd_opus(
        A, W, heads_per_group, kernelId=kernelId, splitK=splitK
    )


# DeepSeek-V4 shapes
DSV4_SHAPES = [
    # (B, H, S, D, G, R)
    (1, 64, 1, 512, 8, 1024),      # decode bs=1
    (1, 64, 4, 512, 8, 1024),      # decode bs=4
    (1, 64, 128, 512, 8, 1024),    # short prefill
    (1, 64, 1024, 512, 8, 1024),   # medium prefill
    (1, 64, 4096, 512, 8, 1024),   # long prefill
    (4, 64, 1, 512, 8, 1024),      # batched decode
    (4, 64, 128, 512, 8, 1024),    # batched short prefill
]


@torch.inference_mode()
def test_accuracy(B, H, S, D, G, R, kid, splitK):
    heads_per_group = H // G
    K = heads_per_group * D

    A = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(G, R, K, dtype=torch.bfloat16, device="cuda")

    ref = reference_einsum(A, W, heads_per_group)

    from aiter.ops.opus.gemm_op_a16w16 import batch_gemm_a16w16_bhsd_opus

    out = batch_gemm_a16w16_bhsd_opus(
        A, W, heads_per_group, kernelId=kid, splitK=splitK
    )

    msg = f"B={B} H={H} S={S} D={D} G={G} R={R} kid={kid} splitK={splitK}"
    checkAllclose(out, ref, msg=msg, rtol=1e-2, atol=1e-2)


@torch.inference_mode()
def test_perf(B, H, S, D, G, R, kid, splitK):
    heads_per_group = H // G
    K = heads_per_group * D

    A = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(G, R, K, dtype=torch.bfloat16, device="cuda")

    ref_out, ref_us = run_reference(A, W, heads_per_group)
    bhsd_out, bhsd_us = run_bhsd_opus(A, W, heads_per_group, kid, splitK)

    M = B * S
    N = R
    K_total = K
    flops = 2 * M * N * K_total * G
    ref_tflops = flops / ref_us / 1e6
    bhsd_tflops = flops / bhsd_us / 1e6
    speedup = ref_us / bhsd_us

    print(
        f"  B={B:2d} S={S:5d} | "
        f"ref {ref_us:8.1f}us ({ref_tflops:6.1f} TFLOPS) | "
        f"bhsd kid={kid} {bhsd_us:8.1f}us ({bhsd_tflops:6.1f} TFLOPS) | "
        f"speedup {speedup:.2f}x"
    )

    checkAllclose(bhsd_out, ref_out, msg=f"perf check B={B} S={S}", rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BHSD batch GEMM test")
    parser.add_argument("--kid", type=int, default=608, help="kernel ID (default: 608)")
    parser.add_argument("--splitK", type=int, default=0, help="split K factor")
    parser.add_argument("--perf", action="store_true", help="run perf benchmarks")
    args = parser.parse_args()

    print("=== Accuracy tests ===")
    for B, H, S, D, G, R in DSV4_SHAPES:
        test_accuracy(B, H, S, D, G, R, args.kid, args.splitK)
    print("All accuracy tests passed!")

    if args.perf:
        print("\n=== Performance benchmarks ===")
        for B, H, S, D, G, R in DSV4_SHAPES:
            test_perf(B, H, S, D, G, R, args.kid, args.splitK)
