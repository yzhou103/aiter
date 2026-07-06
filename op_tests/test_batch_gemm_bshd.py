# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test BSHD-fused batch GEMM for MLA output projection.

BSHD layout: A = [B, S, H, D] (DSv4 default from sparse_attn).
Reference:
    out[b,s,g,r] = sum_d A[b, s, g*hpg+h, d] * W[g, r, h*D+d]
    Equivalent to: torch.einsum("bsgd,grd->bsgr", A.reshape(B,S,G,-1), W)

Two pipeline paths:
  --standard (default): standard a16w16 batch GEMM with strided 3D view [G, S, K]
  --bhsd-remap:         BHSD a_offset remapping with strided 4D view [G, hpg, S, D]
"""

import argparse

import torch
from aiter.test_common import checkAllclose, perftest, run_perftest

TEST_NUM_ITERS = 100


def reference_einsum(A, W, heads_per_group):
    """Torch reference: reshape + einsum (no transpose needed for BSHD)."""
    B, S, H, D = A.shape
    G = H // heads_per_group
    o = A.reshape(B, S, G, heads_per_group * D)
    return torch.einsum("bsgd,grd->bsgr", o.float(), W.float()).to(A.dtype)


@perftest(num_iters=TEST_NUM_ITERS)
def run_reference(A, W, heads_per_group):
    return reference_einsum(A, W, heads_per_group)


@perftest(num_iters=TEST_NUM_ITERS)
def run_bshd_opus(A, W, heads_per_group, kernelId, splitK, use_standard):
    from aiter.ops.opus.gemm_op_a16w16 import batch_gemm_a16w16_bshd_opus

    return batch_gemm_a16w16_bshd_opus(
        A, W, heads_per_group,
        kernelId=kernelId, splitK=splitK,
        use_standard_pipeline=use_standard,
    )


# DeepSeek-V4 shapes: (B, S, H, D, G, R)
DSV4_SHAPES = [
    (1, 1, 64, 512, 8, 1024),      # decode bs=1
    (1, 4, 64, 512, 8, 1024),      # decode bs=4
    (1, 128, 64, 512, 8, 1024),    # short prefill
    (1, 1024, 64, 512, 8, 1024),   # medium prefill
    (1, 4096, 64, 512, 8, 1024),   # long prefill
    (4, 1, 64, 512, 8, 1024),      # batched decode
    (4, 128, 64, 512, 8, 1024),    # batched short prefill
]


@torch.inference_mode()
def test_accuracy(B, S, H, D, G, R, kid, splitK, use_standard):
    heads_per_group = H // G
    K = heads_per_group * D

    A = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(G, R, K, dtype=torch.bfloat16, device="cuda")

    ref = reference_einsum(A, W, heads_per_group)

    from aiter.ops.opus.gemm_op_a16w16 import batch_gemm_a16w16_bshd_opus

    out = batch_gemm_a16w16_bshd_opus(
        A, W, heads_per_group,
        kernelId=kid, splitK=splitK,
        use_standard_pipeline=use_standard,
    )

    pipe = "standard" if use_standard else "bhsd_remap"
    msg = f"B={B} S={S} H={H} D={D} G={G} R={R} kid={kid} splitK={splitK} pipe={pipe}"
    checkAllclose(out, ref, msg=msg, rtol=1e-2, atol=1e-2)


@torch.inference_mode()
def test_perf(B, S, H, D, G, R, kid, splitK, use_standard):
    heads_per_group = H // G
    K = heads_per_group * D

    A = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(G, R, K, dtype=torch.bfloat16, device="cuda")

    ref_out, ref_us = run_reference(A, W, heads_per_group)
    bshd_out, bshd_us = run_bshd_opus(A, W, heads_per_group, kid, splitK, use_standard)

    M = B * S
    N = R
    K_total = K
    flops = 2 * M * N * K_total * G
    ref_tflops = flops / ref_us / 1e6
    bshd_tflops = flops / bshd_us / 1e6
    speedup = ref_us / bshd_us

    pipe = "std" if use_standard else "remap"
    print(
        f"  B={B:2d} S={S:5d} | "
        f"ref {ref_us:8.1f}us ({ref_tflops:6.1f} TFLOPS) | "
        f"bshd[{pipe}] kid={kid} {bshd_us:8.1f}us ({bshd_tflops:6.1f} TFLOPS) | "
        f"speedup {speedup:.2f}x"
    )

    checkAllclose(bshd_out, ref_out, msg=f"perf check B={B} S={S}", rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSHD batch GEMM test")
    parser.add_argument("--kid", type=int, default=None,
                        help="kernel ID (default: 208 for standard, 608 for bhsd-remap)")
    parser.add_argument("--splitK", type=int, default=0, help="split K factor")
    parser.add_argument("--perf", action="store_true", help="run perf benchmarks")
    parser.add_argument("--bhsd-remap", action="store_true",
                        help="use BHSD a_offset remapping pipeline instead of standard")
    parser.add_argument("--both", action="store_true",
                        help="test both pipelines (accuracy only)")
    args = parser.parse_args()

    pipelines = []
    if args.both:
        pipelines = [
            ("standard", True, args.kid or 208),
            ("bhsd_remap", False, args.kid or 608),
        ]
    elif args.bhsd_remap:
        pipelines = [("bhsd_remap", False, args.kid or 608)]
    else:
        pipelines = [("standard", True, args.kid or 208)]

    for pipe_name, use_std, kid in pipelines:
        print(f"=== BSHD Accuracy tests [{pipe_name}] kid={kid} ===")
        for B, S, H, D, G, R in DSV4_SHAPES:
            test_accuracy(B, S, H, D, G, R, kid, args.splitK, use_std)
        print(f"All BSHD accuracy tests passed [{pipe_name}]!")

    if args.perf:
        for pipe_name, use_std, kid in pipelines:
            print(f"\n=== BSHD Performance benchmarks [{pipe_name}] kid={kid} ===")
            for B, S, H, D, G, R in DSV4_SHAPES:
                test_perf(B, S, H, D, G, R, kid, args.splitK, use_std)
