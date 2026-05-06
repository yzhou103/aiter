# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import pandas as pd
import time

import aiter
from aiter import dtypes
from aiter.ops.gemm_op_a16w16 import gemm_a16w16_hip
from aiter.test_common import checkAllclose


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6


configs = [
    ("v2_4x4",       4),   # V2: 4x4, no sched_barrier, BLOCK=128x256, TK=32, LDS=55KB
]


def fmt_result(t, flops):
    tflops = flops / t / 1e6
    return f"{t:.0f}us {tflops:.1f}TF"


def test_gemm_hip(m, n, k, dtype=dtypes.bf16):
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    weight = torch.randn(n, k, dtype=dtype, device="cuda")
    ref = F.linear(x, weight)
    flops = 2.0 * m * n * k

    t_torch = bench(lambda: F.linear(x, weight))

    results = {"M": m, "N": n, "K": k, "torch(hipBLASLt)": fmt_result(t_torch, flops)}

    for name, splitk in configs:
        out = torch.empty(m, n, dtype=dtype, device="cuda")
        try:
            t = bench(lambda: gemm_a16w16_hip(x, weight, out, False, 0, splitk))
            gemm_a16w16_hip(x, weight, out, False, 0, splitk)
            torch.cuda.synchronize()
            ok = torch.allclose(ref, out, atol=0.5, rtol=0.05)
            tflops = flops / t / 1e6
            tag = "" if ok else " FAIL"
            results[name] = f"{t:.0f}us {tflops:.1f}TF{tag}"
        except Exception as e:
            results[name] = f"ERR"

    return results


if __name__ == "__main__":
    shapes = [
        #(256, 4096, 4096),
        (512, 4096, 4096),
        #(1024, 4096, 4096),
        #(4096, 4096, 4096),
        #(1024, 11008, 4096),
    ]

    df = []
    for m, n, k in shapes:
        ret = test_gemm_hip(m, n, k)
        df.append(ret)

    df = pd.DataFrame(df)
    print("\n" + df.to_markdown(index=False))
