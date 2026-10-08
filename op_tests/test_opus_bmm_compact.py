# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compact MXFP8 BMM: general N/K, segmented scales and preserved fast path."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus import opus_bmm
from aiter.ops.opus.gemm_op_a8w8 import _opus_gemm_a8w8_mxscale_bmm_launch_raw as raw
from aiter.ops.opus.launch_plan import _build_a8w8_mxscale_bmm_plan as plan
from aiter.ops.shuffle import shuffle_weight
from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_bmm_bpreshuffle_compact_kernels_list as kernels,
)
from op_tests.test_opus_a8w8_bmm import (
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="gfx950 required"
)
KIDS = tuple(sorted(kernels))
# Includes one-tile K, partial scale dwords, non-power-of-two N, one- and
# two-tile final K segments, more than two segments, and M/batch boundaries.
SHAPES = [
    (1, 1, 128, 128),
    (1, 1, 128, 512),
    (2, 17, 384, 1536),
    (16, 2047, 128, 512),
    (3, 1, 256, 256),
    (2, 63, 384, 384),
    (3, 65, 512, 512),
    (2, 95, 768, 768),
    (4, 97, 512, 1024),
    (1, 193, 2048, 2048),
    (3, 65, 4096, 1024),
    (2, 257, 512, 4096),
    (3, 65, 1024, 4096),
    (3, 65, 384, 4224),
    (2, 97, 512, 4352),
    (3, 65, 384, 4608),
    (2, 97, 512, 9216),
    (1, 65, 256, 4480),
    (4, 193, 2048, 8192),
    (2, 65, 128, 8320),
    (3, 65, 384, 8448),
    (1, 97, 256, 12544),
    (16, 2047, 128, 256),
    (16, 1, 512, 4352),
]


def inputs(b, m, n, k, dtype, seed=41):
    torch.manual_seed(seed)

    def varied(shape):
        values = torch.randn(shape, device="cuda", dtype=torch.float32)
        amplitude = torch.exp2(
            torch.randint(-3, 4, (*shape[:-1], k // 128, 1), device="cuda").float()
        )
        return (
            (values.view(*shape[:-1], k // 128, 128) * amplitude).view(shape).bfloat16()
        )

    x, xs, xs_float = _quant_per_token_e8m0(varied((b, m, k)))
    w, ws, ws_float = _quant_block_e8m0(varied((b, n, k)))
    ref = run_torch(x, w, xs_float, ws_float).transpose(0, 1).to(dtype)
    return (
        x.transpose(0, 1).contiguous(),
        shuffle_weight(w, (16, 16)),
        xs.transpose(0, 1).contiguous(),
        ws,
        ref,
    )


def check_result(y, ref, storage):
    assert torch.isfinite(y).all()
    delta = (y.float() - ref.float()).abs()
    assert (delta.mean() / ref.float().abs().mean().clamp_min(1e-9)).item() < 0.001
    assert (delta > 0.01 + 0.01 * ref.float().abs()).float().mean().item() < 0.001
    assert (storage[:64] == 123).all() and (storage[-64:] == 123).all()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compact_general(shape, dtype):
    b, m, n, k = shape
    x, w, xs, ws, ref = inputs(*shape, dtype)
    storage = torch.full((m * b * n + 128,), 123, dtype=dtype, device="cuda")
    y = storage[64:-64].view(m, b, n)
    for kid in KIDS:
        if k % kernels[kid].B_K:
            continue
        y.fill_(float("nan"))
        raw(x, w, y, xs, ws, workspace=None, kid=kid, split_k=1)
        torch.cuda.synchronize()
        check_result(y, ref, storage)
        first = y.clone()
        y.fill_(float("nan"))
        opus_bmm(
            x.transpose(0, 1),
            w,
            y.transpose(0, 1),
            kid=kid,
            layout="mxscale_bmm",
            x_scale=xs.transpose(0, 1),
            w_scale=ws,
        )
        assert torch.equal(y, first)


@pytest.mark.parametrize("kid", KIDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [4352, 4608])
def test_compact_segment_graph(kid, dtype, k):
    if k % kernels[kid].B_K:
        pytest.skip("K must be divisible by this kid's B_K")
    shape = (3, 65, 384, k)
    x, w, xs, ws, ref = inputs(*shape, dtype, seed=97)
    storage = torch.full((65 * 3 * 384 + 128,), 123, dtype=dtype, device="cuda")
    y = storage[64:-64].view(65, 3, 384)
    raw(x, w, y, xs, ws, workspace=None, kid=kid, split_k=1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw(x, w, y, xs, ws, workspace=None, kid=kid, split_k=1)
    for _ in range(3):
        y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check_result(y, ref, storage)


@pytest.mark.parametrize("kid", KIDS)
def test_compact_plan_and_tuner(kid):
    from csrc.opus_gemm.opus_bmm_mxscale_tune import _applicable

    for b, m, n, k in SHAPES:
        if k % kernels[kid].B_K:
            continue
        plan(
            arch="gfx950",
            kid=kid,
            output_dtype=torch.bfloat16,
            M=m,
            batch=b,
            N=n,
            K=k,
            split_k=1,
        )
        assert _applicable(kid, b, m, n, k) == [1]
        assert _applicable(kid, b, m, n, k, split_ks=[2, 4]) == []
    for b, m, n, k in [
        (17, 65, 512, 2048),
        (3, 2048, 512, 2048),
        (3, 65, 192, 2048),
        (3, 65, 512, 129),
        (1, 1, 1048576, 2048),
        (16, 2047, 65536, 256),
    ]:
        with pytest.raises(ValueError):
            plan(
                arch="gfx950",
                kid=kid,
                output_dtype=torch.bfloat16,
                M=m,
                batch=b,
                N=n,
                K=k,
                split_k=1,
            )
        assert _applicable(kid, b, m, n, k) == []
    with pytest.raises(ValueError):
        plan(
            arch="gfx950",
            kid=kid,
            output_dtype=torch.float32,
            M=65,
            batch=3,
            N=512,
            K=4352,
            split_k=2,
        )


@pytest.mark.parametrize("kid", KIDS)
def test_compact_raw_guards(kid):
    x, w, xs, ws, ref = inputs(3, 65, 512, 4608, torch.bfloat16)
    y = torch.empty_like(ref, memory_format=torch.contiguous_format)
    args = [x, w, y, xs, ws]
    with pytest.raises((ValueError, RuntimeError)):
        raw(*args, workspace=None, kid=kid, split_k=2)
    with pytest.raises((ValueError, RuntimeError)):
        raw(*args, workspace=torch.empty(1, device="cuda"), kid=kid, split_k=1)
    for index, tensor in enumerate(args):
        shifted = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device="cuda")[
            1:
        ].view(tensor.shape)
        bad = args.copy()
        bad[index] = shifted
        with pytest.raises((ValueError, RuntimeError)):
            raw(*bad, workspace=None, kid=kid, split_k=1)
        shape = list(tensor.shape)
        shape[0] *= 2
        bad[index] = torch.empty(shape, dtype=tensor.dtype, device="cuda")[::2]
        with pytest.raises((ValueError, RuntimeError)):
            raw(*bad, workspace=None, kid=kid, split_k=1)
    # K=128 is legal for BK=128 schedules; larger BK must reject it.
    if kernels[kid].B_K > 128:
        x, w, xs, ws, ref = inputs(1, 1, 128, 128, torch.bfloat16)
        with pytest.raises((ValueError, RuntimeError)):
            raw(x, w, torch.empty_like(ref), xs, ws, workspace=None, kid=kid, split_k=1)


@pytest.mark.parametrize(
    "kid,shape",
    [
        (8476, (1, 1, 128, 128)),
        (8476, (3, 65, 384, 4224)),
        (8470, (2, 97, 512, 4352)),
        (8474, (2, 193, 512, 8192)),
        (8477, (2, 17, 384, 4608)),
        (8478, (2, 129, 384, 4608)),
        (8479, (2, 129, 384, 4608)),
        (8480, (2, 129, 384, 4608)),
        (8481, (2, 129, 384, 4608)),
        (8482, (2, 129, 384, 4608)),
        (8483, (2, 129, 384, 4608)),
        (8484, (2, 129, 384, 4608)),
    ],
)
def test_compact_tuner_data_path(kid, shape):
    from csrc.opus_gemm.opus_bmm_mxscale_tune import (
        gen_bmm_mxscale_data,
        run_bmm_mxscale_bench,
    )

    data = gen_bmm_mxscale_data(*shape, 97, torch.bfloat16, kid, 1)
    x, w, y, xs, ws, workspace, ref, w_sh, xs_sh, ws_sh = data
    run_bmm_mxscale_bench(x, w, y, xs, ws, workspace, w_sh, xs_sh, ws_sh, kid, 1)
    torch.cuda.synchronize()
    delta = (y.float() - ref.float()).abs()
    assert (delta.mean() / ref.float().abs().mean()).item() < 0.001
    assert (delta > 0.01 + 0.01 * ref.float().abs()).float().mean().item() < 0.001
