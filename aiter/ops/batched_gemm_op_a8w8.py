# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import pandas as pd
from ..jit.core import (
    compile_ops,
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx
from aiter import logger


def gen_batched_gemm_a8w8_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: int = 0,
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_a8w8",
    fc_name="batched_gemm_a8w8",
    gen_fake=gen_batched_gemm_a8w8_fake_tensors,
)
def batched_gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: int = 0,
) -> Tensor: ...


@functools.lru_cache(maxsize=1024)
def compute_batched_gemm_SplitK(
    M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int
):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


@functools.lru_cache(maxsize=1024)
def get_CKBatchedGEMM_config(
    B: int,
    M: int,
    N: int,
    K: int,
):
    if not hasattr(get_CKBatchedGEMM_config, "ck_batched_gemm_dict"):
        print(
            "Loading CKBatchedGEMM config from:",
            AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
        )
        ck_batched_gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE
        ).drop_duplicates()
        # Use (gfx, cu_num, B, M, N, K) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, B, M, N, K) for old CSVs that pre-date the gfx column.
        if "gfx" in ck_batched_gemm_dict.columns:
            get_CKBatchedGEMM_config.ck_batched_gemm_dict = (
                ck_batched_gemm_dict.set_index(
                    ["gfx", "cu_num", "B", "M", "N", "K"]
                ).to_dict("index")
            )
            get_CKBatchedGEMM_config.has_gfx = True
        else:
            logger.warning(
                f"{AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE} has no 'gfx' column -- "
                "falling back to cu_num-only key. Re-run the tuner or migrate the CSV."
            )
            get_CKBatchedGEMM_config.ck_batched_gemm_dict = (
                ck_batched_gemm_dict.set_index(["cu_num", "B", "M", "N", "K"]).to_dict(
                    "index"
                )
            )
            get_CKBatchedGEMM_config.has_gfx = False
    gfx = get_gfx()
    cu_num = get_cu_num()
    key = (
        (gfx, cu_num, B, M, N, K)
        if get_CKBatchedGEMM_config.has_gfx
        else (cu_num, B, M, N, K)
    )
    config = get_CKBatchedGEMM_config.ck_batched_gemm_dict.get(key, None)
    if config is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is B:{B}, M:{M}, N:{N}, K:{K}, is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE}, kernel name is {config['kernelName']}, splitK is {config['splitK']}!"
            )
        mnk = config["kernelName"].split("_")[3].split("x")[1:]
        config["tile_m"] = int(mnk[0])
        config["tile_n"] = int(mnk[1])
        config["tile_k"] = int(mnk[2])
    else:
        logger.info(
            f"shape is B:{B}, M:{M}, N:{N}, K:{K}, not found tuned config in CKGEMM, will use default config!"
        )
    return config


def batched_gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype=dtypes.bf16,
    splitK: Optional[int] = None,
):
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"

    b = XQ.shape[0]
    m = XQ.shape[1]
    n = WQ.shape[1]
    k = XQ.shape[2]
    ck_config = get_CKBatchedGEMM_config(b, m, n, k)
    if splitK is None:
        if ck_config is not None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(b, m, n, dtype=dtype, device=XQ.device)
    return batched_gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias, splitK)


# ---------------------------------------------------------------------------
# fp8 e8m0 mxscale (block-scale) batched GEMM -- backend-neutral public entry.
#
# This file is the per-family (a8w8 batched) public surface, not a CK-only
# file: like aiter/ops/gemm_op_a8w8.py hosts gemm_a8w8 (CK rowwise) +
# gemm_a8w8_blockscale (ck/cktile/triton/asm) side by side and lazy-imports
# backend impls, we host the mxscale batched entry here too. The concrete
# kernels stay in their backend dirs (opus -> aiter.ops.opus.bmm_op; a future
# flydsl backend lands under aiter.ops.flydsl and adds a `backend` branch
# below + libtype='flydsl' rows in the same tuned CSV).
def batched_gemm_a8w8_mxscale(
    x: Tensor,
    wo_a: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    *,
    backend: Optional[str] = None,
    kernelId: Optional[int] = None,
    splitK: Optional[int] = None,
) -> Tensor:
    """fp8 e8m0 mxscale (128x128 block-scale) batched GEMM, backend-neutral.

    mmajor DSV4 wo_a layout (matches the opus kernels + op test):

    * ``x``       : [M, G, K] fp8 activation (per-token e8m0; transposed view
                    of batch-major [G, M, K]).
    * ``wo_a``    : [G, N, K] fp8 weight (batch-major).
    * ``x_scale`` : [M, G, K/128] uint8 e8m0 activation scale.
    * ``w_scale`` : [G, N/128, K/128] uint8 e8m0 weight scale.
    * ``out``     : optional preallocated [M, G, N] output (fp32 or bf16).

    Note this is *microscaling* (e8m0) block scale -- distinct from
    ``gemm_a8w8_blockscale`` which uses fp32 block scale. Scale type is baked
    into the name so a future fp32-block batched variant stays separate.

    ``backend`` defaults to the only backend available today (opus); a future
    flydsl backend adds a branch below. ``kernelId`` / ``splitK`` are
    opus-backend overrides forwarded when that backend is selected.
    """
    if backend in (None, "opus"):
        from .opus.bmm_op import bmm_a8w8_mxscale_opus

        return bmm_a8w8_mxscale_opus(
            x,
            wo_a,
            x_scale,
            w_scale,
            out,
            dtype=dtype,
            kernelId=kernelId,
            splitK=splitK,
        )
    raise NotImplementedError(
        f"batched_gemm_a8w8_mxscale: backend {backend!r} not implemented "
        f"(available: 'opus'). flydsl/asm backends land under their own "
        f"aiter.ops.<backend> module and register libtype rows in the tuned CSV."
    )


def gen_batched_gemm_a8w8_tune_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor:
    return out


@compile_ops(
    "module_batched_gemm_a8w8_tune",
    fc_name="batched_gemm_a8w8_tune",
    gen_fake=gen_batched_gemm_a8w8_tune_fake_tensors,
)
def batched_gemm_a8w8_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...
