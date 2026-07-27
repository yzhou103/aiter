# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus batched-BMM Python bindings.

This module is intentionally separate from `gemm_op_a16w16.py`: BMM callers use
batch-in-the-middle or grouped layouts (for example DSV4 `wo_a`) while the
underlying kernels still live in the shared opus GEMM backend.
"""

import functools
import logging

import torch

from ...jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ...jit.utils.chip_info import get_gfx_runtime as _get_gfx

logger = logging.getLogger("aiter")


def _gen_bmm_a8w8_scale_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> None:
    # In-place mutation of ``Y``; fake must mirror the void C++ op (full arg
    # list + None return) so torch.compile registers a mutating op, not a
    # tensor-producing one.
    return None


# mmajor fp8 e8m0 mxscale BMM raw binding: x/Y are [M, batch, *], wo_a + w_scale
# batch-major (zero-copy DSV4 wo_a). kid-dispatched; driven by
# bmm_a8w8_mxscale_opus below.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_flatmm_splitk",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_flatmm_splitk_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> None:
    # In-place: result written into ``Y``, void return (``-> None`` keeps it
    # torch.compile-safe as a mutating op). Callers read ``Y``.
    ...


# ---- Shape-driven mxscale flatmm BMM (CSV lookup + heuristic fallback) -----
# The raw binding has no tuning of its own (kernelId=0 -> slow k32 fused). This
# wrapper adds selection: explicit kernelId -> verbatim; else tuned-CSV winner;
# else M-split for large unaligned M; else a coarse M/G heuristic.


@functools.lru_cache(maxsize=1)
def _load_mxscale_bmm_config() -> dict:
    """Load the opus mxscale BMM tuned CSV into a {(gfx,g,m,n,k): row} dict.

    Guards: ``scale`` must be uniformly ``mxscale`` (this is the e8m0 table),
    and rows are filtered to ``libtype == 'opus'`` (backend-selection seam).
    Returns {} if the file is missing so callers fall back to the heuristic.
    """
    import pandas as pd

    path = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    try:
        df = pd.read_csv(path).drop_duplicates()
    except FileNotFoundError:
        logger.warning("opus mxscale BMM tuned CSV not found at %s", path)
        return {}
    if "scale" in df.columns:
        bad = set(df["scale"].unique()) - {"mxscale"}
        assert not bad, (
            f"{path}: expected all rows to have scale=='mxscale' (this file is "
            f"the e8m0 block-scale table); found unexpected scale values {bad}. "
            f"fp32 rowwise-scale configs belong in a separate CSV."
        )
    if "libtype" in df.columns:
        # Backend-selection seam (see docstring): only consume opus rows.
        df = df[df["libtype"] == "opus"]
    return df.set_index(["gfx", "b", "m", "n", "k"])[["kernelId", "splitK"]].to_dict(
        "index"
    )


def _lookup_mxscale_bmm(g: int, m: int, n: int, k: int):
    gfx = _get_gfx()
    cfg = _load_mxscale_bmm_config().get((gfx, g, m, n, k))
    tuned_file = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    if cfg is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is G:{g}, M:{m}, N:{n}, K:{k}, is tuned on gfx = {gfx} in "
                f"{tuned_file}, kernelId is {cfg['kernelId']}, splitK is {cfg['splitK']}!"
            )
    else:
        logger.info(
            f"shape is G:{g}, M:{m}, N:{n}, K:{k}, not found tuned config in "
            f"{tuned_file}, will use heuristic/M-split fallback!"
        )
    return cfg  # None on miss


def _heuristic_mxscale_kid(g: int, m: int, n: int, k: int) -> int:
    """Coarse M/G kid picker for shapes not in the tuned CSV.

    kid 150 (256x256) for large-M/high-G; kid 320/640 for small-M; kid 653 the
    general strong mid/small-M pick; kid 0 (k32 fused) for unaligned shapes.
    """

    def div(a: int, b: int) -> bool:
        return a % b == 0

    if (
        div(m, 256)
        and div(n, 256)
        and div(k, 128)
        and (m >= 2048 or (m >= 1024 and g >= 8))
    ):
        return 150
    # Sub-tile M: B_M=32/64 tiles mask partial M via buffer OOB, so run any M.
    if m < 64:
        return 640 if (div(n, 64) and div(k, 256)) else 653
    if m <= 256 and k <= 1024 and div(m, 64) and div(n, 32) and div(k, 256):
        return 320
    if div(m, 64) and div(n, 64) and div(k, 128):
        return 653
    return 0  # nothing tile-aligned: k32 fused runs arbitrary shapes


def bmm_a8w8_mxscale_opus(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: int | None = None,
    splitK: int | None = None,
) -> torch.Tensor:
    """Shape-driven opus fp8 e8m0 mxscale (block-scale) BMM.

    mmajor DSV4 wo_a layout: ``x`` [M, G, K] fp8, ``wo_a`` [G, N, K] fp8,
    ``x_scale`` [M, G, K/128], ``w_scale`` [G, N/128, K/128], ``out`` optional
    [M, G, N]. ``kernelId`` given -> verbatim; None -> tuned CSV / M-split /
    heuristic. ``splitK`` defaults to the tuned value on a hit, else 1. Returns
    the [M, G, N] output.
    """
    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])

    if out is not None:
        Y = out
    else:
        Y = torch.empty((m, g, n), dtype=dtype, device=x.device)

    if kernelId is None:
        cfg = _lookup_mxscale_bmm(g, m, n, k)
        if cfg is not None:
            kernelId = int(cfg["kernelId"])
            if splitK is None:
                splitK = int(cfg["splitK"])
        elif (
            splitK is None
            and n % 256 == 0
            and k % 128 == 0
            and m >= 512
            and m % 256 != 0
        ):
            # Large tile-unaligned M: the strong large-M tiles (157/150) need
            # M % 256 == 0, so split into an aligned bulk (strong tile) + a
            # <256-row OOB-safe remainder instead of dropping to k32 fused
            # (~0.35x -> ~1.15x bf16). Zero-copy dim0 views; splitK=1 each.
            m_bulk = (m // 256) * 256
            bulk_kid = 157 if k >= 4096 else 150
            tail_kid = _heuristic_mxscale_kid(g, m - m_bulk, n, k)
            _opus_bmm_a8w8_mxscale_flatmm_splitk_raw(
                x[:m_bulk], wo_a, Y[:m_bulk], x_scale[:m_bulk], w_scale, 1, bulk_kid
            )
            _opus_bmm_a8w8_mxscale_flatmm_splitk_raw(
                x[m_bulk:], wo_a, Y[m_bulk:], x_scale[m_bulk:], w_scale, 1, tail_kid
            )
            return Y
        else:
            kernelId = _heuristic_mxscale_kid(g, m, n, k)
    if splitK is None:
        splitK = 1

    _opus_bmm_a8w8_mxscale_flatmm_splitk_raw(
        x, wo_a, Y, x_scale, w_scale, int(splitK), int(kernelId)
    )
    return Y


__all__ = [
    "_opus_bmm_a8w8_mxscale_flatmm_splitk_raw",
    "bmm_a8w8_mxscale_opus",
]
