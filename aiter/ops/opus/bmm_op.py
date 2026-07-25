# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus batched-BMM Python bindings.

This module is intentionally separate from `gemm_op_a16w16.py`: BMM callers use
batch-in-the-middle or grouped layouts (for example DSV4 `wo_a`) while the
underlying kernels still live in the shared opus GEMM backend.
"""

import functools
import logging
from typing import Optional

import torch

from ...jit.core import AITER_CONFIGS, compile_ops
from ...jit.utils.chip_info import get_gfx_runtime as _get_gfx

logger = logging.getLogger("aiter")


def _gen_bmm_a8w8_scale_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return Y


# mmajor fp8 block-scale BMM: x/Y are [M, batch, *] (dim0=M, dim1=batch),
# x_scale [M, batch, K/GROUP_K] (per-token M); wo_a + w_scale stay batch-major.
# Zero-copy DSV4 wo_a fp8 (no caller-side transpose). Y is fp32 today.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_scale",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_scale_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    kernelId: int = 710,
) -> torch.Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale_splitk",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_splitk_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 8,
) -> torch.Tensor: ...


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
) -> torch.Tensor: ...


# ---- Shape-driven mxscale flatmm BMM (CSV lookup + heuristic fallback) -----
#
# The raw flatmm binding above (`_opus_bmm_a8w8_mxscale_flatmm_splitk_raw`)
# has no tuning of its own: kernelId=0 just falls into the C++ `default:` (k32
# fused) branch, so the caller has to hardcode a kid to reach any of the strong
# tiles. This wrapper adds the missing layer, mirroring `gemm_a16w16_opus`:
#
#   1. explicit `kernelId` override -> forward verbatim (tuner / debugging).
#   2. otherwise look up the per-(gfx, g, m, n, k) tuned winner in the opus
#      mxscale CSV (aiter/configs/opus_bmm_a8w8_mxscale_tuned.csv). On hit,
#      launch that kid (+ its tuned splitK).
#   3. CSV miss -> a coarse M/G heuristic that picks the strongest kid whose
#      tile actually divides (m, n, k); anything unaligned falls back to
#      kernelId=0 (the C++ k32 fused kernel, which runs arbitrary shapes).
#
# The CSV is intentionally a separate file per scale type (see the note in
# jit/core.py): e8m0 block-scale here, a future fp32 rowwise-scale opus BMM in
# its own CSV, so keys never collide across scale semantics.


@functools.lru_cache(maxsize=1)
def _load_mxscale_bmm_config() -> dict:
    """Load the opus mxscale BMM tuned CSV into a {(gfx,g,m,n,k): row} dict.

    Keyed on the runtime gfx + shape.

    Two schema guards mirror aiter's established conventions:

    * ``scale`` column -- asserted to be uniformly ``mxscale`` (this file is
      the e8m0 block-scale table; a future fp32 rowwise-scale opus BMM lives
      in a separate CSV). A mis-filed row surfaces loudly.
    * ``libtype`` column -- rows are filtered to ``libtype == 'opus'``, exactly
      like ``aiter/ops/opus/common.py`` does for the a16w16 CSVs. This is the
      backend-selection seam: when a flydsl (or asm/triton) mxscale BMM lands,
      its rows co-exist under the same (gfx,g,m,n,k) keys with
      ``libtype == 'flydsl'`` and a neutral dispatcher routes on that column;
      this opus-backend loader only ever picks its own rows.

    Returns an empty dict (never raises) if the file is missing so callers
    degrade gracefully to the heuristic path.
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
    cfg = _load_mxscale_bmm_config().get((_get_gfx(), g, m, n, k))
    return cfg  # None on miss


def _heuristic_mxscale_kid(g: int, m: int, n: int, k: int) -> int:
    """Coarse M/G kid picker for shapes not in the tuned CSV.

    Follows the dominant pattern in opus_bmm_best_per_shape.md: the 256x256
    scale_pipe tile (kid 150) owns large-M / high-G; the fine m64n32k256 tile
    (kid 320) owns small-M when K is a multiple of 256; the m64n64k128 scale-
    prefetch tile (kid 653) is the general strong mid/small-M pick. Any shape
    that no listed tile divides falls back to kid 0 (C++ k32 fused), which has
    no tile-alignment requirement.
    """

    def div(a: int, b: int) -> bool:
        return a % b == 0

    # 256x256 scale_pipe: dominates once M is large and the tile fits.
    if div(m, 256) and div(n, 256) and div(k, 128):
        if m >= 2048 or (m >= 1024 and g >= 8):
            return 150
    # Sub-tile M (< 64): the main-launcher kids mask partial M via buffer OOB,
    # so a B_M=32/64 tile runs any M. Prefer the small m32n64k256 tile (640,
    # B_M=32, needs K % 256) to minimise wasted rows, else the m64n64k128
    # scale-prefetch tile (653, needs K % 128).
    if m < 64:
        return 640 if (div(n, 64) and div(k, 256)) else 653
    # Small-M at small K (<=1024) with K a multiple of 256: the fine
    # m64n32k256 tile (kid 320) is the winner there. At large K small-M
    # prefers the m64n64k128 scale-prefetch tile (handled below).
    if m <= 256 and k <= 1024 and div(m, 64) and div(n, 32) and div(k, 256):
        return 320
    # General strong mid/small-M pick.
    if div(m, 64) and div(n, 64) and div(k, 128):
        return 653
    # Nothing tile-aligned: k32 fused handles arbitrary shapes.
    return 0


def bmm_a8w8_mxscale_opus(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: Optional[int] = None,
    splitK: Optional[int] = None,
) -> torch.Tensor:
    """Shape-driven opus fp8 e8m0 mxscale (block-scale) BMM.

    mmajor DSV4 wo_a layout (matches the raw binding + op test):

    * ``x``       : [M, G, K] fp8 activation (per-token e8m0, transposed view
                    of the batch-major [G, M, K]).
    * ``wo_a``    : [G, N, K] fp8 weight (batch-major).
    * ``x_scale`` : [M, G, K/128] uint8 e8m0 activation scale.
    * ``w_scale`` : [G, N/128, K/128] uint8 e8m0 weight scale.
    * ``out``     : optional preallocated [M, G, N] output (fp32 or bf16).

    Kernel selection:

    * ``kernelId`` given  -> forwarded verbatim (bypasses lookup/heuristic).
    * ``kernelId`` None   -> per-(gfx, G, M, N, K) tuned CSV lookup; on miss a
      coarse M/G heuristic picks a runnable kid.
    * ``splitK`` defaults to the tuned value on a CSV hit, else 1.

    Returns the [M, G, N] output tensor.
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
    "_opus_bmm_a8w8_mxscale_raw",
    "_opus_bmm_a8w8_mxscale_splitk_raw",
    "_opus_bmm_a8w8_scale_raw",
    "bmm_a8w8_mxscale_opus",
]
