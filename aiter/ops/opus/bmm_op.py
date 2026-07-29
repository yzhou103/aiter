# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus batched-BMM Python bindings.

This module is intentionally separate from `gemm_op_a16w16.py`: BMM callers use
batch-in-the-middle or grouped layouts (for example DSV4 `wo_a`) while the
underlying kernels still live in the shared opus GEMM backend.
"""

import bisect
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

    Rows are filtered to ``libtype == 'opus'`` (backend-selection seam).
    Returns {} if the file is missing so callers fall back to the heuristic.
    """
    import pandas as pd

    path = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
    try:
        df = pd.read_csv(path).drop_duplicates()
    except FileNotFoundError:
        logger.warning("opus mxscale BMM tuned CSV not found at %s", path)
        return {}
    if "libtype" in df.columns:
        # Backend-selection seam (see docstring): only consume opus rows.
        df = df[df["libtype"] == "opus"]
    return df.set_index(["gfx", "b", "m", "n", "k"])[["kernelId", "splitK"]].to_dict(
        "index"
    )


@functools.lru_cache(maxsize=1)
def _mxscale_bmm_buckets() -> dict:
    """Per-(gfx,g,n,k) sorted list of tuned M buckets, for padded-M lookup.

    Mirrors the GEMM ``get_padded_m`` idea but keyed on the BMM batch dim ``g``
    (the tuned table is per-g), so the nearest tuned M is chosen within the same
    (g,n,k) family instead of a g-agnostic global rule.
    """
    buckets: dict = {}
    for gfx, g, m, n, k in _load_mxscale_bmm_config():
        buckets.setdefault((gfx, g, n, k), []).append(m)
    for ms in buckets.values():
        ms.sort()
    return buckets


# OOB-masked sub-tiles (B_M=32/64): partial-M is predicated via buffer OOB, so
# these run any M. The strong pipeline/minterleave tiles (137/139/150/158/163)
# instead grid on M/B_M with a hard ``M % B_M == 0`` assert -> aligned M only.
ARBITRARY_M_KIDS = frozenset({311, 313, 320, 321, 324, 640, 650, 653})


def _lookup_mxscale_bmm(g: int, m: int, n: int, k: int):
    """Exact CSV lookup, then GEMM-style padded-M (round M up to nearest tuned
    bucket in the same (g,n,k) family). Returns ``(cfg, padded_m)``; ``cfg`` is
    ``None`` only when neither exact nor any padded bucket exists.
    """
    gfx = _get_gfx()
    cfgmap = _load_mxscale_bmm_config()
    tuned_file = AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE

    cfg = cfgmap.get((gfx, g, m, n, k))
    if cfg is not None:
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is G:{g}, M:{m}, N:{n}, K:{k}, is tuned on gfx = {gfx} in "
                f"{tuned_file}, kernelId is {cfg['kernelId']}, splitK is {cfg['splitK']}!"
            )
        return cfg, m

    ms = _mxscale_bmm_buckets().get((gfx, g, n, k))
    if ms:
        i = bisect.bisect_left(ms, m)
        if i < len(ms):
            padded_m = ms[i]
            pcfg = cfgmap[(gfx, g, padded_m, n, k)]
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape is G:{g}, M:{m}, N:{n}, K:{k}, exact miss; using "
                    f"padded_M: {padded_m} kernelId {pcfg['kernelId']} "
                    f"(splitK {pcfg['splitK']}) from {tuned_file}!"
                )
            return pcfg, padded_m

    logger.info(
        f"shape is G:{g}, M:{m}, N:{n}, K:{k}, not found tuned/padded config in "
        f"{tuned_file}, will use heuristic/M-split fallback!"
    )
    return None, m


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
    # Sub-tile M: B_M=32/64 tiles mask partial M via buffer OOB, so run any M
    # (no m-alignment needed -- verified 653/321/... run arbitrary unaligned M).
    if m < 64:
        return 640 if (div(n, 64) and div(k, 256)) else 653
    if m <= 256 and k <= 1024 and div(n, 32) and div(k, 256):
        return 320
    if div(n, 64) and div(k, 128):
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
        cfg, padded_m = _lookup_mxscale_bmm(g, m, n, k)
        # Accept the tuned config when it is either an exact hit, or a padded-M
        # hit whose kernel is an OOB-masked sub-tile (runs the real, smaller M
        # with zero pad/copy). A padded-M hit on a strong tile is rejected here:
        # that kernel asserts M % B_M == 0 and would fault on the real M.
        if cfg is not None and (
            padded_m == m or int(cfg["kernelId"]) in ARBITRARY_M_KIDS
        ):
            kernelId = int(cfg["kernelId"])
            if splitK is None:
                splitK = int(cfg["splitK"])
        else:
            # No usable tuned config -> heuristic fallback path. For large tile-
            # unaligned M, the strong large-M tiles (158/150) win big but need
            # M % 256 == 0, so split into an aligned bulk (strong tile) + a
            # <256-row OOB-safe remainder (zero-copy dim0 views). The win is
            # gated on total bulk work g*m, not g alone: measured break-even is
            # g*m ~= 8192 (below it the extra kernel launch + tiny tail make the
            # full-M sub-tile faster; e.g. g=1 m=4100 is 0.76x, but g=1 m=16100
            # is 1.87x and g>=2 wins from ~m=4k). Below the gate we use a single
            # sub-tile heuristic kid (653) on the full real M instead.
            if (
                splitK is None
                and n % 256 == 0
                and k % 128 == 0
                and m >= 512
                and m % 256 != 0
                and g * m >= 8192
            ):
                m_bulk = (m // 256) * 256
                # kid158 preloads the SFA panel into LDS and early-returns for
                # K>8192 (SFA_K_MAX); kid150 (plain, no preload) runs any K. Cap
                # the preload pick at K<=8192 so K>8192 stays correct on kid150.
                bulk_kid = 158 if 4096 <= k <= 8192 else 150
                tail_kid = _heuristic_mxscale_kid(g, m - m_bulk, n, k)
                _opus_bmm_a8w8_mxscale_raw(
                    x[:m_bulk], wo_a, Y[:m_bulk], x_scale[:m_bulk], w_scale, 1, bulk_kid
                )
                _opus_bmm_a8w8_mxscale_raw(
                    x[m_bulk:], wo_a, Y[m_bulk:], x_scale[m_bulk:], w_scale, 1, tail_kid
                )
                return Y
            kernelId = _heuristic_mxscale_kid(g, m, n, k)
    if splitK is None:
        splitK = 1

    _opus_bmm_a8w8_mxscale_raw(x, wo_a, Y, x_scale, w_scale, int(splitK), int(kernelId))
    return Y


__all__ = [
    "_opus_bmm_a8w8_mxscale_raw",
    "bmm_a8w8_mxscale_opus",
]
