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


@functools.cache
def _mxscale_kid_m_align() -> dict[int, int]:
    """kid -> M multiple its launcher requires (1 == it masks a partial M tile).

    Comes from the codegen instance table, which is also what the tuner filters
    candidates on. This used to be a hand-kept kid allowlist here and a second
    hand-kept m_align column in the tuner, and the two disagreed: kid326 was
    dispatched at unaligned M by this file while the tuner never tuned it there,
    which cost ~9% at the wo_a decode shapes.
    """
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): int(inst.m_align)
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
    }


def _kid_runs_m(kid: int, m: int) -> bool:
    """True iff kid's launcher accepts this M (unknown kid -> assume it does not)."""
    align = _mxscale_kid_m_align().get(int(kid))
    return align is not None and m % align == 0


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
        f"{tuned_file}, will use heuristic fallback!"
    )
    return None, m


def _heuristic_mxscale_kid(g: int, m: int, n: int, k: int) -> int:
    """Coarse M/G kid picker for shapes not in the tuned CSV.

    kid 158 (512x256 preload pipeline) for large-M/high-G, falling back to kid 150
    (256x256 plain) for K>8192 where 158 early-returns; kid 320/640 for small-M;
    kid 653 the general strong mid/small-M pick; kid 0 (k32 fused) for shapes that
    are not tile-aligned in N or K.
    """

    def div(a: int, b: int) -> bool:
        return a % b == 0

    if div(n, 256) and div(k, 128) and (m >= 2048 or (m >= 1024 and g >= 8)):
        # Large M: the preload pipeline (kid158) is the tuned winner across this
        # whole region (CSV picks 158 for every aligned m>=2048). No M alignment
        # needed -- the pipeline family masks its partial trailing tile via buffer
        # OOB. kid158 stages the SFA/SFB scales into LDS and early-returns for
        # K>8192 (SFA_K_MAX), so gate the preload pick at K<=8192 and fall back to
        # the plain 256x256 (kid150) for K>8192. Measured on g=2,n=1024,k=4096:
        # kid150 was 34-51% slower than 158 at the untuned m=2560/3072/3584
        # buckets, and on unaligned M a single kid158 launch beats the sub-tile
        # kid653 by 13-34% (g2/m2624, g8/m1000, g16/m600).
        return 158 if 4096 <= k <= 8192 else 150
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
    [M, G, N]. ``kernelId`` given -> verbatim; None -> tuned CSV then heuristic.
    ``splitK`` defaults to the tuned value on a hit, else 1. Returns the [M, G, N]
    output.
    """
    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])

    if out is not None:
        Y = out
    else:
        Y = torch.empty((m, g, n), dtype=dtype, device=x.device)

    if kernelId is None:
        cfg, padded_m = _lookup_mxscale_bmm(g, m, n, k)
        # Accept an exact hit, or a padded-M hit whose kernel accepts the real,
        # smaller M (it runs it with no pad/copy). A padded-M hit on an
        # aligned-only tile is rejected: that kernel's launcher would throw.
        if cfg is not None and (padded_m == m or _kid_runs_m(int(cfg["kernelId"]), m)):
            kernelId = int(cfg["kernelId"])
            if splitK is None:
                splitK = int(cfg["splitK"])
        else:
            kernelId = _heuristic_mxscale_kid(g, m, n, k)
    if splitK is None:
        splitK = 1

    _opus_bmm_a8w8_mxscale_raw(x, wo_a, Y, x_scale, w_scale, int(splitK), int(kernelId))
    return Y


__all__ = [
    "_opus_bmm_a8w8_mxscale_raw",
    "bmm_a8w8_mxscale_opus",
]
