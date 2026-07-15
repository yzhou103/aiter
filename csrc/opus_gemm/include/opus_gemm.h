// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Top-level opus_gemm entry points. Uses aiter_tensor_t (POD,
// torch-free) instead of torch::Tensor so this header costs ~200
// preprocessed lines instead of the ~50K that <torch/all.h> +
// <torch/extension.h> drag in. Mirrors the refactor in PR #2932
// (csrc/include/quant.h). The pybind layer
// (csrc/pybind/opus_gemm_pybind.cu) registers aiter_tensor_t as a
// pybind11 class via AITER_CORE_PYBIND, and Python callers are
// converted with aiter.utility.dtypes.torch_to_aiter_pybind.
#include "aiter_tensor.h"
#include <optional>

void opus_gemm(aiter_tensor_t& XQ,
               aiter_tensor_t& WQ,
               aiter_tensor_t& Y,
               std::optional<aiter_tensor_t> group_layout,
               std::optional<aiter_tensor_t> x_scale,
               std::optional<aiter_tensor_t> w_scale,
               std::optional<aiter_tensor_t> bias);

void opus_gemm_a16w16_tune(aiter_tensor_t& XQ,
                           aiter_tensor_t& WQ,
                           aiter_tensor_t& Y,
                           std::optional<aiter_tensor_t> bias,
                           int kernelId,
                           int splitK);

// BHSD-fused batch GEMM for MLA output projection.
// A: [batch, heads_per_group, seqlen, head_dim], bf16
// W: [batch, N, K], bf16 (K = heads_per_group * head_dim)
// Y: [batch, seqlen, N], bf16 or fp32
void opus_gemm_a16w16_bhsd(aiter_tensor_t& A,
                           aiter_tensor_t& W,
                           aiter_tensor_t& Y,
                           int kernelId,
                           int splitK);

// mmajor batched a16w16 GEMM: A/Y are [M, batch, *] (dim0=M, dim1=batch) so no
// caller-side transpose is needed for batch-in-the-middle layouts. Used by both
// the DeepSeek-V4 output-LoRA path (wo_a_gemm_opus, A=[num_tokens, n_groups, K])
// and batch_gemm_a16w16_bshd_opus. See opus_gemm.cu for the full rationale.
// A:  [M, batch, K], bf16
// B:  [batch, N, K], bf16   (per-batch weight)
// Y:  [M, batch, N], bf16 or fp32
void opus_gemm_a16w16_mmajor(aiter_tensor_t& O,
                            aiter_tensor_t& wo_a,
                            aiter_tensor_t& Y,
                            int kernelId,
                            int splitK);
void opus_gemm_a8w8_blockscale_bpreshuffle_tune(aiter_tensor_t& XQ,
                                                aiter_tensor_t& WQ,
                                                std::optional<aiter_tensor_t> x_scale,
                                                std::optional<aiter_tensor_t> w_scale,
                                                aiter_tensor_t& Y,
                                                int kernelId);

// mmajor fp8 block-scale batched GEMM (zero-copy DSV4 wo_a fp8):
// O/Y are [M, batch, *]; wo_a + w_scale batch-major; x_scale [M, batch, K/GROUP_K].
void opus_gemm_a8w8_scale_mmajor(aiter_tensor_t& O,
                                 aiter_tensor_t& wo_a,
                                 aiter_tensor_t& Y,
                                 aiter_tensor_t& x_scale,
                                 aiter_tensor_t& w_scale);

// Per-stream splitk workspace init. See opus_gemm.cu for rationale.
void opus_gemm_workspace_init();
