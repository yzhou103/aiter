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

// Per-stream splitk workspace init. See opus_gemm.cu for rationale.
void opus_gemm_workspace_init();
