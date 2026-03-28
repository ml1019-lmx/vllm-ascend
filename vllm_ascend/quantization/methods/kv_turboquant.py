#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
from dataclasses import dataclass
from functools import cache

import torch

from .base import AscendAttentionScheme
from .registry import register_scheme

TURBOQUANT_KV_CACHE_BITS = {
    "turboquant25": 2.5,
    "turboquant35": 3.5,
}

TURBOQUANT_OUTLIER_RATIOS = {
    "turboquant25": 0.25,
    "turboquant35": 0.50,
}

TURBOQUANT_GROUP_BITS = {
    "turboquant25": (3, 2),
    "turboquant35": (4, 3),
}

TURBOQUANT_GROUP_ALIGNMENT = 16
TURBOQUANT_VECTOR_NORM_BYTES = 2
TURBOQUANT_RESIDUAL_NORM_BYTES = 2
TURBOQUANT_NORM_BYTES = TURBOQUANT_VECTOR_NORM_BYTES + TURBOQUANT_RESIDUAL_NORM_BYTES
TURBOQUANT_SEED = 20250428
TURBOQUANT_QJL_SEED_OFFSET = 10_000
TURBOQUANT_QJL_SCALE = math.sqrt(math.pi / 2.0)
TURBOQUANT_CODEBOOK_GRID_POINTS = 32768
TURBOQUANT_CODEBOOK_EPS = 1e-6


@dataclass(frozen=True)
class TurboQuantGroupLayout:
    dim: int
    bits: int
    mse_bits: int
    mse_payload_bytes: int
    qjl_payload_bytes: int
    qjl_offset: int
    vector_norm_offset: int
    residual_norm_offset: int
    packed_bytes: int


@dataclass(frozen=True)
class TurboQuantLayout:
    groups: tuple[TurboQuantGroupLayout, TurboQuantGroupLayout]
    packed_dim: int


def is_turboquant_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in TURBOQUANT_KV_CACHE_BITS


def is_turboquant_kv_cache(kv_cache_dtype: str) -> bool:
    """Compatibility alias matching vllm-turboquant naming."""
    return is_turboquant_kv_cache_dtype(kv_cache_dtype)


def get_turboquant_bits(kv_cache_dtype: str) -> float:
    try:
        return TURBOQUANT_KV_CACHE_BITS[kv_cache_dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported TurboQuant KV cache dtype: {kv_cache_dtype}") from e


def canonical_turboquant_dtype(bits_or_dtype: float | int | str) -> str:
    if isinstance(bits_or_dtype, str):
        if not is_turboquant_kv_cache(bits_or_dtype):
            raise ValueError(f"Unsupported TurboQuant KV cache dtype: {bits_or_dtype}")
        return bits_or_dtype

    bits = float(bits_or_dtype)
    if bits == 2.5:
        return "turboquant25"
    if bits == 3.5:
        return "turboquant35"
    raise ValueError(f"Unsupported TurboQuant bit-width: {bits}")


def get_turboquant_outlier_count(head_size: int, kv_cache_dtype: str) -> int:
    if head_size % TURBOQUANT_GROUP_ALIGNMENT != 0:
        raise ValueError(
            "TurboQuant KV cache requires head_size to be a multiple of "
            f"{TURBOQUANT_GROUP_ALIGNMENT}, got {head_size}."
        )

    try:
        ratio = TURBOQUANT_OUTLIER_RATIOS[kv_cache_dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported TurboQuant KV cache dtype: {kv_cache_dtype}") from e

    outlier_count = int(round(head_size * ratio / TURBOQUANT_GROUP_ALIGNMENT) * TURBOQUANT_GROUP_ALIGNMENT)
    if outlier_count <= 0 or outlier_count >= head_size:
        raise ValueError(f"Unsupported TurboQuant head_size {head_size} for {kv_cache_dtype}.")
    return outlier_count


def get_turboquant_group_dims(head_size: int, kv_cache_dtype: str) -> tuple[int, int]:
    outlier_count = get_turboquant_outlier_count(head_size, kv_cache_dtype)
    return outlier_count, head_size - outlier_count


@cache
def _layout_cached(kv_cache_dtype: str, head_size: int) -> TurboQuantLayout:
    group_dims = get_turboquant_group_dims(head_size, kv_cache_dtype)
    group_bits = TURBOQUANT_GROUP_BITS[kv_cache_dtype]
    groups: list[TurboQuantGroupLayout] = []
    cursor = 0
    for group_dim, bits in zip(group_dims, group_bits, strict=True):
        mse_bits = bits - 1
        mse_payload_bytes = (group_dim * mse_bits + 7) // 8
        qjl_payload_bytes = (group_dim + 7) // 8
        qjl_offset = cursor + mse_payload_bytes
        vector_norm_offset = qjl_offset + qjl_payload_bytes
        residual_norm_offset = vector_norm_offset + TURBOQUANT_VECTOR_NORM_BYTES
        packed_bytes = mse_payload_bytes + qjl_payload_bytes + TURBOQUANT_NORM_BYTES
        groups.append(
            TurboQuantGroupLayout(
                dim=group_dim,
                bits=bits,
                mse_bits=mse_bits,
                mse_payload_bytes=mse_payload_bytes,
                qjl_payload_bytes=qjl_payload_bytes,
                qjl_offset=qjl_offset,
                vector_norm_offset=vector_norm_offset,
                residual_norm_offset=residual_norm_offset,
                packed_bytes=packed_bytes,
            )
        )
        cursor += packed_bytes
    return TurboQuantLayout(groups=(groups[0], groups[1]), packed_dim=cursor)


def get_turboquant_layout(kv_cache_dtype: str, head_size: int) -> TurboQuantLayout:
    return _layout_cached(kv_cache_dtype, head_size)


def get_turboquant_packed_dim(head_size: int, bits_or_dtype: float | int | str) -> int:
    """Return per-head packed bytes for TurboQuant KV cache.

    Each group stores:
      1) MSE payload (mse_bits = bits - 1)
      2) QJL sign payload (1 bit per dim)
      3) vector norm + residual norm (4 bytes)
    """
    kv_cache_dtype = canonical_turboquant_dtype(bits_or_dtype)
    return get_turboquant_layout(kv_cache_dtype, head_size).packed_dim


def _turboquant_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Weight loader for TurboQuant KV-cache metadata tensors."""
    loaded_weight = loaded_weight.squeeze()
    if param.data.shape != loaded_weight.shape:
        param.data = loaded_weight.to(param.dtype).clone()
    else:
        param.data.copy_(loaded_weight)


@register_scheme("TurboQuant", "attention")
@register_scheme("TURBOQUANT", "attention")
class AscendTurboQuantAttentionMethod(AscendAttentionScheme):
    """TurboQuant KV cache quantization scaffold for attention layers.

    This class only wires layer metadata and backend binding.
    The actual TurboQuant decode/prefill kernels are implemented in
    ``AscendTurboQuantAttentionBackendImpl``.
    """

    def create_weights(self, layer: torch.nn.Module) -> None:
        # TurboQuant stores quantized KV cache.
        layer.kv_cache_torch_dtype = torch.int8

        # Route this attention layer to the TurboQuant backend implementation.
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import (
                AscendTurboQuantAttentionBackendImpl,
            )

            layer.impl.__class__ = AscendTurboQuantAttentionBackendImpl

        # Placeholder parameters for TurboQuant metadata.
        # Keep loaders for checkpoint compatibility and future expansion.
        layer.turboquant_kv_cache_dtype = "turboquant35"
        layer.turboquant_k_scale = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        layer.turboquant_k_scale.weight_loader = _turboquant_weight_loader
        layer.turboquant_k_offset = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=False
        )
        layer.turboquant_k_offset.weight_loader = _turboquant_weight_loader

        head_size = int(getattr(layer, "head_size", 0))
        if head_size > 0 and head_size % TURBOQUANT_GROUP_ALIGNMENT == 0:
            layer.turboquant_packed_dim = get_turboquant_packed_dim(
                head_size, layer.turboquant_kv_cache_dtype
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.turboquant_k_scale.data = layer.turboquant_k_scale.data.flatten()
        layer.turboquant_k_offset.data = layer.turboquant_k_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendTurboQuantAttentionMethod.apply should not be called. "
            "TurboQuant KV cache quantization is handled by the attention backend."
        )
