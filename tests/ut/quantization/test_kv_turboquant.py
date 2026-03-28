import unittest

import torch
import torch.nn as nn


class TestTurboQuantSchemeRegistration(unittest.TestCase):
    def test_scheme_is_registered(self):
        from vllm_ascend.quantization.methods import get_scheme_class

        scheme_cls = get_scheme_class("TurboQuant", "attention")
        self.assertIsNotNone(scheme_cls)

        scheme_cls_upper = get_scheme_class("TURBOQUANT", "attention")
        self.assertIsNotNone(scheme_cls_upper)


class TestTurboQuantLayoutHelpers(unittest.TestCase):
    def test_is_turboquant_kv_cache_dtype(self):
        from vllm_ascend.quantization.methods.kv_turboquant import is_turboquant_kv_cache_dtype

        self.assertTrue(is_turboquant_kv_cache_dtype("turboquant25"))
        self.assertTrue(is_turboquant_kv_cache_dtype("turboquant35"))
        self.assertFalse(is_turboquant_kv_cache_dtype("int8"))

    def test_group_dims_and_packed_dim(self):
        from vllm_ascend.quantization.methods.kv_turboquant import (
            canonical_turboquant_dtype,
            get_turboquant_group_dims,
            get_turboquant_layout,
            get_turboquant_packed_dim,
        )

        self.assertEqual(get_turboquant_group_dims(128, "turboquant25"), (32, 96))
        self.assertEqual(get_turboquant_group_dims(128, "turboquant35"), (64, 64))
        self.assertEqual(canonical_turboquant_dtype(2.5), "turboquant25")
        self.assertEqual(canonical_turboquant_dtype(3.5), "turboquant35")

        packed_dim_25 = get_turboquant_packed_dim(128, "turboquant25")
        packed_dim_35 = get_turboquant_packed_dim(128, "turboquant35")
        packed_dim_35_from_bits = get_turboquant_packed_dim(128, 3.5)
        self.assertEqual(packed_dim_35, packed_dim_35_from_bits)
        self.assertGreater(packed_dim_35, packed_dim_25)
        self.assertGreater(packed_dim_25, 0)

        layout_35 = get_turboquant_layout("turboquant35", 128)
        self.assertEqual(layout_35.packed_dim, packed_dim_35)
        self.assertEqual(layout_35.groups[0].dim, 64)
        self.assertEqual(layout_35.groups[1].dim, 64)
        self.assertGreater(layout_35.groups[0].qjl_offset, 0)
        self.assertGreater(layout_35.groups[0].packed_bytes, 0)

    def test_invalid_head_size_raises(self):
        from vllm_ascend.quantization.methods.kv_turboquant import get_turboquant_group_dims

        with self.assertRaises(ValueError):
            get_turboquant_group_dims(130, "turboquant25")


class TestAscendTurboQuantAttentionMethod(unittest.TestCase):
    def setUp(self):
        from vllm_ascend.quantization.methods.kv_turboquant import (
            AscendTurboQuantAttentionMethod,
        )

        self.method = AscendTurboQuantAttentionMethod()

    def test_create_weights_registers_metadata(self):
        layer = nn.Module()
        layer.head_size = 128

        self.method.create_weights(layer)

        self.assertEqual(layer.kv_cache_torch_dtype, torch.int8)
        self.assertTrue(hasattr(layer, "turboquant_k_scale"))
        self.assertTrue(hasattr(layer, "turboquant_k_offset"))
        self.assertEqual(layer.turboquant_kv_cache_dtype, "turboquant35")
        self.assertTrue(hasattr(layer, "turboquant_packed_dim"))
        self.assertTrue(hasattr(layer.turboquant_k_scale, "weight_loader"))
        self.assertTrue(hasattr(layer.turboquant_k_offset, "weight_loader"))

    def test_process_weights_after_loading_flattens(self):
        layer = nn.Module()
        layer.turboquant_k_scale = nn.Parameter(torch.ones(1, 1, dtype=torch.float32))
        layer.turboquant_k_offset = nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))

        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.turboquant_k_scale.ndim, 1)
        self.assertEqual(layer.turboquant_k_offset.ndim, 1)

    def test_apply_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.method.apply(None, None, None, None, None, None, None, None, None)
