import unittest

import torch

from nanovllm.layers.attention import store_kvcache


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class StoreKvCacheTest(unittest.TestCase):

    def test_non_power_of_two_width(self):
        num_tokens = 3
        num_heads = 6
        head_dim = 128
        block_size = 256
        key = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key)
        k_cache = torch.zeros(
            2,
            block_size,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_cache = torch.zeros_like(k_cache)
        slot_mapping = torch.tensor(
            [0, 257, 511],
            device="cuda",
            dtype=torch.int32,
        )

        store_kvcache(key, value, k_cache, v_cache, slot_mapping)
        torch.cuda.synchronize()

        expected_k = torch.zeros_like(k_cache).view(
            -1,
            num_heads,
            head_dim,
        )
        expected_v = torch.zeros_like(v_cache).view(
            -1,
            num_heads,
            head_dim,
        )
        expected_k[slot_mapping.long()] = key
        expected_v[slot_mapping.long()] = value
        torch.testing.assert_close(
            k_cache.view_as(expected_k),
            expected_k,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            v_cache.view_as(expected_v),
            expected_v,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
