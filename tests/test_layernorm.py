import unittest

import torch

from nanovllm.layers.layernorm import RMSNorm


class RMSNormTest(unittest.TestCase):

    def test_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "unsupported RMSNorm backend"):
            RMSNorm(128, backend="unknown")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RMSNormCudaTest(unittest.TestCase):

    def test_triton_supports_qkv_split_stride(self):
        num_tokens = 7
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        qkv = torch.randn(
            num_tokens,
            (num_heads + 2 * num_kv_heads) * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        query_elements = num_heads * head_dim
        key_elements = num_kv_heads * head_dim
        query = qkv[:, :query_elements].view(
            num_tokens,
            num_heads,
            head_dim,
        )
        key = qkv[:, query_elements:query_elements + key_elements].view(
            num_tokens,
            num_kv_heads,
            head_dim,
        )
        self.assertFalse(query.is_contiguous())
        self.assertFalse(key.is_contiguous())
        eager = RMSNorm(head_dim, backend="eager").cuda().bfloat16()
        triton = RMSNorm(head_dim, backend="triton").cuda().bfloat16()
        triton.load_state_dict(eager.state_dict())

        for tensor in (query, key):
            expected = eager(tensor)
            actual = triton(tensor)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2e-2,
                atol=2e-2,
            )

    def test_triton_matches_eager_for_qwen3_widths(self):
        for hidden_size in (128, 4096):
            with self.subTest(hidden_size=hidden_size):
                eager = RMSNorm(
                    hidden_size,
                    backend="eager",
                ).cuda().bfloat16()
                triton = RMSNorm(
                    hidden_size,
                    backend="triton",
                ).cuda().bfloat16()
                weight = 1 + 0.1 * torch.randn_like(eager.weight)
                with torch.no_grad():
                    eager.weight.copy_(weight)
                    triton.weight.copy_(weight)
                x = torch.randn(
                    7,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                residual = torch.randn_like(x)

                expected = eager(x)
                actual = triton(x)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-2,
                    atol=2e-2,
                )

                expected, expected_residual = eager(x, residual)
                actual, actual_residual = triton(x, residual)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-2,
                    atol=2e-2,
                )
                torch.testing.assert_close(
                    actual_residual,
                    expected_residual,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
