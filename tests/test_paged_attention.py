import os
import unittest

try:
    import torch
except ImportError:
    torch = None

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()
if CUDA_AVAILABLE:
    from nanovllm.layers.attention import paged_attention
else:
    paged_attention = None


def reference_paged_attention(
    query,
    key_cache,
    value_cache,
    block_tables,
    query_to_request,
    query_positions,
    scale,
):
    outputs = []
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    heads_per_kv = num_query_heads // num_kv_heads
    page_size = key_cache.shape[1]
    for query_index, request_index in enumerate(query_to_request.tolist()):
        position = query_positions[query_index].item()
        logical_tokens = torch.arange(position + 1, device=query.device)
        logical_blocks = logical_tokens // page_size
        block_offsets = logical_tokens % page_size
        physical_blocks = block_tables[request_index, logical_blocks]
        key = key_cache[physical_blocks, block_offsets]
        value = value_cache[physical_blocks, block_offsets]
        key = key.repeat_interleave(heads_per_kv, dim=1).float()
        value = value.repeat_interleave(heads_per_kv, dim=1).float()
        scores = (
            torch.einsum(
                "hd,thd->ht",
                query[query_index].float(),
                key,
            )
            * scale
        )
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, value))
    return torch.stack(outputs).to(query.dtype)


@unittest.skipUnless(
    CUDA_AVAILABLE,
    "CUDA is required",
)
class PagedAttentionTest(unittest.TestCase):
    def run_case(self, page_size, batch_size, context_length):
        torch.manual_seed(0)
        num_query_heads = 32
        num_kv_heads = 8
        head_dim = 128
        pages_per_request = (context_length + page_size - 1) // page_size
        num_blocks = batch_size * pages_per_request
        key_cache = torch.randn(
            num_blocks,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value_cache = torch.randn_like(key_cache)
        block_tables = torch.arange(
            num_blocks,
            device="cuda",
            dtype=torch.int32,
        ).view(batch_size, pages_per_request)
        query = torch.randn(
            batch_size,
            num_query_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        query_to_request = torch.arange(
            batch_size,
            device="cuda",
            dtype=torch.int32,
        )
        query_positions = torch.full(
            (batch_size,),
            context_length - 1,
            device="cuda",
            dtype=torch.int32,
        )
        scale = head_dim**-0.5

        actual = paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_to_request,
            query_positions,
            scale,
            page_size,
        )
        expected = reference_paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_to_request,
            query_positions,
            scale,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_gqa_non_aligned_context_for_all_page_sizes(self):
        for page_size in (16, 32, 64):
            with self.subTest(page_size=page_size):
                self.run_case(page_size, batch_size=8, context_length=133)

    def test_mixed_query_positions(self):
        page_size = 16
        num_query_heads = 32
        num_kv_heads = 8
        head_dim = 128
        key_cache = torch.randn(
            8,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value_cache = torch.randn_like(key_cache)
        block_tables = torch.arange(
            8,
            device="cuda",
            dtype=torch.int32,
        ).view(2, 4)
        query = torch.randn(
            4,
            num_query_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        query_to_request = torch.tensor(
            [0, 0, 0, 1],
            device="cuda",
            dtype=torch.int32,
        )
        query_positions = torch.tensor(
            [5, 6, 17, 31],
            device="cuda",
            dtype=torch.int32,
        )
        scale = head_dim**-0.5

        actual = paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_to_request,
            query_positions,
            scale,
            page_size,
        )
        expected = reference_paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_to_request,
            query_positions,
            scale,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        os.getenv("NANOVLLM_RUN_EXTENDED_KERNEL_TESTS") == "1",
        "set NANOVLLM_RUN_EXTENDED_KERNEL_TESTS=1 for the full matrix",
    )
    def test_extended_acceptance_matrix(self):
        for page_size in (16, 32, 64):
            for batch_size in (1, 8, 32, 128):
                for context_length in (128, 512, 2048, 4096):
                    with self.subTest(
                        page_size=page_size,
                        batch_size=batch_size,
                        context_length=context_length,
                    ):
                        self.run_case(
                            page_size,
                            batch_size,
                            context_length,
                        )


if __name__ == "__main__":
    unittest.main()
