import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class BlockManagerTest(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_cached_block_refcount_and_lru_reuse(self):
        manager = BlockManager(num_blocks=6, block_size=4)
        first = Sequence(list(range(8)))
        manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = 4
        manager.hash_blocks(first)
        shared_block = first.block_table[0]

        second = Sequence(list(range(8)))
        self.assertEqual(manager.can_allocate(second), 1)
        manager.allocate(second, num_cached_blocks=1)
        self.assertEqual(second.block_table[0], shared_block)
        self.assertEqual(manager.blocks[shared_block].ref_count, 2)

        manager.deallocate(first)
        self.assertEqual(manager.blocks[shared_block].ref_count, 1)
        manager.deallocate(second)
        self.assertEqual(manager.blocks[shared_block].ref_count, 0)
        self.assertTrue(manager.blocks[shared_block].in_free_queue)

    def test_hash_collision_checks_tokens(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        first, second = manager.blocks
        first.update(123, [1, 2, 3, 4])
        second.update(123, [5, 6, 7, 8])
        manager._add_hash(first)
        manager._add_hash(second)

        self.assertIs(
            manager._find_cached_block(123, [5, 6, 7, 8]),
            second,
        )
        self.assertIsNone(manager._find_cached_block(123, [9, 9, 9, 9]))

    def test_truncate_releases_complete_lookahead_blocks(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        sequence = Sequence(list(range(5)))
        manager.allocate(sequence, num_cached_blocks=0)
        manager.reserve(sequence, total_tokens=18)
        self.assertEqual(len(sequence.block_table), 5)

        manager.truncate(sequence, total_tokens=6)

        self.assertEqual(len(sequence.block_table), 2)
        self.assertEqual(len(manager.used_block_ids), 2)
        self.assertEqual(len(manager.free_block_ids), 6)

    def test_reserve_is_fail_closed(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        sequence = Sequence([1])
        manager.allocate(sequence, num_cached_blocks=0)

        with self.assertRaisesRegex(RuntimeError, "insufficient KV blocks"):
            manager.reserve(sequence, total_tokens=12)

    def test_shared_prefix_240_hits_block16_but_not_block256(self):
        shared_prefix = list(range(240))
        first_tokens = shared_prefix + [1_000] * 272
        second_tokens = shared_prefix + [2_000] * 272

        Sequence.block_size = 16
        fine_manager = BlockManager(num_blocks=80, block_size=16)
        first = Sequence(first_tokens)
        fine_manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = len(first)
        fine_manager.hash_blocks(first)
        fine_hit = fine_manager.can_allocate(Sequence(second_tokens))

        Sequence.block_size = 256
        coarse_manager = BlockManager(num_blocks=8, block_size=256)
        first = Sequence(first_tokens)
        coarse_manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = len(first)
        coarse_manager.hash_blocks(first)
        coarse_hit = coarse_manager.can_allocate(Sequence(second_tokens))

        self.assertEqual(fine_hit, 15)
        self.assertEqual(coarse_hit, 0)

    def test_fine_pages_reduce_tail_waste_by_more_than_eight_times(self):
        wastes = {}
        for block_size in (16, 256):
            Sequence.block_size = block_size
            manager = BlockManager(num_blocks=32, block_size=block_size)
            sequence = Sequence(list(range(257)))
            manager.allocate(sequence, num_cached_blocks=0)
            sequence.num_cached_tokens = 256
            wastes[block_size] = manager.stats([sequence])["kv_tail_waste_tokens"]

        self.assertGreaterEqual(wastes[256] / wastes[16], 8)

    def test_uncomputed_prompt_capacity_is_not_tail_waste(self):
        Sequence.block_size = 256
        manager = BlockManager(num_blocks=8, block_size=256)
        sequence = Sequence(list(range(1024)))
        manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = 256

        stats = manager.stats([sequence])

        self.assertEqual(stats["kv_allocated_tokens"], 1024)
        self.assertEqual(stats["kv_reserved_tokens"], 1024)
        self.assertEqual(stats["kv_computed_tokens"], 256)
        self.assertEqual(stats["kv_uncomputed_tokens"], 768)
        self.assertEqual(stats["kv_tail_waste_tokens"], 0)

    def test_speculative_lookahead_is_reserved_not_tail_waste(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        sequence = Sequence(list(range(5)))
        manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = 4
        sequence.num_scheduled_tokens = 6
        manager.reserve(sequence, total_tokens=10)

        stats = manager.stats([sequence])

        self.assertEqual(stats["kv_allocated_tokens"], 12)
        self.assertEqual(stats["kv_reserved_tokens"], 10)
        self.assertEqual(stats["kv_uncomputed_tokens"], 6)
        self.assertEqual(stats["kv_tail_waste_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
