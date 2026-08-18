import math
from collections.abc import Iterator

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []
        self.prev_free: Block | None = None
        self.next_free: Block | None = None
        self.in_free_queue = False

    def update(self, block_hash: int, token_ids: list[int]):
        self.hash = block_hash
        self.token_ids = token_ids.copy()

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class FreeBlockQueue:
    """Intrusive FIFO used as an O(1) free list and cached-block LRU."""

    def __init__(self, blocks: list[Block]):
        self.head: Block | None = None
        self.tail: Block | None = None
        self.size = 0
        for block in blocks:
            self.append(block)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        block = self.head
        while block is not None:
            yield block.block_id
            block = block.next_free

    def append(self, block: Block):
        assert not block.in_free_queue
        block.prev_free = self.tail
        block.next_free = None
        if self.tail is None:
            self.head = block
        else:
            self.tail.next_free = block
        self.tail = block
        block.in_free_queue = True
        self.size += 1

    def remove(self, block: Block):
        assert block.in_free_queue
        if block.prev_free is None:
            self.head = block.next_free
        else:
            block.prev_free.next_free = block.next_free
        if block.next_free is None:
            self.tail = block.prev_free
        else:
            block.next_free.prev_free = block.prev_free
        block.prev_free = None
        block.next_free = None
        block.in_free_queue = False
        self.size -= 1

    def popleft(self) -> Block:
        if self.head is None:
            raise IndexError("pop from an empty free-block queue")
        block = self.head
        self.remove(block)
        return block


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_ids: dict[int, set[int]] = {}
        self.free_block_ids = FreeBlockQueue(self.blocks)
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        block_hash = xxhash.xxh64()
        if prefix != -1:
            block_hash.update(prefix.to_bytes(8, "little"))
        block_hash.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return block_hash.intdigest()

    def _add_hash(self, block: Block):
        if block.hash != -1:
            self.hash_to_block_ids.setdefault(block.hash, set()).add(block.block_id)

    def _remove_hash(self, block: Block):
        if block.hash == -1:
            return
        block_ids = self.hash_to_block_ids.get(block.hash)
        if block_ids is None:
            return
        block_ids.discard(block.block_id)
        if not block_ids:
            del self.hash_to_block_ids[block.hash]

    def _find_cached_block(
        self,
        block_hash: int,
        token_ids: list[int],
    ) -> Block | None:
        for block_id in self.hash_to_block_ids.get(block_hash, ()):
            block = self.blocks[block_id]
            if block.token_ids == token_ids:
                return block
        return None

    def _allocate_block(self) -> int:
        block = self.free_block_ids.popleft()
        assert block.ref_count == 0
        self._remove_hash(block)
        block.reset()
        self.used_block_ids.add(block.block_id)
        return block.block_id

    def _touch_block(self, block: Block):
        if block.ref_count == 0:
            self.free_block_ids.remove(block)
            self.used_block_ids.add(block.block_id)
        block.ref_count += 1

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block)

    def _cached_prefix(self, seq: Sequence) -> list[Block]:
        cached: list[Block] = []
        prefix_hash = -1
        for block_index in range(max(0, seq.num_blocks - 1)):
            token_ids = seq.block(block_index)
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block = self._find_cached_block(prefix_hash, token_ids)
            if block is None:
                break
            cached.append(block)
        return cached

    def can_allocate(self, seq: Sequence) -> int:
        cached = self._cached_prefix(seq)
        shared_blocks = sum(block.ref_count > 0 for block in cached)
        required_free_blocks = seq.num_blocks - shared_blocks
        if len(self.free_block_ids) < required_free_blocks:
            return -1
        return len(cached)

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        cached = self._cached_prefix(seq)
        assert len(cached) >= num_cached_blocks
        for block in cached[:num_cached_blocks]:
            self._touch_block(block)
            seq.block_table.append(block.block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.draft_num_cached_tokens = 0
        seq.num_scheduled_tokens = 0
        seq.block_table.clear()

    def can_reserve(self, seq: Sequence, total_tokens: int) -> bool:
        required_blocks = math.ceil(total_tokens / self.block_size)
        return len(self.free_block_ids) >= max(
            0,
            required_blocks - len(seq.block_table),
        )

    def reserve(self, seq: Sequence, total_tokens: int):
        required_blocks = math.ceil(total_tokens / self.block_size)
        missing_blocks = required_blocks - len(seq.block_table)
        if missing_blocks > len(self.free_block_ids):
            raise RuntimeError("insufficient KV blocks for reservation")
        for _ in range(max(0, missing_blocks)):
            seq.block_table.append(self._allocate_block())

    def can_append(self, seq: Sequence) -> bool:
        return self.can_reserve(seq, len(seq))

    def may_append(self, seq: Sequence):
        self.reserve(seq, len(seq))

    def truncate(self, seq: Sequence, total_tokens: int):
        """Release blocks beyond ``total_tokens`` after speculative rollback."""
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        required_blocks = math.ceil(total_tokens / self.block_size)
        while len(seq.block_table) > required_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = min(seq.num_cached_tokens, total_tokens)

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        committed_tokens = seq.num_cached_tokens + seq.num_scheduled_tokens
        end = min(committed_tokens, len(seq)) // self.block_size
        if start == end:
            return
        prefix_hash = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for block_index in range(start, end):
            block = self.blocks[seq.block_table[block_index]]
            token_ids = seq.block(block_index)
            block_hash = self.compute_hash(token_ids, prefix_hash)
            self._remove_hash(block)
            block.update(block_hash, token_ids)
            self._add_hash(block)
            prefix_hash = block_hash

    def stats(self, sequences: list[Sequence]) -> dict[str, int | float]:
        unique_sequences = {seq.seq_id: seq for seq in sequences}.values()
        allocated_tokens = sum(
            len(seq.block_table) * self.block_size for seq in unique_sequences
        )
        computed_tokens = sum(
            min(seq.num_cached_tokens, len(seq)) for seq in unique_sequences
        )
        reserved_tokens = sum(
            min(
                len(seq.block_table) * self.block_size,
                max(
                    len(seq),
                    seq.num_cached_tokens + seq.num_scheduled_tokens,
                ),
            )
            for seq in unique_sequences
        )
        uncomputed_tokens = max(0, reserved_tokens - computed_tokens)
        tail_waste = max(0, allocated_tokens - reserved_tokens)
        utilization = (
            len(self.used_block_ids) / len(self.blocks) if self.blocks else 0.0
        )
        return {
            "kv_used_blocks": len(self.used_block_ids),
            "kv_free_blocks": len(self.free_block_ids),
            "kv_allocated_tokens": allocated_tokens,
            "kv_reserved_tokens": reserved_tokens,
            "kv_computed_tokens": computed_tokens,
            "kv_uncomputed_tokens": uncomputed_tokens,
            "kv_tail_waste_tokens": tail_waste,
            "kv_block_utilization": round(utilization, 6),
        }
