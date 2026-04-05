"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

import heapq

from .base import Tokenizer, get_stats, merge, merge_and_update_stats


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes

        # --- Optimization 1: compute stats once, then maintain incrementally ---
        # The original code called get_stats(ids) from scratch every iteration,
        # costing O(n) per merge for a total of O(n * num_merges).
        # We compute it once here and update it in O(k) per merge, where k is
        # the number of occurrences of the chosen pair (k << n in practice).
        stats = get_stats(ids)

        # --- Optimization 2: max-heap for O(log |stats|) best-pair selection ---
        # The original max(stats, key=stats.get) is O(|stats|) per call.
        # A max-heap with lazy deletion gives O(log |stats|) amortized.
        # Heap entries: (-count, pair). We negate count so heapq (a min-heap)
        # pops the highest-count pair. Stale entries (whose recorded count no
        # longer matches stats) are silently skipped when popped.
        heap = [(-cnt, pair) for pair, cnt in stats.items()]
        heapq.heapify(heap)

        for i in range(num_merges):
            # Find the current best pair, skipping any stale heap entries.
            pair = None
            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                if stats.get(candidate, 0) == -neg_cnt:
                    # Entry is still current — this is our pair.
                    pair = candidate
                    count = -neg_cnt
                    break
                # Entry is stale (stats were updated since it was pushed);
                # discard it and keep looking.
            if pair is None:
                break  # no mergeable pairs remain

            # mint a new token: assign it the next available id
            idx = 256 + i

            # Merge and update stats incrementally.
            # merge_and_update_stats returns the new ids list (stats is
            # mutated in-place) and the set of pairs whose counts changed.
            ids, changed = merge_and_update_stats(ids, pair, idx, stats)

            # Push a fresh heap entry for every pair whose count changed.
            # The old entries for those pairs are now stale and will be
            # skipped lazily; we don't bother removing them explicitly.
            for p in changed:
                cnt = stats.get(p, 0)
                if cnt > 0:
                    heapq.heappush(heap, (-cnt, p))

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {count} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        if len(ids) < 2:
            return ids

        # --- Optimization: maintain stats incrementally ---
        # The original code called get_stats(ids) on every iteration (O(n)
        # each). We compute it once and update it in O(k) per merge.
        stats = get_stats(ids)

        while stats:
            # find the pair with the lowest merge index
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            # stats is mutated in-place; we don't need changed here
            ids, _ = merge_and_update_stats(ids, pair, idx, stats)

        return ids