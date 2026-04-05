"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.

Optimizations over naive implementation:
- Incremental pair stats: only affected pairs are recomputed after each merge
  instead of scanning the full token list from scratch every iteration.
- Linked list via prev/next dicts for O(1) neighbour lookup during merge.
- heapq for O(log n) max-pair retrieval instead of O(n) max() scan.
"""

import heapq
from collections import defaultdict
from tqdm import tqdm
from .base import Tokenizer


def _build_index(ids):
    """
    Build neighbour index and pair->positions map from a flat token list.

    Returns:
        prev      dict  token_pos -> previous token_pos  (-1 = none)
        nxt       dict  token_pos -> next token_pos      (-1 = none)
        pair_pos  dict  (a,b) -> set of positions i where ids[i]==a, ids[i+1]==b
    """
    n = len(ids)
    prev = {i: i - 1 for i in range(n)}
    nxt  = {i: i + 1 for i in range(n)}
    nxt[n - 1] = -1
    prev[0]    = -1

    pair_pos = defaultdict(set)
    for i in range(n - 1):
        pair_pos[(ids[i], ids[i + 1])].add(i)

    return prev, nxt, pair_pos


def _build_heap(pair_pos):
    """Max-heap (negated counts) over all pairs."""
    heap = [(-len(positions), pair) for pair, positions in pair_pos.items()]
    heapq.heapify(heap)
    return heap


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        ids = list(text.encode("utf-8"))   # list of ints 0..255

        # build neighbour index and initial pair stats
        prev, nxt, pair_pos = _build_index(ids)
        heap = _build_heap(pair_pos)

        merges = {}
        vocab  = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges), total=num_merges):
            # --- pop best pair, skipping stale heap entries ---
            while True:
                neg_cnt, pair = heapq.heappop(heap)
                positions = pair_pos.get(pair)
                if positions and -neg_cnt == len(positions):
                    break   # fresh entry
                if positions and len(positions) > 0:
                    # stale count — re-push with correct count
                    heapq.heappush(heap, (-len(positions), pair))
                    continue
                # pair no longer exists; keep popping
                if not heap:
                    break

            if not pair_pos.get(pair):
                break  # nothing left to merge

            idx = 256 + i
            merges[pair] = idx
            vocab[idx]   = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {len(pair_pos[pair])} occurrences")

            # --- apply merge & update index incrementally ---
            affected = set()
            for pos in list(pair_pos[pair]):
                p  = prev[pos]
                n  = nxt[pos]
                nn = nxt.get(n, -1) if n != -1 else -1

                # replace token at pos with idx, remove token at n
                ids[pos] = idx

                # remove old pairs touching pos or n
                if p != -1:
                    old = (ids[p], pair[0])
                    pair_pos[old].discard(p)
                    affected.add(old)
                if nn != -1:
                    old = (pair[1], ids[nn])
                    pair_pos[old].discard(n)
                    affected.add(old)

                # stitch linked list: skip n
                nxt[pos] = nn
                if nn != -1:
                    prev[nn] = pos

                # add new pairs
                if p != -1:
                    new_pair = (ids[p], idx)
                    pair_pos[new_pair].add(p)
                    affected.add(new_pair)
                if nn != -1:
                    new_pair = (idx, ids[nn])
                    pair_pos[new_pair].add(pos)
                    affected.add(new_pair)

            # remove consumed pair
            del pair_pos[pair]

            # push updated counts for affected pairs onto heap
            for ap in affected:
                if pair_pos.get(ap):
                    heapq.heappush(heap, (-len(pair_pos[ap]), ap))

        # reconstruct flat ids (following linked list) for completeness
        self.merges = merges
        self.vocab  = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = {}
            for j in range(len(ids) - 1):
                p = (ids[j], ids[j + 1])
                stats[p] = stats.get(p, 0) + 1
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            # merge
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j+1]) == pair:
                    new_ids.append(idx)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
        return ids