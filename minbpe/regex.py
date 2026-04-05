"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.

Optimizations over naive implementation:
- Per-chunk incremental pair stats: after each merge only the positions
  that changed are updated, avoiding a full O(n) rescan every iteration.
- heapq for O(log k) best-pair lookup instead of O(k) max() over all pairs.
- Lazy-deletion heap: stale heap entries are skipped on pop rather than
  eagerly removed (no decrease-key needed).
"""

import heapq
from collections import defaultdict
import regex as re
from tqdm import tqdm
from .base import Tokenizer


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Incremental merge helpers
# ---------------------------------------------------------------------------

def _build_chunk_index(chunk_ids):
    """
    Build a doubly-linked list + pair->positions index for one chunk.

    Returns:
        tokens    list  mutable token array (only valid positions matter)
        prev      dict  pos -> previous valid pos  (-1 = none)
        nxt       dict  pos -> next valid pos      (-1 = none)
        pair_pos  dict  (a,b) -> set of positions i where token[i]==a, token[i+1]==b
    """
    tokens = list(chunk_ids)
    n = len(tokens)
    prev = {i: i - 1 for i in range(n)}
    nxt  = {i: i + 1 for i in range(n)}
    if n:
        prev[0]    = -1
        nxt[n - 1] = -1

    pair_pos = defaultdict(set)
    for i in range(n - 1):
        pair_pos[(tokens[i], tokens[i + 1])].add(i)

    return tokens, prev, nxt, pair_pos


def _merge_chunk(tokens, prev, nxt, pair_pos, pair, idx):
    """
    Apply one BPE merge in-place on a single chunk's linked-list structure.
    Updates pair_pos incrementally; returns set of pairs whose counts changed.
    """
    affected = set()
    for pos in list(pair_pos.get(pair, [])):
        p  = prev[pos]
        n  = nxt[pos]         # position of right token in the pair
        nn = nxt[n] if n != -1 else -1

        tokens[pos] = idx

        # remove old neighbour pairs
        if p != -1:
            old = (tokens[p], pair[0])
            pair_pos[old].discard(p)
            affected.add(old)
        if nn != -1:
            old = (pair[1], tokens[nn])
            pair_pos[old].discard(n)
            affected.add(old)

        # stitch out the right token
        nxt[pos] = nn
        if nn != -1:
            prev[nn] = pos

        # add new neighbour pairs
        if p != -1:
            np_ = (tokens[p], idx)
            pair_pos[np_].add(p)
            affected.add(np_)
        if nn != -1:
            np_ = (idx, tokens[nn])
            pair_pos[np_].add(pos)
            affected.add(np_)

    pair_pos.pop(pair, None)
    affected.discard(pair)
    return affected


# ---------------------------------------------------------------------------
# RegexTokenizer
# ---------------------------------------------------------------------------

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern          = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens   = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # build per-chunk linked-list index
        chunks = [_build_chunk_index(list(ch.encode("utf-8")))
                  for ch in text_chunks]

        # aggregate pair counts across all chunks
        global_counts = defaultdict(int)
        for _, _, _, pair_pos in chunks:
            for pair, positions in pair_pos.items():
                global_counts[pair] += len(positions)

        # max-heap (lazy deletion via negated counts)
        heap = [(-cnt, pair) for pair, cnt in global_counts.items()]
        heapq.heapify(heap)

        merges = {}
        vocab  = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges), total=num_merges):
            # pop best pair, skip stale entries
            while heap:
                neg_cnt, pair = heapq.heappop(heap)
                real_cnt = global_counts.get(pair, 0)
                if real_cnt > 0 and real_cnt == -neg_cnt:
                    break   # fresh
                if real_cnt > 0:
                    heapq.heappush(heap, (-real_cnt, pair))
            else:
                break

            if global_counts.get(pair, 0) == 0:
                break

            idx = 256 + i
            merges[pair] = idx
            vocab[idx]   = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {global_counts[pair]} occurrences")

            del global_counts[pair]

            # apply merge to every chunk, collect affected pairs
            all_affected = defaultdict(int)
            for tokens, prev, nxt, pair_pos in chunks:
                old_counts = {p: len(pair_pos[p]) for p in pair_pos if p in pair_pos}
                affected = _merge_chunk(tokens, prev, nxt, pair_pos, pair, idx)
                for ap in affected:
                    new_cnt = len(pair_pos.get(ap, set()))
                    old_cnt = old_counts.get(ap, 0)
                    all_affected[ap] += new_cnt - old_cnt

            # update global counts and push changed pairs onto heap
            for ap, delta in all_affected.items():
                global_counts[ap] = global_counts.get(ap, 0) + delta
                if global_counts[ap] > 0:
                    heapq.heappush(heap, (-global_counts[ap], ap))
                elif ap in global_counts:
                    del global_counts[ap]

        self.merges = merges
        self.vocab  = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens         = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = {}
            for j in range(len(ids) - 1):
                p = (ids[j], ids[j + 1])
                stats[p] = stats.get(p, 0) + 1
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            new_ids, j = [], 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
                    new_ids.append(idx)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items()
                       if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks  = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids