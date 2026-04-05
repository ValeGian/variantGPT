"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.

Optimizations over naive implementation:
- Per-chunk incremental pair stats via doubly-linked list: after each merge
  only the positions that changed are updated, avoiding a full O(n) rescan.
- _merge_chunk returns per-pair *deltas* directly, eliminating the expensive
  old_counts dict comprehension that previously scanned every pair in the
  chunk on every merge iteration.
- heapq for O(log k) best-pair lookup instead of O(k) max() over all pairs.
- Lazy-deletion heap: stale entries are skipped on pop.
- Validity check on merge positions to handle overlapping pairs correctly
  (e.g. merging (a,a) in the sequence a,a,a).
- _encode_chunk uses incremental stats via merge_and_update_stats.
- decode uses bytearray for O(1)-amortised concatenation.
"""

import heapq
from collections import defaultdict

import regex as re
from tqdm import tqdm

from .base import Tokenizer, get_stats, merge_and_update_stats

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
        prev      list  pos -> previous valid pos  (-1 = none)
        nxt       list  pos -> next valid pos      (-1 = none)
        pair_pos  dict  (a,b) -> set of positions where token[i]==a, token[nxt[i]]==b
    """
    tokens = list(chunk_ids)
    n = len(tokens)
    # lists are faster than dicts for integer-keyed dense access
    prev = list(range(-1, n - 1))   # prev[0] = -1, prev[i] = i-1
    nxt  = list(range(1, n + 1))    # nxt[i] = i+1
    if n:
        nxt[n - 1] = -1

    pair_pos = defaultdict(set)
    for i in range(n - 1):
        pair_pos[(tokens[i], tokens[i + 1])].add(i)

    return tokens, prev, nxt, pair_pos


def _merge_chunk(tokens, prev, nxt, pair_pos, pair, idx):
    """
    Apply one BPE merge in-place on a single chunk's linked-list structure.
    Returns dict {affected_pair: count_delta} so the caller can update
    global_counts with a simple addition — no per-chunk old_counts scan.
    """
    deltas = {}
    positions = list(pair_pos.get(pair, ()))

    for pos in positions:
        # --- validity check (handles overlapping pairs, e.g. (a,a) in a,a,a) ---
        n = nxt[pos]
        if n == -1 or tokens[pos] != pair[0] or tokens[n] != pair[1]:
            # This position was already consumed by an earlier iteration
            # within this same merge pass.  Adjust the delta for the merged
            # pair since global_counts assumed it existed.
            deltas[pair] = deltas.get(pair, 0) - 1
            continue

        p  = prev[pos]
        nn = nxt[n] if n != -1 else -1

        # replace left token of the pair with the new merged token
        tokens[pos] = idx

        # --- remove old neighbour pairs ---
        if p != -1:
            old_left = (tokens[p], pair[0])
            pair_pos[old_left].discard(p)
            deltas[old_left] = deltas.get(old_left, 0) - 1

        if nn != -1:
            old_right = (pair[1], tokens[nn])
            pair_pos[old_right].discard(n)
            deltas[old_right] = deltas.get(old_right, 0) - 1

        # --- stitch out the right token of the pair ---
        nxt[pos] = nn
        if nn != -1:
            prev[nn] = pos

        # --- add new neighbour pairs ---
        if p != -1:
            new_left = (tokens[p], idx)
            pair_pos[new_left].add(p)
            deltas[new_left] = deltas.get(new_left, 0) + 1

        if nn != -1:
            new_right = (idx, tokens[nn])
            pair_pos[new_right].add(pos)
            deltas[new_right] = deltas.get(new_right, 0) + 1

    pair_pos.pop(pair, None)
    deltas.pop(pair, None)  # the merged pair is handled separately by caller
    return deltas


# ---------------------------------------------------------------------------
# RegexTokenizer
# ---------------------------------------------------------------------------

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split text into chunks via regex
        text_chunks = re.findall(self.compiled_pattern, text)

        # build per-chunk linked-list index
        chunks = [_build_chunk_index(list(ch.encode("utf-8")))
                  for ch in text_chunks]

        # aggregate pair counts across all chunks
        global_counts = defaultdict(int)
        for _, _, _, pair_pos in chunks:
            for pair, positions in pair_pos.items():
                global_counts[pair] += len(positions)

        # max-heap with lazy deletion
        heap = [(-cnt, pair) for pair, cnt in global_counts.items()]
        heapq.heapify(heap)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges), total=num_merges):
            # pop best pair, skip stale entries
            pair = None
            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                real_cnt = global_counts.get(candidate, 0)
                if real_cnt == -neg_cnt and real_cnt > 0:
                    pair = candidate
                    break
                # push back if count is still positive but stale
                if real_cnt > 0:
                    heapq.heappush(heap, (-real_cnt, candidate))

            if pair is None:
                break

            count = global_counts[pair]
            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                      f"({vocab[idx]}) had {count} occurrences")

            del global_counts[pair]

            # ---- apply merge to every chunk, collect deltas ----
            # _merge_chunk now returns {pair: delta} directly.
            # No more O(|pair_pos|) old_counts scan per chunk.
            all_deltas = defaultdict(int)
            for tokens, prev_ll, nxt_ll, pair_pos in chunks:
                chunk_deltas = _merge_chunk(
                    tokens, prev_ll, nxt_ll, pair_pos, pair, idx)
                for p, d in chunk_deltas.items():
                    all_deltas[p] += d

            # update global counts and push changed pairs onto heap
            for ap, delta in all_deltas.items():
                new_cnt = global_counts.get(ap, 0) + delta
                if new_cnt > 0:
                    global_counts[ap] = new_cnt
                    heapq.heappush(heap, (-new_cnt, ap))
                else:
                    global_counts.pop(ap, None)

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # bytearray: O(1)-amortised append vs list-of-bytes + join
        buf = bytearray()
        vocab = self.vocab
        inv = self.inverse_special_tokens
        for idx in ids:
            if idx in vocab:
                buf.extend(vocab[idx])
            elif idx in inv:
                buf.extend(inv[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        return buf.decode("utf-8", errors="replace")

    def _encode_chunk(self, text_bytes):
        """Encode a single chunk using incremental stats from base."""
        ids = list(text_bytes)
        if len(ids) < 2:
            return ids

        # compute pair stats once, then maintain incrementally
        stats = get_stats(ids)
        merges = self.merges  # local ref avoids repeated attribute lookup

        while stats:
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            ids, _ = merge_and_update_stats(ids, pair, idx, stats)

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
        allowed_special: "all" | "none" | "none_raise" | set of strings
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
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
