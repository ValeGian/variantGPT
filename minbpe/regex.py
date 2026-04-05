"""
Parallel (byte-level) Byte Pair Encoding tokenizer.

Parallelisation strategy
------------------------
Training:  Chunks are sharded across persistent worker *processes* (one per
           CPU core).  Each worker owns its shard's linked-list state for the
           entire training run.  Per merge the main process broadcasts
           (pair, idx) and each worker replies with a {pair: delta} dict.
           No linked-list data is serialised after startup.

Encoding:  Each regex chunk is independent, so encode_ordinary fans them
           out across a process pool with Pool.map.

All single-process optimisations from the previous version are retained:
incremental delta tracking in _merge_chunk, overlapping-pair validity
check, heap with lazy deletion, bytearray decode, incremental stats in
_encode_chunk.
"""

import heapq
import os
from collections import defaultdict
from multiprocessing import Process, Pipe, cpu_count

import regex as re
from tqdm import tqdm

from .base import Tokenizer, get_stats, merge_and_update_stats

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Linked-list helpers  (same as optimised single-threaded version)
# ---------------------------------------------------------------------------

def _build_chunk_index(chunk_ids):
    tokens = list(chunk_ids)
    n = len(tokens)
    prev = list(range(-1, n - 1))
    nxt  = list(range(1, n + 1))
    if n:
        nxt[n - 1] = -1
    pair_pos = defaultdict(set)
    for i in range(n - 1):
        pair_pos[(tokens[i], tokens[i + 1])].add(i)
    return tokens, prev, nxt, pair_pos


def _merge_chunk(tokens, prev, nxt, pair_pos, pair, idx):
    """Apply one merge in-place, return {pair: delta} dict."""
    deltas = {}
    for pos in list(pair_pos.get(pair, ())):
        rn = nxt[pos]
        if rn == -1 or tokens[pos] != pair[0] or tokens[rn] != pair[1]:
            deltas[pair] = deltas.get(pair, 0) - 1
            continue
        p  = prev[pos]
        nn = nxt[rn] if rn != -1 else -1
        tokens[pos] = idx
        if p != -1:
            old = (tokens[p], pair[0])
            pair_pos[old].discard(p)
            deltas[old] = deltas.get(old, 0) - 1
        if nn != -1:
            old = (pair[1], tokens[nn])
            pair_pos[old].discard(rn)
            deltas[old] = deltas.get(old, 0) - 1
        nxt[pos] = nn
        if nn != -1:
            prev[nn] = pos
        if p != -1:
            np_ = (tokens[p], idx)
            pair_pos[np_].add(p)
            deltas[np_] = deltas.get(np_, 0) + 1
        if nn != -1:
            np_ = (idx, tokens[nn])
            pair_pos[np_].add(pos)
            deltas[np_] = deltas.get(np_, 0) + 1
    pair_pos.pop(pair, None)
    deltas.pop(pair, None)
    return deltas


# ---------------------------------------------------------------------------
# Persistent shard worker (training)
# ---------------------------------------------------------------------------

class _ShardWorker(Process):
    """
    Long-lived subprocess that owns a shard of chunks.

    Protocol over the Pipe:
        startup  -> worker sends {"init_counts": {pair: count, ...}}
        main sends (pair, idx) -> worker sends {pair: delta, ...}
        main sends None         -> worker exits
    """

    def __init__(self, conn, raw_chunks):
        """
        Parameters
        ----------
        conn : Connection
            One end of a multiprocessing.Pipe.
        raw_chunks : list[list[int]]
            Byte-id lists for each chunk in this shard.  Linked-list
            structures are built *inside* the subprocess to avoid
            serialising them.
        """
        super().__init__(daemon=True)
        self.conn = conn
        self.raw_chunks = raw_chunks

    def run(self):
        # Build linked lists inside this process — never serialised again.
        chunks = [_build_chunk_index(ids) for ids in self.raw_chunks]

        # Send initial pair counts back to the main process.
        counts = defaultdict(int)
        for _, _, _, pair_pos in chunks:
            for pair, positions in pair_pos.items():
                counts[pair] += len(positions)
        self.conn.send(dict(counts))

        # Main merge loop — receive commands until None.
        while True:
            msg = self.conn.recv()
            if msg is None:
                break
            pair, idx = msg
            shard_deltas = defaultdict(int)
            for tokens, prev_ll, nxt_ll, pair_pos in chunks:
                for p, d in _merge_chunk(
                        tokens, prev_ll, nxt_ll, pair_pos, pair, idx).items():
                    shard_deltas[p] += d
            self.conn.send(dict(shard_deltas))


# ---------------------------------------------------------------------------
# Module-level helpers for parallel encoding
# ---------------------------------------------------------------------------

_encode_merges = None  # set by pool initialiser


def _init_encode_worker(merges):
    global _encode_merges
    _encode_merges = merges


def _encode_chunk_worker(text_bytes):
    """Standalone function for Pool.map — uses module-global merges."""
    ids = list(text_bytes)
    if len(ids) < 2:
        return ids
    stats = get_stats(ids)
    merges = _encode_merges
    while stats:
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids, _ = merge_and_update_stats(ids, pair, merges[pair], stats)
    return ids


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

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size, verbose=False, n_workers=None):
        """
        Train BPE.

        Parameters
        ----------
        n_workers : int | None
            Number of worker processes for parallel chunk merging.
            Defaults to cpu_count().  Set to 0 or 1 for single-process
            mode (useful for debugging or small texts).
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        if n_workers is None:
            n_workers = cpu_count()

        # split text into chunks, convert to byte-id lists
        text_chunks = re.findall(self.compiled_pattern, text)
        raw_chunks = [list(ch.encode("utf-8")) for ch in text_chunks]

        if n_workers <= 1 or len(raw_chunks) < n_workers:
            self._train_single(raw_chunks, num_merges, vocab_size, verbose)
        else:
            self._train_parallel(raw_chunks, num_merges, vocab_size,
                                 verbose, n_workers)

    # ---- single-process training (unchanged from optimised version) --------
    def _train_single(self, raw_chunks, num_merges, vocab_size, verbose):
        chunks = [_build_chunk_index(ids) for ids in raw_chunks]

        global_counts = defaultdict(int)
        for _, _, _, pair_pos in chunks:
            for pair, positions in pair_pos.items():
                global_counts[pair] += len(positions)

        heap = [(-cnt, pair) for pair, cnt in global_counts.items()]
        heapq.heapify(heap)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges), total=num_merges):
            pair = None
            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                real = global_counts.get(candidate, 0)
                if real == -neg_cnt and real > 0:
                    pair = candidate
                    break
                if real > 0:
                    heapq.heappush(heap, (-real, candidate))
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

            all_deltas = defaultdict(int)
            for tokens, prev_ll, nxt_ll, pair_pos in chunks:
                for p, d in _merge_chunk(
                        tokens, prev_ll, nxt_ll, pair_pos, pair, idx).items():
                    all_deltas[p] += d

            for ap, delta in all_deltas.items():
                new_cnt = global_counts.get(ap, 0) + delta
                if new_cnt > 0:
                    global_counts[ap] = new_cnt
                    heapq.heappush(heap, (-new_cnt, ap))
                else:
                    global_counts.pop(ap, None)

        self.merges = merges
        self.vocab = vocab

    # ---- multi-process training -------------------------------------------
    def _train_parallel(self, raw_chunks, num_merges, vocab_size,
                        verbose, n_workers):
        # Round-robin shard the chunks across workers.
        shards = [[] for _ in range(n_workers)]
        for i, rc in enumerate(raw_chunks):
            shards[i % n_workers].append(rc)

        # Spawn persistent workers.
        workers = []
        conns   = []
        for shard in shards:
            parent_conn, child_conn = Pipe()
            w = _ShardWorker(child_conn, shard)
            w.start()
            workers.append(w)
            conns.append(parent_conn)

        try:
            # Collect initial counts from every worker.
            global_counts = defaultdict(int)
            for conn in conns:
                shard_counts = conn.recv()
                for pair, cnt in shard_counts.items():
                    global_counts[pair] += cnt

            heap = [(-cnt, pair) for pair, cnt in global_counts.items()]
            heapq.heapify(heap)
            merges = {}
            vocab = {idx: bytes([idx]) for idx in range(256)}

            for i in tqdm(range(num_merges), total=num_merges):
                # Find best pair.
                pair = None
                while heap:
                    neg_cnt, candidate = heapq.heappop(heap)
                    real = global_counts.get(candidate, 0)
                    if real == -neg_cnt and real > 0:
                        pair = candidate
                        break
                    if real > 0:
                        heapq.heappush(heap, (-real, candidate))
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

                # Broadcast merge command to all workers.
                for conn in conns:
                    conn.send((pair, idx))

                # Collect deltas from all workers.
                all_deltas = defaultdict(int)
                for conn in conns:
                    shard_deltas = conn.recv()
                    for p, d in shard_deltas.items():
                        all_deltas[p] += d

                for ap, delta in all_deltas.items():
                    new_cnt = global_counts.get(ap, 0) + delta
                    if new_cnt > 0:
                        global_counts[ap] = new_cnt
                        heapq.heappush(heap, (-new_cnt, ap))
                    else:
                        global_counts.pop(ap, None)

            self.merges = merges
            self.vocab = vocab
        finally:
            # Shut down workers.
            for conn in conns:
                try:
                    conn.send(None)
                except BrokenPipeError:
                    pass
            for w in workers:
                w.join(timeout=5)

    # ----------------------------------------------------------------- tokens
    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    # ---------------------------------------------------------------- decode
    def decode(self, ids):
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

    # ---------------------------------------------------------------- encode
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        if len(ids) < 2:
            return ids
        stats = get_stats(ids)
        merges = self.merges
        while stats:
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids, _ = merge_and_update_stats(ids, pair, merges[pair], stats)
        return ids

    def encode_ordinary(self, text, n_workers=None):
        """
        Encode text, ignoring special tokens.

        Parameters
        ----------
        n_workers : int | None
            Worker processes for parallel chunk encoding.
            Defaults to cpu_count().  Set to 0 or 1 for single-process.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        if n_workers is None:
            n_workers = cpu_count()

        byte_chunks = [chunk.encode("utf-8") for chunk in text_chunks]

        if n_workers <= 1 or len(byte_chunks) < n_workers:
            # single-process fast path
            ids = []
            for bc in byte_chunks:
                ids.extend(self._encode_chunk(bc))
            return ids

        # Parallel encoding: each chunk is fully independent.
        from multiprocessing import Pool
        with Pool(n_workers, initializer=_init_encode_worker,
                  initargs=(self.merges,)) as pool:
            # chunksize keeps IPC overhead manageable for many small chunks
            cs = max(1, len(byte_chunks) // (n_workers * 4))
            results = pool.map(_encode_chunk_worker, byte_chunks,
                               chunksize=cs)
        ids = []
        for r in results:
            ids.extend(r)
        return ids

    def encode(self, text, allowed_special="none_raise", n_workers=None):
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
            return self.encode_ordinary(text, n_workers=n_workers)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part, n_workers=n_workers))
        return ids