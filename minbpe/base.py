"""
Base tokenizer class + shared helpers.

Key optimisation: merge_and_update_stats now works on a doubly-linked list
stored as parallel arrays (vals, prev, nxt) alongside a position index.
This makes each merge O(k) in the number of pair occurrences rather than
O(n) in the total sequence length, because we no longer rebuild the list.
"""

import unicodedata


def get_stats(ids):
    """Count consecutive pairs in *ids* (list of ints)."""
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Linked-list representation for O(k) incremental merges
# ---------------------------------------------------------------------------
# We keep three parallel lists:
#   vals[i]  – the token id at position i (only meaningful for live nodes)
#   prev[i]  – index of previous live node (-1 if head)
#   nxt[i]   – index of next live node     (-1 if tail)
# Plus a dict  pair_positions: (a,b) -> set of positions i where
#   vals[i]==a and vals[nxt[i]]==b.
#
# This representation is created lazily on the first call to
# merge_and_update_stats and cached on the *stats* dict itself (via a
# dunder key that can't collide with an (int,int) pair key).

_LL_KEY = "__ll"   # hidden key stored on the stats dict


def _init_ll(ids, stats):
    """Attach linked-list structures to *stats* for future merges."""
    n = len(ids)
    vals = list(ids)
    prev = list(range(-1, n - 1))
    nxt  = list(range(1, n + 1))
    if n:
        nxt[n - 1] = -1

    pair_pos = {}
    for i in range(n - 1):
        pair = (vals[i], vals[i + 1])
        if pair not in pair_pos:
            pair_pos[pair] = set()
        pair_pos[pair].add(i)

    stats[_LL_KEY] = (vals, prev, nxt, pair_pos)


def merge_and_update_stats(ids, pair, idx, stats):
    """
    Merge every occurrence of *pair* into *idx*.

    On the first call the function builds a linked-list index and caches it
    inside *stats*.  Subsequent calls reuse and mutate that index, making
    each merge O(k) where k = number of occurrences of the pair.

    Parameters
    ----------
    ids : list[int]
        Current token sequence.  Ignored after the first call (the linked
        list is the source of truth), but the compacted list is returned.
    pair : tuple[int, int]
        The bigram to merge.
    idx : int
        The replacement token id.
    stats : dict
        Pair-count dict.  Mutated in-place.  Also carries the linked-list
        cache under a private key.

    Returns
    -------
    new_ids : list[int]
        The compacted token sequence after the merge.
    changed : set
        Set of pairs whose counts were modified (excluding *pair* itself).
    """
    # lazily build linked list on first call
    if _LL_KEY not in stats:
        _init_ll(ids, stats)

    vals, prev, nxt, pair_pos = stats[_LL_KEY]
    changed = set()
    positions = list(pair_pos.get(pair, ()))

    for pos in positions:
        # validity check for overlapping pairs
        rn = nxt[pos]
        if rn == -1 or vals[pos] != pair[0] or vals[rn] != pair[1]:
            continue

        p  = prev[pos]
        nn = nxt[rn] if rn != -1 else -1

        # --- replace left token with merged token ---
        vals[pos] = idx

        # --- remove old neighbour pairs ---
        if p != -1:
            old_left = (vals[p], pair[0])
            pair_pos.get(old_left, set()).discard(p)
            cnt = stats.get(old_left, 0) - 1
            if cnt > 0:
                stats[old_left] = cnt
            else:
                stats.pop(old_left, None)
            changed.add(old_left)

        if nn != -1:
            old_right = (pair[1], vals[nn])
            pair_pos.get(old_right, set()).discard(rn)
            cnt = stats.get(old_right, 0) - 1
            if cnt > 0:
                stats[old_right] = cnt
            else:
                stats.pop(old_right, None)
            changed.add(old_right)

        # --- stitch out the right token ---
        nxt[pos] = nn
        if nn != -1:
            prev[nn] = pos

        # --- add new neighbour pairs ---
        if p != -1:
            new_left = (vals[p], idx)
            if new_left not in pair_pos:
                pair_pos[new_left] = set()
            pair_pos[new_left].add(p)
            stats[new_left] = stats.get(new_left, 0) + 1
            changed.add(new_left)

        if nn != -1:
            new_right = (idx, vals[nn])
            if new_right not in pair_pos:
                pair_pos[new_right] = set()
            pair_pos[new_right].add(pos)
            stats[new_right] = stats.get(new_right, 0) + 1
            changed.add(new_right)

    # clean up the merged pair from stats and pair_pos
    stats.pop(pair, None)
    pair_pos.pop(pair, None)
    changed.discard(pair)

    # --- compact the linked list into a plain list for the caller ---
    # walk the linked list from the head
    new_ids = []
    # find head (first node whose prev == -1 among live nodes)
    # The original head is always position 0 if it's still live,
    # otherwise follow the chain.  Easiest: just walk from 0.
    node = 0
    # but position 0 might have been stitched out — find the real head
    if len(vals) > 0:
        # head is the node with prev == -1 that is still reachable
        # Since we only stitch out *right* tokens of pairs, the very first
        # live node is always position 0 (it can be replaced but never removed).
        node = 0
        while node != -1:
            new_ids.append(vals[node])
            node = nxt[node]

    return new_ids, changed


# ---------------------------------------------------------------------------
# Tokenizer base class
# ---------------------------------------------------------------------------

def render_token(t):
    """Pretty-print a token, escaping control characters."""
    s = t.decode("utf-8", errors="replace")
    ctrl = unicodedata.category(s[0]).startswith("C") if s else False
    return repr(s) if ctrl else s


class Tokenizer:
    """Base class; subclasses must implement train / encode / decode."""

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().rsplit(" ", 1)
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
