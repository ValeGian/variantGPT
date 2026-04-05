"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def merge_and_update_stats(ids, pair, idx, stats):
    """
    Merge all occurrences of pair -> idx in ids, updating stats
    incrementally rather than recomputing from scratch.

    Mutates stats in-place. Returns (new_ids, changed_pairs).
    changed_pairs is a set of every pair whose count was modified —
    callers that maintain a priority queue can use it to push fresh
    entries without rebuilding the whole structure.

    Why this is faster than merge() + get_stats():
    - merge() + get_stats() touch every element of ids twice: O(n) each.
    - This function also touches every element once for the merge scan,
      but the stats update is O(k) where k = number of occurrences of
      pair (k << n for rare pairs), not O(n).
    - Total per-iteration cost: O(n) scan + O(k) stats vs O(n) + O(n).

    Correctness notes for the incremental update:
      For each merged position, three kinds of pairs change:
        1. Left boundary:  (left_neighbor, p0) → (left_neighbor, idx)
        2. Right boundary: (p1, right_neighbor) → (idx, right_neighbor)
        3. Between two consecutive merges: (p1, p0) → (idx, idx)
      Cases 1 and 2 are skipped when the neighbor is itself a just-merged
      idx (consecutive merges), to avoid double-counting.
    """
    p0, p1 = pair
    newids = []
    merged_at = []   # positions in newids where idx was placed

    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
            merged_at.append(len(newids))
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    n_merged = len(merged_at)
    if n_merged == 0:
        return newids, set()

    changed = set()

    def dec(p):
        """Decrement count of p, removing the entry when it hits 0."""
        c = stats.get(p, 0) - 1
        if c <= 0:
            stats.pop(p, None)
        else:
            stats[p] = c
        changed.add(p)

    def inc(p):
        stats[p] = stats.get(p, 0) + 1
        changed.add(p)

    for j, pos in enumerate(merged_at):
        is_consecutive   = j > 0      and merged_at[j - 1] == pos - 1
        next_consecutive = j < n_merged - 1 and merged_at[j + 1] == pos + 1

        # --- left boundary ---
        # Skip when the left neighbor is another idx we just merged
        # (that update is handled by the "between" case of the prior step).
        if pos > 0 and not is_consecutive:
            left = newids[pos - 1]
            dec((left, p0))   # old left-boundary pair disappears
            inc((left, idx))  # new left-boundary pair appears

        # --- right boundary ---
        # Skip when the right neighbor is another idx we just merged
        # (that update is handled by the "between" case of this step).
        if pos < len(newids) - 1 and not next_consecutive:
            right = newids[pos + 1]
            dec((p1, right))   # old right-boundary pair disappears
            inc((idx, right))  # new right-boundary pair appears

        # --- between two consecutive merged positions ---
        if next_consecutive:
            dec((p1, p0))    # bridging pair disappears
            inc((idx, idx))  # new pair between two adjacent idx tokens

    # Remove all n_merged direct occurrences of pair that were consumed.
    # (Neighboring dec() calls above may have already partially reduced
    # stats[pair] when pair's own tokens appear as neighbors — e.g. (a,a)
    # adjacent to another (a,a) — so we adjust the remainder here.)
    remaining = stats.get(pair, 0) - n_merged
    if remaining <= 0:
        stats.pop(pair, None)
    else:
        stats[pair] = remaining
    changed.add(pair)

    return newids, changed


# first two helper functions...


def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class


class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
