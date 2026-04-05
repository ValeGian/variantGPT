"""
Correctness tests for BasicTokenizer and RegexTokenizer.

Run:  pytest tests/test_tokenizers.py -v
"""

import pytest

from minbpe.basic import BasicTokenizer
from minbpe.regex import RegexTokenizer


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

SIMPLE = "aaabdaaabac"
UNICODE = "Hello, world! 你好世界 🌍🔥 café naïve résumé"
REPEATED = "the cat sat on the mat. " * 200
OVERLAP = "aaaaaaaaa"

CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells by the sea shore. "
    "Peter Piper picked a peck of pickled peppers. "
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood? "
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo. "
) * 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_trip(tok, text):
    """Encode then decode; assert we recover the original text."""
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


# ---------------------------------------------------------------------------
# BasicTokenizer
# ---------------------------------------------------------------------------

class TestBasicTokenizer:

    @pytest.fixture
    def tok(self):
        t = BasicTokenizer()
        t.train(CORPUS, vocab_size=300, verbose=False)
        return t

    def test_merge_count(self, tok):
        assert len(tok.merges) == 300 - 256

    def test_vocab_size(self, tok):
        assert len(tok.vocab) == 300

    def test_round_trip_corpus(self, tok):
        round_trip(tok, CORPUS)

    @pytest.mark.parametrize("text", [
        SIMPLE, UNICODE, OVERLAP, "", "x", "ab",
    ], ids=["simple", "unicode", "overlap", "empty", "single_char", "two_chars"])
    def test_round_trip_unseen(self, tok, text):
        round_trip(tok, text)

    def test_compression(self, tok):
        raw_len = len(CORPUS.encode("utf-8"))
        enc_len = len(tok.encode(CORPUS))
        assert enc_len < raw_len

    def test_determinism(self):
        t1 = BasicTokenizer()
        t1.train(CORPUS, vocab_size=300, verbose=False)
        t2 = BasicTokenizer()
        t2.train(CORPUS, vocab_size=300, verbose=False)
        assert t1.merges == t2.merges


# ---------------------------------------------------------------------------
# RegexTokenizer
# ---------------------------------------------------------------------------

class TestRegexTokenizer:

    @pytest.fixture
    def tok(self):
        t = RegexTokenizer()
        t.train(CORPUS, vocab_size=300, verbose=False, n_workers=1)
        return t

    def test_merge_count(self, tok):
        assert len(tok.merges) == 300 - 256

    def test_vocab_size(self, tok):
        assert len(tok.vocab) == 300

    @pytest.mark.parametrize("text", [
        CORPUS, SIMPLE, UNICODE, REPEATED, OVERLAP, "", "x", "   \n\t  ",
    ], ids=[
        "corpus", "simple", "unicode", "repeated",
        "overlap", "empty", "single_char", "whitespace",
    ])
    def test_round_trip(self, tok, text):
        round_trip(tok, text)

    def test_compression(self, tok):
        raw_len = len(CORPUS.encode("utf-8"))
        enc_len = len(tok.encode(CORPUS))
        assert enc_len < raw_len

    def test_parallel_train_matches_serial(self, tok):
        tok_par = RegexTokenizer()
        tok_par.train(CORPUS, vocab_size=300, verbose=False, n_workers=4)
        assert tok.merges == tok_par.merges

    @pytest.mark.parametrize("text", [CORPUS, UNICODE, REPEATED, OVERLAP, ""])
    def test_parallel_encode_matches_serial(self, tok, text):
        ids_ser = tok.encode_ordinary(text, n_workers=1)
        ids_par = tok.encode_ordinary(text, n_workers=4)
        assert ids_ser == ids_par

    def test_determinism(self):
        t1 = RegexTokenizer()
        t1.train(CORPUS, vocab_size=300, verbose=False, n_workers=1)
        t2 = RegexTokenizer()
        t2.train(CORPUS, vocab_size=300, verbose=False, n_workers=1)
        assert t1.merges == t2.merges


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

class TestSpecialTokens:

    @pytest.fixture
    def tok(self):
        t = RegexTokenizer()
        t.train(CORPUS, vocab_size=280, verbose=False, n_workers=1)
        t.register_special_tokens({
            "<|endoftext|>": 280,
            "<|pad|>": 281,
        })
        return t

    def test_special_tokens_in_ids(self, tok):
        text = "hello<|endoftext|>world<|pad|>!"
        ids = tok.encode(text, allowed_special="all")
        assert 280 in ids
        assert 281 in ids

    def test_round_trip_with_special(self, tok):
        text = "hello<|endoftext|>world<|pad|>!"
        ids = tok.encode(text, allowed_special="all")
        assert tok.decode(ids) == text

    def test_none_raise_rejects_special(self, tok):
        text = "hello<|endoftext|>world"
        with pytest.raises(AssertionError):
            tok.encode(text, allowed_special="none_raise")

    def test_none_treats_as_ordinary(self, tok):
        text = "hello<|endoftext|>world<|pad|>!"
        ids = tok.encode(text, allowed_special="none")
        assert tok.decode(ids) == text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_merges_at_vocab_256(self):
        tok = BasicTokenizer()
        tok.train("hello", vocab_size=256, verbose=False)
        assert len(tok.merges) == 0
        ids = tok.encode("hello")
        assert ids == list(b"hello")
        assert tok.decode(ids) == "hello"

    def test_single_merge(self):
        tok = BasicTokenizer()
        tok.train("aaaa", vocab_size=257, verbose=False)
        assert len(tok.merges) == 1
        ids = tok.encode("aa")
        assert len(ids) == 1
        assert tok.decode(ids) == "aa"

    def test_multibyte_utf8(self):
        tok = RegexTokenizer()
        tok.train("日本語のテスト " * 100, vocab_size=300, verbose=False, n_workers=1)
        text = "日本語テスト"
        assert tok.decode(tok.encode(text)) == text

    def test_overlap_stress(self):
        tok = BasicTokenizer()
        tok.train("a" * 1000, vocab_size=266, verbose=False)
        text = "a" * 100
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert len(ids) < 100


# ---------------------------------------------------------------------------
# Performance (not assertions on speed, just verify correctness)
# ---------------------------------------------------------------------------

class TestPerformance:

    @pytest.fixture
    def big_text(self):
        return CORPUS * 10

    def test_parallel_train_matches_on_large_corpus(self, big_text):
        tok1 = RegexTokenizer()
        tok1.train(big_text, vocab_size=400, verbose=False, n_workers=1)

        tok4 = RegexTokenizer()
        tok4.train(big_text, vocab_size=400, verbose=False, n_workers=4)

        assert tok1.merges == tok4.merges