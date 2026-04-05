"""
Correctness tests for BasicTokenizer and RegexTokenizer.

Run from the repo root:
    python -m minbpe.test_tokenizers
    (or adjust the imports to match your package layout)
"""

import sys
import time

from minbpe.basic import BasicTokenizer
from minbpe.regex import RegexTokenizer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def assert_eq(a, b, msg=""):
    if a != b:
        detail = f"\n  got:      {a!r}\n  expected: {b!r}"
        raise AssertionError(f"FAIL: {msg}{detail}")


def round_trip(tok, text, label=""):
    """Encode then decode; verify we get the original text back."""
    ids = tok.encode(text) if hasattr(tok, 'encode') else tok.encode(text)
    decoded = tok.decode(ids)
    assert_eq(decoded, text, f"round-trip {label}")
    return ids


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# test corpus
# ---------------------------------------------------------------------------

SIMPLE   = "aaabdaaabac"
UNICODE  = "Hello, world! 你好世界 🌍🔥 café naïve résumé"
REPEATED = "the cat sat on the mat. " * 200
OVERLAP  = "aaaaaaaaa"  # overlapping pairs: (a,a) in a run

# A longer corpus for more thorough training
CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells by the sea shore. "
    "Peter Piper picked a peck of pickled peppers. "
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood? "
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo. "
) * 50


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_basic_tokenizer():
    print("=" * 60)
    print("BasicTokenizer")
    print("=" * 60)

    tok = BasicTokenizer()

    # --- train ---
    tok.train(CORPUS, vocab_size=300, verbose=False)
    assert len(tok.merges) == 300 - 256, "wrong number of merges"
    assert len(tok.vocab) == 300, "wrong vocab size"
    print(f"  train: {len(tok.merges)} merges learned ✓")

    # --- round-trip on training corpus ---
    round_trip(tok, CORPUS, "training corpus")
    print("  round-trip (training corpus) ✓")

    # --- round-trip on unseen text ---
    for label, text in [("simple", SIMPLE),
                        ("unicode", UNICODE),
                        ("overlap", OVERLAP),
                        ("empty", ""),
                        ("single char", "x"),
                        ("two chars", "ab")]:
        round_trip(tok, text, label)
    print("  round-trip (unseen texts) ✓")

    # --- encode should compress the repeated corpus ---
    raw_len = len(CORPUS.encode("utf-8"))
    enc_len = len(tok.encode(CORPUS))
    assert enc_len < raw_len, "encoding should compress repeated text"
    print(f"  compression: {raw_len} bytes → {enc_len} tokens "
          f"({enc_len / raw_len:.1%}) ✓")

    # --- determinism: training twice gives the same merges ---
    tok2 = BasicTokenizer()
    tok2.train(CORPUS, vocab_size=300, verbose=False)
    assert_eq(tok.merges, tok2.merges, "determinism")
    print("  determinism ✓")

    print()


def test_regex_tokenizer():
    print("=" * 60)
    print("RegexTokenizer")
    print("=" * 60)

    tok = RegexTokenizer()

    # --- train (single-process) ---
    tok.train(CORPUS, vocab_size=300, verbose=False, n_workers=1)
    assert len(tok.merges) == 300 - 256
    assert len(tok.vocab) == 300
    print(f"  train (1 worker): {len(tok.merges)} merges ✓")

    # --- round-trip ---
    for label, text in [("corpus", CORPUS),
                        ("simple", SIMPLE),
                        ("unicode", UNICODE),
                        ("repeated", REPEATED),
                        ("overlap", OVERLAP),
                        ("empty", ""),
                        ("single char", "x"),
                        ("whitespace", "   \n\t  ")]:
        round_trip(tok, text, label)
    print("  round-trip (various texts) ✓")

    # --- compression ---
    raw_len = len(CORPUS.encode("utf-8"))
    enc_len = len(tok.encode(CORPUS))
    assert enc_len < raw_len
    print(f"  compression: {raw_len} bytes → {enc_len} tokens "
          f"({enc_len / raw_len:.1%}) ✓")

    # --- parallel train produces identical merges ---
    tok_par = RegexTokenizer()
    tok_par.train(CORPUS, vocab_size=300, verbose=False, n_workers=4)
    assert_eq(tok.merges, tok_par.merges, "parallel vs serial merges")
    print("  parallel train (4 workers) matches serial ✓")

    # --- parallel encode matches serial ---
    for text in [CORPUS, UNICODE, REPEATED, OVERLAP, ""]:
        ids_ser = tok.encode_ordinary(text, n_workers=1)
        ids_par = tok.encode_ordinary(text, n_workers=4)
        assert_eq(ids_ser, ids_par, "parallel vs serial encode")
    print("  parallel encode matches serial ✓")

    # --- determinism ---
    tok2 = RegexTokenizer()
    tok2.train(CORPUS, vocab_size=300, verbose=False, n_workers=1)
    assert_eq(tok.merges, tok2.merges, "determinism")
    print("  determinism ✓")

    print()


def test_special_tokens():
    print("=" * 60)
    print("RegexTokenizer — special tokens")
    print("=" * 60)

    tok = RegexTokenizer()
    tok.train(CORPUS, vocab_size=280, verbose=False, n_workers=1)

    tok.register_special_tokens({
        "<|endoftext|>": 280,
        "<|pad|>":       281,
    })

    # encode with special tokens allowed
    text = "hello<|endoftext|>world<|pad|>!"
    ids = tok.encode(text, allowed_special="all")
    assert 280 in ids, "endoftext token should appear"
    assert 281 in ids, "pad token should appear"
    decoded = tok.decode(ids)
    assert_eq(decoded, text, "round-trip with special tokens")
    print("  encode/decode with special tokens ✓")

    # none_raise should raise on special tokens in text
    try:
        tok.encode(text, allowed_special="none_raise")
        raise AssertionError("should have raised on special token")
    except AssertionError:
        pass
    print("  none_raise correctly rejects special tokens ✓")

    # none should silently treat them as ordinary text
    ids_none = tok.encode(text, allowed_special="none")
    decoded_none = tok.decode(ids_none)
    assert_eq(decoded_none, text, "round-trip with allowed_special='none'")
    print("  allowed_special='none' round-trip ✓")

    print()


def test_edge_cases():
    print("=" * 60)
    print("Edge cases")
    print("=" * 60)

    # --- vocab_size == 256: no merges ---
    tok = BasicTokenizer()
    tok.train("hello", vocab_size=256, verbose=False)
    assert len(tok.merges) == 0
    ids = tok.encode("hello")
    assert_eq(ids, list(b"hello"), "no-merge encode")
    assert_eq(tok.decode(ids), "hello", "no-merge decode")
    print("  vocab_size=256 (no merges) ✓")

    # --- single-byte input ---
    tok = BasicTokenizer()
    tok.train("aaaa", vocab_size=257, verbose=False)
    assert len(tok.merges) == 1
    ids = tok.encode("aa")
    assert_eq(len(ids), 1, "single merge should fuse 'aa' into 1 token")
    assert_eq(tok.decode(ids), "aa", "single merge decode")
    print("  single merge on 'aaaa' ✓")

    # --- multi-byte UTF-8 ---
    tok = RegexTokenizer()
    tok.train("日本語のテスト " * 100, vocab_size=300, verbose=False, n_workers=1)
    text = "日本語テスト"
    assert_eq(tok.decode(tok.encode(text)), text, "Japanese round-trip")
    print("  multi-byte UTF-8 round-trip ✓")

    # --- overlapping pairs stress test ---
    tok = BasicTokenizer()
    tok.train("a" * 1000, vocab_size=266, verbose=False)
    text = "a" * 100
    ids = tok.encode(text)
    assert_eq(tok.decode(ids), text, "overlap stress round-trip")
    assert len(ids) < 100, "repeated merges should compress 'aaa...'"
    print(f"  overlap stress: 100 a's → {len(ids)} tokens ✓")

    print()


def test_performance():
    print("=" * 60)
    print("Performance (informational)")
    print("=" * 60)

    big_text = CORPUS * 10  # ~100k chars
    raw_bytes = len(big_text.encode("utf-8"))

    # --- train timing ---
    tok = RegexTokenizer()
    _, t_train_1 = timed(tok.train, big_text, vocab_size=400,
                         verbose=False, n_workers=1)
    print(f"  train  (1 worker,  {raw_bytes:,} bytes, 400 vocab): "
          f"{t_train_1:.2f}s")

    tok_par = RegexTokenizer()
    _, t_train_n = timed(tok_par.train, big_text, vocab_size=400,
                         verbose=False, n_workers=4)
    print(f"  train  (4 workers, {raw_bytes:,} bytes, 400 vocab): "
          f"{t_train_n:.2f}s")

    # verify they agree
    assert_eq(tok.merges, tok_par.merges, "perf: parallel merges match")
    print("  parallel merges match ✓")

    # --- encode timing ---
    _, t_enc_1 = timed(tok.encode_ordinary, big_text, n_workers=1)
    _, t_enc_n = timed(tok.encode_ordinary, big_text, n_workers=4)
    print(f"  encode (1 worker):  {t_enc_1:.2f}s")
    print(f"  encode (4 workers): {t_enc_n:.2f}s")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print()
    passed = 0
    failed = 0
    tests = [
        test_basic_tokenizer,
        test_regex_tokenizer,
        test_special_tokens,
        test_edge_cases,
        test_performance,
    ]
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_fn.__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed "
          f"out of {passed + failed}")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
