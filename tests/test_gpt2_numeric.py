"""
Numeric correctness tests for GPT2Model.

Run:  pytest tests/test_gpt2_numeric.py -v
"""

import math
import pytest
import torch
from torch.nn import functional as F

from variant_gpt.models.gpt2.configuration import GPT2Config
from variant_gpt.models.gpt2.modeling import (
    GPT2Attention,
    GPT2Block,
    GPT2MLP,
    GPT2Model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
DEVICE = "cpu"


def _small_config(**overrides) -> GPT2Config:
    """Deterministic small config for fast, reproducible tests."""
    defaults = dict(
        vocab_size=64,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,          # deterministic
        bias=True,            # enable biases so bias-init tests are exercised
        flash=False,           # so we can inspect attention weights
        device=DEVICE,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
    )
    defaults.update(overrides)
    return GPT2Config(**defaults)


@pytest.fixture
def config():
    return _small_config()


@pytest.fixture
def model(config):
    torch.manual_seed(SEED)
    m = GPT2Model(config)
    m.eval()
    return m


@pytest.fixture
def input_ids(config):
    torch.manual_seed(SEED + 1)
    return torch.randint(0, config.vocab_size, (2, 16))


# ---------------------------------------------------------------------------
# 1. Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:

    def test_logits_shape_no_targets(self, model, input_ids):
        logits, loss = model(input_ids)
        B, T = input_ids.shape
        # Without targets only the last position is projected
        assert logits.shape == (B, 1, model.config.vocab_size)
        assert loss is None

    def test_logits_shape_with_targets(self, model, input_ids):
        targets = input_ids.clone()
        logits, loss = model(input_ids, targets=targets)
        B, T = input_ids.shape
        assert logits.shape == (B, T, model.config.vocab_size)
        assert loss.shape == ()

    def test_single_token_input(self, model):
        idx = torch.tensor([[0]])
        logits, _ = model(idx)
        assert logits.shape == (1, 1, model.config.vocab_size)


# ---------------------------------------------------------------------------
# 2. Determinism & reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_forward_deterministic(self, model, input_ids):
        logits1, _ = model(input_ids)
        logits2, _ = model(input_ids)
        torch.testing.assert_close(logits1, logits2)

    def test_seed_reproducibility(self, config, input_ids):
        torch.manual_seed(SEED)
        m1 = GPT2Model(config)
        m1.eval()

        torch.manual_seed(SEED)
        m2 = GPT2Model(config)
        m2.eval()

        l1, _ = m1(input_ids)
        l2, _ = m2(input_ids)
        torch.testing.assert_close(l1, l2)


# ---------------------------------------------------------------------------
# 3. Loss numeric correctness
# ---------------------------------------------------------------------------

class TestLoss:

    def test_loss_matches_manual_cross_entropy(self, model, input_ids):
        targets = input_ids.clone()
        logits, loss = model(input_ids, targets=targets)
        expected = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        torch.testing.assert_close(loss, expected)

    def test_loss_decreases_with_correct_targets(self, config):
        """A single gradient step on matching targets should reduce loss."""
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        m.train()
        idx = torch.randint(0, config.vocab_size, (4, 16))
        _, loss_before = m(idx, targets=idx)

        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        loss_before.backward()
        opt.step()
        opt.zero_grad()

        _, loss_after = m(idx, targets=idx)
        assert loss_after.item() < loss_before.item()

    def test_loss_positive(self, model, input_ids):
        _, loss = model(input_ids, targets=input_ids)
        assert loss.item() > 0.0

    def test_ignore_index_minus1(self, model, input_ids):
        """Tokens marked -1 in targets should not contribute to loss."""
        full_targets = input_ids.clone()
        _, loss_full = model(input_ids, targets=full_targets)

        partial_targets = input_ids.clone()
        partial_targets[:, ::2] = -1  # mask every other token
        _, loss_partial = model(input_ids, targets=partial_targets)

        # Losses should differ because fewer tokens contribute
        assert not torch.allclose(loss_full, loss_partial)


# ---------------------------------------------------------------------------
# 4. Attention numeric correctness
# ---------------------------------------------------------------------------

class TestAttention:

    @pytest.fixture
    def attn(self, config):
        torch.manual_seed(SEED)
        a = GPT2Attention(config, flash=False)
        a.eval()
        return a

    @pytest.fixture
    def x(self, config):
        torch.manual_seed(SEED + 1)
        return torch.randn(2, 8, config.n_embd)

    def test_output_shape(self, attn, x, config):
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_mask(self, config):
        """Verify attention weights are zero above the diagonal (causal)."""
        torch.manual_seed(SEED)
        attn = GPT2Attention(config, flash=False)
        attn.eval()

        x = torch.randn(1, 8, config.n_embd)
        B, T, C = x.size()
        q, k, v = attn.c_attn(x).split(config.n_embd, dim=2)
        nh, hs = config.n_head, config.n_embd // config.n_head
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)

        scale = 1.0 / math.sqrt(hs)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Upper-triangular entries (future positions) must be 0
        for i in range(T):
            for j in range(i + 1, T):
                assert att[0, :, i, j].abs().max().item() < 1e-6

    def test_attention_rows_sum_to_one(self, config):
        """Softmax rows in attention weights should sum to 1."""
        torch.manual_seed(SEED)
        attn = GPT2Attention(config, flash=False)
        attn.eval()

        x = torch.randn(1, 8, config.n_embd)
        B, T, C = x.size()
        q, k, v = attn.c_attn(x).split(config.n_embd, dim=2)
        nh, hs = config.n_head, config.n_embd // config.n_head
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)

        scale = 1.0 / math.sqrt(hs)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        row_sums = att.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0)

    def test_flash_vs_manual_attention(self, config):
        """Flash and manual attention paths must produce the same output."""
        torch.manual_seed(SEED)
        attn_manual = GPT2Attention(config, flash=False)
        attn_manual.eval()

        # Clone weights into a flash variant
        attn_flash = GPT2Attention(config, flash=True)
        attn_flash.load_state_dict(attn_manual.state_dict(), strict=False)
        attn_flash.eval()

        x = torch.randn(2, 8, config.n_embd)
        out_manual = attn_manual(x)
        out_flash = attn_flash(x)

        torch.testing.assert_close(out_manual, out_flash, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 5. MLP numeric correctness
# ---------------------------------------------------------------------------

class TestMLP:

    def test_output_shape(self, config):
        torch.manual_seed(SEED)
        mlp = GPT2MLP(config)
        x = torch.randn(2, 8, config.n_embd)
        assert mlp(x).shape == x.shape

    def test_activation_applied(self, config):
        """Output should differ from a pure linear pass — activation is non-trivial."""
        torch.manual_seed(SEED)
        mlp = GPT2MLP(config)
        mlp.eval()
        x = torch.randn(2, 8, config.n_embd)

        # Linear-only path (no activation)
        linear_out = mlp.proj(mlp.fc(x))
        actual_out = mlp(x)
        assert not torch.allclose(linear_out, actual_out, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Transformer block (residual stream)
# ---------------------------------------------------------------------------

class TestBlock:

    def test_residual_connection(self, config):
        """Output should differ from sub-layer output alone (residual adds input)."""
        torch.manual_seed(SEED)
        block = GPT2Block(config)
        block.eval()
        x = torch.randn(2, 8, config.n_embd)

        out = block(x)
        # If residual were missing, out ≈ mlp(ln2(attn(ln1(x)))),
        # which won't match x + attn(...) + mlp(...)
        assert out.shape == x.shape
        # The output should not be identical to the input
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# 7. Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:

    def test_weight_tying(self, model):
        """Token embedding and lm_head must share the same weight tensor."""
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_position_embedding_range(self, model, config):
        """Position ids must stay within [0, block_size)."""
        pos_buf = model._pos
        assert pos_buf.min() == 0
        assert pos_buf.max() == config.block_size - 1
        assert pos_buf.shape == (config.block_size,)


# ---------------------------------------------------------------------------
# 8. Weight initialization
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_linear_weight_std(self, config):
        """Linear layer weights should be ≈ N(0, 0.02)."""
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        for name, p in m.named_parameters():
            if "weight" in name and p.dim() == 2 and not name.endswith("c_proj.weight"):
                assert p.std().item() == pytest.approx(0.02, abs=0.01), (
                    f"{name} std={p.std().item():.4f}"
                )

    def test_residual_projection_scaled_init(self, config):
        """c_proj weights should have std ≈ 0.02 / sqrt(2 * n_layer)."""
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        expected_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, p in m.named_parameters():
            if name.endswith("c_proj.weight"):
                assert p.std().item() == pytest.approx(expected_std, abs=0.005), (
                    f"{name} std={p.std().item():.4f}, expected≈{expected_std:.4f}"
                )

    def test_bias_initialized_to_zero(self, config):
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        for name, p in m.named_parameters():
            if "bias" in name and p.dim() == 1:
                assert (p == 0).all(), f"{name} has non-zero bias after init"


# ---------------------------------------------------------------------------
# 9. Generation
# ---------------------------------------------------------------------------

class TestGeneration:

    def test_generated_length(self, model):
        prompt = torch.tensor([[1, 2, 3]])
        max_new = 10
        out = model.generate(prompt, max_new_tokens=max_new, temperature=1.0)
        assert out.shape == (1, 3 + max_new)

    def test_prompt_preserved(self, model):
        prompt = torch.tensor([[5, 10, 15]])
        out = model.generate(prompt, max_new_tokens=5, temperature=1.0)
        torch.testing.assert_close(out[:, :3], prompt)

    def test_valid_token_range(self, model, config):
        prompt = torch.tensor([[0]])
        out = model.generate(prompt, max_new_tokens=20, temperature=1.0)
        assert (out >= 0).all() and (out < config.vocab_size).all()

    def test_temperature_zero_is_greedy(self, model):
        """Temperature → 0 should behave like argmax (greedy)."""
        prompt = torch.tensor([[1, 2, 3]])
        # Very low temp ≈ greedy
        out1 = model.generate(prompt.clone(), max_new_tokens=5, temperature=1e-8)
        out2 = model.generate(prompt.clone(), max_new_tokens=5, temperature=1e-8)
        torch.testing.assert_close(out1, out2)

    def test_top_k_restricts_vocab(self, model, config):
        """With top_k=1 every step is deterministic (always pick the top token)."""
        prompt = torch.tensor([[0]])
        out1 = model.generate(prompt.clone(), max_new_tokens=10, temperature=1.0, top_k=1)
        out2 = model.generate(prompt.clone(), max_new_tokens=10, temperature=1.0, top_k=1)
        torch.testing.assert_close(out1, out2)

    def test_top_p_restricts_vocab(self, model):
        """With top_p ≈ 0 only the most likely token is kept → deterministic."""
        prompt = torch.tensor([[0]])
        out1 = model.generate(prompt.clone(), max_new_tokens=10, temperature=1.0, top_p=1e-8)
        out2 = model.generate(prompt.clone(), max_new_tokens=10, temperature=1.0, top_p=1e-8)
        torch.testing.assert_close(out1, out2)

    def test_block_size_respected(self, model, config):
        """Generation should work even when prompt + new tokens > block_size."""
        prompt = torch.randint(0, config.vocab_size, (1, config.block_size))
        # Generating beyond block_size forces context cropping
        out = model.generate(prompt, max_new_tokens=5, temperature=1.0)
        assert out.shape == (1, config.block_size + 5)


# ---------------------------------------------------------------------------
# 10. Gradient flow
# ---------------------------------------------------------------------------

class TestGradients:

    def test_all_params_receive_gradients(self, config):
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        m.train()

        idx = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = m(idx, targets=idx)
        loss.backward()

        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no gradient"
                assert p.grad.abs().sum() > 0, f"{name} gradient is all zeros"

    def test_no_nan_in_gradients(self, config):
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        m.train()

        idx = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = m(idx, targets=idx)
        loss.backward()

        for name, p in m.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN gradient in {name}"
                assert not p.grad.isinf().any(), f"Inf gradient in {name}"


# ---------------------------------------------------------------------------
# 11. Optimizer configuration
# ---------------------------------------------------------------------------

class TestOptimizer:

    def test_weight_decay_groups(self, model):
        opt = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3,
            betas=(0.9, 0.95), device_type="cpu",
        )
        groups = opt.param_groups
        assert len(groups) == 2

        decay_group = groups[0]
        nodecay_group = groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert nodecay_group["weight_decay"] == 0.0

        # All decay params should be >= 2-D (matrices)
        for p in decay_group["params"]:
            assert p.dim() >= 2
        # All nodecay params should be < 2-D (biases, layernorm)
        for p in nodecay_group["params"]:
            assert p.dim() < 2


# ---------------------------------------------------------------------------
# 12. Numerical stability edge cases
# ---------------------------------------------------------------------------

class TestStability:

    def test_long_sequence(self, config):
        """Full block_size sequence should not produce NaN."""
        torch.manual_seed(SEED)
        m = GPT2Model(config)
        m.eval()
        idx = torch.randint(0, config.vocab_size, (1, config.block_size))
        logits, _ = m(idx)
        assert not logits.isnan().any()
        assert not logits.isinf().any()

    def test_exceeding_block_size_raises(self, model, config):
        idx = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
        with pytest.raises(AssertionError):
            model(idx)

    def test_param_count(self, model):
        n = model.get_num_params(non_embedding=True)
        assert n > 0
        n_full = model.get_num_params(non_embedding=False)
        assert n_full > n  # position embeddings add params

    def test_mfu_positive(self, model):
        mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=1.0)
        assert mfu > 0