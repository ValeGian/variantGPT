# variantGPT

A research framework for training GPT-2 with interchangeable attention mechanisms. The goal is to make it easy to swap in a different attention variant — from standard multi-head to DeepSeek-V2's MLA — and compare them under identical training conditions.

## Attention variants

| Key | Name | Description |
|-----|------|-------------|
| `mha` | Multi-Head Attention | Standard scaled dot-product attention |
| `mqa` | Multi-Query Attention | Single shared K/V head across all query heads |
| `gqa` | Grouped-Query Attention | `n_kv_head` groups of K/V heads; generalises MHA and MQA |
| `local` | Local (Sliding Window) | Causal attention within a fixed `window_size` |
| `linear` | Linear Attention | Kernel-feature-map attention, O(T) compute via chunked parallel form |
| `sparse` | Sparse Attention | BigBird-style: global tokens + sliding window + random connections |
| `mla` | Multi-Head Latent Attention | DeepSeek-V2: low-rank KV compression + decoupled RoPE |

## Setup

Requires Python ≥ 3.11 and PyTorch 2.10.

```bash
pip install -e .
```

## Architecture

```
src/variant_gpt/
├── attention/          # all attention variants + registry
│   ├── config.py       # AttentionConfig — single dataclass covering all variants
│   ├── base.py         # CausalSelfAttention abstract base class
│   ├── mha.py, mqa.py, gqa.py, local.py, linear.py, sparse.py, mla.py
│   └── __init__.py     # _REGISTRY, build_attention(), register_attention()
├── models/gpt2/
│   ├── configuration.py  # GPT2Config (includes all attention params)
│   └── modeling.py       # GPT2Model, GPT2Block, GPT2MLP
└── activations.py      # activation registry (gelu_new, silu, relu, …)

pretrain/
├── config.py   # TrainConfig — all training hyperparameters as a dataclass
├── train.py    # training loop: single-GPU or multi-GPU via DDP
└── data.py     # PretrainDataset (memory-mapped .bin files) + DataLoaders

notebooks/      # step-by-step pipeline: data cleaning → BPE → pretraining → fine-tuning
minbpe/         # Karpathy's RegexTokenizer (vendored)
```

**Attention registry.** Variants are registered with `@register_attention("name")` and instantiated through `build_attention(name, config)`. Adding a new variant means subclassing `CausalSelfAttention`, decorating it, and adding the relevant config fields to `AttentionConfig` — no changes to the model or training code needed.

**Model.** `GPT2Block` constructs an `AttentionConfig` from `GPT2Config` and calls `build_attention` to wire in the chosen variant. The rest of the block is standard pre-norm transformer (LayerNorm → attn → residual → MLP → residual). `GPT2Model` supports `torch.compile`, Flash Attention (`flash=True`), fused AdamW, weight tying, and MFU estimation.

**Training.** `TrainConfig` drives everything. All fields are CLI-overridable. The loop supports gradient accumulation, `bfloat16`/`float16` mixed precision, DDP, rotating checkpoints, early stopping, and MLflow tracking. Data is memory-mapped flat binary (`train.bin` / `val.bin`) of concatenated token IDs.

## Running experiments

```bash
# Single GPU
python pretrain/train.py --attention_type mha

# Multi-GPU
torchrun --standalone --nproc_per_node=8 pretrain/train.py --attention_type gqa --n_kv_head 4

# Resume (auto-detects latest checkpoint)
python pretrain/train.py --run_name my_run

# MLA example (requires explicit dim config)
python pretrain/train.py \
  --attention_type mla \
  --kv_lora_rank 192 \
  --qk_nope_head_dim 64 \
  --qk_rope_head_dim 32 \
  --v_head_dim 64
```

MLflow credentials go in `pretrain/.env`. Training logs per variant (`mha.log`, `mla.log`, …) are written to `pretrain/`.

## Tests

```bash
pytest                                      # all tests
pytest tests/test_gpt2_numeric.py          # model correctness
```
