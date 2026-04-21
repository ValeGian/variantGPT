"""
Training configuration for GPT-2 pretraining.

All hyperparameters are collected here so the training script stays clean
and experiments are easy to reproduce by swapping config values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TrainConfig:
    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: str = "../output/encoded_data"
    tokenizer_path: str = "../output/tokenizer/fineweb_tokenizer.model"
    output_dir: str = "../output/pre_training"
    run_name: str = "run_1"

    # ── Model ─────────────────────────────────────────────────────────────
    dropout: float = 0.0        # typically 0 during pretraining, >0 for fine-tuning
    flash: bool = True
    block_size: int = 1024      # context / sequence length

    attention_type: str = "mha"
    n_kv_head: int | None = None  # only read by GQA-family variants; None => MHA, 1 => MQA, 1<k<n_head => GQA

    # ── Optimiser ─────────────────────────────────────────────────────────
    learning_rate: float = 6e-4          # peak LR  (GPT-2 124M sweet spot)
    min_lr: float = 6e-5                 # 10% of peak is a common floor
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ── LR schedule ───────────────────────────────────────────────────────
    #  "cosine"  – warmup → cosine decay to min_lr   (GPT-3 / Chinchilla)
    #  "linear"  – warmup → linear decay to min_lr   (simpler baseline)
    #  "rsqrt"   – warmup → 1/√step decay            (PaLM-style)
    lr_schedule: Literal["cosine", "linear", "rsqrt"] = "cosine"
    warmup_fraction: float = 0.05    # fraction of total_steps used for LR warmup

    # ── Batch / throughput ────────────────────────────────────────────────
    micro_batch_size: int = 52       # per-GPU batch size
    grad_accum_steps: int = 3        # effective_batch = micro * accum * n_gpus
    #  e.g. 52 * 3 * 8 = 1248 sequences → 1.28 M tokens / step

    # ── Training budget ───────────────────────────────────────────────────
    num_epochs: int = 10             # number of passes over the training set
    val_interval: int = 100          # validate every N steps
    val_steps: int = 50              # micro-batches per validation
    log_interval: int = 10           # print loss every N steps

    # ── Checkpointing ────────────────────────────────────────────────────
    ckpt_interval: int = 1_000       # save every N steps
    keep_last_n: int = 2             # rotating checkpoint window

    # ── Early stopping ────────────────────────────────────────────────────
    patience: int = 7                # N validations without improvement → stop
    min_delta: float = 0.001         # minimum val-loss decrease to count

    # ── Mixed precision ──────────────────────────────────────────────────
    dtype: str = "bfloat16"          # "bfloat16" | "float16" | "float32"

    # ── Resume ────────────────────────────────────────────────────────────
    resume_from: str | None = None   # path to checkpoint_latest.pth (auto-detected)

    # ── Derived (set in __post_init__ or at runtime) ──────────────────────
    run_dir: Path = field(init=False)
    total_steps: int = field(init=False, default=0)    # set by set_total_steps()
    warmup_steps: int = field(init=False, default=0)   # set by set_total_steps()

    def __post_init__(self) -> None:
        self.run_dir = Path(self.output_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def set_total_steps(self, steps_per_epoch: int) -> None:
        """Call once the dataset size is known to finalise the schedule."""
        self.total_steps = steps_per_epoch * self.num_epochs
        self.warmup_steps = int(self.total_steps * self.warmup_fraction)

    # ── LR computation ────────────────────────────────────────────────────
    def get_lr(self, step: int) -> float:
        """Return the learning rate for a given optimiser step."""
        # 1) linear warmup
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps

        # 2) after total_steps, hold at min_lr
        if step >= self.total_steps:
            return self.min_lr

        # progress in [0, 1] over the decay phase
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)

        if self.lr_schedule == "cosine":
            # half-cosine anneal (GPT-3, Chinchilla, LLaMA)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.learning_rate - self.min_lr) * coeff

        elif self.lr_schedule == "linear":
            return self.learning_rate - (self.learning_rate - self.min_lr) * progress

        elif self.lr_schedule == "rsqrt":
            # 1/√t style used by PaLM; normalised so it starts at peak LR
            t = step - self.warmup_steps + 1
            decay = 1.0 / math.sqrt(t)
            return max(self.min_lr, self.learning_rate * decay)

        raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
