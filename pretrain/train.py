#!/usr/bin/env python3
"""
GPT-2 pretraining script.

Single GPU:
    python train.py

Multi-GPU (8× A40 on one node):
    torchrun --standalone --nproc_per_node=8 train.py

Resume from checkpoint:
    python train.py                          # auto-detects latest checkpoint
    python train.py --resume_from path.pth   # explicit path

Override any config field from the CLI:
    torchrun --standalone --nproc_per_node=8 train.py \
        --micro_batch_size 32 --grad_accum_steps 2 --run_name run_2
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ── Project imports ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / ".."))

from config import TrainConfig  # noqa: E402
from data import create_dataloaders  # noqa: E402
from minbpe import RegexTokenizer  # noqa: E402
from variant_gpt.models import GPT2Config, GPT2Model  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_vocab_size(tokenizer: RegexTokenizer) -> int:
    return len(tokenizer.vocab) + len(tokenizer.special_tokens)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log(msg: str) -> None:
    """Print only on rank 0."""
    if is_main_process():
        print(msg, flush=True)


def fmt_duration(seconds: float) -> str:
    """Format seconds into a compact human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


def setup_distributed() -> tuple[int, int, int, str]:
    """
    Initialise DDP if launched via torchrun, else fall back to single-GPU.
    Returns (rank, local_rank, world_size, device).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        return rank, local_rank, world_size, device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 0, 1, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════════
#  MLflow helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_mlflow(cfg: TrainConfig, resume_run_id: str | None = None) -> str:
    """
    Start (or resume) an MLflow run and log all hyperparameters.
    Returns the active run_id so it can be persisted in checkpoints.
    """
    # Experiment is set via MLFLOW_EXPERIMENT_NAME env var;
    # URI + credentials via MLFLOW_TRACKING_URI / USERNAME / PASSWORD.
    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id)
        log(f"MLflow: resumed run {resume_run_id}")
    else:
        mlflow.start_run(run_name=cfg.run_name)
        log(f"MLflow: started new run {mlflow.active_run().info.run_id}")

    # Log every config field as a param (safe on resume — skip if already set)
    import dataclasses
    params = {f.name: getattr(cfg, f.name)
              for f in dataclasses.fields(cfg)
              if f.name != "run_dir"}
    try:
        mlflow.log_params(params)
    except mlflow.exceptions.MlflowException:
        # Params already logged from a previous run segment — that's fine
        pass

    return mlflow.active_run().info.run_id


def log_metrics_mlflow(metrics: dict[str, float], step: int) -> None:
    """Convenience wrapper — only calls MLflow on rank 0."""
    if is_main_process():
        mlflow.log_metrics(metrics, step=step)


def end_mlflow() -> None:
    if is_main_process() and mlflow.active_run():
        mlflow.end_run()


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint management
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
    train_loss: float,
    mlflow_run_id: str | None = None,
    tag: str = "step",
) -> Path:
    """
    Save a checkpoint and rotate old ones (keep last N).
    Only called on rank 0.
    """
    raw_model = model.module if isinstance(model, DDP) else model

    state = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "train_loss": train_loss,
        "mlflow_run_id": mlflow_run_id,
        "config": {k: v for k, v in vars(cfg).items() if not isinstance(v, Path)},
    }

    ckpt_path = cfg.run_dir / f"checkpoint_{tag}_{step}.pth"
    torch.save(state, ckpt_path)

    # Symlink "latest" for easy resume
    latest_link = cfg.run_dir / "checkpoint_latest.pth"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(ckpt_path.name)

    # Rotate: keep only the last N step-checkpoints (don't touch "best")
    step_ckpts = sorted(
        cfg.run_dir.glob("checkpoint_step_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(step_ckpts) > cfg.keep_last_n:
        old = step_ckpts.pop(0)
        old.unlink(missing_ok=True)

    return ckpt_path


def save_best(cfg: TrainConfig, model: torch.nn.Module, step: int, val_loss: float) -> Path:
    raw_model = model.module if isinstance(model, DDP) else model
    path = cfg.run_dir / "checkpoint_best.pth"
    torch.save({
        "step": step,
        "val_loss": val_loss,
        "model_state_dict": raw_model.state_dict(),
    }, path)
    log(f"  ★ New best model saved (val_loss={val_loss:.4f})")
    return path


def find_latest_checkpoint(cfg: TrainConfig) -> Path | None:
    """Auto-detect the most recent checkpoint in the run directory."""
    latest = cfg.run_dir / "checkpoint_latest.pth"
    if latest.exists():
        return latest.resolve()
    return None


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load checkpoint and restore model + optimizer state."""
    log(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    cfg: TrainConfig,
    device: str,
    amp_ctx,
) -> float:
    """Run a fixed number of val micro-batches and return mean loss."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.val_steps:
            break
        x = x.to(device, non_blocking=True).long()
        y = y.to(device, non_blocking=True).long()
        with amp_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()

    avg_loss = sum(losses) / len(losses) if losses else float("inf")

    # Average across ranks for consistent early-stopping decisions
    if dist.is_initialized():
        t = torch.tensor([avg_loss], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg_loss = t.item()

    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: TrainConfig) -> None:
    # ── Distributed setup ─────────────────────────────────────────────────
    rank, local_rank, world_size, device = setup_distributed()
    distributed = world_size > 1
    master = rank == 0

    log("=" * 70)
    log(f"  GPT-2 Pretraining  |  {world_size} GPU(s)  |  run: {cfg.run_name}")
    log("=" * 70)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = RegexTokenizer()
    tokenizer.load(cfg.tokenizer_path)
    vocab_size = get_vocab_size(tokenizer)
    del tokenizer
    log(f"Vocab size: {vocab_size:,}")

    # Determine numpy dtype (must match what the tokeniser pipeline wrote)
    max_token = vocab_size
    np_dtype = np.uint16 if max_token < (1 << 16) else np.uint32

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        data_dir=cfg.data_dir,
        block_size=cfg.block_size,
        micro_batch_size=cfg.micro_batch_size,
        num_workers=4,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        dtype=np_dtype,
    )
    log(f"Train samples: {len(train_loader.dataset):,}  "
        f"Val samples: {len(val_loader.dataset):,}")

    tokens_per_step = (
        cfg.micro_batch_size * cfg.block_size * cfg.grad_accum_steps * world_size
    )
    log(f"Tokens per optimiser step: {tokens_per_step:,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = GPT2Config(
        vocab_size=vocab_size,
        dropout=cfg.dropout,
        device=device,
        flash=cfg.flash,
    )
    model = GPT2Model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}  ({n_params/1e6:.1f}M)")

    # Compile for speed (PyTorch ≥ 2.0)
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile …")
        model = torch.compile(model)

    # ── Optimiser ─────────────────────────────────────────────────────────
    device_type = "cuda" if "cuda" in device else "cpu"
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type,
    )

    # ── Mixed precision context ───────────────────────────────────────────
    if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
        pt_dtype = torch.bfloat16
    elif cfg.dtype == "float16":
        pt_dtype = torch.float16
    else:
        pt_dtype = torch.float32

    amp_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=pt_dtype)
        if device_type == "cuda"
        else nullcontext()
    )
    # GradScaler is only needed for float16 (bfloat16 doesn't need it)
    scaler = (
        torch.amp.GradScaler(device_type)
        if pt_dtype == torch.float16
        else None
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    epoch = 0
    resume_run_id = None

    ckpt_path = Path(cfg.resume_from) if cfg.resume_from else find_latest_checkpoint(cfg)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        epoch = ckpt.get("epoch", 0)
        resume_run_id = ckpt.get("mlflow_run_id", None)
        log(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")
    else:
        log("Starting from scratch (no checkpoint found)")

    # ── MLflow (rank 0 only) ──────────────────────────────────────────────
    mlflow_run_id: str | None = None
    if master:
        mlflow_run_id = setup_mlflow(cfg, resume_run_id=resume_run_id)
        mlflow.set_tags({
            "model_params": f"{n_params:,}",
            "world_size": world_size,
            "device_type": device_type,
            "dtype": cfg.dtype,
            "tokens_per_step": tokens_per_step,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        })

    # ── DDP wrapper (after loading weights so all ranks match) ────────────
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # ── Training state ────────────────────────────────────────────────────
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t_start = time.perf_counter()  # wall clock for elapsed / ETA
    t0 = time.perf_counter()       # interval timer for tok/s

    log(f"\nStarting training from step {start_step} → {cfg.max_steps}")
    log(f"LR schedule: {cfg.lr_schedule}  warmup={cfg.warmup_steps}  "
        f"peak={cfg.learning_rate}  min={cfg.min_lr}")
    log("")

    for step in range(start_step, cfg.max_steps):
        # ── Set LR for this step ──────────────────────────────────────────
        lr = cfg.get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation loop ────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(cfg.grad_accum_steps):
            # Fetch next batch; wrap around epochs
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                if distributed:
                    train_loader.sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()

            # In DDP, only sync gradients on the last micro-step
            sync_ctx = (
                model.no_sync()
                if distributed and micro_step < cfg.grad_accum_steps - 1
                else nullcontext()
            )

            with sync_ctx:
                with amp_ctx:
                    _, loss = model(x, y)
                    # Scale loss by accum steps so the effective loss is
                    # the mean over all micro-batches
                    scaled_loss = loss / cfg.grad_accum_steps

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            accum_loss += loss.item()

        # ── Gradient clipping ─────────────────────────────────────────────
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # ── Optimiser step ────────────────────────────────────────────────
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        train_loss = accum_loss / cfg.grad_accum_steps
        running_loss += train_loss

        if (step + 1) % cfg.log_interval == 0 and master:
            dt = time.perf_counter() - t0
            avg_loss = running_loss / cfg.log_interval
            tok_per_sec = tokens_per_step * cfg.log_interval / dt
            tokens_seen = (step + 1) * tokens_per_step
            ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow

            # Elapsed and ETA based on steps done in this session
            elapsed = time.perf_counter() - t_start
            steps_done_session = step + 1 - start_step
            steps_remaining = cfg.max_steps - (step + 1)
            if steps_done_session > 0:
                sec_per_step = elapsed / steps_done_session
                eta = sec_per_step * steps_remaining
            else:
                eta = 0.0

            print(
                f"step {step+1:>7d}/{cfg.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"ppl {ppl:.1f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.2f} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"elapsed {fmt_duration(elapsed)} | "
                f"eta {fmt_duration(eta)}"
            )
            log_metrics_mlflow({
                "train/loss": avg_loss,
                "train/perplexity": ppl,
                "train/learning_rate": lr,
                "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen": tokens_seen,
                "train/epoch": epoch,
                "train/elapsed_hours": elapsed / 3600,
                "train/eta_hours": eta / 3600,
            }, step=step + 1)
            running_loss = 0.0
            t0 = time.perf_counter()

        # ── Validation + early stopping ───────────────────────────────────
        if (step + 1) % cfg.val_interval == 0:
            val_loss = validate(model, val_loader, cfg, device, amp_ctx)
            val_ppl = math.exp(min(val_loss, 20))
            log(f"  → val_loss = {val_loss:.4f}  val_ppl = {val_ppl:.1f}  "
                f"(best = {best_val_loss:.4f}, "
                f"patience = {patience_counter}/{cfg.patience})")

            log_metrics_mlflow({
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                "val/best_loss": best_val_loss,
                "val/patience_counter": patience_counter,
            }, step=step + 1)

            improved = val_loss < best_val_loss - cfg.min_delta
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                if master:
                    save_best(cfg, model, step + 1, val_loss)
                    log_metrics_mlflow({"val/best_loss": best_val_loss}, step=step + 1)
            else:
                patience_counter += 1

            # Early stopping (all ranks must agree, counter is synchronised)
            if patience_counter >= cfg.patience:
                log(f"Early stopping triggered at step {step+1}.")
                if master:
                    save_checkpoint(
                        cfg, model, optimizer, step + 1, epoch,
                        best_val_loss, patience_counter, train_loss,
                        mlflow_run_id=mlflow_run_id,
                    )
                    mlflow.set_tag("stop_reason", "early_stopping")
                    mlflow.log_metric("final/step", step + 1)
                break

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (step + 1) % cfg.ckpt_interval == 0 and master:
            save_checkpoint(
                cfg, model, optimizer, step + 1, epoch,
                best_val_loss, patience_counter, train_loss,
                mlflow_run_id=mlflow_run_id,
            )
            log(f"  Checkpoint saved at step {step+1}")

    # ── Final save ────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    if master:
        save_checkpoint(
            cfg, model, optimizer, step + 1, epoch,
            best_val_loss, patience_counter, train_loss,
            mlflow_run_id=mlflow_run_id, tag="final",
        )
        log(f"\nTraining complete in {fmt_duration(total_elapsed)}. "
            f"Best val loss: {best_val_loss:.4f}")
        log(f"Outputs in: {cfg.run_dir}")

        # ── Final MLflow logging ──────────────────────────────────────────
        mlflow.log_metrics({
            "final/best_val_loss": best_val_loss,
            "final/best_val_perplexity": math.exp(min(best_val_loss, 20)),
            "final/total_steps": step + 1,
            "final/total_tokens_seen": (step + 1) * tokens_per_step,
            "final/elapsed_hours": total_elapsed / 3600,
        }, step=step + 1)
        if not mlflow.active_run().info.tags.get("stop_reason"):
            mlflow.set_tag("stop_reason", "max_steps")

    end_mlflow()
    cleanup_distributed()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> TrainConfig:
    """
    Create a TrainConfig from CLI overrides.
    Any TrainConfig field can be set via --field_name value.
    """
    import dataclasses

    # Map stringified type annotations → argparse-compatible types
    _TYPE_MAP = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "str | None": str,
    }

    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="GPT-2 Pretraining")

    for f in dataclasses.fields(cfg):
        if f.name == "run_dir":
            continue  # derived field
        annotation = f.type if isinstance(f.type, str) else f.type.__name__
        arg_type = _TYPE_MAP.get(annotation, str)  # fall back to str
        default = getattr(cfg, f.name)
        parser.add_argument(f"--{f.name}", type=arg_type, default=default)

    args = parser.parse_args()
    return TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cfg) if f.name != "run_dir"})


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
