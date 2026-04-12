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
    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id)
        log(f"MLflow: resumed run {resume_run_id}")
    else:
        mlflow.start_run(run_name=cfg.run_name)
        log(f"MLflow: started new run {mlflow.active_run().info.run_id}")

    import dataclasses
    params = {f.name: getattr(cfg, f.name)
              for f in dataclasses.fields(cfg)
              if f.name not in ("run_dir", "total_steps", "warmup_steps")}
    try:
        mlflow.log_params(params)
    except mlflow.exceptions.MlflowException:
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

    tokens_per_step = cfg.micro_batch_size * cfg.block_size * cfg.grad_accum_steps * world_size

    # ── Compute total steps from dataset size and num_epochs ──────────────
    # Each optimiser step consumes grad_accum_steps micro-batches.
    # In DDP each rank sees len(train_loader) micro-batches per epoch.
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    cfg.set_total_steps(steps_per_epoch)

    log(f"Tokens per optimiser step: {tokens_per_step:,}")
    log(f"Steps per epoch: {steps_per_epoch:,}  |  "
        f"Total steps ({cfg.num_epochs} epochs): {cfg.total_steps:,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = GPT2Config(
        vocab_size=vocab_size,
        dropout=cfg.dropout,
        device=device,
        flash=cfg.flash,
    )
    model = GPT2Model(model_cfg)
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
    scaler = (
        torch.amp.GradScaler(device_type)
        if pt_dtype == torch.float16
        else None
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    resume_run_id = None

    ckpt_path = Path(cfg.resume_from) if cfg.resume_from else find_latest_checkpoint(cfg)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
        start_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)
        resume_run_id = ckpt.get("mlflow_run_id", None)
        log(f"Resumed from step {start_step} (epoch {start_epoch}), "
            f"best_val_loss={best_val_loss:.4f}")
    else:
        log("Starting from scratch (no checkpoint found)")

    # ── MLflow (rank 0 only) ──────────────────────────────────────────────
    mlflow_run_id: str | None = None
    if master:
        mlflow_run_id = setup_mlflow(cfg, resume_run_id=resume_run_id)
        mlflow.log_param("total_steps", cfg.total_steps)
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
    running_loss = 0.0
    train_loss = 0.0
    t_start = time.perf_counter()  # wall clock for elapsed / ETA
    t0 = time.perf_counter()       # interval timer for tok/s
    step = start_step
    early_stopped = False

    log(f"\nStarting training from epoch {start_epoch} (step {start_step}) "
        f"→ {cfg.num_epochs} epochs ({cfg.total_steps} steps)")
    log(f"LR schedule: {cfg.lr_schedule}  "
        f"warmup={cfg.warmup_fraction:.1%} ({cfg.warmup_steps} steps)  "
        f"peak={cfg.learning_rate}  min={cfg.min_lr}")
    log("")

    for epoch in range(start_epoch, cfg.num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        micro_iter = iter(train_loader)
        micro_consumed = 0  # micro-batches consumed this epoch

        # If resuming mid-epoch, skip already-seen micro-batches
        micro_to_skip = (start_step - epoch * steps_per_epoch) * cfg.grad_accum_steps
        if epoch == start_epoch and micro_to_skip > 0:
            for _ in range(min(micro_to_skip, len(train_loader))):
                try:
                    next(micro_iter)
                    micro_consumed += 1
                except StopIteration:
                    break

        log(f"── Epoch {epoch + 1}/{cfg.num_epochs} ──")

        while True:
            # ── Collect grad_accum_steps micro-batches ────────────────────
            batches = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(micro_iter)
                    micro_consumed += 1
                    batches.append(batch)
                except StopIteration:
                    break  # epoch exhausted

            if not batches:
                break  # move to next epoch

            # If the epoch ended mid-accumulation we still train on what
            # we got, but scale the loss accordingly.
            actual_accum = len(batches)

            # ── Set LR for this step ──────────────────────────────────────
            lr = cfg.get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Gradient accumulation loop ────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step, (x, y) in enumerate(batches):
                x = x.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()

                sync_ctx = (
                    model.no_sync()
                    if distributed and micro_step < actual_accum - 1
                    else nullcontext()
                )

                with sync_ctx:
                    with amp_ctx:
                        _, loss = model(x, y)
                        scaled_loss = loss / actual_accum

                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                accum_loss += loss.item()

            # ── Gradient clipping ─────────────────────────────────────────
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,)

            # ── Optimiser step ────────────────────────────────────────────
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            step += 1

            # ── Logging ───────────────────────────────────────────────────
            train_loss = accum_loss / actual_accum
            running_loss += train_loss

            if step % cfg.log_interval == 0 and master:
                dt = time.perf_counter() - t0
                avg_loss = running_loss / cfg.log_interval
                tok_per_sec = tokens_per_step * cfg.log_interval / dt
                tokens_seen = step * tokens_per_step
                ppl = math.exp(min(avg_loss, 20))

                elapsed = time.perf_counter() - t_start
                steps_done_session = step - start_step
                steps_remaining = cfg.total_steps - step
                if steps_done_session > 0:
                    sec_per_step = elapsed / steps_done_session
                    eta = sec_per_step * steps_remaining
                else:
                    eta = 0.0

                print(
                    f"step {step:>7d}/{cfg.total_steps} | "
                    f"epoch {epoch+1}/{cfg.num_epochs} | "
                    f"loss {avg_loss:.4f} | "
                    f"ppl {ppl:.1f} | "
                    f"lr {lr:.6e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"{tok_per_sec:,.0f} tok/s | "
                    f"elapsed {fmt_duration(elapsed)} | "
                    f"eta {fmt_duration(eta)}"
                )
                log_metrics_mlflow({
                    "train/loss": avg_loss,
                    "train/perplexity": ppl,
                    "train/learning_rate": lr,
                    "train/grad_norm": (grad_norm.item()
                                        if torch.is_tensor(grad_norm)
                                        else grad_norm),
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_seen": tokens_seen,
                    "train/epoch": epoch,
                    "train/elapsed_hours": elapsed / 3600,
                    "train/eta_hours": eta / 3600,
                }, step=step)
                running_loss = 0.0
                t0 = time.perf_counter()

            # ── Validation + early stopping ───────────────────────────────
            if step % cfg.val_interval == 0:
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
                }, step=step)

                improved = val_loss < best_val_loss - cfg.min_delta
                if improved:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if master:
                        save_best(cfg, model, step, val_loss)
                        log_metrics_mlflow(
                            {"val/best_loss": best_val_loss}, step=step,
                        )
                else:
                    patience_counter += 1

                if patience_counter >= cfg.patience:
                    log(f"Early stopping triggered at step {step}.")
                    if master:
                        save_checkpoint(
                            cfg, model, optimizer, step, epoch,
                            best_val_loss, patience_counter, train_loss,
                            mlflow_run_id=mlflow_run_id,
                        )
                        mlflow.set_tag("stop_reason", "early_stopping")
                        mlflow.log_metric("final/step", step)
                    early_stopped = True
                    break

            # ── Periodic checkpoint ───────────────────────────────────────
            if step % cfg.ckpt_interval == 0 and master:
                save_checkpoint(
                    cfg, model, optimizer, step, epoch,
                    best_val_loss, patience_counter, train_loss,
                    mlflow_run_id=mlflow_run_id,
                )
                log(f"  Checkpoint saved at step {step}")

        if early_stopped:
            break

        # ── End-of-epoch validation (skip if already done at this step) ───
        if step > 0 and step % cfg.val_interval != 0:
            val_loss = validate(model, val_loader, cfg, device, amp_ctx)
            val_ppl = math.exp(min(val_loss, 20))
            log(f"  [end-of-epoch {epoch+1}] val_loss = {val_loss:.4f}  "
                f"val_ppl = {val_ppl:.1f}  "
                f"(best = {best_val_loss:.4f}, "
                f"patience = {patience_counter}/{cfg.patience})")

            log_metrics_mlflow({
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                "val/best_loss": best_val_loss,
                "val/patience_counter": patience_counter,
            }, step=step)

            improved = val_loss < best_val_loss - cfg.min_delta
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                if master:
                    save_best(cfg, model, step, val_loss)
                    log_metrics_mlflow(
                        {"val/best_loss": best_val_loss}, step=step,
                    )
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                log(f"Early stopping triggered at end of epoch {epoch+1}.")
                if master:
                    save_checkpoint(
                        cfg, model, optimizer, step, epoch,
                        best_val_loss, patience_counter, train_loss,
                        mlflow_run_id=mlflow_run_id,
                    )
                    mlflow.set_tag("stop_reason", "early_stopping")
                    mlflow.log_metric("final/step", step)
                early_stopped = True
                break

    # ── Final save ────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    if master:
        save_checkpoint(
            cfg, model, optimizer, step, epoch,
            best_val_loss, patience_counter, train_loss,
            mlflow_run_id=mlflow_run_id, tag="final",
        )
        log(f"\nTraining complete in {fmt_duration(total_elapsed)}. "
            f"Best val loss: {best_val_loss:.4f}")
        log(f"Outputs in: {cfg.run_dir}")

        mlflow.log_metrics({
            "final/best_val_loss": best_val_loss,
            "final/best_val_perplexity": math.exp(min(best_val_loss, 20)),
            "final/total_steps": step,
            "final/total_epochs": epoch + 1,
            "final/total_tokens_seen": step * tokens_per_step,
            "final/elapsed_hours": total_elapsed / 3600,
        }, step=step)
        if not mlflow.active_run().info.tags.get("stop_reason"):
            mlflow.set_tag("stop_reason", "completed_epochs")

    end_mlflow()
    cleanup_distributed()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> TrainConfig:
    """
    Create a TrainConfig from CLI overrides.
    Any TrainConfig field can be set via --field_name value.

    Reads defaults directly from the dataclass field definitions so no
    throwaway TrainConfig is instantiated (which would trigger
    __post_init__ and create a spurious run directory).
    """
    import dataclasses

    _TYPE_MAP = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "str | None": str,
    }

    parser = argparse.ArgumentParser(description="GPT-2 Pretraining")

    skip = {"run_dir", "total_steps", "warmup_steps"}  # derived fields
    for f in dataclasses.fields(TrainConfig):
        if f.name in skip:
            continue
        annotation = f.type if isinstance(f.type, str) else f.type.__name__
        arg_type = _TYPE_MAP.get(annotation, str)
        default = f.default if f.default is not dataclasses.MISSING else None
        parser.add_argument(f"--{f.name}", type=arg_type, default=default)

    args = parser.parse_args()
    return TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig) if f.name not in skip})


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
