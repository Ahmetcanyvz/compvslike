"""Train nanochat GPT model using lm-trainer's data pipeline.

Raw PyTorch training loop (no Lightning), matching nanochat's base_train.py exactly.
Only the data loading uses lm-trainer's pre-tokenized datasets.

Usage:
    uv run python -m src.train_nanochat <config.yaml> [--seed 42]
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import time
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add nanochat to path
NANOCHAT_DIR = Path(__file__).parent.parent / "nanochat"
if str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON

from src.data import DataModule

app = typer.Typer()

# Model configs matching nanochat's depth-based sizing
MODEL_CONFIGS = {
    "nc57M":  dict(n_layer=6,  n_embd=768,  n_head=12, n_kv_head=12),
    "nc100M": dict(n_layer=12, n_embd=768,  n_head=12, n_kv_head=12),
    "nc340M": dict(n_layer=20, n_embd=1280, n_head=20, n_kv_head=4),
    "nc500M": dict(n_layer=26, n_embd=1280, n_head=20, n_kv_head=4),
    "nc1B":   dict(n_layer=22, n_embd=2048, n_head=32, n_kv_head=4),
}


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to training config YAML"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override seed"),
) -> None:
    """Train a nanochat GPT model with lm-trainer data."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    tc = config.get("training", {})
    mc = config.get("model", {})
    hc = config.get("hardware", {})
    cc = config.get("checkpoint", {})

    # Seed
    if seed is None:
        seed = tc.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device} | COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    # Tokenizer
    tokenizer_path = paths.get("tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id or 0
    print(f"Tokenizer: {tokenizer_path} | vocab_size: {vocab_size:,}")

    # Model config
    nc_model = mc.get("nanochat_model", "nc100M")
    dims = MODEL_CONFIGS[nc_model]
    seq_len = tc.get("sequence_length", 2048)
    window_pattern = mc.get("window_pattern", "SSSL")

    gpt_config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=dims["n_layer"],
        n_head=dims["n_head"],
        n_kv_head=dims["n_kv_head"],
        n_embd=dims["n_embd"],
        window_pattern=window_pattern,
    )

    # Build model on meta device, then materialize and init weights (nanochat pattern)
    with torch.device("meta"):
        model = GPT(gpt_config)
    model.to_empty(device=device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {nc_model} | {num_params / 1e6:.1f}M params | layers={dims['n_layer']} dim={dims['n_embd']}")

    # Compile model
    use_compile = mc.get("torch_compile", True)
    orig_model = model
    if use_compile:
        try:
            model = torch.compile(model, dynamic=False)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without")
            model = orig_model

    # Training params
    device_batch_size = tc.get("batch_size", 32)
    grad_accum = tc.get("gradient_accumulation", 1)
    total_batch_size = device_batch_size * grad_accum * seq_len
    print(f"Batch: {device_batch_size} x {grad_accum} x {seq_len} = {total_batch_size:,} tokens/step")

    # Compute num_iterations
    max_tokens = tc.get("max_tokens")
    max_steps = tc.get("max_steps")
    if max_tokens is not None:
        num_iterations = max_tokens // total_batch_size
    elif max_steps is not None:
        num_iterations = max_steps
    else:
        num_iterations = 50000
    print(f"Training for {num_iterations:,} steps ({num_iterations * total_batch_size / 1e9:.2f}B tokens)")

    # Output dir
    base_output = Path(paths.get("output_dir", "./outputs"))
    tok_name = Path(tokenizer_path).name
    if max_tokens and max_tokens >= 1_000_000_000:
        run_name = f"{nc_model}_{tok_name}_{max_tokens // 1_000_000_000}Btok_seed{seed}"
    else:
        run_name = f"{nc_model}_{tok_name}_{num_iterations}steps_seed{seed}"
    output_dir = base_output / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / cc.get("save_dir", ".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(asdict(gpt_config), f, indent=2)

    # Optimizer (nanochat's MuonAdamW)
    optim_cfg = tc.get("optimizer", {})
    B_REF = 2**19
    batch_lr_scale = (total_batch_size / B_REF) ** 0.5
    weight_decay = optim_cfg.get("weight_decay", 0.28)
    weight_decay_scaled = weight_decay * math.sqrt(total_batch_size / B_REF)

    optimizer = orig_model.setup_optimizer(
        unembedding_lr=optim_cfg.get("unembedding_lr", 0.008) * batch_lr_scale,
        embedding_lr=optim_cfg.get("embedding_lr", 0.3) * batch_lr_scale,
        scalar_lr=optim_cfg.get("scalar_lr", 0.5) * batch_lr_scale,
        matrix_lr=optim_cfg.get("matrix_lr", 0.02) * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )

    # Schedules (exactly as nanochat)
    warmup_steps = optim_cfg.get("warmup_steps", 40)
    warmdown_ratio = optim_cfg.get("warmdown_ratio", 0.65)
    final_lr_frac = optim_cfg.get("final_lr_frac", 0.05)

    def get_lr_multiplier(it):
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_steps:
            return (it + 1) / warmup_steps
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    def get_muon_momentum(it):
        warmdown_iters = round(warmdown_ratio * num_iterations)
        warmdown_start = num_iterations - warmdown_iters
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def get_weight_decay(it):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

    # Data loader (lm-trainer's PackedTokenDataset)
    data_module = DataModule(
        train_data_path=paths.get("train_data"),
        val_data_path=paths.get("val_data"),
        test_data_path=paths.get("test_data"),
        seq_len=seq_len,
        eos_token_id=eos_token_id,
        shuffle_seed=seed,
        batch_size=device_batch_size,
        num_workers=hc.get("num_workers", 4),
    )
    data_module.setup("fit")
    train_loader = iter(data_module.train_dataloader())

    # Logging
    save_every = cc.get("save_every_n_steps", 5000)
    log_every = tc.get("log_loss_every_n_steps", 100)
    log_file = open(output_dir / "training_log.txt", "w")
    log_file.write(f"Training {nc_model} | {num_params/1e6:.1f}M params | {num_iterations:,} steps\n")

    # =========================================================================
    # Training loop (matches nanochat's base_train.py)
    # =========================================================================
    model.train()
    smooth_train_loss = 0.0
    total_training_time = 0.0

    print(f"\nStarting training...")
    for step in range(num_iterations):
        torch.cuda.synchronize()
        t0 = time.time()

        # Gradient accumulation
        for micro_step in range(grad_accum):
            try:
                batch = next(train_loader)
            except StopIteration:
                train_loader = iter(data_module.train_dataloader())
                batch = next(train_loader)

            input_ids = batch["input_ids"].to(device)
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]
            loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum
            loss.backward()

        # Update optimizer with schedules
        lrm = get_lr_multiplier(step)
        muon_momentum = get_muon_momentum(step)
        muon_wd = get_weight_decay(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

        if step > 10:
            total_training_time += dt

        if step % log_every == 0:
            tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
            pct = 100 * step / num_iterations
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

            # ETA
            if step > 10:
                avg_dt = total_training_time / (step - 10)
                eta_min = (num_iterations - step) * avg_dt / 60
                eta_str = f" | eta: {eta_min:.1f}m"
            else:
                eta_str = ""

            msg = f"step {step:05d}/{num_iterations:05d} ({pct:.1f}%) | loss: {debiased_loss:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | mem: {peak_mem:.0f}MiB{eta_str}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        # Checkpointing
        if save_every > 0 and step > 0 and step % save_every == 0:
            ckpt_path = checkpoint_dir / f"step{step}.pt"
            torch.save(orig_model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # GC management (from nanochat)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    # Save final checkpoint
    if cc.get("save_last", True):
        ckpt_path = checkpoint_dir / "last.pt"
        torch.save(orig_model.state_dict(), ckpt_path)
        print(f"Saved final checkpoint: {ckpt_path}")

    # Save model config for loading later
    with open(checkpoint_dir / "model_config.json", "w") as f:
        json.dump(asdict(gpt_config), f, indent=2)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"\nTraining complete!")
    print(f"Peak memory: {peak_mem:.0f} MiB")
    print(f"Total training time: {total_training_time/60:.1f}m")
    print(f"Final loss: {debiased_loss:.4f}")
    print(f"Output: {output_dir}")

    log_file.close()


if __name__ == "__main__":
    app()
