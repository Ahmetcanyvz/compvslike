"""Debug a mid-epoch resume hang. Self-instrumenting: if any phase hangs,
faulthandler dumps ALL thread stacks every 30s so we see the exact stuck line.

Reproduces the resume path from src/train.py in isolated phases so we can tell
WHERE it freezes: metadata build, dataloader skip, first batch fetch, or the
actual trainer.fit resume.

Usage:
    uv run python scripts/debug_resume.py <config.yaml> <checkpoint.ckpt> [num_workers]

Example:
    uv run python scripts/debug_resume.py \
        configs/me340M-tied_bpe-8k_seed42.yaml \
        outputs/me340M-tied_bpe-8k_7Btok_seed42/.checkpoints/step15000.ckpt 0
"""

import faulthandler
import signal
import sys
import time
from pathlib import Path

# --- Watchdog: dump every thread's stack every 30s if we're still alive. ---
# A hang anywhere will print the exact frozen frames to stderr on a timer.
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
try:
    faulthandler.register(signal.SIGUSR1)  # manual: kill -USR1 <pid>
except Exception:
    pass

_t0 = time.time()


def log(msg: str) -> None:
    print(f"[debug +{time.time() - _t0:7.1f}s] {msg}", flush=True)


def phase(name: str):
    log(f"==== BEGIN {name} ====")
    return time.time()


def done(name: str, t: float):
    log(f"==== END   {name}  ({time.time() - t:.1f}s) ====")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    config_path = Path(sys.argv[1])
    ckpt_path = Path(sys.argv[2])
    force_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    log(f"config={config_path}")
    log(f"ckpt={ckpt_path}")
    log(f"force num_workers={force_workers}")

    t = phase("imports (torch/lightning/src)")
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from src.data import DataModule
    from src.model import LanguageModel
    from src.train import (
        compute_max_steps,
        load_config,
        load_model_config,
        setup_trainer,
    )
    from transformers import AutoTokenizer
    done("imports", t)

    t = phase("load config + seed")
    config = load_config(config_path)
    seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed, workers=True)
    training_config = config.get("training", {})
    paths_config = config.get("paths", {})
    max_steps = compute_max_steps(training_config)
    training_config["max_steps"] = max_steps
    log(f"max_steps={max_steps}")
    done("load config", t)

    t = phase("tokenizer + model config")
    tok = AutoTokenizer.from_pretrained(paths_config["tokenizer"])
    vocab_size = len(tok)
    eos_token_id = tok.eos_token_id or 0
    mcfg_path = Path(config["model"]["config_path"])
    if not mcfg_path.is_absolute():
        mcfg_path = config_path.parent / mcfg_path
    march = load_model_config(mcfg_path)
    march["vocab_size"] = vocab_size
    march["max_position_embeddings"] = training_config.get("sequence_length", 2048)
    done("tokenizer + model config", t)

    workers = force_workers if force_workers is not None else config.get("hardware", {}).get("num_workers", 4)
    t = phase(f"build DataModule (num_workers={workers})")
    dm = DataModule(
        train_data_path=paths_config.get("train_data"),
        val_data_path=paths_config.get("val_data"),
        test_data_path=paths_config.get("test_data"),
        seq_len=training_config.get("sequence_length", 2048),
        eos_token_id=eos_token_id,
        shuffle_seed=seed,
        batch_size=training_config.get("batch_size", 32),
        eval_batch_size=training_config.get("eval_batch_size"),
        num_workers=workers,
    )
    done("build DataModule", t)

    t = phase("dm.setup('fit')  [metadata build / offsets]")
    dm.setup("fit")
    log(f"train sequences = {len(dm.train_ds):,}")
    done("dm.setup", t)

    t = phase("load ckpt + extract batch_completed")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    batch_completed = (
        ckpt.get("loops", {})
        .get("fit_loop", {})
        .get("epoch_loop.batch_progress", {})
        .get("total", {})
        .get("completed", 0)
    )
    resume_global_step = int(ckpt.get("global_step", 0))
    log(f"global_step={resume_global_step}  batch_completed={batch_completed}")
    dm._skip_batches = batch_completed
    del ckpt
    done("load ckpt", t)

    t = phase("build train_dataloader (applies skip)")
    dl = dm.train_dataloader()
    log(f"dataloader batch_sampler len = {len(dl.batch_sampler)}")
    done("build train_dataloader", t)

    # ---- ISOLATED TEST: does the dataloader itself hang on the skip / first batch? ----
    t = phase("ISOLATED: iterate dataloader, pull first 5 post-skip batches")
    it = iter(dl)
    for i in range(5):
        b0 = time.time()
        batch = next(it)
        shp = tuple(batch["input_ids"].shape)
        log(f"  batch {i}: shape={shp}  fetch={time.time() - b0:.2f}s")
    done("ISOLATED dataloader", t)
    del it, dl

    # ---- build model ----
    t = phase("build LanguageModel")
    optim_config = {
        "learning_rate": training_config.get("learning_rate", 3e-4),
        "weight_decay": training_config.get("weight_decay", 0.1),
        "beta1": training_config.get("beta1", 0.9),
        "beta2": training_config.get("beta2", 0.95),
        "warmup_steps": training_config.get("warmup_steps", 2000),
        "decay_steps": training_config.get("decay_steps", 10000),
        "min_lr_ratio": training_config.get("min_lr_ratio", 0.1),
        "z_loss_weight": training_config.get("z_loss_weight"),
    }
    ms = config.get("model", {})
    model = LanguageModel(
        config=march,
        optim_config=optim_config,
        use_flash_attention=ms.get("use_flash_attention", True),
        use_liger_kernel=ms.get("use_liger_kernel", False),
        torch_compile=ms.get("torch_compile", False),
        max_steps=max_steps,
    )
    done("build LanguageModel", t)

    # ---- REAL REPRODUCTION: trainer.fit with ckpt_path, capped to a few steps PAST resume ----
    # Remove max_tokens so compute_max_steps uses our explicit max_steps (otherwise it
    # recomputes from tokens and can land BELOW the resume step -> fit exits instantly).
    cap = resume_global_step + 5
    config["training"].pop("max_tokens", None)
    config["training"]["max_steps"] = cap
    log(f"set max_steps={cap} (resume_step {resume_global_step} + 5) for the fit reproduction")

    t = phase("setup_trainer")
    output_dir = Path("outputs/_debug_resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = setup_trainer(config, output_dir)
    done("setup_trainer", t)

    t = phase("trainer.fit(ckpt_path=...)  [THE REAL RESUME — watch for hang]")
    log("if this phase stalls, the 30s watchdog will dump the stuck stack below")
    dm2 = DataModule(
        train_data_path=paths_config.get("train_data"),
        val_data_path=paths_config.get("val_data"),
        test_data_path=paths_config.get("test_data"),
        seq_len=training_config.get("sequence_length", 2048),
        eos_token_id=eos_token_id,
        shuffle_seed=seed,
        batch_size=training_config.get("batch_size", 32),
        eval_batch_size=training_config.get("eval_batch_size"),
        num_workers=workers,
    )
    trainer.fit(model=model, datamodule=dm2, ckpt_path=str(ckpt_path))
    done("trainer.fit", t)

    log("ALL PHASES COMPLETED — resume did NOT hang in this run.")


if __name__ == "__main__":
    main()
