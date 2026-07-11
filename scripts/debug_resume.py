"""Replicate the EXACT run_train_satay.sh launch, with a hang-detector.

run_train_satay.sh runs, per model:
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.train train <config> --seed 42 --resume <ckpt>

This wrapper calls that identical entry point (same code path, same config,
same --resume), but first installs a faulthandler watchdog that dumps EVERY
thread's stack every 30s. So if the resume hangs, the exact frozen lines get
printed on a timer — no py-spy needed.

Run it with the same single GPU the real script uses:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/debug_resume.py <config> <ckpt> [seed]
"""

import faulthandler
import signal
import sys
import time

# Watchdog: dump all thread stacks every 30s. A hang prints its exact frames.
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
try:
    faulthandler.register(signal.SIGUSR1)  # manual dump: kill -USR1 <pid>
except Exception:
    pass


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    config = sys.argv[1]
    ckpt = sys.argv[2]
    seed = sys.argv[3] if len(sys.argv) > 3 else "42"

    print(f"[debug +{0.0:.1f}s] replicating: python -m src.train train {config} --seed {seed} --resume {ckpt}", flush=True)
    print("[debug] faulthandler watchdog active: dumps ALL thread stacks every 30s if it hangs", flush=True)
    print("[debug] when it stalls, the stack dump that appears IS the hang location", flush=True)

    # Identical code path to run_train_satay.sh's per-model launch.
    from src.train import app

    sys.argv = ["src.train", "train", config, "--seed", str(seed), "--resume", ckpt]
    _t0 = time.time()
    try:
        app()
    finally:
        print(f"[debug +{time.time() - _t0:.1f}s] app() returned/exited", flush=True)


if __name__ == "__main__":
    main()
