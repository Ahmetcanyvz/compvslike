"""Export eval results to JSON summary."""

import json
from pathlib import Path

import polars as pl
import typer


def main(
    eval_dir: Path = typer.Argument(..., help="Path to eval_results directory"),
    models: list[str] = typer.Option(None, "--model", "-m", help="Model folder names (default: all)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON path (default: eval_dir/summary.json)"),
) -> None:
    if models:
        model_dirs = [eval_dir / m for m in models]
    else:
        model_dirs = sorted(p for p in eval_dir.iterdir() if p.is_dir())

    results = {}
    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"[SKIP] {model_dir} not found")
            continue

        name = model_dir.name
        entry = {}

        # BLiMP
        blimp = model_dir / "blimp.parquet"
        if blimp.exists():
            df = pl.read_parquet(blimp)
            entry["blimp_overall"] = round(df["correct"].mean(), 4)
            # Per-task accuracy
            per_task = (
                df.group_by("task")
                .agg(pl.col("correct").mean().alias("accuracy"))
                .sort("task")
            )
            entry["blimp_per_task"] = {
                row["task"]: round(row["accuracy"], 4) for row in per_task.iter_rows(named=True)
            }

        # BPB (English-only model: bpb.parquet)
        bpb = model_dir / "bpb.parquet"
        if bpb.exists():
            df = pl.read_parquet(bpb)
            total_loss = df["loss_nats"].sum()
            total_bytes = df["num_bytes"].sum()
            entry["bpb"] = round(total_loss / total_bytes / 0.6931472, 4)
            entry["perplexity"] = round(2 ** entry["bpb"], 2)
            entry["num_docs"] = len(df)

        # Per-language BPB (multilingual: bpb_<lang>.parquet)
        per_lang = {}
        for lang_file in sorted(model_dir.glob("bpb_*.parquet")):
            lang = lang_file.stem.replace("bpb_", "")
            df = pl.read_parquet(lang_file)
            total_loss = df["loss_nats"].sum()
            total_bytes = df["num_bytes"].sum()
            bpb_val = round(total_loss / total_bytes / 0.6931472, 4)
            per_lang[lang] = {
                "bpb": bpb_val,
                "perplexity": round(2 ** bpb_val, 2),
                "num_docs": len(df),
            }
        if per_lang:
            entry["bpb_per_language"] = per_lang

        if entry:
            results[name] = entry
            print(f"{name}: BLiMP={entry.get('blimp_overall', 'N/A')}, PPL={entry.get('perplexity', 'N/A')}")

    out_path = output or (eval_dir / "summary.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    typer.run(main)
