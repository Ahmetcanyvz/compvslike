"""MultiBLiMP (Multilingual Benchmark of Linguistic Minimal Pairs) evaluation.

MultiBLiMP tests grammatical knowledge across languages by presenting minimal
pairs of sentences: one grammatical, one ungrammatical. The model should
assign higher probability to the grammatical sentence.

Reference: Jumelet, Weissweiler & Bisazza (2025), "MultiBLiMP 1.0" (arXiv:2504.02768).
Dataset: https://huggingface.co/datasets/jumelet/multiblimp (one config per ISO 639-3 lang).
"""

import json
from pathlib import Path

import polars as pl
import torch
import typer
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.model import load_model_from_checkpoint

app = typer.Typer(help="Run MultiBLiMP evaluation on trained models.")
console = Console()


def compute_sentence_logprob(model: torch.nn.Module, input_ids: torch.Tensor) -> float:
    """Sum of token log-probabilities under the model (causal LM)."""
    if len(input_ids) <= 1:
        return float("-inf")
    input_ids = input_ids.unsqueeze(0)
    logits = model(input_ids[:, :-1]).logits
    labels = input_ids[:, 1:]
    loss = cross_entropy(logits.permute(0, 2, 1), labels, reduction="sum")
    return -loss.item()


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    tokenizer_path: Path = typer.Argument(..., help="Path to tokenizer"),
    language: str = typer.Argument(..., help="ISO 639-3 language code (e.g. deu, spa, tur, cmn, eng)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output parquet file"),
    dataset_name: str = typer.Option("jumelet/multiblimp", "--dataset", help="HF dataset id"),
    split: str = typer.Option("train", "--split", help="Dataset split"),
    device: str = typer.Option("cuda", "--device", help="cuda/cpu"),
) -> None:
    """Run MultiBLiMP evaluation for a single language."""
    console.print(f"[green]Loading model from {checkpoint}[/green]")
    model = load_model_from_checkpoint(checkpoint)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    console.print(f"[blue]Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters[/blue]")

    console.print(f"[green]Loading tokenizer from {tokenizer_path}[/green]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    console.print(f"[blue]Tokenizer vocab size: {len(tokenizer)}[/blue]")

    console.print(f"[green]Loading {dataset_name} config={language} split={split}[/green]")
    dataset = load_dataset(dataset_name, language, split=split)
    console.print(f"[blue]{len(dataset)} minimal pairs[/blue]")

    results = []
    per_phenomenon = {}

    with torch.inference_mode():
        for example in tqdm(dataset, desc=f"MultiBLiMP {language}"):
            sentence_good = example["sen"]
            sentence_bad = example["wrong_sen"]
            phenomenon = example.get("phenomenon", "unknown")

            ids_good = tokenizer.encode(sentence_good, return_tensors="pt", add_special_tokens=False)
            ids_bad = tokenizer.encode(sentence_bad, return_tensors="pt", add_special_tokens=False)
            ids_good = ids_good.squeeze(0).to(device)
            ids_bad = ids_bad.squeeze(0).to(device)

            logprob_good = compute_sentence_logprob(model, ids_good)
            logprob_bad = compute_sentence_logprob(model, ids_bad)
            is_correct = logprob_good > logprob_bad

            results.append({
                "language": language,
                "phenomenon": phenomenon,
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "logprob_good": logprob_good,
                "logprob_bad": logprob_bad,
                "correct": is_correct,
            })

            entry = per_phenomenon.setdefault(phenomenon, {"correct": 0, "total": 0})
            entry["correct"] += int(is_correct)
            entry["total"] += 1

    total_correct = sum(v["correct"] for v in per_phenomenon.values())
    total_examples = sum(v["total"] for v in per_phenomenon.values())
    overall_accuracy = total_correct / total_examples if total_examples else 0.0

    table = Table(title=f"MultiBLiMP {language} Results")
    table.add_column("Phenomenon", style="cyan")
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    for phen, stats in sorted(per_phenomenon.items(), key=lambda x: -x[1]["correct"] / max(1, x[1]["total"])):
        acc = stats["correct"] / max(1, stats["total"])
        table.add_row(phen, f"{acc:.1%}", str(stats["correct"]), str(stats["total"]))
    table.add_row("", "", "", "", style="dim")
    table.add_row("OVERALL", f"{overall_accuracy:.1%}", str(total_correct), str(total_examples), style="bold")
    console.print(table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(results).write_parquet(output)
        console.print(f"[green]Results saved to {output}[/green]")
        summary = {
            "language": language,
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_examples": total_examples,
            "phenomenon_accuracies": {
                k: v["correct"] / max(1, v["total"]) for k, v in per_phenomenon.items()
            },
        }
        with open(output.with_suffix(".summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"[green]Summary saved to {output.with_suffix('.summary.json')}[/green]")


if __name__ == "__main__":
    app()
