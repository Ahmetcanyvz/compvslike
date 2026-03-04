"""BLiMP (Benchmark of Linguistic Minimal Pairs) evaluation.

BLiMP tests grammatical knowledge by presenting minimal pairs of sentences:
one grammatical, one ungrammatical. The model should assign higher
probability to the grammatical sentence.

Reference: Warstadt et al. (2020) "BLiMP: The Benchmark of Linguistic Minimal Pairs for English"
"""

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

app = typer.Typer(help="Run BLiMP evaluation on trained models.")
console = Console()


def compute_sentence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> float:
    """Compute log-probability of a sentence.

    Args:
        model: The language model.
        input_ids: Token IDs of shape (seq_len,).

    Returns:
        Sum of log-probabilities (negative cross-entropy).
    """
    if len(input_ids) <= 1:
        return float("-inf")

    input_ids = input_ids.unsqueeze(0)  # Add batch dim

    # Compute logits
    logits = model(input_ids[:, :-1]).logits
    labels = input_ids[:, 1:]

    # Compute per-token loss
    loss = cross_entropy(
        logits.permute(0, 2, 1),
        labels,
        reduction="sum",
    )

    # Return negative loss (log-probability)
    return -loss.item()


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    tokenizer_path: Path = typer.Argument(..., help="Path to tokenizer"),
    output: Path = typer.Option(None, "--output", "-o", help="Output parquet file (optional)"),
    tasks: str = typer.Option(None, "--tasks", "-t", help="Comma-separated task names (default: all)"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)"),
) -> None:
    """Run BLiMP evaluation.

    Downloads BLiMP dataset from HuggingFace and evaluates model accuracy
    on each linguistic task.
    """
    console.print(f"[green]Loading model from {checkpoint}[/green]")

    # Load model
    model = load_model_from_checkpoint(checkpoint)
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    console.print(f"[blue]Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters[/blue]")

    # Load tokenizer
    console.print(f"[green]Loading tokenizer from {tokenizer_path}[/green]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    console.print(f"[blue]Tokenizer vocab size: {len(tokenizer)}[/blue]")

    # Load BLiMP dataset
    console.print("[green]Loading BLiMP dataset from HuggingFace...[/green]")

    # Get list of all BLiMP tasks
    all_tasks = [
        "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
        "animate_subject_passive", "animate_subject_trans", "causative",
        "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction", "determiner_noun_agreement_1",
        "determiner_noun_agreement_2", "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2", "determiner_noun_agreement_with_adj_2",
        "determiner_noun_agreement_with_adj_irregular_1", "determiner_noun_agreement_with_adj_irregular_2",
        "determiner_noun_agreement_with_adjective_1", "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause", "drop_argument", "ellipsis_n_bar_1",
        "ellipsis_n_bar_2", "existential_there_object_raising", "existential_there_quantifiers_1",
        "existential_there_quantifiers_2", "existential_there_subject_raising", "expletive_it_object_raising",
        "inchoative", "intransitive", "irregular_past_participle_adjectives",
        "irregular_past_participle_verbs", "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2", "left_branch_island_echo_question",
        "left_branch_island_simple_question", "matrix_question_npi_licensor_present",
        "npi_present_1", "npi_present_2", "only_npi_licensor_present",
        "only_npi_scope", "passive_1", "passive_2", "principle_A_c_command",
        "principle_A_case_1", "principle_A_case_2", "principle_A_domain_1",
        "principle_A_domain_2", "principle_A_domain_3", "principle_A_reconstruction",
        "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
        "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope",
        "sentential_subject_island", "superlative_quantifiers_1", "superlative_quantifiers_2",
        "tough_vs_raising_1", "tough_vs_raising_2", "transitive", "wh_island",
        "wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ]

    if tasks:
        selected_tasks = [t.strip() for t in tasks.split(",")]
    else:
        selected_tasks = all_tasks

    # Evaluate each task
    results = []
    task_accuracies = {}

    with torch.inference_mode():
        for task_name in tqdm(selected_tasks, desc="Tasks"):
            try:
                dataset = load_dataset("blimp", task_name, split="train")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load task {task_name}: {e}[/yellow]")
                continue

            correct = 0
            total = 0

            for example in dataset:
                sentence_good = example["sentence_good"]
                sentence_bad = example["sentence_bad"]

                # Tokenize
                ids_good = tokenizer.encode(sentence_good, return_tensors="pt", add_special_tokens=False)
                ids_bad = tokenizer.encode(sentence_bad, return_tensors="pt", add_special_tokens=False)

                ids_good = ids_good.squeeze(0).to(device)
                ids_bad = ids_bad.squeeze(0).to(device)

                # Compute log-probabilities
                logprob_good = compute_sentence_logprob(model, ids_good)
                logprob_bad = compute_sentence_logprob(model, ids_bad)

                # Check if model prefers grammatical sentence
                is_correct = logprob_good > logprob_bad

                results.append({
                    "task": task_name,
                    "sentence_good": sentence_good,
                    "sentence_bad": sentence_bad,
                    "logprob_good": logprob_good,
                    "logprob_bad": logprob_bad,
                    "correct": is_correct,
                })

                if is_correct:
                    correct += 1
                total += 1

            if total > 0:
                accuracy = correct / total
                task_accuracies[task_name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                }

    # Compute overall accuracy
    total_correct = sum(t["correct"] for t in task_accuracies.values())
    total_examples = sum(t["total"] for t in task_accuracies.values())
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0

    # Print results
    console.print("\n")
    table = Table(title="BLiMP Results")
    table.add_column("Task", style="cyan")
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    # Sort by accuracy
    sorted_tasks = sorted(task_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    for task_name, stats in sorted_tasks:
        table.add_row(
            task_name,
            f"{stats['accuracy']:.1%}",
            str(stats["correct"]),
            str(stats["total"]),
        )

    table.add_row("", "", "", "", style="dim")
    table.add_row(
        "OVERALL",
        f"{overall_accuracy:.1%}",
        str(total_correct),
        str(total_examples),
        style="bold",
    )

    console.print(table)

    # Save results if requested
    if output:
        df = pl.DataFrame(results)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        console.print(f"\n[green]Results saved to {output}[/green]")

        # Also save summary
        summary_path = output.with_suffix(".summary.json")
        import json

        summary = {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_examples": total_examples,
            "task_accuracies": {k: v["accuracy"] for k, v in task_accuracies.items()},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"[green]Summary saved to {summary_path}[/green]")


if __name__ == "__main__":
    app()
