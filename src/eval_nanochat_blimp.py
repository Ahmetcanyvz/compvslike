"""BLiMP evaluation for nanochat GPT models."""

import json
import sys
from pathlib import Path

import torch
import typer
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm

NANOCHAT_DIR = Path(__file__).parent.parent / "nanochat"
if str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

from nanochat.gpt import GPT, GPTConfig

app = typer.Typer()
console = Console()


def load_nanochat_model(checkpoint_path: Path, meta_path: Path, device: str = "cuda") -> GPT:
    """Load a nanochat GPT model from checkpoint."""
    with open(meta_path) as f:
        meta = json.load(f)
    config = GPTConfig(**meta["model_config"])
    model = GPT(config)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model


def load_tokenizer(tokenizer_dir: Path):
    """Load tokenizer — try HuggingFace json first, fall back to tiktoken pkl."""
    json_path = tokenizer_dir / "tokenizer.json"
    pkl_path = tokenizer_dir / "tokenizer.pkl"

    if json_path.exists():
        try:
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast(tokenizer_file=str(json_path))
            return tok, "hf"
        except Exception:
            pass

    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            tok = pickle.load(f)
        return tok, "tiktoken"

    raise FileNotFoundError(f"No tokenizer found in {tokenizer_dir}")


def encode(tokenizer, tok_type: str, text: str) -> list[int]:
    """Encode text with either HF or tiktoken tokenizer."""
    if tok_type == "hf":
        return tokenizer.encode(text, add_special_tokens=False)
    else:
        return tokenizer.encode(text)


def compute_sentence_logprob(model: GPT, input_ids: torch.Tensor) -> float:
    """Compute log-probability of a sentence."""
    if len(input_ids) <= 1:
        return float("-inf")
    input_ids = input_ids.unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_ids[:, :-1])
        labels = input_ids[:, 1:]
        loss = cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
    return -loss.item()


ALL_BLIMP_TASKS = [
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


@app.command()
def evaluate(
    checkpoint_dir: Path = typer.Argument(..., help="Directory containing model_*.pt and meta_*.json"),
    tokenizer_dir: Path = typer.Argument(..., help="Directory containing tokenizer"),
    output: Path = typer.Option(None, "--output", "-o", help="Output parquet file"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    """Run BLiMP evaluation on a nanochat model."""
    import polars as pl

    # Find checkpoint and meta files
    model_files = sorted(checkpoint_dir.glob("model_*.pt"))
    meta_files = sorted(checkpoint_dir.glob("meta_*.json"))
    if not model_files or not meta_files:
        console.print(f"[red]No model/meta files found in {checkpoint_dir}[/red]")
        raise typer.Exit(1)

    model_path = model_files[-1]
    meta_path = meta_files[-1]
    console.print(f"[green]Loading model from {model_path}[/green]")

    model = load_nanochat_model(model_path, meta_path, device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"[blue]Model: {num_params:.1f}M parameters[/blue]")

    tokenizer, tok_type = load_tokenizer(tokenizer_dir)
    console.print(f"[green]Loaded {tok_type} tokenizer from {tokenizer_dir}[/green]")

    results = []
    task_accuracies = {}

    with torch.inference_mode():
        for task_name in tqdm(ALL_BLIMP_TASKS, desc="Tasks"):
            try:
                dataset = load_dataset("blimp", task_name, split="train")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load {task_name}: {e}[/yellow]")
                continue

            correct = 0
            total = 0

            for example in dataset:
                ids_good = torch.tensor(encode(tokenizer, tok_type, example["sentence_good"]), device=device)
                ids_bad = torch.tensor(encode(tokenizer, tok_type, example["sentence_bad"]), device=device)

                logprob_good = compute_sentence_logprob(model, ids_good)
                logprob_bad = compute_sentence_logprob(model, ids_bad)

                is_correct = logprob_good > logprob_bad
                results.append({
                    "task": task_name,
                    "sentence_good": example["sentence_good"],
                    "sentence_bad": example["sentence_bad"],
                    "logprob_good": logprob_good,
                    "logprob_bad": logprob_bad,
                    "correct": is_correct,
                })

                if is_correct:
                    correct += 1
                total += 1

            if total > 0:
                task_accuracies[task_name] = {"accuracy": correct / total, "correct": correct, "total": total}

    total_correct = sum(t["correct"] for t in task_accuracies.values())
    total_examples = sum(t["total"] for t in task_accuracies.values())
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0

    table = Table(title="BLiMP Results")
    table.add_column("Task", style="cyan")
    table.add_column("Accuracy", style="green", justify="right")
    for task_name, stats in sorted(task_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        table.add_row(task_name, f"{stats['accuracy']:.1%}")
    table.add_row("", "")
    table.add_row("OVERALL", f"{overall_accuracy:.1%}", style="bold")
    console.print(table)

    if output:
        df = pl.DataFrame(results)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        summary_path = output.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({"overall_accuracy": overall_accuracy, "task_accuracies": {k: v["accuracy"] for k, v in task_accuracies.items()}}, f, indent=2)
        console.print(f"[green]Saved to {output} and {summary_path}[/green]")


if __name__ == "__main__":
    app()
