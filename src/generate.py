"""Generate text from a trained language model."""

from pathlib import Path

import torch
import typer
from rich.console import Console
from transformers import AutoTokenizer

from src.model import LanguageModel

app = typer.Typer(help="Generate text from trained models.")
console = Console()


@app.command()
def generate(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint"),
    tokenizer_path: Path = typer.Argument(..., help="Path to tokenizer"),
    prompt: str = typer.Option("The meaning of life is", "--prompt", "-p", help="Prompt text"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-m", help="Max tokens to generate"),
    temperature: float = typer.Option(0.8, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling"),
) -> None:
    """Generate text from a trained model."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    console.print(f"[blue]Loaded tokenizer with {vocab_size} tokens[/blue]")

    # Get model config from checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ckpt["hyper_parameters"]["config"]
    console.print(f"[blue]Model: {model_config.get('name', 'unknown')}[/blue]")

    # Load model
    model = LanguageModel.load_from_checkpoint(
        checkpoint,
        config=model_config,
        use_flash_attention=False,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    console.print(f"[blue]Using device: {device}[/blue]")

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    console.print(f"\n[yellow]Prompt:[/yellow] {prompt}\n")

    # Generate
    with torch.no_grad():
        output = model.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    console.print(f"[green]Generated:[/green]\n{generated_text}")


if __name__ == "__main__":
    app()
