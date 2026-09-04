"""Print a markdown table from eval summary JSON."""

import json
from pathlib import Path

import typer


def main(summary_json: Path = typer.Argument(..., help="Path to summary.json")) -> None:
    data = json.loads(summary_json.read_text())

    # Detect languages present (from any multilingual entry)
    langs: list[str] = []
    for entry in data.values():
        per_lang = entry.get("bpb_per_language", {})
        if per_lang:
            langs = sorted(per_lang.keys())
            break

    # Header
    cols = ["Model", "BLiMP"]
    if langs:
        cols += [f"BPB ({l})" for l in langs]
    cols += ["BPB", "PPL"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")

    for name, entry in data.items():
        blimp = entry.get("blimp_overall", "—")
        per_lang = entry.get("bpb_per_language", {})
        row = [name, f"{blimp:.4f}" if isinstance(blimp, float) else blimp]
        if langs:
            for l in langs:
                v = per_lang.get(l, {}).get("bpb")
                row.append(f"{v:.4f}" if v is not None else "—")
        bpb = entry.get("bpb", "—")
        ppl = entry.get("perplexity", "—")
        row.append(f"{bpb:.4f}" if isinstance(bpb, float) else bpb)
        row.append(f"{ppl:.2f}" if isinstance(ppl, float) else ppl)
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    typer.run(main)
