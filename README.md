# comp-vs-like

Code and reproduction pipeline for **"Objective vs. Search: Decomposing What Makes a Good Tokeniser"**

Ahmetcan Yavuz (ETH Zürich), Clara Meister (EPFL), Tiago Pimentel (ETH Zürich)

The two dominant tokenisation algorithms, BPE and UnigramLM, differ along two orthogonal axes at once:

|                      | **Compression objective** | **Log-likelihood objective** |
|----------------------|---------------------------|------------------------------|
| **Bottom-up merging**| `BPE`                     | `BottomUpLL`  *(ours)*       |
| **Top-down pruning** | `TopDownComp`  *(ours)*   | `UnigramLM`                  |

Existing comparisons confound the two axes. This repository fills the empty cells with two new
tokenisers so the axes can be varied independently, and contains everything needed to reproduce
the paper end to end: data preparation, tokeniser training, language-model training, and evaluation.

The 20 trained tokenisers used for the main results ship in `tokenizers/`, named after the paper.

## Repository layout

```
comp-vs-like/
├── env.sh                      # all machine-specific paths live here
├── tokenizers/                 #    the 20 trained tokenisers (ready to use)
├── data_prep/                  # 1. download + tokenise the corpora
├── tokenizer_training/
│   ├── fork/                   #    modified HuggingFace tokenizers (Rust) — the algorithms
│   └── scripts/                # 2. train the tokenisers
├── src/                        # 3. model, data pipeline, training and eval CLIs
├── configs/models/             #    model architectures
├── training/                   # 3. training launchers
├── evaluation/                 # 4. BPB and BLiMP launchers
├── intrinsic/                  #    tokeniser-intrinsic analysis (+ figures/)
├── analysis/                   #    bootstrap, seed variance, result tables
├── slurm/                      #    cluster job array (host) + containerised inner script
└── tests/
```

## Where the algorithms live

Both new tokenisers are implemented in the vendored Rust fork, not in Python:

| Tokeniser   | Directory             | Implementation                                                       |
|-------------|-----------------------|----------------------------------------------------------------------|
| BPE         | `bpe-*`               | `fork/tokenizers/src/models/bpe/trainer.rs`, `score_by="count"`       |
| BottomUpLL  | `bottomupll-exact-*`  | same file, `score_by="greedy_ll_exact"`                               |
| BottomUpLL  | `bottomupll-approx-*` | same file, `score_by="greedy_ll_approx"`                              |
| TopDownComp | `topdowncomp-*`       | `fork/tokenizers/src/models/unigram/compression_trainer.rs`           |
| UnigramLM   | `unigramlm-*`         | upstream `fork/tokenizers/src/models/unigram/trainer.rs` (unmodified) |

Each exists at `8k`, `32k` and `128k`, plus a `-multi-128k` variant trained on the multilingual
corpus: 20 tokenisers covering the main results.

The `score_by` values keep their original spellings (`greedy_ll_exact`, `greedy_ll_approx`): those
are the Rust API, not display names.

All four therefore share one code path for corpus reading, normalisation and pre-tokenisation;
only the scoring and search differ. See `NOTICE` for the full list of modifications.

## Setup

```bash
git clone https://github.com/Ahmetcanyvz/comp-vs-like.git && cd comp-vs-like
uv sync                                    # add --extra flash for FlashAttention
```

Build and install the tokeniser fork. This replaces the PyPI `tokenizers` package with the
modified build; it stays API-compatible, so `transformers` continues to work.

```bash
uv pip install maturin
cd tokenizer_training/fork/bindings/python && maturin develop --release && cd -
python -c "from tokenizers.trainers import CompressionTrainer; print('fork OK')"
```

Point the pipeline at your storage. Everything reads from `env.sh`, which takes its values from
`env.local.sh` (gitignored) — there are no paths baked into any script.

```bash
cat > env.local.sh <<'EOF'
export CVL_DATA=/scratch/me/compvslike/data
export CVL_TOKENIZERS=/scratch/me/compvslike/tokenizers
export CVL_OUTPUTS=/scratch/me/compvslike/outputs
export CVL_NGPU=4
EOF
```

## Reproduction pipeline

### 1. Data

```bash
python data_prep/download_english.py -o "$CVL_RAW_EN" --target-tokens 20_000_000_000
python data_prep/download_multilingual.py -o "$CVL_RAW_MULTI" --tokens-per-lang 2_500_000_000
```

Validation and test are carved from the **first 2B tokens** under a seed-42 shuffle, giving 47,384
documents each; they never grow when the training set is extended. All reported BPB and BLiMP
numbers are computed on those held-out sets.

Pass `--revision <sha>` to pin the HuggingFace dataset snapshot. FineWeb-Edu and FineWeb-2 are
updated over time, and the default `main` will not stay byte-identical to what the paper used.

Then tokenise the corpus once per tokeniser:

```bash
python data_prep/prepare_all.py --tokenize-only \
    --raw-data-dir "$CVL_RAW_EN" -o "$CVL_DATA" \
    -t "$CVL_TOKENIZERS/bpe-128k" -t "$CVL_TOKENIZERS/topdowncomp-128k" \
    -t "$CVL_TOKENIZERS/bottomupll-exact-128k" -t "$CVL_TOKENIZERS/unigramlm-128k"
```

### 2. Tokenisers

```bash
python tokenizer_training/scripts/train_bpe_family.py     --vocab-sizes 8000 32000 128000
python tokenizer_training/scripts/train_unigram_family.py --vocab-sizes 8000 32000 128000
```

The first trains `bpe`, `bottomupll-exact` and `bottomupll-approx`; the second trains `topdowncomp`
and `unigramlm`. Multilingual equivalents are `train_multilingual_*.py`.

All four tokenisers are trained on the same corpus with identical settings — NFC normalisation,
byte-level pre-tokenisation without a prefix space, `max_piece_length=16`, and a matched pruning
rate (`prune_ratio=0.1` for TopDownComp, `shrinking_factor=0.9` for UnigramLM). Special tokens are
`<|endoftext|>` (id 0) and `<|padding|>` (id 1).

### 3. Model training

```bash
# single GPU per model, CVL_NGPU models in parallel (as used for 100M/340M/500M and 1B)
MODELS="me340M-tied" VOCABS="128k 32k 8k" SEEDS="42 43 44" ./training/train_models.sh

# multi-GPU DDP on a SLURM cluster (as used for the 1B runs)
MODEL=me1B-tied SEEDS="43 44" ./slurm/submit.sh
```

The two paths are genuinely different launchers, not one script with a flag. The single-GPU path
runs `python -m src.train` directly; the cluster path is a job array whose batch shell stays on the
**host** (so the `USR1` auto-requeue trap can call `scontrol` ~180 s before the 12 h walltime) while
the workload runs in the container under `torchrun`. `slurm/train_inner.sh` also pre-creates the
`PackedTokenDataset` metadata single-process before launching, so the DDP ranks load it rather than
racing to build it.

Every run used a **global batch of 128 sequences per optimiser step**; the machines reached it
differently, so `gradient_accumulation` is derived rather than hardcoded:

| Machine  | Model | Per-device batch | Accum | GPUs | Sequences/step |
|----------|-------|------------------|-------|------|----------------|
| satay    | 340M  | 16               | 8     | 1    | 128            |
| satay    | 100M  | 32               | 4     | 1    | 128            |
| bourbon  | 1B    | 16               | 8     | 1    | 128            |
| clariden | 1B    | 16               | 2     | 4    | 128            |

Architectures (Llama-style, GQA, tied embeddings):

| Config        | Params | Hidden | Layers | Heads | KV heads | Token budget |
|---------------|--------|--------|--------|-------|----------|--------------|
| `me100M-tied` | 100M   | 576    | 30     | 9     | 3        | 2B           |
| `me340M-tied` | 340M   | 960    | 32     | 15    | 5        | 7B           |
| `me500M-tied` | 500M   | 1280   | 26     | 16    | 4        | 10B          |
| `me1B-tied`   | 1B     | 2048   | 22     | 32    | 4        | 20B          |

Training uses sequence length 2048, LR 6e-4 with 2000 warmup and 2000 decay steps to 1% of peak,
AdamW (0.9, 0.95), weight decay 0.1, gradient clipping 1.0, and bf16. Budgets follow ~20 tokens
per parameter.

### 4. Evaluation

```bash
./evaluation/run_bpb.sh                                  # bits per byte, English test
TEST_DATA="$CVL_RAW_MULTI/deu/test" ./evaluation/run_bpb.sh
./evaluation/run_blimp.sh                                # BLiMP (English), all models
./evaluation/run_multiblimp.sh bpe-multi-128k      # MultiBLiMP + ZhoBLiMP, one model
```

`run_multiblimp.sh` loops the four MultiBLiMP languages (`eng deu spa tur`) and then runs ZhoBLiMP
for Chinese, because MultiBLiMP has no `cmn` config.

Both discover every trained model under `$CVL_OUTPUTS`, run one per GPU, and skip anything already
evaluated, so they are safe to re-run.

Corpus BPB is byte-weighted — `sum(loss_nats) / (sum(num_bytes) * ln 2)` — not the mean of
per-document BPB. Then:

```bash
python analysis/gather_results.py                        # merge BPB + BLiMP per model
python analysis/bootstrap_bpb_compare.py A.parquet B.parquet   # paired bootstrap, 95% CI
python analysis/bpb_seed_std.py                          # mean +/- std across seeds
```

`bootstrap_bpb_compare.py` resamples documents with **shared indices** across the two models, so
the interval is over the paired difference rather than the two marginals.

### 5. Intrinsic analysis

Properties of the tokenisers themselves, independent of any trained model. All of these read
`intrinsic/config.py`, which resolves tokeniser and corpus paths from the same `CVL_*` variables:

```bash
python intrinsic/compression_stats.py        # bytes per token, fertility
python intrinsic/token_length_dist.py        # token length distribution
python intrinsic/zipf_and_entropy.py         # Zipf alpha, Shannon entropy, coverage
python intrinsic/vocab_differences.py        # vocabulary comparison across methods
python intrinsic/merge_overlap.py            # merge overlap and ordering (bottom-up methods only)
python intrinsic/absorbed_tokens.py          # tokens absorbed by later merges
python intrinsic/absorbed_zero_usage.py      # absorbed tokens with zero usage on held-out text
python intrinsic/tokenization_examples.py    # qualitative side-by-side segmentations
python intrinsic/multi_bytes.py              # UTF-8 byte counts for the multilingual test sets

CVL_VARIANT=multi python intrinsic/compression_stats.py   # multilingual tokenisers
```

The two figures in the paper:

```bash
python intrinsic/figures/zipf_curves.py -o zipf_curves.pdf
python intrinsic/figures/vocab_overlap_heatmap.py -o vocab_overlap_heatmap.pdf
```

The heatmap needs only the shipped tokenisers, so it runs immediately after cloning and reproduces
the block structure the paper reports: overlap within a search family (BPE / BottomUpLL 82-94%,
TopDownComp / UnigramLM 61%) is far higher than across families (41-50%). `zipf_curves.py`
additionally needs the raw English test split and caches its token counts under
`intrinsic/_counts_cache/`.

## Reproducibility notes

- **Tokeniser training is deterministic.** Merge ties break on token-pair id, not heap order, so a
  given corpus and vocabulary size always produce the same vocabulary.
- **`REBUILD_INTERVAL` is a performance knob only.** The trainer's pop path re-validates every
  entry against current counts and re-pushes stale ones, so heap-rebuild frequency cannot change
  which merge is selected.
- **Directories were renamed to the paper's names after the experiments ran.** The code and the
  trained checkpoint directories originally used development names. Map old to new with:
  `greedyll-exact` -> `bottomupll-exact`, `greedyll-approx` -> `bottomupll-approx`,
  `compmax` -> `topdowncomp`, `bpe_count` -> `bpe`. The eval launchers derive the tokeniser name
  from the checkpoint directory name, so checkpoint dirs must use the new names (for example
  `me1B-tied_topdowncomp-128k_20Btok_seed42`) for the lookup to resolve.
- **`bpe_count-*` and `bpe-*` were byte-identical** at 8k, 32k and 128k, so only one copy ships,
  as `bpe-*`. The paper's BPE baseline is `score_by="count"`.
- **The paper's runs used SDPA, not FlashAttention.** Every generated config sets
  `use_flash_attention: true`, but that flag only *requests* flash: `get_attention_implementation()`
  returns `"flash_attention_2"` only if `flash_attn` is importable and silently falls back to
  `"sdpa"` otherwise. `flash-attn` is an optional extra that no setup path installed
  (`Dockerfile.clariden` does not list it, and `setup_bourbon.sh` runs a bare `uv sync`), so the
  runs used SDPA. The configs are kept as they were; if you install `--extra flash` you will get a
  different attention kernel than the paper did.
- **Pin your dataset revision.** This is the one step that will silently drift; see step 1.

## Citation

See `CITATION.cff`.

## Licence

MIT, except `tokenizer_training/fork/`, which is a modified copy of HuggingFace `tokenizers` and
remains under Apache 2.0. See `LICENSE` and `NOTICE`.
