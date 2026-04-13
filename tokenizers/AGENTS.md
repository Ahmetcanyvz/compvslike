# Repository Guidelines

## Project Structure & Module Organization
Each top-level directory represents a tokenizer family and target vocab size (for example, `bpe-32k`, `bpe_count-8k`, `compmax-128k`, `greedyll-approx-32k`, `greedyll-exact-8k`, `unigramlm-128k`). Every bundle stores a `tokenizer.json`, its `tokenizer_config.json`, and—where applicable—a `special_tokens_map.json`. Keep additions consistent with this `algorithm-size` naming so downstream consumers can glob for their preferred recipe. Place experimental exports in a new directory with the same trio of files; do not mix multiple algorithms inside a single folder.

## Build, Test, and Development Commands
Tokenizers are validated with Hugging Face tooling. Run `python -m json.tool bpe-32k/tokenizer.json > /dev/null` before committing to catch malformed JSON. Use a quick smoke-load with transformers: `python - <<'PY'\nfrom transformers import AutoTokenizer\nAutoTokenizer.from_pretrained("greedyll-exact-32k")\nprint("loaded")\nPY` to ensure configs and special tokens align. When regenerating assets, document the exact `tokenizers` or `transformers` version used (store it in your pull request) so artifacts are reproducible.

## Coding Style & Naming Conventions
JSON files are indented with two spaces, keys use snake_case to match Hugging Face expectations, and special tokens stay in the `<|token|>` pattern already present in every config. Keep vocabulary sizes encoded in the directory name only; the JSON should not repeat it unless the upstream trainer requires it. Never remove the `<|endoftext|>` and `<|padding|>` entries from `special_tokens_map.json`—additions must be appended with explicit flags (`special: true`).

## Testing Guidelines
Before opening a pull request, validate any modified tokenizer JSON via both strict JSON parsing and a round-trip encode/decode: `python - <<'PY' ... tokenizer.encode("smoke test")`. Compare vocabulary counts against the expected size (`len(tokenizer) == 32000` for `*-32k`). Run at least one detokenization sample per special token to confirm padding and BOS/EOS mappings. Capture these checks in your PR description rather than committing ad-hoc scripts.

## Commit & Pull Request Guidelines
Commits follow the short, imperative style already in history (for example, "Add data preparation command"). Group all files for a single tokenizer export in one commit to keep diffs reviewable. Pull requests should describe the training recipe used (dataset, trainer flags, vocab size), link to any upstream issue or experiment log, and attach the validation commands/output above. If previews are large, provide a summary of key diffs (`git diff --stat` per directory) so reviewers can focus on intentional changes.
