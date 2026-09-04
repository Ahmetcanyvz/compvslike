//! Greedy compression-based trainer for the Unigram model.
//!
//! This trainer minimizes the total number of tokens under **unit-cost decoding**.
//! It iteratively deletes token(s) that minimize
//!
//!   ΔL(t) = c[t] * ( d[t] - 1 )
//!
//! where
//!   - `c[t]` is the current corpus count of `t` in the best (unit-cost) segmentations,
//!   - `d[t]` is the shortest decomposition length of the *string of `t`* using `V \ {t}`.
//!
//! After deleting a token, we **only re-segment the sentences that used it**, and we
//! **only recompute d[·]** for tokens whose decomposition depended on it.
//!
//! Notes:
//! - We realize *unit-cost decoding* by giving **every token the same negative score** (e.g. `-1.0`).
//!   Viterbi maximizes the total score, so with per-token cost `-1.0`, it prefers **fewer tokens**.
//! - We don't require any change to `Lattice` or `Trie` by filtering disallowed nodes
//!   out of the lattice **after** populating it.
//!
//! This file is self-contained w.r.t. lattice/trie implementations.

use crate::models::unigram::{lattice::Lattice, model::Unigram};
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::parallelism::*;
use crate::utils::progress::{ProgressBar, ProgressStyle};

use ahash::{AHashMap, AHashSet};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;

// ----------------------------- Local types ------------------------------------

/// A full sentence/span and its count within the dataset
type Sentence = (String, u32);

/// A token candidate and a (dummy) score
type SentencePiece = (String, f64);

// ----------------------------- Trainer ----------------------------------------

/// Trainer for greedy compression-based Unigram.
/// Configuration mirrors the style of `UnigramTrainer` but replaces EM with greedy deletion.
#[derive(Builder, Debug, Clone, Serialize, Deserialize)]
pub struct CompressionTrainer {
    /// Show progress bars while training
    #[builder(default = "true")]
    pub show_progress: bool,

    /// Target vocabulary size (including special tokens)
    #[builder(default = "8000")]
    pub vocab_size: u32,

    /// Max piece length considered when seeding multi-char candidates
    #[builder(default = "16")]
    pub max_piece_length: usize,

    /// Special tokens to prepend in the vocabulary (kept; never deleted)
    #[builder(default = "vec![]")]
    pub special_tokens: Vec<AddedToken>,

    /// Characters to force-include in Σ (in addition to those seen in data)
    #[builder(default = "AHashSet::new()")]
    pub initial_alphabet: AHashSet<char>,

    /// Upper bound on seed list size (same default as UnigramTrainer)
    #[builder(default = "1_000_000")]
    pub seed_size: usize,

    /// If provided, training starts **exactly** from this vocabulary (scores forced to -1.0).
    /// You must include Σ yourself if you want single-char coverage guaranteed.
    #[builder(default)]
    pub seed_vocab: Option<Vec<String>>,

    /// Batch pruning: remove about `ceil(prune_ratio * remaining)` tokens per outer pass
    /// (but at least `min_prune`, and never more than the remaining tokens needed).
    /// When `batch_recompute = true`, we recompute Δ after **each** deletion (default, exact);
    /// otherwise we compute Δ once per pass and delete the top-K (faster, approximate).
    #[builder(default = "0.0")]
    pub prune_ratio: f32,
    #[builder(default = "1")]
    pub min_prune: usize,
    #[builder(default = "true")]
    pub batch_recompute: bool,

    /// If true, enable Unigram byte fallback in the produced model(s).
    /// Note: the underlying Unigram handles fallback internally; we don't insert explicit byte tokens here.
    #[builder(default = "false")]
    pub byte_fallback: bool,

    /// If true, and if fallback tokens are materialized by the underlying implementation,
    /// they are considered non-deletable. (No-op if the model does not export such tokens.)
    #[builder(default = "true")]
    pub keep_byte_fallback: bool,

    /// If true, use random sampling to estimate token removal cost (rand_compression method).
    /// Instead of computing d[t] from the token string, sample spans that use the token
    /// and measure actual cost difference when resegmenting without that token.
    #[builder(default = "false")]
    pub rand_scoring: bool,

    /// Number of spans to sample per token for rand_scoring.
    #[builder(default = "100")]
    pub rand_sample_size: usize,

    /// If true, use forward-backward expected counts instead of Viterbi hard counts.
    /// This considers all possible segmentations weighted by their length.
    #[builder(default = "false")]
    pub use_expected_counts: bool,

    /// Temperature for expected counts. Controls sharpness of length preference.
    /// T → 0: Approaches Viterbi (hard assignment to shortest path)
    /// T = 1: Standard expected counts with unit-cost
    /// T → ∞: Uniform weighting across all paths
    #[builder(default = "1.0")]
    pub temperature: f64,

    /// Internal: word/span counts (populated by `feed`)
    #[builder(default = "AHashMap::new()")]
    words: AHashMap<String, u32>,
}

impl Default for CompressionTrainer {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("CompressionTrainer::default()")
    }
}

impl CompressionTrainer {
    /// Builder entry point, like other trainers in this crate.
    pub fn builder() -> CompressionTrainerBuilder {
        CompressionTrainerBuilder::default()
    }

    // ----------------------------- Progress helpers -----------------------------

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<36!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, _p: &Option<ProgressBar>, len: usize, message: &str) {
        if self.show_progress {
            eprintln!("[CompressionTrainer] {} (n={})", message, len);
        }
    }

    /// Finish the progress bar
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
        }
    }

    // ----------------------------- Utilities -----------------------------------

    /// Required characters Σ = set of all characters seen in data + initial_alphabet.
    fn required_chars(&self, word_counts: &[Sentence]) -> AHashSet<String> {
        word_counts
            .iter()
            .flat_map(|(s, _)| s.chars())
            .chain(self.initial_alphabet.iter().copied())
            .map(|c| c.to_string())
            .collect()
    }

    /// Basic validity check for a candidate sentencepiece.
    fn is_valid_sentencepiece(&self, chars: &[char]) -> bool {
        let n = chars.len();
        if n == 0 || n > self.max_piece_length {
            return false;
        }
        true
    }

    /// Generate seed candidates: include Σ and frequent substrings (via suffix array).
    /// We mirror `UnigramTrainer::make_seed_sentence_pieces` in style.
    fn seed_pieces(&self, sentences: &[Sentence]) -> Vec<SentencePiece> {
        // Flatten sentences with a boundary char to prevent crossing spans
        let c_sentence_boundary = '\0';
        let boundary = c_sentence_boundary.to_string();

        // Collect char counts and build a flat string
        let total: usize = sentences
            .iter()
            .map(|(s, _)| s.chars().count())
            .sum::<usize>()
            + sentences.len();
        let mut flat = String::with_capacity(total);
        let mut all_chars: AHashMap<char, u32> = AHashMap::new();

        for (s, n) in sentences {
            if s.is_empty() {
                continue;
            }
            flat.push_str(s);
            // Keep the boundary to avoid cross-span substrings in suffix-array results
            flat.push_str(&boundary);
            // Count characters excluding boundary
            for c in s.chars() {
                if c != c_sentence_boundary {
                    *all_chars.entry(c).or_default() += *n;
                }
            }
        }
        flat.shrink_to_fit();

        // Compute substrings via suffix array (fast or pure-Rust fallback)
        #[cfg(feature = "esaxx_fast")]
        let suffix = esaxx_rs::suffix(&flat).expect("esaxx suffix failed");
        #[cfg(not(feature = "esaxx_fast"))]
        let suffix = esaxx_rs::suffix_rs(&flat).expect("esaxx_rs suffix_rs failed");

        // 1) Single characters, sorted by decreasing frequency
        //    Also include initial_alphabet chars (with freq 0 if not in corpus)
        let mut seed: Vec<SentencePiece> = Vec::with_capacity(self.seed_size);

        // Add initial_alphabet chars that aren't in corpus (with freq 0)
        for c in &self.initial_alphabet {
            all_chars.entry(*c).or_insert(0);
        }

        let mut sall_chars: Vec<(u32, char)> = all_chars.into_iter().map(|(c, f)| (f, c)).collect();
        // Reversed order by frequency
        sall_chars.sort_by_key(|&a| Reverse(a));
        for (_freq, ch) in sall_chars {
            seed.push((ch.to_string(), 0.0));
            if seed.len() >= self.seed_size {
                return seed;
            }
        }

        // 2) Multi-char substrings from suffix array, scored by freq * length (for ordering)
        //    Use `.iter()` (NOT `.into_iter()`) since `Suffix<T>` is not an iterator.
        let mut substr_index: Vec<_> = suffix
            .iter()
            .filter_map(|(string, freq)| {
                if string.len() <= 1 {
                    return None;
                }
                if string.contains(&c_sentence_boundary) {
                    return None;
                }
                if !self.is_valid_sentencepiece(string) {
                    return None;
                }
                // `freq` is a `u32`; no deref.
                let score = freq * (string.len() as u32);
                Some((score, string))
            })
            .collect();

        // sort by decreasing score (keep compatibility with unigram::trainer approach)
        substr_index.sort_by_key(|&a| Reverse(a));
        for (_score, char_string) in substr_index {
            // Just in case
            debug_assert!(self.is_valid_sentencepiece(char_string));
            // Build a String from `&[char]` / `&Vec<char>`
            let string: String = char_string.iter().copied().collect();
            seed.push((string, 0.0));
            if seed.len() >= self.seed_size {
                break;
            }
        }

        seed
    }

    /// Run Viterbi on `s` and return token ids. Simple, no filtering.
    fn segment_sentence(model: &Unigram, s: &str) -> Vec<usize> {
        let mut lattice = Lattice::from(s, model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);
        let path = lattice.viterbi();
        path.into_iter().map(|n| n.borrow().id).collect()
    }

    /// Compute d[t] = segment token t's string, excluding token t itself.
    /// Returns (d, deps) where deps are the token ids used in decomposition.
    fn compute_dt_with_deps(model: &Unigram, t: usize) -> (usize, Vec<usize>) {
        let tok_str = &model.vocab[t].0;

        // Build lattice and populate
        let mut lattice = Lattice::from(tok_str, model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        // Remove only token t from lattice
        let len = lattice.len();
        for pos in 0..=len {
            lattice.begin_nodes[pos].retain(|node_rc| node_rc.borrow().id != t);
            lattice.end_nodes[pos].retain(|node_rc| node_rc.borrow().id != t);
        }

        // Viterbi
        let path = lattice.viterbi();
        let ids: Vec<usize> = path.into_iter().map(|n| n.borrow().id).collect();

        let d = if ids.is_empty() {
            tok_str.chars().count().max(1)
        } else {
            ids.len()
        };
        (d, ids)
    }

    /// Compute expected counts using forward-backward algorithm.
    /// This considers all possible segmentations weighted by their probability.
    /// With unit-cost mode and temperature T, P(path) ∝ exp(-length/T).
    fn compute_expected_counts_parallel(model: &Unigram, sentences: &[Sentence]) -> Vec<f64> {
        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);

        sentences
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_expected: Vec<f64> = vec![0.0; model.len()];
                for (s, cnt) in chunk {
                    let mut lattice = Lattice::from(s, model.bos_id, model.eos_id);
                    model.populate_nodes(&mut lattice);
                    // populate_marginal accumulates expected counts weighted by freq
                    lattice.populate_marginal(*cnt as f64, &mut local_expected);
                }
                local_expected
            })
            .reduce(
                || vec![0.0; model.len()],
                |mut acc, local| {
                    for (i, v) in local.into_iter().enumerate() {
                        acc[i] += v;
                    }
                    acc
                },
            )
    }

    /// Segment entire corpus in parallel, return c[t] counts
    fn segment_corpus_parallel(model: &Unigram, sentences: &[Sentence]) -> Vec<u32> {
        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);

        sentences
            .maybe_par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_ct: Vec<u32> = vec![0; model.len()];
                for (s, cnt) in chunk {
                    let ids = Self::segment_sentence(model, s);
                    for id in ids {
                        local_ct[id] = local_ct[id].saturating_add(*cnt);
                    }
                }
                local_ct
            })
            .reduce(
                || vec![0u32; model.len()],
                |mut acc, local| {
                    for (i, v) in local.into_iter().enumerate() {
                        acc[i] = acc[i].saturating_add(v);
                    }
                    acc
                },
            )
    }

    /// Segment entire corpus and return:
    /// - c[t]: token counts
    /// - reverse_index: token_id -> list of (span_index, token_count_in_span)
    /// - segmentations: span_index -> (token_ids, token_count)
    fn segment_corpus_with_index(
        model: &Unigram,
        sentences: &[Sentence],
    ) -> (Vec<u32>, Vec<Vec<(usize, u32)>>, Vec<(Vec<usize>, u32)>) {
        let mut ct: Vec<u32> = vec![0; model.len()];
        let mut reverse_index: Vec<Vec<(usize, u32)>> = vec![Vec::new(); model.len()];
        let mut segmentations: Vec<(Vec<usize>, u32)> = Vec::with_capacity(sentences.len());

        for (span_idx, (s, cnt)) in sentences.iter().enumerate() {
            let ids = Self::segment_sentence(model, s);

            // Count token occurrences in this span
            let mut local_counts: AHashMap<usize, u32> = AHashMap::new();
            for &id in &ids {
                *local_counts.entry(id).or_insert(0) += 1;
            }

            // Update global counts and reverse index
            for (&token_id, &token_cnt) in &local_counts {
                ct[token_id] = ct[token_id].saturating_add(token_cnt * cnt);
                reverse_index[token_id].push((span_idx, token_cnt * cnt));
            }

            segmentations.push((ids, *cnt));
        }

        (ct, reverse_index, segmentations)
    }

    /// Segment a span without using a specific token.
    fn segment_without_token(model: &Unigram, s: &str, exclude_token: usize) -> Vec<usize> {
        let mut lattice = Lattice::from(s, model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        // Remove the excluded token from lattice
        let len = lattice.len();
        for pos in 0..=len {
            lattice.begin_nodes[pos].retain(|node_rc| node_rc.borrow().id != exclude_token);
            lattice.end_nodes[pos].retain(|node_rc| node_rc.borrow().id != exclude_token);
        }

        let path = lattice.viterbi();
        path.into_iter().map(|n| n.borrow().id).collect()
    }

    /// Compute ΔL for all tokens using random sampling (u32 counts version).
    /// For each token t with c[t] > 0:
    ///   - Sample up to `sample_size` spans that use t
    ///   - Resegment each span without t
    ///   - Compute average extra tokens
    ///   - ΔL[t] = avg_extra × c[t]
    fn compute_rand_scores(
        model: &Unigram,
        sentences: &[Sentence],
        ct: &[u32],
        reverse_index: &[Vec<(usize, u32)>],
        segmentations: &[(Vec<usize>, u32)],
        sample_size: usize,
        non_deletable: &AHashSet<String>,
    ) -> Vec<f64> {
        let ct_f64: Vec<f64> = ct.iter().map(|&c| c as f64).collect();
        Self::compute_rand_scores_f64(
            model,
            sentences,
            &ct_f64,
            reverse_index,
            segmentations,
            sample_size,
            non_deletable,
        )
    }

    /// Compute ΔL for all tokens using random sampling (f64 counts version).
    /// Supports expected counts which are floating point.
    fn compute_rand_scores_f64(
        model: &Unigram,
        sentences: &[Sentence],
        ct: &[f64],
        reverse_index: &[Vec<(usize, u32)>],
        segmentations: &[(Vec<usize>, u32)],
        sample_size: usize,
        non_deletable: &AHashSet<String>,
    ) -> Vec<f64> {
        let mut scores: Vec<f64> = vec![0.0; model.len()];

        for t in 0..model.len() {
            // Skip non-deletable tokens
            if non_deletable.contains(&model.vocab[t].0) {
                continue;
            }

            // Skip tokens not used in corpus (or very low expected count)
            if ct[t] < 0.001 {
                continue;
            }

            let spans_using_t = &reverse_index[t];
            if spans_using_t.is_empty() {
                continue;
            }

            // Sample spans (take first `sample_size` for now, could randomize)
            let sample_count = spans_using_t.len().min(sample_size);
            let mut total_extra: f64 = 0.0;
            let mut total_weight: f64 = 0.0;

            for &(span_idx, weight) in spans_using_t.iter().take(sample_count) {
                let (ref old_ids, _span_cnt) = segmentations[span_idx];
                let old_len = old_ids.len();

                // Resegment without token t
                let span_str = &sentences[span_idx].0;
                let new_ids = Self::segment_without_token(model, span_str, t);
                let new_len = new_ids.len();

                let extra = (new_len as f64) - (old_len as f64);
                total_extra += extra * (weight as f64);
                total_weight += weight as f64;
            }

            // Average extra cost per usage
            let avg_extra = if total_weight > 0.0 {
                total_extra / total_weight
            } else {
                0.0
            };

            // ΔL[t] = avg_extra × c[t]
            scores[t] = avg_extra * ct[t];
        }

        scores
    }

    /// Build the initial Unigram model for training:
    /// - If `seed_vocab` is provided, use it **as-is** (plus special tokens), all scores `-1.0`.
    /// - Otherwise, start from Σ and substrings discovered via suffix array (scores `-1.0`).
    fn build_initial_model(
        &self,
        sentences: &[Sentence],
        required: &AHashSet<String>,
    ) -> Result<Unigram> {
        if let Some(seed) = &self.seed_vocab {
            // Use the provided seed exactly (plus special tokens), all scores = -1.0.
            let mut seen = AHashSet::new();
            let mut final_seed: Vec<(String, f64)> =
                Vec::with_capacity(seed.len() + self.special_tokens.len());

            // Special tokens first (kept forever)
            for t in &self.special_tokens {
                if seen.insert(t.content.clone()) {
                    final_seed.push((t.content.clone(), -1.0));
                }
            }
            // Then user-provided seed_vocab order
            for s in seed {
                if seen.insert(s.clone()) {
                    final_seed.push((s.clone(), -1.0));
                }
            }
            // No automatic Σ injection here: caller controls it explicitly via seed_vocab.
            let model = Unigram::from(final_seed, None, self.byte_fallback)?;
            return Ok(model);
        }

        // Fallback: automatic seeding (Σ + substrings)
        let mut pieces = self.seed_pieces(sentences);

        // Ensure Σ is included first (deterministic order). Insert any missing required char at front.
        for ch in required.iter() {
            if !pieces.iter().any(|(s, _)| s == ch) {
                pieces.insert(0, (ch.clone(), 0.0));
            }
        }

        // Prepend special tokens (kept; never deleted)
        let mut specials: Vec<SentencePiece> = self
            .special_tokens
            .iter()
            .map(|t| (t.content.clone(), 0.0))
            .collect();
        specials.append(&mut pieces);

        // Convert every score to -1.0 so maximizing total score gives **minimal number of tokens**
        let final_seed: Vec<(String, f64)> =
            specials.into_iter().map(|(s, _)| (s, -1.0)).collect();

        // No UNK during training; byte_fallback as requested.
        let model = Unigram::from(final_seed, /*unk_id*/ None, /*byte_fallback*/ self.byte_fallback)?;
        Ok(model)
    }

    #[inline]
    fn batch_size_for(&self, remaining: usize) -> usize {
        if remaining == 0 {
            return 0;
        }
        let mut k = if self.prune_ratio > 0.0 {
            ((remaining as f32) * self.prune_ratio).ceil() as usize
        } else {
            1
        };
        if k < self.min_prune {
            k = self.min_prune;
        }
        if k > remaining {
            k = remaining;
        }
        k
    }

    /// Main training logic; separated to keep `train` small.
    pub fn do_train(
        &self,
        sentences: Vec<Sentence>,
        model: &mut Unigram,
    ) -> Result<Vec<AddedToken>> {
        // 1) Prepare Σ and initial model
        let required = self.required_chars(&sentences); // Σ
        let mut current_model = self.build_initial_model(&sentences, &required)?;

        // Define non-deletable strings (Σ + special tokens)
        let non_deletable_strings: AHashSet<String> = required
            .iter()
            .cloned()
            .chain(self.special_tokens.iter().map(|t| t.content.clone()))
            .collect();

        let target = self.vocab_size as usize;

        if current_model.len() <= target {
            *model = current_model;
            return Ok(self.special_tokens.clone());
        }

        let method_name = if self.rand_scoring {
            "rand_compression"
        } else if self.use_expected_counts {
            "compression (expected counts)"
        } else {
            "compression"
        };
        if self.show_progress {
            eprintln!(
                "[CompressionTrainer] Starting ({}): {} sentences, {} initial vocab, {} target",
                method_name, sentences.len(), current_model.len(), target
            );
            if self.use_expected_counts {
                eprintln!("[CompressionTrainer] Temperature: {}", self.temperature);
            }
        }

        // Configure model for unit-cost mode with temperature
        current_model.set_unit_cost(true);
        current_model.set_temperature(self.temperature);

        // Branch based on scoring method
        if self.rand_scoring {
            // ==================== RAND_SCORING METHOD ====================
            return self.do_train_rand_scoring(sentences, model, current_model, non_deletable_strings, target);
        }

        // ==================== ORIGINAL METHOD (d[t] based) ====================

        // 2) Initial segmentation (parallel) - use expected counts or Viterbi
        if self.show_progress {
            if self.use_expected_counts {
                eprintln!("[CompressionTrainer] Computing expected counts (forward-backward)...");
            } else {
                eprintln!("[CompressionTrainer] Initial segmentation...");
            }
        }
        let mut ct_f64: Vec<f64> = if self.use_expected_counts {
            Self::compute_expected_counts_parallel(&current_model, &sentences)
        } else {
            Self::segment_corpus_parallel(&current_model, &sentences)
                .into_iter()
                .map(|c| c as f64)
                .collect()
        };

        // 3) Initial d[t] computation + track deps
        if self.show_progress {
            eprintln!("[CompressionTrainer] Computing initial d[t]...");
        }
        let mut dt: Vec<usize> = vec![1; current_model.len()];
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); current_model.len()];

        for t in 0..current_model.len() {
            if non_deletable_strings.contains(&current_model.vocab[t].0) {
                dt[t] = 1;
                continue;
            }
            let (d, used) = Self::compute_dt_with_deps(&current_model, t);
            dt[t] = d;
            deps[t] = used;
        }

        // 4) Batch deletion loop
        let mut pass = 0;

        while current_model.len() > target {
            pass += 1;
            let remaining = current_model.len() - target;
            let k_pass = self.batch_size_for(remaining);
            if k_pass == 0 {
                break;
            }

            // Compute ΔL(t) = c[t] * (d[t] - 1) for all deletable tokens
            // Use f64 for scoring to support expected counts
            let mut candidates: Vec<(u64, usize)> = Vec::new();
            for t in 0..current_model.len() {
                if non_deletable_strings.contains(&current_model.vocab[t].0) {
                    continue;
                }
                let c_t = ct_f64[t];
                let d_t = dt[t] as f64;
                let delta = c_t * (d_t - 1.0);
                // Convert to u64 for sorting (multiply by 1000 for precision)
                let delta_int = (delta * 1000.0).round().max(0.0) as u64;
                candidates.push((delta_int, t));
            }

            if candidates.is_empty() {
                break;
            }

            // Sort by ΔL ascending (lowest = best to delete)
            candidates.sort_by_key(|&(delta, _)| delta);

            // Get tokens to delete (by their string, since ids will change after rebuild)
            let to_delete_strings: AHashSet<String> = candidates
                .iter()
                .take(k_pass)
                .map(|&(_, t)| current_model.vocab[t].0.clone())
                .collect();

            // Save d[t] and deps by string BEFORE rebuilding model
            let mut dt_by_string: AHashMap<String, usize> = AHashMap::new();
            let mut deps_by_string: AHashMap<String, Vec<String>> = AHashMap::new();
            let mut need_recompute: AHashSet<String> = AHashSet::new();

            for t in 0..current_model.len() {
                let tok_str = &current_model.vocab[t].0;
                if to_delete_strings.contains(tok_str) {
                    continue;
                }
                dt_by_string.insert(tok_str.clone(), dt[t]);

                // Convert dep ids to strings and check if any dep was deleted
                let dep_strings: Vec<String> = deps[t]
                    .iter()
                    .map(|&id| current_model.vocab[id].0.clone())
                    .collect();

                let needs_recompute = dep_strings.iter().any(|s| to_delete_strings.contains(s));
                if needs_recompute {
                    need_recompute.insert(tok_str.clone());
                }
                deps_by_string.insert(tok_str.clone(), dep_strings);
            }

            // Build new model with remaining tokens
            let new_pieces: Vec<(String, f64)> = current_model
                .vocab
                .iter()
                .filter(|(s, _)| !to_delete_strings.contains(s))
                .map(|(s, _)| (s.clone(), -1.0))
                .collect();

            current_model = Unigram::from(new_pieces, None, self.byte_fallback)?;
            // Re-apply unit_cost and temperature to new model
            current_model.set_unit_cost(true);
            current_model.set_temperature(self.temperature);

            if self.show_progress {
                eprintln!(
                    "[CompressionTrainer] Pass {}: deleted {}, vocab_size={}",
                    pass, to_delete_strings.len(), current_model.len()
                );
            }

            if current_model.len() <= target {
                break;
            }

            // Re-segment corpus with new model (parallel) - use expected counts or Viterbi
            if self.show_progress {
                if self.use_expected_counts {
                    eprintln!("[CompressionTrainer] Pass {} expected counts...", pass);
                } else {
                    eprintln!("[CompressionTrainer] Pass {} segmentation...", pass);
                }
            }
            ct_f64 = if self.use_expected_counts {
                Self::compute_expected_counts_parallel(&current_model, &sentences)
            } else {
                Self::segment_corpus_parallel(&current_model, &sentences)
                    .into_iter()
                    .map(|c| c as f64)
                    .collect()
            };

            // Rebuild dt and deps arrays for new model
            dt = vec![1; current_model.len()];
            deps = vec![Vec::new(); current_model.len()];

            // Build string->id map for new model
            let str_to_id: AHashMap<&str, usize> = current_model
                .vocab
                .iter()
                .enumerate()
                .map(|(id, (s, _))| (s.as_str(), id))
                .collect();

            let recompute_count = need_recompute.len();
            if self.show_progress {
                eprintln!("[CompressionTrainer] Pass {} d[t] ({} to recompute)...", pass, recompute_count);
            }

            for t in 0..current_model.len() {
                let tok_str = &current_model.vocab[t].0;

                if non_deletable_strings.contains(tok_str) {
                    dt[t] = 1;
                    continue;
                }

                if need_recompute.contains(tok_str) {
                    // Recompute
                    let (d, used) = Self::compute_dt_with_deps(&current_model, t);
                    dt[t] = d;
                    deps[t] = used;
                } else {
                    // Reuse old values
                    if let Some(&old_d) = dt_by_string.get(tok_str) {
                        dt[t] = old_d;
                    }
                    if let Some(old_dep_strings) = deps_by_string.get(tok_str) {
                        // Convert strings back to new ids
                        deps[t] = old_dep_strings
                            .iter()
                            .filter_map(|s| str_to_id.get(s.as_str()).copied())
                            .collect();
                    }
                }
            }
        }

        *model = current_model;
        Ok(self.special_tokens.clone())
    }

    /// Training loop using rand_scoring method.
    /// Instead of computing d[t] from token strings, we sample spans that use each token
    /// and measure actual cost difference when resegmenting without that token.
    fn do_train_rand_scoring(
        &self,
        sentences: Vec<Sentence>,
        model: &mut Unigram,
        mut current_model: Unigram,
        non_deletable_strings: AHashSet<String>,
        target: usize,
    ) -> Result<Vec<AddedToken>> {
        let mut pass = 0;

        while current_model.len() > target {
            pass += 1;
            let remaining = current_model.len() - target;
            let k_pass = self.batch_size_for(remaining);
            if k_pass == 0 {
                break;
            }

            // Segment corpus and build reverse index (always uses Viterbi for span tracking)
            if self.show_progress {
                eprintln!("[CompressionTrainer] Pass {} segmentation with index...", pass);
            }
            let (ct_viterbi, reverse_index, segmentations) =
                Self::segment_corpus_with_index(&current_model, &sentences);

            // Get counts for scoring - use expected counts if enabled, otherwise Viterbi counts
            let ct: Vec<f64> = if self.use_expected_counts {
                if self.show_progress {
                    eprintln!("[CompressionTrainer] Pass {} computing expected counts...", pass);
                }
                Self::compute_expected_counts_parallel(&current_model, &sentences)
            } else {
                ct_viterbi.iter().map(|&c| c as f64).collect()
            };

            // Compute scores using random sampling
            if self.show_progress {
                eprintln!("[CompressionTrainer] Pass {} computing rand scores (sample_size={})...", pass, self.rand_sample_size);
            }
            let scores = Self::compute_rand_scores_f64(
                &current_model,
                &sentences,
                &ct,
                &reverse_index,
                &segmentations,
                self.rand_sample_size,
                &non_deletable_strings,
            );

            // Build candidates list (score, token_id)
            let mut candidates: Vec<(u64, usize)> = Vec::new();
            for t in 0..current_model.len() {
                if non_deletable_strings.contains(&current_model.vocab[t].0) {
                    continue;
                }
                // Convert f64 score to u64 for sorting (multiply by 1000 for precision)
                let score_int = (scores[t] * 1000.0).round() as u64;
                candidates.push((score_int, t));
            }

            if candidates.is_empty() {
                break;
            }

            // Sort by score ascending (lowest = best to delete)
            candidates.sort_by_key(|&(score, _)| score);

            // Get tokens to delete
            let to_delete_strings: AHashSet<String> = candidates
                .iter()
                .take(k_pass)
                .map(|&(_, t)| current_model.vocab[t].0.clone())
                .collect();

            // Build new model with remaining tokens
            let new_pieces: Vec<(String, f64)> = current_model
                .vocab
                .iter()
                .filter(|(s, _)| !to_delete_strings.contains(s))
                .map(|(s, _)| (s.clone(), -1.0))
                .collect();

            current_model = Unigram::from(new_pieces, None, self.byte_fallback)?;
            // Re-apply unit_cost and temperature to new model
            current_model.set_unit_cost(true);
            current_model.set_temperature(self.temperature);

            if self.show_progress {
                eprintln!(
                    "[CompressionTrainer] Pass {}: deleted {}, vocab_size={}",
                    pass, to_delete_strings.len(), current_model.len()
                );
            }
        }

        *model = current_model;
        Ok(self.special_tokens.clone())
    }
}

// ----------------------------- Trainer impl -----------------------------------

impl Trainer for CompressionTrainer {
    type Model = Unigram;

    /// Public training entry point; delegates to `do_train`.
    fn train(&self, model: &mut Unigram) -> Result<Vec<AddedToken>> {
        let sentences: Vec<_> = self.words.iter().map(|(s, i)| (s.to_owned(), *i)).collect();
        self.do_train(sentences, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    /// Collect the word/span counts from an iterator, mirroring `UnigramTrainer::feed`.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        // For each input sequence, `process` returns a vector of "words"/spans.
        // We count their frequencies and merge across the dataset, in parallel when available.
        let words: Result<AHashMap<String, u32>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(word).or_default() += 1;
                }
                Ok(map)
            })
            .reduce(
                || Ok(AHashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        *acc.entry(k).or_default() += v;
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}