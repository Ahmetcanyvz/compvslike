#![allow(clippy::map_entry)]

use super::{BpeTelemetry, MergeEvent, Pair, ScoreItem, ScoreSnapshot, WithFirstLastIterator, Word, BPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

/// How to rank candidate merges in the heap.
///
/// * `Count`           — original BPE: pick the most frequent pair.
/// * `GreedyLLExact`   — rank by exact ΔLL.
/// * `GreedyLLApprox`  — rank by approx ΔLL (PMI-like).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BpeScoreBy {
    #[serde(rename = "count")]
    Count,
    #[serde(rename = "greedy_ll_exact")]
    GreedyLLExact,
    #[serde(rename = "greedy_ll_approx")]
    GreedyLLApprox,
}

/// When to stop training.
///
/// * `VocabSize`       — stop when vocab size hits `vocab_size` (original behavior).
/// * `DeltaLLExact`    — additionally stop when best exact ΔLL ≤ 0.
/// * `DeltaLLApprox`   — additionally stop when best approx ΔLL ≤ 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BpeStopBy {
    #[serde(rename = "vocab_size")]
    VocabSize,
    #[serde(rename = "delta_ll_exact")]
    DeltaLLExact,
    #[serde(rename = "delta_ll_approx")]
    DeltaLLApprox,
}

#[inline]
fn xlogx(x: u64) -> f64 {
    if x == 0 {
        0.0
    } else {
        let f = x as f64;
        f * f.ln()
    }
}

/// Exact ΔLL for merging (b,c) with counts measured on the current stream.
#[inline]
fn delta_ll_exact(nb: u64, nc: u64, nbc: u64, n: u64) -> f64 {
    // ΔLL(b,c) = (nb - nbc)log(nb - nbc) - nb log nb
    //          + (nc - nbc)log(nc - nbc) - nc log nc
    //          + nbc log nbc
    //          - (N - nbc)log(N - nbc) + N log N
    xlogx(nb.saturating_sub(nbc))
        - xlogx(nb)
        + xlogx(nc.saturating_sub(nbc))
        - xlogx(nc)
        + xlogx(nbc)
        - xlogx(n.saturating_sub(nbc))
        + xlogx(n)
}

/// First-order Taylor approximation of exact ΔLL around small n_bc:
///   n_bc * ( log( n_bc * N / (n_b * n_c) ) - 1 )  =  n_bc * (PMI - 1)
#[inline]
fn delta_ll_approx(nb: u64, nc: u64, nbc: u64, n: u64) -> f64 {
    if nbc == 0 || nb == 0 || nc == 0 {
        0.0
    } else {
        let num = (nbc as f64) * (n as f64);
        let den = (nb as f64) * (nc as f64);
        (nbc as f64) * ((num / den).ln() - 1.0)
    }
}

#[inline]
fn score_of(policy: BpeScoreBy, nb: u64, nc: u64, nbc: u64, n: u64) -> f64 {
    match policy {
        BpeScoreBy::Count => nbc as f64,
        BpeScoreBy::GreedyLLExact => delta_ll_exact(nb, nc, nbc, n),
        BpeScoreBy::GreedyLLApprox => delta_ll_approx(nb, nc, nbc, n),
    }
}

/// Heap element: pair + count + score + positions.
#[derive(Debug)]
struct Merge {
    pair: Pair,
    count: u64,
    score: f64, // ranking key according to `scoring`
    pos: AHashSet<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
            && self.pair == other.pair
            && self.score.to_bits() == other.score.to_bits()
    }
}
impl Eq for Merge {}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by score; when equal, use original deterministic tie-breaker on pair.
        match self
            .score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => other.pair.cmp(&self.pair),
            ord => ord,
        }
    }
}

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: AHashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,

    // NEW:
    scoring: BpeScoreBy,
    stop_by: BpeStopBy,
    track_ll: bool,
    score_snapshot_every: Option<usize>,
    score_sample_size: Option<usize>,
}

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,
                vocab_size: 30000,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: AHashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,

                // defaults: preserve original behavior
                scoring: BpeScoreBy::Count,
                stop_by: BpeStopBy::VocabSize,
                track_ll: false,
                score_snapshot_every: None,
                score_sample_size: None,
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut initial_alphabet = AHashSet::with_capacity(alphabet.len());
        initial_alphabet.extend(alphabet);
        self.config.initial_alphabet = initial_alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }

    /// Set max_token_length
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    /// Set scoring policy
    #[must_use]
    pub fn score_by(mut self, scoring: BpeScoreBy) -> Self {
        self.config.scoring = scoring;
        self
    }

    /// Set stopping policy
    #[must_use]
    pub fn stop_by(mut self, stop_by: BpeStopBy) -> Self {
        self.config.stop_by = stop_by;
        self
    }

    /// Enable / disable LL tracking (API only; current trainer ignores this flag)
    #[must_use]
    pub fn track_ll(mut self, track_ll: bool) -> Self {
        self.config.track_ll = track_ll;
        self
    }

    /// Configure how often to snapshot scores (API only; trainer currently ignores this)
    #[must_use]
    pub fn score_snapshot_every(mut self, step: Option<usize>) -> Self {
        self.config.score_snapshot_every = step;
        self
    }

    /// Configure how many scores to keep per snapshot (API only; trainer currently ignores this)
    #[must_use]
    pub fn score_sample_size(mut self, sz: Option<usize>) -> Self {
        self.config.score_sample_size = sz;
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,

            scoring: self.config.scoring,
            stop_by: self.config.stop_by,
            track_ll: self.config.track_ll,
            score_snapshot_every: self.config.score_snapshot_every,
            score_sample_size: self.config.score_sample_size,

            words: AHashMap::new(),
        }
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::{BPE, BpeTrainer};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = BPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    pub min_frequency: u64,
    /// The target vocabulary size
    pub vocab_size: usize,
    /// Whether to show progress while training
    pub show_progress: bool,
    /// A list of special tokens that the model should know of
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    pub initial_alphabet: AHashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An optional parameter to limit the max length of any single token
    pub max_token_length: Option<usize>,

    /// How to rank pairs in the heap (default: Count)
    pub scoring: BpeScoreBy,
    /// When to stop training (default: VocabSize)
    pub stop_by: BpeStopBy,
    /// Whether to track LL (API only; currently not used in trainer)
    pub track_ll: bool,
    /// Score snapshot frequency (API only; currently not used)
    pub score_snapshot_every: Option<usize>,
    /// Number of scores per snapshot (API only; currently not used)
    pub score_sample_size: Option<usize>,

    words: AHashMap<CompactString, u64>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(
        &self,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for token in &self.special_tokens {
            // get hash of content
            if !w2id.contains_key(&CompactString::from(&token.content)) {
                id2w.push(CompactString::from(&token.content));
                w2id.insert(CompactString::from(&token.content), (id2w.len() - 1) as u32);
            }
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        // Compute the alphabet from seen words
        let mut alphabet: AHashMap<char, usize> = AHashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += *count as usize;
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            *alphabet.entry(*c).or_default() = usize::MAX;
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| alphabet.len().saturating_sub(limit))
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| *k.0 as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            /*
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
            */
            // u64 hash version
            if !w2id.contains_key(&CompactString::from(&s)) {
                id2w.push(CompactString::from(&s));
                w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
            }
        });
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&CompactString::from(&s)) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s.insert_str(0, prefix);
                        }
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s.push_str(suffix);
                        }
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&CompactString::from(&s)) {
                        id2w.push(CompactString::from(&s));
                        w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&CompactString::from(&s)], 1); // We do not care about the len here
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    // Then update counts
                    *pair_counts.entry(cur_pair).or_default() += counts[i] as i32;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    pub fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        // Initialize telemetry if tracking is enabled
        let mut telemetry = if self.track_ll {
            Some(BpeTelemetry::default())
        } else {
            None
        };
        let snapshot_every = self.score_snapshot_every.unwrap_or(0);
        let snapshot_sample_size = self.score_sample_size.unwrap_or(10_000);

        let mut word_to_id: AHashMap<CompactString, u32> = AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<CompactString> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 3. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(&progress, words.len());

        //
        // 3.5. Compute initial symbol marginals and total tokens (once)
        //
        let mut sym_counts: AHashMap<u32, u64> = AHashMap::new();
        let mut total_tokens: u64 = 0;
        for (w, &cnt) in words.iter().zip(&counts) {
            let chars = w.get_chars();
            total_tokens += (chars.len() as u64) * cnt as u64;
            for c in chars {
                *sym_counts.entry(c).or_default() += cnt as u64;
            }
        }

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);

        // Build reverse index: symbol -> set of pairs containing that symbol
        // This is needed for ΔLL-based scoring to update affected pairs when symbol counts change
        let mut symbol_to_pairs: AHashMap<u32, AHashSet<Pair>> = AHashMap::new();

        // Maintain pair -> positions mapping separately from the heap
        // This allows us to re-push pairs with correct positions when scores change
        let mut pair_to_pos: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

        // Insert them in the queue (with score according to the chosen policy)
        let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                let nb = *sym_counts.get(&pair.0).unwrap_or(&0);
                let nc = *sym_counts.get(&pair.1).unwrap_or(&0);
                let score = score_of(self.scoring, nb, nc, count as u64, total_tokens);
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    score,
                    pos: pos.clone(),
                });
                // Store positions separately
                pair_to_pos.insert(pair, pos);
                // Add to reverse index
                symbol_to_pairs.entry(pair.0).or_default().insert(pair);
                symbol_to_pairs.entry(pair.1).or_default().insert(pair);
            }
        });
        self.finalize_progress(&progress, words.len());

        //
        // 5. Do merges
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];
        let mut merge_step: u32 = 0;
        loop {
            // Hard cap: never grow vocab beyond vocab_size
            if word_to_id.len() >= self.vocab_size {
                break;
            }

            let Some(mut top) = queue.pop() else {
                break;
            };

            // Lazy refresh of count & score for the popped top
            // Note: pair_counts uses i32, so we must handle negative values (treat as 0)
            let cur_count = pair_counts
                .get(&top.pair)
                .copied()
                .unwrap_or(0)
                .max(0) as u64;
            if cur_count == 0 {
                // Pair disappeared, skip
                continue;
            }
            let nb_now = *sym_counts.get(&top.pair.0).unwrap_or(&0);
            let nc_now = *sym_counts.get(&top.pair.1).unwrap_or(&0);
            let cur_score = score_of(self.scoring, nb_now, nc_now, cur_count, total_tokens);

            if cur_count != top.count || (cur_score - top.score).abs() > 1e-12 {
                top.count = cur_count;
                top.score = cur_score;
                queue.push(top);
                continue;
            }

            // If using ΔLL-based stopping, stop when best achievable ΔLL ≤ 0.
            if !matches!(self.stop_by, BpeStopBy::VocabSize) {
                let stop_score = match self.stop_by {
                    BpeStopBy::VocabSize => f64::INFINITY,
                    BpeStopBy::DeltaLLExact => {
                        delta_ll_exact(nb_now, nc_now, cur_count, total_tokens)
                    }
                    BpeStopBy::DeltaLLApprox => {
                        delta_ll_approx(nb_now, nc_now, cur_count, total_tokens)
                    }
                };
                if stop_score <= 0.0 {
                    break;
                }
            }

            if top.count < 1 {
                // Pair count is zero or negative, skip it
                continue;
            }
            if self.min_frequency > top.count {
                // For count-based scoring, if the top pair is below min_frequency,
                // all remaining pairs will also be below (since heap is sorted by count).
                // For ΔLL-based scoring, other pairs might still be above min_frequency,
                // so we should continue instead of break.
                if matches!(self.scoring, BpeScoreBy::Count) {
                    break;
                } else {
                    continue;
                }
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].as_str();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix {
                if let Some(rest) = part_b.strip_prefix(prefix) {
                    part_b = rest;
                }
            }

            // Insert new token if it does not already exist
            let new_token = format!("{part_a}{part_b}");
            let new_token_id = word_to_id
                .get(&CompactString::from(&new_token))
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&CompactString::from(&new_token)) {
                id_to_word.push(CompactString::from(&new_token));
                word_to_id.insert(CompactString::from(&new_token), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // Record telemetry for this merge
            if let Some(ref mut tel) = telemetry {
                // Compute exact ΔLL for telemetry (even if scoring uses approx)
                let delta_ll_value = delta_ll_exact(nb_now, nc_now, cur_count, total_tokens);
                tel.merge_trace.push(MergeEvent {
                    step: merge_step,
                    pair: top.pair,
                    new_id: new_token_id,
                    count: cur_count,
                    score: cur_score,
                    delta_ll: Some(delta_ll_value),
                    total_tokens: Some(total_tokens),
                    n_a: Some(nb_now),
                    n_b: Some(nc_now),
                });

                // Take score snapshot periodically
                if snapshot_every > 0 && merge_step % snapshot_every as u32 == 0 {
                    // Sample top candidates from the queue
                    // We need to peek at the heap without modifying it, so we collect and re-push
                    let mut sampled_items: Vec<ScoreItem> = Vec::new();
                    let mut temp_popped: Vec<Merge> = Vec::new();

                    // Pop up to snapshot_sample_size items
                    for _ in 0..snapshot_sample_size {
                        if let Some(m) = queue.pop() {
                            // Check if this entry is still valid
                            let pc = pair_counts.get(&m.pair).copied().unwrap_or(0);
                            if pc > 0 {
                                sampled_items.push(ScoreItem {
                                    pair: m.pair,
                                    score: m.score,
                                    count: m.count,
                                });
                            }
                            temp_popped.push(m);
                        } else {
                            break;
                        }
                    }

                    // Re-push all popped items
                    for m in temp_popped {
                        queue.push(m);
                    }

                    tel.score_snapshots.push(ScoreSnapshot {
                        step: merge_step,
                        items: sampled_items,
                    });
                }
            }
            merge_step += 1;

            // Merge the new pair in every word
            // Use pair_to_pos as the authoritative source for positions
            let pos = pair_to_pos.get(&top.pair).cloned().unwrap_or_default();

            let words_len = words.len();
            struct WordPtr(*mut Word);
            // Safety: We do not actually use this for concurrent access to the same memory,
            // only to different chunks within the same allocation.
            unsafe impl Sync for WordPtr {}
            let word_start = WordPtr(words.as_mut_ptr());

            // Collect both changes and merge counts from each word
            let results: Vec<(Vec<((Pair, i32), usize)>, u64)> = (&pos)
                .maybe_par_iter()
                .map(|&i| {
                    // We can merge each of these words in parallel here because each position
                    // can be there only once (AHashSet). So this is safe.
                    unsafe {
                        assert!(i < words_len);
                        // This is words[i], but avoids needing to go through &T (which triggers UB)
                        let word = word_start.0.add(i);
                        let (changes, merge_count) =
                            (*word).merge(top.pair.0, top.pair.1, new_token_id, max_token_length);
                        let word_freq = counts[i];
                        // Actual token reduction = merge_count * word_frequency
                        let token_reduction = (merge_count as u64) * word_freq;
                        let changes_with_word_idx: Vec<((Pair, i32), usize)> =
                            changes.into_iter().map(|c| (c, i)).collect();
                        (changes_with_word_idx, token_reduction)
                    }
                })
                .collect();

            // Aggregate changes and total token reduction
            let mut changes: Vec<((Pair, i32), usize)> = Vec::new();
            let mut applied_bc: u64 = 0;
            for (word_changes, token_reduction) in results {
                changes.extend(word_changes);
                applied_bc += token_reduction;
            }

            // Introduce new formed pairs & update pair counts
            for ((pair, change), iw) in changes {
                let count_delta = change * counts[iw] as i32;
                *pair_counts.entry(pair).or_default() += count_delta;
                if change > 0 {
                    // Only pairs whose counts increased need a refreshed 'pos'
                    where_to_update.entry(pair).or_default().insert(iw);
                }
            }

            // Remove the merged pair from pair_counts entirely (it's been fully consumed)
            pair_counts.remove(&top.pair);
            // Update symbol marginals and total token count incrementally
            if applied_bc > 0 {
                if let Some(v) = sym_counts.get_mut(&top.pair.0) {
                    *v = v.saturating_sub(applied_bc);
                }
                if let Some(v) = sym_counts.get_mut(&top.pair.1) {
                    *v = v.saturating_sub(applied_bc);
                }
                let new_sym_count = sym_counts.entry(new_token_id).or_default();
                *new_sym_count = new_sym_count.saturating_add(applied_bc);
                total_tokens = total_tokens.saturating_sub(applied_bc);
            }

            // Remove the merged pair from pair_to_pos and reverse index
            pair_to_pos.remove(&top.pair);
            if let Some(set) = symbol_to_pairs.get_mut(&top.pair.0) {
                set.remove(&top.pair);
            }
            if let Some(set) = symbol_to_pairs.get_mut(&top.pair.1) {
                set.remove(&top.pair);
            }

            // For ΔLL-based scoring: recompute scores for all pairs affected by n_x/n_y change
            // When we merged (x,y), both n_x and n_y decreased, which affects scores of all pairs
            // containing x or y (since their scores depend on symbol marginals)
            if !matches!(self.scoring, BpeScoreBy::Count) && applied_bc > 0 {
                let empty_set = AHashSet::new();
                let pairs_with_x = symbol_to_pairs.get(&top.pair.0).unwrap_or(&empty_set);
                let pairs_with_y = symbol_to_pairs.get(&top.pair.1).unwrap_or(&empty_set);

                // Collect affected pairs (excluding the pair we just merged)
                for &affected_pair in pairs_with_x.iter().chain(pairs_with_y.iter()) {
                    if affected_pair == top.pair {
                        continue; // Skip the merged pair itself
                    }
                    let count = pair_counts.get(&affected_pair).copied().unwrap_or(0).max(0) as u64;
                    if count > 0 {
                        let na = *sym_counts.get(&affected_pair.0).unwrap_or(&0);
                        let nb = *sym_counts.get(&affected_pair.1).unwrap_or(&0);
                        let score = score_of(self.scoring, na, nb, count, total_tokens);
                        // Re-push with updated score; pos is looked up from pair_to_pos when needed
                        let affected_pos = pair_to_pos.get(&affected_pair).cloned().unwrap_or_default();
                        queue.push(Merge {
                            pair: affected_pair,
                            count,
                            score,
                            pos: affected_pos,
                        });
                    }
                }
            }

            // Reinsert newly formed/changed pairs with fresh scores
            where_to_update.drain().for_each(|(pair, new_pos)| {
                let count = pair_counts.get(&pair).copied().unwrap_or(0).max(0) as u64;
                if count > 0 {
                    let nb = *sym_counts.get(&pair.0).unwrap_or(&0);
                    let nc = *sym_counts.get(&pair.1).unwrap_or(&0);
                    let score = score_of(self.scoring, nb, nc, count, total_tokens);
                    // Update pair_to_pos with the new positions
                    pair_to_pos.entry(pair).or_default().extend(new_pos.iter().cloned());
                    let pos = pair_to_pos.get(&pair).cloned().unwrap_or_default();
                    queue.push(Merge {
                        pair,
                        count,
                        score,
                        pos,
                    });
                    // Update reverse index for new pairs
                    symbol_to_pairs.entry(pair.0).or_default().insert(pair);
                    symbol_to_pairs.entry(pair.1).or_default().insert(pair);
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }

            // Periodic heap rebuild to eliminate stale entries
            // This prevents unbounded heap growth in ΔLL-based scoring modes
            const REBUILD_INTERVAL: u32 = 1_000;
            if merge_step % REBUILD_INTERVAL == 0 {
                // Rebuild heap from scratch using current pair_counts
                queue = OctonaryHeap::with_capacity(pair_counts.len());
                for (&pair, &count) in pair_counts.iter() {
                    if count > 0 {
                        let nb = *sym_counts.get(&pair.0).unwrap_or(&0);
                        let nc = *sym_counts.get(&pair.1).unwrap_or(&0);
                        let score = score_of(self.scoring, nb, nc, count as u64, total_tokens);
                        let pos = pair_to_pos.get(&pair).cloned().unwrap_or_default();
                        queue.push(Merge {
                            pair,
                            count: count as u64,
                            score,
                            pos,
                        });
                    }
                }
            }
        }
        self.finalize_progress(&progress, merges.len());

        // Transfer new vocab & options to model
        model.vocab = word_to_id
            .into_iter()
            // we have to look up the string in id_to_word because the key in word_to_id is a hash
            .map(|(_key, val)| (id_to_word[val as usize].to_string(), val))
            .collect();
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();

        model.continuing_subword_prefix = self.continuing_subword_prefix.clone();
        model.end_of_word_suffix = self.end_of_word_suffix.clone();

        // Attach telemetry to model if it was collected
        if let Some(tel) = telemetry {
            model.telemetry = Some(tel);
        }

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for BpeTrainer {
    type Model = BPE;

    /// Train a BPE model
    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<AHashMap<CompactString, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(CompactString::from(word)).or_default() += 1;
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

#[cfg(test)]
mod tests {
    use super::{BpeTrainer, Pair, BPE};
    use ahash::AHashMap;
    use compact_str::CompactString;

    #[test]
    fn test_train() {
        let word_counts: AHashMap<CompactString, u64> = [
            ("roses".into(), 1),
            ("are".into(), 2),
            ("red".into(), 1),
            ("voilets".into(), 1),
            ("blue".into(), 1),
            ("BERT".into(), 1),
            ("is".into(), 2),
            ("big".into(), 1),
            ("and".into(), 1),
            ("so".into(), 1),
            ("GPT-2".into(), 1),
        ]
        .iter()
        .cloned()
        .collect();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(2)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&word_counts, &mut model).unwrap();

        // Vocab should contain all of the characters from the `word_counts` mapping
        // as well as three merges: 're', 'are', and 'is'.
        let expected_vocab: AHashMap<String, u32> = [
            ("-".into(), 0),
            ("2".into(), 1),
            ("B".into(), 2),
            ("E".into(), 3),
            ("G".into(), 4),
            ("P".into(), 5),
            ("R".into(), 6),
            ("T".into(), 7),
            ("a".into(), 8),
            ("b".into(), 9),
            ("d".into(), 10),
            ("e".into(), 11),
            ("g".into(), 12),
            ("i".into(), 13),
            ("l".into(), 14),
            ("n".into(), 15),
            ("o".into(), 16),
            ("r".into(), 17),
            ("s".into(), 18),
            ("t".into(), 19),
            ("u".into(), 20),
            ("v".into(), 21),
            ("re".into(), 22),
            ("are".into(), 23),
            ("is".into(), 24),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.vocab, expected_vocab);

        // The keys in `merges` are pairs of symbols, the values are tuples of (rank, id),
        // where 'rank' determines the order in which this merge will be applied during
        // tokenization, and 'id' is the vocab id of the symbol resulting from merging
        // the pair of symbols in the corresponding key.
        let expected_merges: AHashMap<Pair, (u32, u32)> = [
            ((17, 11), (0, 22)), // 'r' + 'e'  -> 're'
            ((8, 22), (1, 23)),  // 'a' + 're' -> 'are'
            ((13, 18), (2, 24)), // 'i' + 's'  -> 'is'
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.merges, expected_merges);
    }

    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length series of tests test the max_token_length flag of bpetrainer
        // this is the more robust version that only tests max length of learned tokens
        // (pre) tokenizer settings or vocab can be easily modified when necessary
         */

        let max_token_length = 16;
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("singlelongtokenwithoutcasechange", 2),
            ("singleLongTokenWithCamelCaseChange", 2),
            ("Longsingletokenwithpunctu@t!onwithin", 2),
            ("Anotherlongsingletokenwithnumberw1th1n", 2),
            ("짧은한글문자열짧은한", 2),             // korean 10 char
            ("긴한글문자열긴한글문자열긴한글문", 2), // korean 16 char
            ("短字符串短字符串短字", 2),             //simplified chinese 10 char
            ("长字符串长字符串长字符串长字符串", 2), // simp. chinese 16 char
            ("短い文字列短い文字列", 2),             // japanese 10 char
            ("長い文字列長い文字列長い文字列長", 2), // japanese 16 char
            ("so", 2),
            ("GPT-2", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let vocab = model.get_vocab();
        for token in vocab.keys() {
            assert!(
                token.chars().count() <= max_token_length,
                "token too long : {} , chars().count() = {}",
                token,
                token.chars().count()
            )
        }
    }

    #[test]
    fn bpe_test_max_token_length_direct_assert() {
        /* more direct version of bpe_test_max_token_length test
        // directly compares tokens with known expected values.
        // maybe unstable depending on specific settings or changes.
         */
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(2))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let trained_vocab: AHashMap<String, u32> = model.get_vocab().into_iter().collect();
        let expected_vocab: AHashMap<String, u32> = [
            ("短", 12),
            ("n", 6),
            ("i", 5),
            ("s", 8),
            ("字符", 23),
            ("長", 14),
            ("긴", 17),
            ("い文", 22),
            ("L", 2),
            ("in", 21),
            ("o", 7),
            ("은한", 29),
            ("S", 4),
            ("P", 3),
            ("so", 27),
            ("符", 13),
            ("文", 11),
            ("字", 10),
            ("짧", 19),
            ("GP", 25),
            ("글", 16),
            ("G", 1),
            ("An", 24),
            ("长", 15),
            ("A", 0),
            ("Lo", 26),
            ("긴한", 28),
            ("い", 9),
            ("한", 20),
            ("은", 18),
        ]
        .iter()
        .cloned()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(trained_vocab, expected_vocab)
    }
}