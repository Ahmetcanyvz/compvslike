//! [Unigram](https://arxiv.org/abs/1804.10959) model.
// Core building blocks that already existed
mod lattice;
mod model;
mod serialization;
mod trainer;
mod trie;

// NEW: our compression-based trainer (greedy deletion with affected-sentences-only resegmentation)
mod compression_trainer; // add the module so it can be used from outside

// Re-export public items so users can `use tokenizers::models::unigram::*;`
pub use lattice::*;                    // expose Lattice & related types
pub use model::*;                      // expose Unigram model
pub use trainer::*;                    // expose original EM-based UnigramTrainer
pub use compression_trainer::CompressionTrainer; // NEW: expose the compression trainer
