use std::sync::{Arc, RwLock};

use crate::models::PyModel;
use crate::tokenizer::PyAddedToken;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};
use tk::models::TrainerWrapper;
use tk::Trainer;
use tokenizers as tk;

// Match your Rust trainer enums
use tk::models::bpe::trainer::{BpeScoreBy, BpeStopBy};

/// Base class for all trainers
#[pyclass(module = "tokenizers.trainers", name = "Trainer", subclass)]
#[derive(Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct PyTrainer {
    pub trainer: Arc<RwLock<TrainerWrapper>>,
}

impl PyTrainer {
    #[cfg(test)]
    pub(crate) fn new(trainer: Arc<RwLock<TrainerWrapper>>) -> Self {
        PyTrainer { trainer }
    }
    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        let base = self.clone();
        Ok(match *self.trainer.as_ref().read().unwrap() {
            TrainerWrapper::BpeTrainer(_) => Py::new(py, (PyBpeTrainer {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            TrainerWrapper::WordPieceTrainer(_) => Py::new(py, (PyWordPieceTrainer {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            TrainerWrapper::WordLevelTrainer(_) => Py::new(py, (PyWordLevelTrainer {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            TrainerWrapper::UnigramTrainer(_) => Py::new(py, (PyUnigramTrainer {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            // NEW: surface the CompressionTrainer subtype in Python
            TrainerWrapper::CompressionTrainer(_) => Py::new(py, (PyCompressionTrainer {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
        })
    }
}

#[pymethods]
impl PyTrainer {
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.trainer).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle PyTrainer: {e}"
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                let unpickled = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle PyTrainer: {e}"
                    ))
                })?;
                self.trainer = unpickled;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }
}

impl Trainer for PyTrainer {
    type Model = PyModel;

    fn should_show_progress(&self) -> bool {
        self.trainer.read().unwrap().should_show_progress()
    }

    fn train(&self, model: &mut PyModel) -> tk::Result<Vec<tk::AddedToken>> {
        self.trainer
            .read()
            .unwrap()
            .train(&mut model.model.write().unwrap())
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> tk::Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> tk::Result<Vec<String>> + Sync,
    {
        self.trainer.write().unwrap().feed(iterator, process)
    }
}

impl<I> From<I> for PyTrainer
where
    I: Into<TrainerWrapper>,
{
    fn from(trainer: I) -> Self {
        PyTrainer {
            trainer: Arc::new(RwLock::new(trainer.into())),
        }
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        if let TrainerWrapper::$variant(ref trainer) = *super_.trainer.read().unwrap() {
            trainer.$($name)+
        } else {
            unreachable!()
        }
    }};
}

macro_rules! setter {
    ($self: ident, $variant: ident, $name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let TrainerWrapper::$variant(ref mut trainer) = *super_.trainer.write().unwrap() {
            trainer.$name = $value;
        }
    }};
    ($self: ident, $variant: ident, @$name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let TrainerWrapper::$variant(ref mut trainer) = *super_.trainer.write().unwrap() {
            trainer.$name($value);
        }
    }};
}

// ----------------- helpers: map enums <-> Python -----------------

fn score_by_from_py(s: &str) -> PyResult<BpeScoreBy> {
    match s.to_ascii_lowercase().as_str() {
        "count" | "bpe" | "frequency" => Ok(BpeScoreBy::Count),
        "exact_ll" | "greedy_ll_exact" | "exact" => Ok(BpeScoreBy::GreedyLLExact),
        "approx_ll" | "greedy_ll_approx" | "approx" => Ok(BpeScoreBy::GreedyLLApprox),
        _ => Err(exceptions::PyValueError::new_err(format!(
            "Invalid score_by: {s}. Expected one of 'count' | 'exact_ll' | 'approx_ll'"
        ))),
    }
}

fn score_by_to_py(s: &BpeScoreBy) -> &'static str {
    match s {
        BpeScoreBy::Count => "count",
        BpeScoreBy::GreedyLLExact => "exact_ll",
        BpeScoreBy::GreedyLLApprox => "approx_ll",
    }
}

fn stop_by_from_py(obj: &Bound<'_, PyAny>) -> PyResult<BpeStopBy> {
    let s: String = obj.extract()?;
    match s.to_ascii_lowercase().as_str() {
        "vocab" | "vocab_size" => Ok(BpeStopBy::VocabSize),
        "delta_ll_exact" | "exact_ll_stop" | "dll_exact" => Ok(BpeStopBy::DeltaLLExact),
        "delta_ll_approx" | "approx_ll_stop" | "dll_approx" => Ok(BpeStopBy::DeltaLLApprox),
        other => Err(exceptions::PyValueError::new_err(format!(
            "Invalid stop_by: {other}. Expected 'vocab_size' | 'delta_ll_exact' | 'delta_ll_approx'"
        ))),
    }
}

fn stop_by_to_py<'py>(py: Python<'py>, s: &BpeStopBy) -> PyObject {
    let tag = match s {
        BpeStopBy::VocabSize => "vocab_size",
        BpeStopBy::DeltaLLExact => "delta_ll_exact",
        BpeStopBy::DeltaLLApprox => "delta_ll_approx",
    };
    PyString::new(py, tag).into_any().unbind().into()
}

/// Trainer capable of training a BPE model
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "BpeTrainer")]
pub struct PyBpeTrainer {}

#[pymethods]
impl PyBpeTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> usize {
        getter!(self_, BpeTrainer, vocab_size)
    }
    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: usize) {
        setter!(self_, BpeTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_min_frequency(self_: PyRef<Self>) -> u64 {
        getter!(self_, BpeTrainer, min_frequency)
    }
    #[setter]
    fn set_min_frequency(self_: PyRef<Self>, freq: u64) {
        setter!(self_, BpeTrainer, min_frequency, freq);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, BpeTrainer, show_progress)
    }
    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, BpeTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            BpeTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }
    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            BpeTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_limit_alphabet(self_: PyRef<Self>) -> Option<usize> {
        getter!(self_, BpeTrainer, limit_alphabet)
    }
    #[setter]
    fn set_limit_alphabet(self_: PyRef<Self>, limit: Option<usize>) {
        setter!(self_, BpeTrainer, limit_alphabet, limit);
    }

    #[getter]
    fn get_max_token_length(self_: PyRef<Self>) -> Option<usize> {
        getter!(self_, BpeTrainer, max_token_length)
    }
    #[setter]
    fn set_max_token_length(self_: PyRef<Self>, limit: Option<usize>) {
        setter!(self_, BpeTrainer, max_token_length, limit);
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            BpeTrainer,
            initial_alphabet.iter().map(|c| c.to_string()).collect()
        )
    }
    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            BpeTrainer,
            initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BpeTrainer, continuing_subword_prefix.clone())
    }
    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, prefix: Option<String>) {
        setter!(self_, BpeTrainer, continuing_subword_prefix, prefix);
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BpeTrainer, end_of_word_suffix.clone())
    }
    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, suffix: Option<String>) {
        setter!(self_, BpeTrainer, end_of_word_suffix, suffix);
    }

    // ----- scoring / stop_by / track_ll -----

    #[getter]
    fn get_score_by(self_: PyRef<Self>) -> PyResult<String> {
        let super_ = self_.as_ref();
        let guard = super_.trainer.read().unwrap();
        if let TrainerWrapper::BpeTrainer(ref tr) = *guard {
            Ok(score_by_to_py(&tr.scoring).to_string())
        } else {
            unreachable!()
        }
    }
    #[setter]
    fn set_score_by(self_: PyRef<Self>, how: &str) -> PyResult<()> {
        let sb = score_by_from_py(how)?;
        let super_ = self_.as_ref();
        if let TrainerWrapper::BpeTrainer(ref mut tr) = *super_.trainer.write().unwrap() {
            tr.scoring = sb;
        }
        Ok(())
    }

    #[getter]
    fn get_stop_by<'py>(self_: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyObject> {
        let super_ = self_.as_ref();
        let guard = super_.trainer.read().unwrap();
        if let TrainerWrapper::BpeTrainer(ref tr) = *guard {
            Ok(stop_by_to_py(py, &tr.stop_by))
        } else {
            unreachable!()
        }
    }
    #[setter]
    fn set_stop_by(self_: PyRef<Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let sb = stop_by_from_py(value)?;
        let super_ = self_.as_ref();
        if let TrainerWrapper::BpeTrainer(ref mut tr) = *super_.trainer.write().unwrap() {
            tr.stop_by = sb;
        }
        Ok(())
    }

    #[getter]
    fn get_track_ll(self_: PyRef<Self>) -> bool {
        getter!(self_, BpeTrainer, track_ll)
    }
    #[setter]
    fn set_track_ll(self_: PyRef<Self>, flag: bool) {
        setter!(self_, BpeTrainer, track_ll, flag);
    }

    // ----- telemetry toggles (runtime, optional) -----
    // Backward compatibility: 'trace_merges' now maps to 'track_ll'
    #[getter]
    fn get_trace_merges(self_: PyRef<Self>) -> bool {
        getter!(self_, BpeTrainer, track_ll)
    }
    #[setter]
    fn set_trace_merges(self_: PyRef<Self>, flag: bool) {
        setter!(self_, BpeTrainer, track_ll, flag);
    }

    // Snapshot knobs are Option<usize> in Rust; expose as int with 0 meaning None.
    #[getter]
    fn get_score_snapshot_every(self_: PyRef<Self>) -> usize {
        let super_ = self_.as_ref();
        if let TrainerWrapper::BpeTrainer(ref tr) = *super_.trainer.read().unwrap() {
            tr.score_snapshot_every.unwrap_or(0)
        } else {
            unreachable!()
        }
    }
    #[setter]
    fn set_score_snapshot_every(self_: PyRef<Self>, every: usize) {
        let super_ = self_.as_ref();
        if let TrainerWrapper::BpeTrainer(ref mut tr) = *super_.trainer.write().unwrap() {
            tr.score_snapshot_every = if every == 0 { None } else { Some(every) };
        }
    }

    #[getter]
    fn get_score_sample_size(self_: PyRef<Self>) -> usize {
        let super_ = self_.as_ref();
        if let TrainerWrapper::BpeTrainer(ref tr) = *super_.trainer.read().unwrap() {
            tr.score_sample_size.unwrap_or(0)
        } else {
            unreachable!()
        }
    }
    #[setter]
    fn set_score_sample_size(self_: PyRef<Self>, n: usize) {
        let super_ = self_.as_ref();
        if let TrainerWrapper::BpeTrainer(ref mut tr) = *super_.trainer.write().unwrap() {
            tr.score_sample_size = if n == 0 { None } else { Some(n.max(1)) };
        }
    }

    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "\
(self, \
vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], \
limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix=None, end_of_word_suffix=None, \
max_token_length=None, \
score_by='count', stop_by='vocab_size', track_ll=False, \
trace_merges=False, score_snapshot_every=0, score_sample_size=256)"
    )]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::bpe::BpeTrainer::builder();

        let mut score_by: Option<BpeScoreBy> = None;
        let mut stop_by: Option<BpeStopBy> = None;
        let mut track_ll: Option<bool> = None;
        let mut trace_merges: Option<bool> = None;
        let mut score_snapshot_every: Option<usize> = None;
        let mut score_sample_size: Option<usize> = None;

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.downcast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "max_token_length" => builder = builder.max_token_length(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),

                    // new knobs
                    "score_by" => {
                        let s: String = val.extract()?;
                        score_by = Some(score_by_from_py(&s)?);
                    }
                    "stop_by" => {
                        stop_by = Some(stop_by_from_py(&val)?);
                    }
                    "track_ll" => {
                        track_ll = Some(val.extract()?);
                    }

                    // telemetry knobs (runtime optional)
                    "trace_merges" => {
                        trace_merges = Some(val.extract()?);
                    }
                    "score_snapshot_every" => {
                        let every: usize = val.extract()?;
                        score_snapshot_every = Some(every);
                    }
                    "score_sample_size" => {
                        let n: usize = val.extract()?;
                        score_sample_size = Some(n);
                    }

                    _ => println!("Ignored unknown kwargs option {key}"),
                };
            }
        }

        if let Some(sb) = score_by {
            builder = builder.score_by(sb);
        }
        if let Some(st) = stop_by {
            builder = builder.stop_by(st);
        }
        if let Some(flag) = track_ll {
            builder = builder.track_ll(flag);
        }
        if let Some(flag) = trace_merges {
            // Backward-compat: map trace_merges to track_ll in the builder too
            builder = builder.track_ll(flag);
        }
        if let Some(every) = score_snapshot_every {
            builder = builder.score_snapshot_every(Some(every));
        }
        if let Some(n) = score_sample_size {
            builder = builder.score_sample_size(Some(n));
        }

        Ok((PyBpeTrainer {}, builder.build().into()))
    }
}

/// WordPiece trainer
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "WordPieceTrainer")]
pub struct PyWordPieceTrainer {}

#[pymethods]
impl PyWordPieceTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> usize {
        getter!(self_, WordPieceTrainer, vocab_size())
    }
    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: usize) {
        setter!(self_, WordPieceTrainer, @set_vocab_size, vocab_size);
    }

    #[getter]
    fn get_min_frequency(self_: PyRef<Self>) -> u64 {
        getter!(self_, WordPieceTrainer, min_frequency())
    }
    #[setter]
    fn set_min_frequency(self_: PyRef<Self>, freq: u64) {
        setter!(self_, WordPieceTrainer, @set_min_frequency, freq);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, WordPieceTrainer, show_progress())
    }
    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, WordPieceTrainer, @set_show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            WordPieceTrainer,
            special_tokens()
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }
    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            WordPieceTrainer,
            @set_special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_limit_alphabet(self_: PyRef<Self>) -> Option<usize> {
        getter!(self_, WordPieceTrainer, limit_alphabet())
    }
    #[setter]
    fn set_limit_alphabet(self_: PyRef<Self>, limit: Option<usize>) {
        setter!(self_, WordPieceTrainer, @set_limit_alphabet, limit);
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            WordPieceTrainer,
            initial_alphabet().iter().map(|c| c.to_string()).collect()
        )
    }
    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            WordPieceTrainer,
            @set_initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, WordPieceTrainer, continuing_subword_prefix().clone())
    }
    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, prefix: Option<String>) {
        setter!(self_, WordPieceTrainer, @set_continuing_subword_prefix, prefix);
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, WordPieceTrainer, end_of_word_suffix().clone())
    }
    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, suffix: Option<String>) {
        setter!(self_, WordPieceTrainer, @set_end_of_word_suffix, suffix);
    }

    #[new]
    #[pyo3(
        signature = (** kwargs),
        text_signature = "(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix=\"##\", end_of_word_suffix=None)"
    )]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::wordpiece::WordPieceTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.downcast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {key}"),
                };
            }
        }

        Ok((PyWordPieceTrainer {}, builder.build().into()))
    }
}

/// WordLevel trainer
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "WordLevelTrainer")]
pub struct PyWordLevelTrainer {}

#[pymethods]
impl PyWordLevelTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> usize {
        getter!(self_, WordLevelTrainer, vocab_size)
    }
    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: usize) {
        setter!(self_, WordLevelTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_min_frequency(self_: PyRef<Self>) -> u64 {
        getter!(self_, WordLevelTrainer, min_frequency)
    }
    #[setter]
    fn set_min_frequency(self_: PyRef<Self>, freq: u64) {
        setter!(self_, WordLevelTrainer, min_frequency, freq);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, WordLevelTrainer, show_progress)
    }
    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, WordLevelTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            WordLevelTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }
    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            WordLevelTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "special_tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[])"
    )]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::wordlevel::WordLevelTrainer::builder();

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => {
                        builder.vocab_size(val.extract()?);
                    }
                    "min_frequency" => {
                        builder.min_frequency(val.extract()?);
                    }
                    "show_progress" => {
                        builder.show_progress(val.extract()?);
                    }
                    "special_tokens" => {
                        builder.special_tokens(
                            val.downcast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    _ => println!("Ignored unknown kwargs option {key}"),
                }
            }
        }

        Ok((
            PyWordLevelTrainer {},
            builder
                .build()
                .expect("WordLevelTrainerBuilder cannot fail")
                .into(),
        ))
    }
}

/// Trainer capable of training a Unigram model (EM-based)
///
/// Args:
///     vocab_size (:obj:`int`):
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     show_progress (:obj:`bool`):
///         Whether to show progress bars while training.
///
///     special_tokens (:obj:`List[Union[str, AddedToken]]`):
///         A list of special tokens the model should know of.
///
///     initial_alphabet (:obj:`List[str]`):
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
///     shrinking_factor (:obj:`float`):
///         The shrinking factor used at each step of the training to prune the
///         vocabulary.
///
///     unk_token (:obj:`str`):
///         The token used for out-of-vocabulary tokens.
///
///     max_piece_length (:obj:`int`):
///         The maximum length of a given token.
///
///     n_sub_iterations (:obj:`int`):
///         The number of iterations of the EM algorithm to perform before
///         pruning the vocabulary.
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "UnigramTrainer")]
pub struct PyUnigramTrainer {}

#[pymethods]
impl PyUnigramTrainer {
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> u32 {
        getter!(self_, UnigramTrainer, vocab_size)
    }
    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: u32) {
        setter!(self_, UnigramTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, UnigramTrainer, show_progress)
    }
    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, UnigramTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            UnigramTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }
    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            UnigramTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            UnigramTrainer,
            initial_alphabet.iter().map(|c| c.to_string()).collect()
        )
    }
    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            UnigramTrainer,
            initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "(self, vocab_size=8000, show_progress=True, special_tokens=[], initial_alphabet=[], shrinking_factor=0.75, unk_token=None, max_piece_length=16, n_sub_iterations=2, seed_size=1000000, seed_vocab=None)"
    )]
    pub fn new(kwargs: Option<Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::unigram::UnigramTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => builder.vocab_size(val.extract()?),
                    "show_progress" => builder.show_progress(val.extract()?),
                    "n_sub_iterations" => builder.n_sub_iterations(val.extract()?),
                    "shrinking_factor" => builder.shrinking_factor(val.extract()?),
                    "unk_token" => builder.unk_token(val.extract()?),
                    "max_piece_length" => builder.max_piece_length(val.extract()?),
                    "seed_size" => builder.seed_size(val.extract()?),
                    "seed_vocab" => {
                        let sv: Option<Vec<String>> = val.extract()?;
                        builder.seed_vocab(sv)
                    }
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        )
                    }
                    "special_tokens" => builder.special_tokens(
                        val.downcast::<PyList>()?
                            .into_iter()
                            .map(|token| {
                                if let Ok(content) = token.extract::<String>() {
                                    Ok(PyAddedToken::from(content, Some(true)).get_token())
                                } else if let Ok(mut token) =
                                    token.extract::<PyRefMut<PyAddedToken>>()
                                {
                                    token.special = true;
                                    Ok(token.get_token())
                                } else {
                                    Err(exceptions::PyTypeError::new_err(
                                        "special_tokens must be a List[Union[str, AddedToken]]",
                                    ))
                                }
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    ),
                    _ => {
                        println!("Ignored unknown kwargs option {key}");
                        &mut builder
                    }
                };
            }
        }

        let trainer: tokenizers::models::unigram::UnigramTrainer = builder.build().map_err(|e| {
            exceptions::PyException::new_err(format!("Cannot build UnigramTrainer: {e}"))
        })?;
        Ok((PyUnigramTrainer {}, trainer.into()))
    }
}

/// Trainer capable of training a Unigram model by **compression maximization**
/// (greedy deletion with affected-sentences-only resegmentation).
///
/// Args:
///     vocab_size (:obj:`int`):
///         Target final vocabulary size (includes special tokens).
///     show_progress (:obj:`bool`):
///         Whether to show progress bars during training.
///     special_tokens (:obj:`List[Union[str, AddedToken]]`):
///         Special tokens that are kept and never deleted.
///     initial_alphabet (:obj:`List[str]`):
///         Characters to force-include in Σ, even if not present in data.
///     max_piece_length (:obj:`int`):
///         Maximum token length considered when seeding substrings.
///     seed_size (:obj:`int`):
///         Upper bound on the number of seed candidates.
///     seed_vocab (:obj:`List[str]`, optional):
///         If provided, force the initial seed vocabulary to exactly these tokens.
///     prune_ratio (:obj:`float`, optional):
///         Per-iteration batch prune ratio in [0,1] (ceil(prune_ratio * remaining)).
///     min_prune (:obj:`int`, optional):
///         Minimum tokens to prune per iteration (default 1).
///     batch_recompute (:obj:`bool`, optional):
///         If True (default), recompute Δ after each deletion; otherwise delete top-K based on a single pass.
///     byte_fallback (:obj:`bool`, optional):
///         If True, enable Unigram byte fallback in produced model.
///     keep_byte_fallback (:obj:`bool`, optional):
///         If True (default), treat fallback tokens (if any) as non-deletable.
///     rand_scoring (:obj:`bool`, optional):
///         If True, use random sampling to estimate token removal cost (rand_compression method).
///     rand_sample_size (:obj:`int`, optional):
///         Number of spans to sample per token for rand_scoring (default 100).
///     use_expected_counts (:obj:`bool`, optional):
///         If True, use forward-backward expected counts instead of Viterbi hard counts.
///         This considers all possible segmentations weighted by their length.
///     temperature (:obj:`float`, optional):
///         Temperature for expected counts (default 1.0). Controls sharpness of length preference.
///         T→0: approaches Viterbi, T=1: standard expected counts, T→∞: uniform weighting.
#[pyclass(extends=PyTrainer, module = "tokenizers.trainers", name = "CompressionTrainer")]
pub struct PyCompressionTrainer {}

#[pymethods]
impl PyCompressionTrainer {
    // ---- Basic getters/setters wired with the standard macros ----
    #[getter]
    fn get_vocab_size(self_: PyRef<Self>) -> u32 {
        getter!(self_, CompressionTrainer, vocab_size)
    }

    #[setter]
    fn set_vocab_size(self_: PyRef<Self>, vocab_size: u32) {
        setter!(self_, CompressionTrainer, vocab_size, vocab_size);
    }

    #[getter]
    fn get_show_progress(self_: PyRef<Self>) -> bool {
        getter!(self_, CompressionTrainer, show_progress)
    }

    #[setter]
    fn set_show_progress(self_: PyRef<Self>, show_progress: bool) {
        setter!(self_, CompressionTrainer, show_progress, show_progress);
    }

    #[getter]
    fn get_max_piece_length(self_: PyRef<Self>) -> usize {
        getter!(self_, CompressionTrainer, max_piece_length)
    }

    #[setter]
    fn set_max_piece_length(self_: PyRef<Self>, v: usize) {
        setter!(self_, CompressionTrainer, max_piece_length, v);
    }

    #[getter]
    fn get_seed_size(self_: PyRef<Self>) -> usize {
        getter!(self_, CompressionTrainer, seed_size)
    }

    #[setter]
    fn set_seed_size(self_: PyRef<Self>, v: usize) {
        setter!(self_, CompressionTrainer, seed_size, v);
    }

    #[getter]
    fn get_prune_ratio(self_: PyRef<Self>) -> f32 {
        getter!(self_, CompressionTrainer, prune_ratio)
    }

    #[setter]
    fn set_prune_ratio(self_: PyRef<Self>, r: f32) {
        setter!(self_, CompressionTrainer, prune_ratio, r);
    }

    #[getter]
    fn get_min_prune(self_: PyRef<Self>) -> usize {
        getter!(self_, CompressionTrainer, min_prune)
    }

    #[setter]
    fn set_min_prune(self_: PyRef<Self>, k: usize) {
        setter!(self_, CompressionTrainer, min_prune, k);
    }

    #[getter]
    fn get_batch_recompute(self_: PyRef<Self>) -> bool {
        getter!(self_, CompressionTrainer, batch_recompute)
    }

    #[setter]
    fn set_batch_recompute(self_: PyRef<Self>, v: bool) {
        setter!(self_, CompressionTrainer, batch_recompute, v);
    }

    #[getter]
    fn get_byte_fallback(self_: PyRef<Self>) -> bool {
        getter!(self_, CompressionTrainer, byte_fallback)
    }

    #[setter]
    fn set_byte_fallback(self_: PyRef<Self>, v: bool) {
        setter!(self_, CompressionTrainer, byte_fallback, v);
    }

    #[getter]
    fn get_keep_byte_fallback(self_: PyRef<Self>) -> bool {
        getter!(self_, CompressionTrainer, keep_byte_fallback)
    }

    #[setter]
    fn set_keep_byte_fallback(self_: PyRef<Self>, v: bool) {
        setter!(self_, CompressionTrainer, keep_byte_fallback, v);
    }

    #[getter]
    fn get_initial_alphabet(self_: PyRef<Self>) -> Vec<String> {
        getter!(
            self_,
            CompressionTrainer,
            initial_alphabet.iter().map(|c| c.to_string()).collect()
        )
    }

    #[setter]
    fn set_initial_alphabet(self_: PyRef<Self>, alphabet: Vec<char>) {
        setter!(
            self_,
            CompressionTrainer,
            initial_alphabet,
            alphabet.into_iter().collect()
        );
    }

    #[getter]
    fn get_special_tokens(self_: PyRef<Self>) -> Vec<PyAddedToken> {
        getter!(
            self_,
            CompressionTrainer,
            special_tokens
                .iter()
                .map(|tok| tok.clone().into())
                .collect()
        )
    }

    #[setter]
    fn set_special_tokens(self_: PyRef<Self>, special_tokens: &Bound<'_, PyList>) -> PyResult<()> {
        setter!(
            self_,
            CompressionTrainer,
            special_tokens,
            special_tokens
                .into_iter()
                .map(|token| {
                    if let Ok(content) = token.extract::<String>() {
                        Ok(tk::tokenizer::AddedToken::from(content, true))
                    } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                        token.special = true;
                        Ok(token.get_token())
                    } else {
                        Err(exceptions::PyTypeError::new_err(
                            "Special tokens must be a List[Union[str, AddedToken]]",
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()?
        );
        Ok(())
    }

    #[getter]
    fn get_seed_vocab(self_: PyRef<Self>) -> Option<Vec<String>> {
        getter!(self_, CompressionTrainer, seed_vocab.clone())
    }

    #[setter]
    fn set_seed_vocab(self_: PyRef<Self>, sv: Option<Vec<String>>) {
        setter!(self_, CompressionTrainer, seed_vocab, sv);
    }

    #[getter]
    fn get_rand_scoring(self_: PyRef<Self>) -> bool {
        getter!(self_, CompressionTrainer, rand_scoring)
    }

    #[setter]
    fn set_rand_scoring(self_: PyRef<Self>, v: bool) {
        setter!(self_, CompressionTrainer, rand_scoring, v);
    }

    #[getter]
    fn get_rand_sample_size(self_: PyRef<Self>) -> usize {
        getter!(self_, CompressionTrainer, rand_sample_size)
    }

    #[setter]
    fn set_rand_sample_size(self_: PyRef<Self>, v: usize) {
        setter!(self_, CompressionTrainer, rand_sample_size, v);
    }

    #[getter]
    fn get_use_expected_counts(self_: PyRef<Self>) -> bool {
        getter!(self_, CompressionTrainer, use_expected_counts)
    }

    #[setter]
    fn set_use_expected_counts(self_: PyRef<Self>, v: bool) {
        setter!(self_, CompressionTrainer, use_expected_counts, v);
    }

    #[getter]
    fn get_temperature(self_: PyRef<Self>) -> f64 {
        getter!(self_, CompressionTrainer, temperature)
    }

    #[setter]
    fn set_temperature(self_: PyRef<Self>, v: f64) {
        setter!(self_, CompressionTrainer, temperature, v);
    }

    // ---- Python constructor ----
    #[new]
    #[pyo3(
        signature = (**kwargs),
        text_signature = "(self, vocab_size=8000, show_progress=True, special_tokens=[], initial_alphabet=[], max_piece_length=16, seed_size=1000000, seed_vocab=None, prune_ratio=0.0, min_prune=1, batch_recompute=True, byte_fallback=False, keep_byte_fallback=True, rand_scoring=False, rand_sample_size=100, use_expected_counts=False, temperature=1.0)"
    )]
    pub fn new(kwargs: Option<Bound<'_, PyDict>>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::unigram::CompressionTrainer::builder();

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "vocab_size" => {
                        builder.vocab_size(val.extract()?);
                    }
                    "show_progress" => {
                        builder.show_progress(val.extract()?);
                    }
                    "max_piece_length" => {
                        builder.max_piece_length(val.extract()?);
                    }
                    "seed_size" => {
                        builder.seed_size(val.extract()?);
                    }
                    "prune_ratio" => {
                        builder.prune_ratio(val.extract()?);
                    }
                    "min_prune" => {
                        builder.min_prune(val.extract()?);
                    }
                    "batch_recompute" => {
                        builder.batch_recompute(val.extract()?);
                    }
                    "byte_fallback" => {
                        builder.byte_fallback(val.extract()?);
                    }
                    "keep_byte_fallback" => {
                        builder.keep_byte_fallback(val.extract()?);
                    }
                    "rand_scoring" => {
                        builder.rand_scoring(val.extract()?);
                    }
                    "rand_sample_size" => {
                        builder.rand_sample_size(val.extract()?);
                    }
                    "use_expected_counts" => {
                        builder.use_expected_counts(val.extract()?);
                    }
                    "temperature" => {
                        builder.temperature(val.extract()?);
                    }
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .filter_map(|s| s.chars().next())
                                .collect(),
                        );
                    }
                    "special_tokens" => {
                        builder.special_tokens(
                            val.downcast::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.special = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    // Exact seed vocab from Python
                    "seed_vocab" => {
                        let sv: Option<Vec<String>> = val.extract()?;
                        builder.seed_vocab(sv);
                    }
                    _ => println!("Ignored unknown kwargs option {key}"),
                }
            }
        }

        // Build the Rust trainer and wrap it in PyTrainer using the generic `From` impl
        let trainer: tokenizers::models::unigram::CompressionTrainer =
            builder.build().map_err(|e| {
                exceptions::PyException::new_err(format!(
                    "Cannot build CompressionTrainer: {e}"
                ))
            })?;
        Ok((PyCompressionTrainer {}, trainer.into()))
    }
}

/// Trainers Module
#[pymodule]
pub fn trainers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyBpeTrainer>()?;
    m.add_class::<PyWordPieceTrainer>()?;
    m.add_class::<PyWordLevelTrainer>()?;
    m.add_class::<PyUnigramTrainer>()?;
    // NEW: export the compression-based trainer in the Python API
    m.add_class::<PyCompressionTrainer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk::models::bpe::trainer::BpeTrainer;

    #[test]
    fn get_subtype() {
        Python::with_gil(|py| {
            let py_trainer = PyTrainer::new(Arc::new(RwLock::new(BpeTrainer::default().into())));
            let py_bpe = py_trainer.get_as_subtype(py).unwrap();
            assert_eq!("BpeTrainer", py_bpe.bind(py).get_type().qualname().unwrap());
        })
    }
}