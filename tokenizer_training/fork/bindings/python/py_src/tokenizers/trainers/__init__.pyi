# Generated content DO NOT EDIT
class Trainer:
    """
    Base class for all trainers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Trainer will return an instance of this class when instantiated.
    """

class BpeTrainer(Trainer):
    """
    Trainer capable of training a BPE model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.

        max_token_length (:obj:`int`, `optional`):
            Prevents creating tokens longer than the specified size.
            This can help with reducing polluting your vocabulary with
            highly repetitive tokens like `======` for wikipedia

    """
    def __init__(
        self,
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        limit_alphabet=None,
        initial_alphabet=[],
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
        max_token_length=None,
        words={},
    ):
        pass


class UnigramTrainer(Trainer):
    """
    Trainer capable of training a Unigram model

    Args:
        vocab_size (:obj:`int`):
            The size of the final vocabulary, including all tokens and alphabet.

        show_progress (:obj:`bool`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.

        initial_alphabet (:obj:`List[str]`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        shrinking_factor (:obj:`float`):
            The shrinking factor used at each step of the training to prune the
            vocabulary.

        unk_token (:obj:`str`):
            The token used for out-of-vocabulary tokens.

        max_piece_length (:obj:`int`):
            The maximum length of a given token.

        n_sub_iterations (:obj:`int`):
            The number of iterations of the EM algorithm to perform before
            pruning the vocabulary.

        seed_size (:obj:`int`, `optional`):
            Upper bound on the number of seed candidates considered when initializing
            Unigram (mirrors the Rust builder).
    """
    def __init__(
        self,
        vocab_size=8000,
        show_progress=True,
        special_tokens=[],
        initial_alphabet=[],
        shrinking_factor=0.75,
        unk_token=None,
        max_piece_length=16,
        n_sub_iterations=2,
        seed_size=None,
    ):
        pass


class WordLevelTrainer(Trainer):
    """
    Trainer capable of training a WorldLevel model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.
    """
    def __init__(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[]):
        pass


class WordPieceTrainer(Trainer):
    """
    Trainer capable of training a WordPiece model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.
    """
    def __init__(
        self,
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        limit_alphabet=None,
        initial_alphabet=[],
        continuing_subword_prefix="##",
        end_of_word_suffix=None,
    ):
        pass


class CompressionTrainer(Trainer):
    """
    Greedy compression-based trainer for the Unigram model.

    This trainer minimizes the total number of tokens under **unit-cost decoding**
    by iteratively deleting the token `t` that minimizes:

        ΔL(t) = c[t] * ( d[t] - 1 )

    where `c[t]` is the corpus count of `t` in the best (unit-cost) segmentations,
    and `d[t]` is the shortest decomposition length of the string of `t`
    using `V \\ {t}`. After deleting a token, only the sentences that used it
    are re-segmented, and only the impacted decomposition lengths are recomputed.

    Args:
        vocab_size (:obj:`int`):
            Target final vocabulary size (includes special tokens).

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            Special tokens to prepend in the vocabulary (kept; never deleted).

        initial_alphabet (:obj:`List[str]`, `optional`):
            Characters to force-include in Σ even if not present in the data.
            If a string has more than one character, only the first one is kept.

        max_piece_length (:obj:`int`, `optional`):
            Maximum token length considered when seeding substrings.

        seed_size (:obj:`int`, `optional`):
            Upper bound on the number of seed candidates (same default as UnigramTrainer).

        seed_vocab (:obj:`List[str]`, `optional`):
            If provided, training starts **exactly** from this vocabulary (scores forced to -1.0).
            You must include single-character tokens yourself if you want full Σ coverage.
    """
    def __init__(
        self,
        vocab_size=8000,
        show_progress=True,
        special_tokens=[],
        initial_alphabet=[],
        max_piece_length=16,
        seed_size=1000000,
        seed_vocab=None,
    ):
        pass